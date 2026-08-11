"""
indexer.py
──────────
Handles:
  - Downloading single patent PDFs from GCS
  - Uploading PDFs to Gemini Files API and extracting full text
  - Extracting filing/grant dates from cover page (in parallel with text extraction)
    → Supports scanned/image-only PDFs via OCR-aware Gemini Vision prompts
    → 3-tier fallback: vision → text extraction → OCR-focused vision
  - Chunking text and generating embeddings
  - Storing chunks + sentinel records in ChromaDB
    → Dates stored in EVERY chunk's metadata and the sentinel upfront.
      No backfill step needed — copies automatically carry dates.
  - Cross-collection deduplication (copy instead of re-index)
"""

import asyncio
import hashlib
import json
import random
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types
import chromadb

from .gcs_lister import get_gcs_client, GCS_BUCKET_NAME

# ─────────────────────────────────────────────
# Gemini + ChromaDB clients
# ─────────────────────────────────────────────

import os
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key)

CHROMA_DB_PATH = str(Path(__file__).parent / "chroma_patent_db")
chroma_client  = chromadb.PersistentClient(path=CHROMA_DB_PATH)
print(f"[CHROMADB] Initialized at: {CHROMA_DB_PATH}")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHUNK_SIZE_CHARS           = 2000
OVERLAP_CHARS              = 400
_MAX_UPLOAD_RETRIES        = 3
_MAX_EMBED_RETRIES         = 3
_GEMINI_FILE_SIZE_LIMIT_MB = 20

# Cover page render DPI — higher values produce sharper images for
# Gemini Vision to read small-font fields like "(22) Filed:" and
# "(45) Date of Patent:".  150 was often too low for dense USPTO
# cover pages; 250 works reliably.
COVER_PAGE_DPI = int(os.getenv("COVER_PAGE_DPI", "300"))

DATE_EXTRACTION_PROMPT = """You are a patent document parser. This image is the cover page of a patent or patent application (it may be a scanned image with no embedded text — use OCR to read it).

Extract ONLY these two dates:
1. Filing date: look for ANY of these labels (in any language):
   - "(22) Filed:" (US patents)
   - "(22) International Filing Date:" (WO/PCT applications)
   - "(22) Date de dépôt international:" (French WO)
   - "(22) Fecha de presentación:" or "Fecha de depósito:" (Spanish/MX patents)
   - "(22) Data de Depósito:" or "Data de apresentação:" (Brazilian/PT patents)
   - "Filing Date:", "Date Filed:", "PCT Filed:"
   - Any field labelled with INID code (22)
2. Grant/Publication date: look for ANY of these labels (in any language):
   - "(45) Date of Patent:" (US granted patents)
   - "(43) International Publication Date:" (WO/PCT applications)
   - "(43) Date de la publication internationale:" (French WO)
   - "(43) Fecha de publicación internacional:" (Spanish WO/MX)
   - "(43) Data da Publicação Internacional:" (Brazilian/PT WO)
   - "(45) Дата публикации:" (Eurasian/Russian EA patents)
   - "(43) Дата международной публикации:" (Russian WO)
   - "Grant Date:", "Published:", "Publication Date:", "Data de Publicação:"
   - Any field labelled with INID code (43) or (45)

Rules:
- Return ONLY the dates for THIS patent/application (not cited prior art references)
- For WO/PCT, EP, BR, MX, EA, JP applications: use the publication date as grant_date (or null if not found)
- If a date is missing or unclear -> use null
- Convert any date format (e.g. "Dec. 14, 2023", "14.12.2023", "14 December 2023", "14.12.2023", "2023.12.14") to YYYY-MM-DD format

Return ONLY valid JSON with no markdown, no explanation:
{
  "filing_date": "YYYY-MM-DD or null",
  "grant_date":  "YYYY-MM-DD or null"
}
"""

# Fallback prompt: extract dates from raw text of the cover page
DATE_EXTRACTION_TEXT_PROMPT = """You are a patent document parser. Extract ONLY the filing date and grant/publication date from the text below.

Look for (in any language):
- Filing date (INID code 22): "(22) Filed:", "(22) International Filing Date:", "Filing Date:", "Date Filed:", "PCT Filed:", "Fecha de presentación:", "Data de Depósito:", "Дата подачи:"
- Grant/Publication date (INID code 43 or 45): "(45) Date of Patent:", "(43) International Publication Date:", "Grant Date:", "Published:", "Publication Date:", "Fecha de publicación:", "Data de Publicação:", "Дата публикации:"

Rules:
- Return ONLY the dates for THIS patent/application (not cited prior art references or foreign patent documents)
- For WO/PCT, EP, BR, MX, EA, JP applications: use the publication date as grant_date (or null if not found)
- If a date is missing or unclear -> use null
- Format all dates as YYYY-MM-DD

Return ONLY valid JSON with no markdown, no explanation:
{
  "filing_date": "YYYY-MM-DD or null",
  "grant_date":  "YYYY-MM-DD or null"
}

TEXT:
"""

# OCR-focused prompt for scanned/image-only patents (used as final fallback)
_OCR_DATE_PROMPT = """This is a scanned image of a patent cover page. Please carefully read ALL text in the image using OCR.

The document may be a US patent, a WO/PCT international application, or another type.

Then extract ONLY these two dates:
1. Filing date — look for:
   - "(22) Filed:" (US patents — usually left side)
   - "(22) International Filing Date:" (WO/PCT — usually near the top)
2. Grant/Publication date — look for:
   - "(45) Date of Patent:" (US patents — usually right side near the top)
   - "(43) International Publication Date:" (WO/PCT — usually near the top)

Convert any dates you find to YYYY-MM-DD format.
Dates may appear as "Dec. 14, 2023", "14 December 2023", "14.12.2023", etc.

Return ONLY valid JSON:
{"filing_date": "YYYY-MM-DD or null", "grant_date": "YYYY-MM-DD or null"}
"""


# ─────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────

def _clean_date(val) -> Optional[str]:
    """
    Normalize a date value from Gemini's JSON response.

    Handles common Gemini quirks:
      - String "null" / "None" / "N/A" → None
      - Empty string → None
      - Validates YYYY-MM-DD format
      - Salvages non-ISO formats like "Dec. 14, 2023"
    """
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.lower() in ("null", "none", "n/a", "unknown", ""):
            return None
        # Validate YYYY-MM-DD pattern
        if re.match(r"^\d{4}-\d{2}-\d{2}$", stripped):
            return stripped
        # Try to salvage partial dates like "Dec. 14, 2023" or "14.12.2023"
        try:
            from datetime import datetime
            for fmt in (
                "%B %d, %Y", "%b. %d, %Y", "%b %d, %Y",
                "%m/%d/%Y", "%d/%m/%Y",
                "%d.%m.%Y",   # European dot-separated: 14.12.2023
                "%Y.%m.%d",   # ISO-ish: 2023.12.14
                "%d-%m-%Y",   # dash-separated: 14-12-2023
            ):
                try:
                    dt = datetime.strptime(stripped, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception:
            pass
        print(f"[WARN] Could not parse date value: '{stripped}'")
        return None
    return None


def _has_valid_dates(meta: dict) -> bool:
    """Check whether metadata has at least one non-empty date."""
    filing = meta.get("filing_date", "")
    grant  = meta.get("grant_date", "")
    return bool(filing and filing not in ("", "null", "None")) or \
           bool(grant and grant not in ("", "null", "None"))


def _parse_gemini_date_response(raw: str) -> Dict:
    """
    Parse a Gemini response that should contain date JSON.

    Handles:
      - Clean JSON
      - JSON wrapped in markdown code fences
      - Empty responses (returns nulls)
    """
    if not raw or not raw.strip():
        return {"filing_date": None, "grant_date": None}

    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

    # Try to extract JSON from the response even if there's surrounding text
    # Use re.DOTALL so { ... } matches across newlines
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        dates = json.loads(cleaned)
        return {
            "filing_date": _clean_date(dates.get("filing_date")),
            "grant_date":  _clean_date(dates.get("grant_date")),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"filing_date": None, "grant_date": None}


async def _call_gemini_for_dates(contents: list, filename: str) -> Dict:
    """
    Call Gemini to extract dates, with automatic retry.

    First tries with response_mime_type="application/json" (structured output).
    If that returns empty, retries WITHOUT the constraint so Gemini can
    produce free-form text that we parse manually.
    """
    # Attempt 1: structured JSON output
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=256,
            ),
        )
        raw = response.text.strip() if response.text else ""
        if raw:
            dates = _parse_gemini_date_response(raw)
            if dates.get("filing_date") or dates.get("grant_date"):
                return dates
        print(f"  [DATES] Structured JSON response was empty for {filename} — retrying without constraint")
    except Exception as e:
        print(f"  [DATES] Structured call failed for {filename}: {e} — retrying without constraint")

    # Attempt 2: free-form (no response_mime_type constraint)
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
            ),
        )
        raw = response.text.strip() if response.text else ""
        if raw:
            dates = _parse_gemini_date_response(raw)
            return dates
    except Exception as e:
        print(f"  [DATES] Free-form call also failed for {filename}: {e}")

    return {"filing_date": None, "grant_date": None}


# ─────────────────────────────────────────────
# GCS download
# ─────────────────────────────────────────────

def download_single_patent_pdf(blob_name: str, filename: str, drug_name: str) -> Optional[dict]:
    if not GCS_BUCKET_NAME:
        print("[GCS] GCS_BUCKET_NAME not set — cannot download")
        return None
    try:
        client     = get_gcs_client()
        bucket     = client.bucket(GCS_BUCKET_NAME)
        blob       = bucket.blob(blob_name)
        tmp_dir    = Path(tempfile.mkdtemp(prefix=f"patents_{drug_name}_"))
        # filename may be a subfolder-relative path (e.g. "US/patent.pdf");
        # use only the basename for the local file to avoid missing-dir errors.
        local_path = tmp_dir / Path(filename).name
        blob.download_to_filename(str(local_path))
        print(f"[GCS] Downloaded {filename}")
        return {"filename": filename, "path": str(local_path), "tmp_dir": str(tmp_dir)}
    except Exception as e:
        print(f"[GCS] Failed to download {filename}: {e}")
        return None


# ─────────────────────────────────────────────
# Gemini file upload + text extraction
# ─────────────────────────────────────────────

async def upload_pdf_to_gemini(file_path: str) -> Optional[object]:
    path = Path(file_path)
    print(f"[UPLOAD] Uploading {path.name}...")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[UPLOAD] File size: {file_size_mb:.1f} MB")
    if file_size_mb > _GEMINI_FILE_SIZE_LIMIT_MB:
        print(f"[ERROR] {path.name} is {file_size_mb:.1f} MB — exceeds {_GEMINI_FILE_SIZE_LIMIT_MB} MB limit.")
        return None

    loop = asyncio.get_running_loop()

    for attempt in range(1, _MAX_UPLOAD_RETRIES + 1):
        try:
            uploaded_file = await loop.run_in_executor(
                None,
                lambda: gemini_client.files.upload(
                    file=file_path,
                    config=dict(mime_type="application/pdf"),
                )
            )

            max_wait, wait_time = 60, 0
            while uploaded_file.state == "PROCESSING" and wait_time < max_wait:
                await asyncio.sleep(2 + random.uniform(0, 0.5))
                _file_name = uploaded_file.name  # capture by value to avoid closure bug
                uploaded_file = await loop.run_in_executor(
                    None, lambda n=_file_name: gemini_client.files.get(name=n)
                )
                wait_time += 2
                print(f"[UPLOAD] Processing... ({wait_time}s)")

            if uploaded_file.state == "FAILED":
                print(f"[ERROR] Gemini failed to process {path.name}")
                return None

            print(f"[UPLOAD] Ready: {path.name}")
            return uploaded_file

        except Exception as e:
            if attempt == _MAX_UPLOAD_RETRIES:
                print(f"[ERROR] Upload failed for {path.name} after {_MAX_UPLOAD_RETRIES} attempts: {e}")
                return None
            backoff = (2 ** attempt) + random.uniform(0, 1)
            print(f"[UPLOAD] Attempt {attempt} failed: {e} — retrying in {backoff:.1f}s")
            await asyncio.sleep(backoff)

    return None


async def extract_text_via_gemini(uploaded_file: object, filename: str) -> Optional[str]:
    """
    Extract full plain text from uploaded PDF via Gemini.
    max_output_tokens=65536 ensures claims section at end of patent is captured.
    """
    print(f"[TEXT EXTRACTION] Extracting text from {filename}...")
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                uploaded_file,
                "Extract ALL text from this patent document exactly as it appears. "
                "Include every section: cover page, patent number, all dates, "
                "inventors, assignee, claims, description, abstract. "
                "Return only plain text with no commentary or formatting.",
            ],
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=65536),
        )

        try:
            finish_reason = response.candidates[0].finish_reason
            if str(finish_reason) in ("MAX_TOKENS", "2"):
                print(
                    f"[WARNING] Text extraction for {filename} hit MAX_TOKENS — "
                    "document may be partially indexed. Consider splitting the PDF."
                )
        except (IndexError, AttributeError):
            pass

        text = response.text
        print(f"[TEXT EXTRACTION] Extracted {len(text)} characters from {filename}")
        return text

    except Exception as e:
        print(f"[ERROR] Text extraction failed for {filename}: {e}")
        return None


async def cleanup_uploaded_file(uploaded_file: object):
    loop = asyncio.get_running_loop()
    _file_name = uploaded_file.name  # capture by value
    try:
        await loop.run_in_executor(
            None, lambda: gemini_client.files.delete(name=_file_name)
        )
        print(f"[UPLOAD] Cleaned up {_file_name}")
    except Exception as e:
        print(f"[WARNING] Could not clean up {_file_name}: {e}")


# ─────────────────────────────────────────────
# Date extraction
# ─────────────────────────────────────────────

def render_cover_page_as_png(file_path: str, dpi: int = COVER_PAGE_DPI) -> Optional[bytes]:
    """
    Render page 1 of a PDF as a PNG byte string using PyMuPDF.

    Raises ImportError explicitly if pymupdf is missing so the caller
    gets a clear signal (instead of silently returning None).
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for cover-page date extraction. "
            "Install it with:  pip install pymupdf --break-system-packages"
        )

    doc       = fitz.open(file_path)
    page      = doc[0]
    mat       = fitz.Matrix(dpi / 72, dpi / 72)
    pix       = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def render_cover_pages_as_pngs(
    file_path: str, dpi: int = COVER_PAGE_DPI, max_pages: int = 2
) -> List[bytes]:
    """
    Render first N pages of a PDF as PNG byte strings using PyMuPDF.

    Returns a list of PNG byte strings (one per page).
    Some patents split cover info (e.g., filing date on page 2),
    so we render multiple pages to improve extraction reliability.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for cover-page date extraction. "
            "Install it with:  pip install pymupdf --break-system-packages"
        )

    doc = fitz.open(file_path)
    pages_to_render = min(max_pages, len(doc))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pngs = []
    for i in range(pages_to_render):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        pngs.append(pix.tobytes("png"))
    doc.close()
    return pngs


async def extract_dates_from_pdf(file_path: str, filename: str) -> Dict:
    """
    Extract filing/grant dates from a patent PDF.

    Strategy (4-tier fallback):
      1. Upload the raw PDF to Gemini and ask it to extract dates directly.
         This is the most reliable approach because Gemini can natively
         parse PDFs — including scanned/image-only ones with unselectable
         text. It sees the original vector paths and embedded fonts that
         PyMuPDF's get_text() cannot decode.
      2. If the PDF upload fails or returns no dates, render the first 2
         cover pages as high-DPI PNGs and send them to Gemini Vision.
      3. If vision fails and PyMuPDF can extract text → text-based prompt.
      4. If PDF is image-only (no text layer) → OCR-focused vision prompt
         with the rendered PNGs.
    """
    if not file_path or not Path(file_path).exists():
        print(f"[DATE EXTRACTION] No local file for {filename} — dates will be null")
        return {"filing_date": None, "grant_date": None}

    loop = asyncio.get_running_loop()

    # ── Step 1: Native PDF upload to Gemini ──────────────────────────
    # This handles scanned/image-only PDFs where text is unselectable
    # because Gemini processes the PDF natively (not via rendered image).
    print(f"[DATE EXTRACTION] Trying native PDF upload for {filename}...")
    try:
        pdf_bytes = Path(file_path).read_bytes()
        # Only send first ~2MB to keep it fast (cover pages are at the start)
        max_bytes = 2 * 1024 * 1024
        if len(pdf_bytes) > max_bytes:
            # Use PyMuPDF to extract just the first 2 pages into a new PDF
            try:
                import fitz
                doc = fitz.open(file_path)
                new_doc = fitz.open()
                for i in range(min(2, len(doc))):
                    new_doc.insert_pdf(doc, from_page=i, to_page=i)
                pdf_bytes = new_doc.tobytes()
                new_doc.close()
                doc.close()
                print(f"[DATE EXTRACTION] Trimmed PDF to first 2 pages ({len(pdf_bytes)} bytes)")
            except ImportError:
                # pymupdf not available — send full PDF if under limit
                if len(pdf_bytes) > 5 * 1024 * 1024:
                    print(f"[DATE EXTRACTION] PDF too large for native upload without pymupdf, skipping")
                    pdf_bytes = None
            except Exception as e:
                print(f"[DATE EXTRACTION] PDF trim failed: {e}, using full PDF")

        if pdf_bytes:
            dates = await _call_gemini_for_dates(
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    DATE_EXTRACTION_PROMPT,
                ],
                filename=filename,
            )
            if dates.get("filing_date") or dates.get("grant_date"):
                print(f"[DATE EXTRACTION] {filename} -> Filed: {dates['filing_date']} | Granted: {dates['grant_date']} (native PDF)")
                return dates
            print(f"[DATE EXTRACTION] Native PDF upload returned no dates for {filename} — trying vision fallback")
    except Exception as e:
        print(f"[DATE EXTRACTION] Native PDF upload failed for {filename}: {e} — trying vision fallback")

    # ── Step 2: Multi-page vision-based extraction ───────────────────
    print(f"[DATE EXTRACTION] Rendering cover pages of {filename} at {COVER_PAGE_DPI} DPI...")
    png_list: List[bytes] = []
    try:
        png_list = await loop.run_in_executor(
            None, render_cover_pages_as_pngs, file_path
        )
    except ImportError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[DATE EXTRACTION] Cover page render failed for {filename}: {e}")

    if png_list:
        # Build contents with all rendered pages
        contents = []
        for i, png_bytes in enumerate(png_list):
            contents.append(
                types.Part.from_bytes(data=png_bytes, mime_type="image/png")
            )
        contents.append(DATE_EXTRACTION_PROMPT)

        dates = await _call_gemini_for_dates(
            contents=contents,
            filename=filename,
        )
        if dates.get("filing_date") or dates.get("grant_date"):
            print(f"[DATE EXTRACTION] {filename} -> Filed: {dates['filing_date']} | Granted: {dates['grant_date']} (vision)")
            return dates
        print(f"[DATE EXTRACTION] Vision returned no dates for {filename} — trying text/OCR fallback")

    # ── Step 3 & 4: Text extraction then OCR ─────────────────────────
    return await _extract_dates_fallback(file_path, filename, png_list)


async def _extract_dates_fallback(
    file_path: str, filename: str, png_list: Optional[List[bytes]] = None
) -> Dict:
    """
    Fallback date extraction for when the primary methods fail.

    Strategy (in order):
      1. Try pymupdf text extraction from first 2 pages → Gemini text prompt
         (works for text-layer PDFs, fast and cheap)
      2. If PDF is image-only (no text layer), use OCR-via-Gemini-Vision
         with a more explicit prompt and all rendered pages.
      3. If pymupdf is missing entirely, use OCR-via-vision as well.
    """
    # ── Try text extraction first ──────────────────────────────────
    cover_text = ""
    pymupdf_available = True
    try:
        import fitz
        doc = fitz.open(file_path)
        for page_num in range(min(2, len(doc))):
            cover_text += doc[page_num].get_text("text") + "\n"
        doc.close()
    except ImportError:
        pymupdf_available = False
    except Exception as e:
        print(f"[DATE EXTRACTION] Text extraction from PDF failed for {filename}: {e}")

    if cover_text.strip():
        # Has embedded text — use text-based prompt
        dates = await _call_gemini_for_dates(
            contents=[DATE_EXTRACTION_TEXT_PROMPT + cover_text[:3000]],
            filename=filename,
        )
        if dates.get("filing_date") or dates.get("grant_date"):
            print(f"[DATE EXTRACTION] {filename} -> Filed: {dates['filing_date']} | Granted: {dates['grant_date']} (text fallback)")
            return dates
    else:
        print(f"[DATE EXTRACTION] PDF is image-only (no text layer) — using OCR fallback for {filename}")

    # ── OCR-via-Vision fallback ────────────────────────────────────
    if not png_list:
        if not pymupdf_available:
            print(f"[ERROR] pymupdf not available and no cover image — cannot extract dates for {filename}")
            return {"filing_date": None, "grant_date": None}
        try:
            loop = asyncio.get_running_loop()
            png_list = await loop.run_in_executor(
                None, render_cover_pages_as_pngs, file_path
            )
        except Exception as e:
            print(f"[ERROR] Could not render cover pages for OCR fallback: {e}")
            return {"filing_date": None, "grant_date": None}

    if not png_list:
        return {"filing_date": None, "grant_date": None}

    # Build contents with all rendered pages for OCR
    contents = []
    for png_bytes in png_list:
        contents.append(
            types.Part.from_bytes(data=png_bytes, mime_type="image/png")
        )
    contents.append(_OCR_DATE_PROMPT)

    dates = await _call_gemini_for_dates(
        contents=contents,
        filename=filename,
    )
    print(f"[DATE EXTRACTION] {filename} -> Filed: {dates.get('filing_date')} | Granted: {dates.get('grant_date')} (OCR fallback)")
    return dates


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk_text(
    text:       str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap:    int = OVERLAP_CHARS,
) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    chunks, start = [], 0
    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            bp = max(chunk.rfind("."), chunk.rfind("\n"))
            if bp > chunk_size // 2:
                chunk = chunk[: bp + 1]
                end   = start + bp + 1
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────

async def generate_embeddings(
    texts: List[str],
    model: str = "gemini-embedding-001",
) -> List[List[float]]:
    print(f"[EMBEDDINGS] Generating for {len(texts)} chunks...")
    loop = asyncio.get_running_loop()

    try:
        embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]

            for attempt in range(1, _MAX_EMBED_RETRIES + 1):
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda b=batch: gemini_client.models.embed_content(
                            model=model,
                            contents=b,
                            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                        ),
                    )
                    for emb in result.embeddings:
                        embeddings.append(emb.values)
                    break

                except Exception as e:
                    if attempt == _MAX_EMBED_RETRIES:
                        raise RuntimeError(
                            f"Embedding batch {i}-{i+len(batch)} failed after "
                            f"{_MAX_EMBED_RETRIES} attempts: {e}"
                        ) from e
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[EMBEDDINGS] Attempt {attempt} failed — retrying in {backoff:.1f}s")
                    await asyncio.sleep(backoff)

        print(f"[EMBEDDINGS] Generated {len(embeddings)} embeddings")
        return embeddings

    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        return []


# ─────────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────────

def sanitize_collection_name(drug_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", drug_name.strip())
    safe = re.sub(r"[_\-]{2,}", "_", safe)
    safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
    safe = safe.ljust(3, "x")
    safe = safe[:55]
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
    return f"patents_{safe}"


def get_or_create_collection(drug_name: str):
    name = sanitize_collection_name(drug_name)
    print(f"[CHROMADB] Collection name: {name}")
    try:
        col = chroma_client.get_collection(name=name)
        print(f"[CHROMADB] Using existing: {name}")
    except Exception:
        col = chroma_client.create_collection(
            name=name,
            metadata={"description": f"Patent embeddings for {drug_name}"},
        )
        print(f"[CHROMADB] Created new: {name}")
    return col


def sentinel_exists(collection, filename: str) -> bool:
    try:
        sid    = hashlib.md5(filename.encode()).hexdigest() + "_complete"
        result = collection.get(ids=[sid])
        return bool(result["ids"])
    except Exception:
        return False


def get_dates_from_chromadb(collection, filename: str) -> dict:
    """
    Read filing/grant dates for a patent.

    Priority:
      1. Sentinel record (chunk_index = -1)
      2. First chunk (chunk_index = 0) — fallback, also carries dates
    Both contain dates for all patents indexed after this update.
    """
    file_hash = hashlib.md5(filename.encode()).hexdigest()

    # 1. Sentinel
    try:
        sid    = file_hash + "_complete"
        result = collection.get(ids=[sid], include=["metadatas"])
        if result["metadatas"]:
            meta   = result["metadatas"][0]
            filing = meta.get("filing_date") or None
            grant  = meta.get("grant_date")  or None
            if filing or grant:
                return {"filing_date": filing, "grant_date": grant}
    except Exception as e:
        print(f"[DATES] Sentinel read failed for {filename}: {e}")

    # 2. First chunk fallback
    try:
        result = collection.get(ids=[f"{file_hash}_chunk_0"], include=["metadatas"])
        if result["metadatas"]:
            meta   = result["metadatas"][0]
            filing = meta.get("filing_date") or None
            grant  = meta.get("grant_date")  or None
            if filing or grant:
                print(f"[DATES] Dates read from chunk_0 for {filename}")
                return {"filing_date": filing, "grant_date": grant}
    except Exception as e:
        print(f"[DATES] Chunk-0 read failed for {filename}: {e}")

    return {"filing_date": None, "grant_date": None}


# ─────────────────────────────────────────────
# Core indexing
# ─────────────────────────────────────────────

async def index_text(
    drug_name: str,
    filename:  str,
    text:      str,
    collection,
    dates:     dict = None,
) -> bool:
    """
    Chunk and index a patent into ChromaDB.

    Dates are stored in:
      - EVERY chunk's metadata  -> copies carry dates automatically
      - The sentinel record     -> fast date lookup

    No post-indexing backfill needed.
    """
    file_hash   = hashlib.md5(filename.encode()).hexdigest()
    sentinel_id = f"{file_hash}_complete"

    try:
        if collection.get(ids=[sentinel_id])["ids"]:
            print(f"[INDEXING] Already fully indexed: {filename}")
            return True
    except Exception:
        pass

    try:
        stale = collection.get(where={"filename": filename}, include=["ids"])
        if stale["ids"]:
            print(f"[INDEXING] Incomplete index for {filename} — clearing and re-indexing")
            collection.delete(where={"filename": filename})
    except Exception:
        pass

    chunks = chunk_text(text)
    print(f"[CHUNKING] {filename} -> {len(chunks)} chunks")
    if not chunks:
        return False

    embeddings = await generate_embeddings(chunks)
    if not embeddings:
        return False

    dates       = dates or {}
    filing_date = _clean_date(dates.get("filing_date")) or ""
    grant_date  = _clean_date(dates.get("grant_date"))  or ""

    # Store dates in every chunk so cross-collection copies carry them
    collection.add(
        documents  = chunks,
        embeddings = embeddings,
        metadatas  = [
            {
                "filename":     filename,
                "drug":         drug_name,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "filing_date":  filing_date,
                "grant_date":   grant_date,
            }
            for i in range(len(chunks))
        ],
        ids=[f"{file_hash}_chunk_{i}" for i in range(len(chunks))],
    )

    # Sentinel — completeness guard + dates
    collection.add(
        documents  = ["__index_complete__"],
        embeddings = [[0.0] * len(embeddings[0])],
        metadatas  = [{
            "filename":     filename,
            "drug":         drug_name,
            "chunk_index":  -1,
            "total_chunks": len(chunks),
            "filing_date":  filing_date,
            "grant_date":   grant_date,
        }],
        ids=[sentinel_id],
    )

    print(
        f"[INDEXING] {filename} — {len(chunks)} chunks stored | "
        f"Filed: {filing_date or 'unknown'} | Granted: {grant_date or 'unknown'}"
    )
    return True


# ─────────────────────────────────────────────
# Cross-collection deduplication
# ─────────────────────────────────────────────

def find_in_any_collection(filename: str) -> Optional[str]:
    sid = hashlib.md5(filename.encode()).hexdigest() + "_complete"
    for col in chroma_client.list_collections():
        if not col.name.startswith("patents_"):
            continue
        try:
            existing = chroma_client.get_collection(col.name)
            if existing.get(ids=[sid])["ids"]:
                print(f"[CROSS-CHECK] '{filename}' found in '{col.name}'")
                return col.name
        except Exception:
            continue
    return None


def copy_from_collection(
    filename:         str,
    source_name:      str,
    target_collection,
    target_drug:      str,
) -> bool:
    """
    Copies all chunks + sentinel from source to target collection.
    Dates are embedded in chunk metadata so they transfer automatically.
    """
    try:
        source    = chroma_client.get_collection(source_name)
        file_hash = hashlib.md5(filename.encode()).hexdigest()

        chunks = source.get(
            where={
                "$and": [
                    {"filename":    {"$eq": filename}},
                    {"chunk_index": {"$gte": 0}},
                ]
            },
            include=["documents", "metadatas", "embeddings"],
        )
        if chunks["ids"]:
            target_collection.add(
                documents  = chunks["documents"],
                embeddings = chunks["embeddings"],
                metadatas  = [{**m, "drug": target_drug} for m in chunks["metadatas"]],
                ids        = chunks["ids"],
            )
            print(f"[COPY] {len(chunks['ids'])} chunks -> '{target_collection.name}'")

        sentinel_id = f"{file_hash}_complete"
        sentinel    = source.get(
            ids=[sentinel_id], include=["documents", "metadatas", "embeddings"]
        )
        if sentinel["ids"]:
            target_collection.add(
                documents  = sentinel["documents"],
                embeddings = sentinel["embeddings"],
                metadatas  = [{**sentinel["metadatas"][0], "drug": target_drug}],
                ids        = [sentinel_id],
            )
            print(f"[COPY] Sentinel copied for '{filename}'")

        return True

    except Exception as e:
        print(f"[COPY] Failed to copy '{filename}' from '{source_name}': {e}")
        return False


# ─────────────────────────────────────────────
# Fix dates for already-indexed patents
# ─────────────────────────────────────────────

async def fix_dates_for_file(
    filename:   str,
    file_path:  str,
    collection,
) -> bool:
    """
    Re-extract dates for a single patent and update all its records
    in ChromaDB in-place (no re-embedding needed).

    Returns True if dates were successfully fixed.
    """
    file_hash   = hashlib.md5(filename.encode()).hexdigest()
    sentinel_id = f"{file_hash}_complete"

    # Check if already has valid dates
    try:
        result = collection.get(ids=[sentinel_id], include=["metadatas"])
        if result["metadatas"] and _has_valid_dates(result["metadatas"][0]):
            meta = result["metadatas"][0]
            print(
                f"[FIX-DATES] {filename} already has dates: "
                f"Filed={meta.get('filing_date')} | Granted={meta.get('grant_date')}"
            )
            return True
    except Exception:
        pass

    # Re-extract dates
    dates = await extract_dates_from_pdf(file_path, filename)
    filing = _clean_date(dates.get("filing_date")) or ""
    grant  = _clean_date(dates.get("grant_date"))  or ""

    if not filing and not grant:
        print(f"[FIX-DATES] Still could not extract dates for {filename}")
        return False

    # Update all records in-place
    try:
        all_records = collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        if all_records["ids"]:
            updated_metas = []
            for m in all_records["metadatas"]:
                m["filing_date"] = filing
                m["grant_date"]  = grant
                updated_metas.append(m)

            collection.update(
                ids       = all_records["ids"],
                metadatas = updated_metas,
            )
            print(
                f"[FIX-DATES] {filename}: {len(all_records['ids'])} records updated — "
                f"Filed={filing} | Granted={grant}"
            )
            return True
    except Exception as e:
        print(f"[FIX-DATES] Update failed for {filename}: {e}")

    return False


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

# Maximum number of PDFs processed concurrently by run_indexing.
# Each worker slot covers: GCS download + Gemini upload + text/date extraction
# + ChromaDB write. Raise this if Gemini rate limits allow; lower it to
# reduce peak memory pressure on very large patent sets.
INDEXING_WORKERS = int(os.getenv("INDEXING_WORKERS", "5"))


async def run_indexing(
    drug_name:  str,
    pdf_refs:   List[dict],
    collection,
    reindex:    bool = False,
    on_indexed: Optional[callable] = None,
) -> List[dict]:
    """
    For each PDF ref:
      1. Skip if already indexed AND has valid dates        (no worker slot used)
      2. Copy from another collection if found              (no worker slot used)
      3. If indexed but missing filing date → fix-dates     (worker slot used)
      4. Download → upload → text + dates in parallel → index (worker slot used)

    Steps 3 and 4 run concurrently across up to INDEXING_WORKERS (default 5)
    workers via asyncio.Semaphore. Steps 1 and 2 are fast local/DB operations
    and run without throttling.

    A threading.Lock serialises all ChromaDB writes so concurrent workers
    don't interleave collection.add() calls.

    Dates are stored upfront during indexing.
    No backfill step is needed or performed.

    Args:
        drug_name:  Drug name string
        pdf_refs:   List of {"filename": str, "blob_name": str} from gcs_lister
        collection: ChromaDB collection object
        reindex:    If True, force re-indexing even if sentinel exists
        on_indexed: Optional async callback(filename: str) invoked immediately
                    after each patent is confirmed ready in ChromaDB — whether
                    it was freshly indexed, copied, or already present.
                    Use this to pipeline downstream work (e.g. Step 1 analysis)
                    so it starts per-patent rather than waiting for the full batch.

    Returns:
        List of {"filename": str, "path": str | None, "tmp_dir": str | None}
        in the same order as pdf_refs.
    """
    import shutil
    import threading

    semaphore   = asyncio.Semaphore(INDEXING_WORKERS)
    chroma_lock = threading.Lock()

    print(
        f"\n[INDEXER] Checking index status for {len(pdf_refs)} file(s) "
        f"({INDEXING_WORKERS} parallel workers)..."
    )

    # ── Thread-safe wrappers for ChromaDB writes ──────────────────────────────

    def _locked_index_text_sync(drug_name, filename, text, collection, dates):
        """Run index_text synchronously inside the chroma_lock."""
        # index_text is async; we need a new event loop to run it from a thread.
        # Instead we inline the sync ChromaDB calls here using the lock.
        import hashlib as _hashlib
        from . import indexer as _self  # avoid circular — use module-level helpers

        file_hash   = _hashlib.md5(filename.encode()).hexdigest()
        sentinel_id = f"{file_hash}_complete"

        try:
            if collection.get(ids=[sentinel_id])["ids"]:
                print(f"[INDEXING] Already fully indexed: {filename}")
                return True
        except Exception:
            pass

        try:
            stale = collection.get(where={"filename": filename}, include=["ids"])
            if stale["ids"]:
                print(f"[INDEXING] Incomplete index for {filename} — clearing and re-indexing")
                with chroma_lock:
                    collection.delete(where={"filename": filename})
        except Exception:
            pass

        return None  # signal: caller should proceed with async index_text

    async def _index_text_locked(drug_name, filename, text, collection, dates):
        """Call index_text (which writes to ChromaDB) under chroma_lock."""
        loop = asyncio.get_running_loop()
        # Run the ChromaDB .add() calls inside a thread that holds the lock.
        # index_text itself is async but only awaits generate_embeddings;
        # the actual collection.add() calls are synchronous. We therefore
        # await index_text normally — the lock is acquired just before the
        # add calls via a wrapper that acquires it in the executor.
        # Simpler and correct: run the whole index_text under the lock since
        # generate_embeddings is the slow part and doesn't touch ChromaDB.
        embeddings_done = asyncio.Event()

        async def _run():
            return await index_text(drug_name, filename, text, collection, dates=dates)

        # Acquire lock in executor so we don't block the event loop while waiting
        acquired = await loop.run_in_executor(None, chroma_lock.acquire)
        try:
            result = await _run()
        finally:
            chroma_lock.release()
        return result

    async def _fix_dates_locked(filename, file_path, collection):
        """Call fix_dates_for_file (which writes to ChromaDB) under chroma_lock."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, chroma_lock.acquire)
        try:
            return await fix_dates_for_file(filename, file_path, collection)
        finally:
            chroma_lock.release()

    def _copy_locked(filename, source_col, collection, drug_name):
        """copy_from_collection under chroma_lock (it's synchronous)."""
        with chroma_lock:
            return copy_from_collection(filename, source_col, collection, drug_name)

    # ── Per-file coroutine ────────────────────────────────────────────────────

    async def _process_one(ref: dict) -> dict:
        """
        Classify and process a single PDF ref.
        Returns {"filename": str, "path": str|None, "tmp_dir": str|None}.
        """
        filename = ref["filename"]

        # ── Fast path: already indexed + has dates ────────────────────────
        if not reindex and sentinel_exists(collection, filename):
            existing_dates = get_dates_from_chromadb(collection, filename)
            filing = existing_dates.get("filing_date")
            if filing and filing not in ("", "null", "None"):
                print(f"[SKIP] {filename} — already indexed (Filed: {filing})")
                if on_indexed:
                    await on_indexed(filename)
                return {"filename": filename, "path": None, "tmp_dir": None}

            # Indexed but missing filing date — fix under semaphore
            print(f"[FIX-DATES] {filename} — indexed but missing filing date, re-extracting...")
            async with semaphore:
                loop = asyncio.get_running_loop()
                pf = await loop.run_in_executor(
                    None, download_single_patent_pdf, ref["blob_name"], filename, drug_name
                )
                if pf:
                    try:
                        fixed = await _fix_dates_locked(filename, pf["path"], collection)
                        if fixed:
                            print(f"[FIX-DATES] {filename} — dates updated successfully")
                        else:
                            print(f"[FIX-DATES] {filename} — could not extract dates (will retry on next run)")
                    except Exception as e:
                        print(f"[FIX-DATES] {filename} — error: {e}")
                    finally:
                        if pf.get("tmp_dir"):
                            shutil.rmtree(pf["tmp_dir"], ignore_errors=True)
                else:
                    print(f"[FIX-DATES] {filename} — could not download for date re-extraction")

            if on_indexed:
                await on_indexed(filename)
            return {"filename": filename, "path": None, "tmp_dir": None}

        # ── Fast path: cross-collection copy ─────────────────────────────
        if not reindex:
            source_col = find_in_any_collection(filename)
            if source_col:
                print(f"[COPY] {filename} — found in '{source_col}', copying (dates included)...")
                _copy_locked(filename, source_col, collection, drug_name)
                if on_indexed:
                    await on_indexed(filename)
                return {"filename": filename, "path": None, "tmp_dir": None}

        # ── Full index path (Gemini-heavy — throttled by semaphore) ───────
        async with semaphore:
            print(f"[INDEX] {filename} — downloading...")
            loop = asyncio.get_running_loop()
            pf = await loop.run_in_executor(
                None, download_single_patent_pdf, ref["blob_name"], filename, drug_name
            )
            if not pf:
                print(f"[WARNING] Could not download {filename}")
                return {"filename": filename, "path": None, "tmp_dir": None}

            try:
                uploaded_file = await upload_pdf_to_gemini(pf["path"])
                if not uploaded_file:
                    print(f"[WARNING] Upload failed for {filename}")
                    return {"filename": filename, "path": None, "tmp_dir": None}

                # Text extraction and date extraction run in parallel
                text, dates = await asyncio.gather(
                    extract_text_via_gemini(uploaded_file, filename),
                    extract_dates_from_pdf(pf["path"], filename),
                )

                if text:
                    await _index_text_locked(drug_name, filename, text, collection, dates)
                    print(
                        f"[INDEX] {filename} — stored | "
                        f"Filed: {(dates or {}).get('filing_date') or 'unknown'} | "
                        f"Granted: {(dates or {}).get('grant_date') or 'unknown'}"
                    )
                    if on_indexed:
                        await on_indexed(filename)
                else:
                    print(f"[WARNING] No text extracted from {filename}")

                await cleanup_uploaded_file(uploaded_file)
                await asyncio.sleep(1 + random.uniform(0, 0.5))

                return pf

            except Exception as e:
                print(f"[ERROR] Processing failed for {filename}: {e}")
                return {"filename": filename, "path": None, "tmp_dir": None}

            finally:
                if pf.get("tmp_dir"):
                    shutil.rmtree(pf["tmp_dir"], ignore_errors=True)
                    print(f"[GCS] Cleaned up temp dir for {filename}")
                    pf["tmp_dir"] = None

    # ── Dispatch all files concurrently, collect results in order ─────────────
    results: List[dict] = await asyncio.gather(
        *[_process_one(ref) for ref in pdf_refs],
        return_exceptions=False,
    )

    print(
        f"[INDEXER] Done — {sum(1 for r in results if r.get('path'))} new file(s) indexed, "
        f"{sum(1 for r in results if not r.get('path'))} skipped/copied."
    )
    return results
