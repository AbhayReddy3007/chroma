"""
step7.py — Patent Claim Decomposition Engine
=============================================
Reads ONE patent at a time from step6's charting queue.

TEXT SOURCE (priority order):
  1. ChromaDB chunks  — already indexed full text from the PDF pipeline.
                        Reassembled from ordered chunks, no web call needed.
  2. Google Search grounding — fallback if patent not in ChromaDB
                                (Gemini searches Google Patents live).

If step6 output is not available for a drug, step6 is run first automatically.

Model: gemini-2.5-flash-preview-05-20

Usage:
    python step7.py --drug Axitinib                   # all patents in queue
    python step7.py --drug Axitinib --priority 1      # highest priority only
    python step7.py --drug Axitinib --patent US10123456
    python step7.py --drug Axitinib --rerun_step6
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────
# Paths — must match step6.py and indexer.py
# ─────────────────────────────────────────────────────────────

EXCEL_OUTPUT_DIR = Path(os.getenv("EXCEL_OUTPUT_DIR", Path(__file__).parent / "patent_exports"))
STEP6_OUTPUT_DIR = Path(os.getenv("STEP6_OUTPUT_DIR", Path(__file__).parent / "step6_output"))
STEP7_OUTPUT_DIR = Path(os.getenv("STEP7_OUTPUT_DIR", Path(__file__).parent / "step7_output"))
CHROMA_DB_PATH   = str(Path(__file__).parent / "chroma_patent_db")

# ─────────────────────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────────────────────

MODEL          = "gemini-2.5-flash-preview-05-20"
_gemini_client = None
_WORKERS       = int(os.getenv("PIPELINE_WORKERS", "6"))  # shared across steps 7-9

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable is not set.\n"
                "Add it to your .env file or set it in your shell before running."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

# ─────────────────────────────────────────────────────────────
# ChromaDB — lazy init so import doesn't crash if chromadb absent
# ─────────────────────────────────────────────────────────────

_chroma_client = None

def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _chroma_client


def _sanitize_collection_name(drug_name: str) -> str:
    """Must match indexer.py sanitize_collection_name exactly."""
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", drug_name.strip())
    safe = re.sub(r"[_\-]{2,}", "_", safe)
    safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
    safe = safe.ljust(3, "x")
    safe = safe[:55]
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
    return f"patents_{safe}"


def _get_chunks_from_chroma(drug_name: str, filename: str) -> Optional[str]:
    """
    Retrieve and reassemble the full patent text from ChromaDB chunks.

    Chunks are stored with metadata:
        {"filename": filename, "chunk_index": i, "total_chunks": N, ...}
    Sentinel has chunk_index = -1 and document = "__index_complete__".

    Returns the full reconstructed text, or None if not found.
    """
    try:
        chroma      = _get_chroma()
        col_name    = _sanitize_collection_name(drug_name)
        collection  = chroma.get_collection(name=col_name)

        result = collection.get(
            where   = {"filename": {"$eq": filename}},
            include = ["documents", "metadatas"],
        )

        if not result["ids"]:
            print(f"[ChromaDB] No chunks found for '{filename}' in collection '{col_name}'")
            return None

        # Separate content chunks from sentinel
        chunks_with_idx = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            idx = meta.get("chunk_index", -1)
            if idx == -1 or doc == "__index_complete__":
                continue    # skip sentinel
            chunks_with_idx.append((idx, doc))

        if not chunks_with_idx:
            print(f"[ChromaDB] Only sentinel found for '{filename}' — no text chunks")
            return None

        # Sort by chunk_index and join
        chunks_with_idx.sort(key=lambda x: x[0])
        full_text = "\n".join(text for _, text in chunks_with_idx)

        print(
            f"[ChromaDB] ✓ Retrieved '{filename}' — "
            f"{len(chunks_with_idx)} chunks, {len(full_text):,} chars"
        )
        return full_text

    except Exception as e:
        print(f"[ChromaDB] Error retrieving '{filename}': {e}")
        return None


def _find_filename_in_chroma(drug_name: str, patent_number: str) -> Optional[str]:
    """
    The charting queue stores filename as a GCS-relative path
    (e.g. "US/US10123456B2.pdf" or "US10123456B2.pdf").
    This helper searches the collection for any chunk whose filename
    contains the patent_number, returning the exact stored filename.
    """
    try:
        chroma     = _get_chroma()
        col_name   = _sanitize_collection_name(drug_name)
        collection = chroma.get_collection(name=col_name)

        # Get all unique filenames via sentinel records (chunk_index = -1)
        # ChromaDB doesn't support "contains" queries, so we fetch all metas
        # with a small limit first to check if collection is populated
        all_meta = collection.get(include=["metadatas"])
        if not all_meta["ids"]:
            return None

        patent_norm = patent_number.replace("-", "").replace(" ", "").upper()
        for meta in all_meta["metadatas"]:
            fn = meta.get("filename", "")
            fn_norm = Path(fn).stem.replace("-", "").replace(" ", "").upper()
            if patent_norm in fn_norm or fn_norm in patent_norm:
                return fn

        return None

    except Exception as e:
        print(f"[ChromaDB] Collection lookup error for '{drug_name}': {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Step 6 bootstrap
# ─────────────────────────────────────────────────────────────

def _ensure_step6(drug_name: str, force: bool = False) -> Optional[list]:
    """
    Return the charting queue for drug_name from step6 output.
    Runs step6 first if the queue file is missing or force=True.
    """
    safe       = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    queue_path = STEP6_OUTPUT_DIR / f"{safe}_charting_queue.json"

    if force or not queue_path.exists():
        print(f"[Step 7] Step 6 output not found for '{drug_name}' — running step6 first...")
        try:
            from step6 import run_for_drug
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "step6", Path(__file__).parent / "step6.py"
            )
            step6_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(step6_mod)
            run_for_drug = step6_mod.run_for_drug

        STEP6_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result = run_for_drug(
            drug_name  = drug_name,
            excel_dir  = EXCEL_OUTPUT_DIR,
            output_dir = STEP6_OUTPUT_DIR,
        )
        if result is None:
            print(f"[Step 7] Step 6 failed for '{drug_name}'. Cannot continue.")
            return None

    if not queue_path.exists():
        print(f"[Step 7] Charting queue still missing after step6 run: {queue_path}")
        return None

    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)

    print(f"[Step 7] Loaded charting queue: {len(queue)} patent(s) for '{drug_name}'")
    return queue


# ─────────────────────────────────────────────────────────────
# Gemini prompt — always uses both ChromaDB text + Google Search
# ─────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a patent claim decomposition engine for IP litigation analysis.
You do NOT assess prior art. You read claims only — not specification prose.

PATENT DETAILS
==============
Patent number     : {patent_number}
Jurisdiction      : {jurisdiction}
Drug              : {drug_name}
Priority date     : {filing_date}
Blocking category : {blocking_category}

{text_block}

TASK
====

STEP 1 — Identify and verify claims
{step1_instruction}

STEP 2 — Keep independent claims only
Independent claims do NOT contain phrases like "claim 1", "as recited in",
"according to claim", or "of claim N". List each independent claim number and
its FULL verbatim text exactly as written.

If the PDF text and Google Patents disagree on claim wording, prefer the PDF
text (it is the granted/published document). Use Google Patents only to fill
gaps or verify completeness.

STEP 3 — Decompose each independent claim into limitations
Use litigation-style labels:
  [<n>.P]              = preamble (everything before "comprising:" /
                         "consisting of:" / "consisting essentially of:")
  [<n>.a], [<n>.b], ... = each element/wherein/said/at-least clause

Rules:
- ONE testable technical feature per limitation.
- Split compound clauses ("X at pH 5.5-7.5 AND 5-10 mg/mL") into SEPARATE
  limitations.
- Quote claim words VERBATIM. Do not paraphrase or summarise.
- Preserve antecedent basis exactly as written.
- Tag each limitation with ONE limitation_type from:
    compound_structure | salt_form | excipient | concentration |
    pH | device_feature | dosing | method_step | process_step
- Markush/genus claim → decompose the independent scaffold and add flag
  "MARKUSH - structural search required in Step 8".

OUTPUT FORMAT
=============
Return ONLY a single valid JSON object — no markdown fences, no prose.

{{
  "drug_name": "{drug_name}",
  "patent_number": "{patent_number}",
  "jurisdiction": "{jurisdiction}",
  "priority_date": "{filing_date}",
  "blocking_category": "{blocking_category}",
  "source_file": "{source_file}",
  "text_source": "{text_source}",
  "independent_claims": [
    {{
      "claim_number": 1,
      "claim_text_verbatim": "...",
      "limitations": [
        {{
          "limitation_id": "1.P",
          "limitation_text_verbatim": "...",
          "limitation_type": "compound_structure",
          "flags": []
        }}
      ]
    }}
  ]
}}

If claims are completely unavailable from both the PDF text and Google Patents,
return exactly:
  {{"error": "CLAIMS_UNAVAILABLE - route to counsel",
    "patent_number": "{patent_number}", "text_source": "{text_source}"}}
"""


# ─────────────────────────────────────────────────────────────
# Core decomposition
# ─────────────────────────────────────────────────────────────

async def decompose_patent(item: dict) -> Optional[dict]:
    """
    Decompose one charting queue item.

    Always uses Google Search grounding so Gemini can verify and cross-reference
    claims on Google Patents. When ChromaDB chunks exist, the full PDF-extracted
    text is included in the prompt as the primary source — Gemini uses web search
    to fill gaps, verify completeness, and resolve OCR artefacts.
    """
    patent_number     = item["patent_number"]
    jurisdiction      = item["jurisdiction"]
    drug_name         = item["drug_name"]
    filing_date       = item.get("filing_date", "Unknown")
    blocking_category = item.get("blocking_category", "")
    source_file       = item.get("source_file", "")

    print(f"\n[Step 7] ── {patent_number} ({jurisdiction}) | {drug_name} ──")

    # ── Retrieve ChromaDB text (may be None) ──────────────────
    patent_text    = None
    exact_filename = source_file

    if exact_filename:
        patent_text = _get_chunks_from_chroma(drug_name, exact_filename)

    if patent_text is None:
        matched_fn = _find_filename_in_chroma(drug_name, patent_number)
        if matched_fn and matched_fn != exact_filename:
            print(f"[ChromaDB] Fuzzy-matched filename: {matched_fn}")
            patent_text = _get_chunks_from_chroma(drug_name, matched_fn)

    # ── Build prompt with both sources ────────────────────────
    if patent_text:
        # Truncate if needed — keep tail (claims at end) + head for context
        MAX_CHARS = 300_000
        if len(patent_text) > MAX_CHARS:
            head = patent_text[:50_000]
            tail = patent_text[-(MAX_CHARS - 50_000):]
            patent_text = head + "\n\n[... middle truncated for length ...]\n\n" + tail
            print(f"[Step 7] Text truncated to {MAX_CHARS:,} chars (head+tail)")

        text_block = (
            "FULL PATENT TEXT (extracted from the original PDF — primary source)\n"
            "===================================================================\n"
            f"{patent_text}"
        )
        step1_instruction = (
            "You have two sources for this patent's claims:\n"
            "  (A) The PDF-extracted text above (primary — this is the granted document).\n"
            "  (B) Google Patents (use Google Search to look up \"{patent_number}\").\n\n"
            "Use source (A) as the primary text. Cross-reference with source (B) to:\n"
            "  - Verify you have ALL independent claims (not truncated or missing).\n"
            "  - Fill in any text corrupted by OCR artefacts in the PDF.\n"
            "  - Confirm claim numbering matches the published patent.\n"
            "If they conflict on wording, prefer the PDF text (A)."
        ).format(patent_number=patent_number)
        text_source = "chromadb+search"
        print(f"[Step 7] ✓ ChromaDB text ({len(patent_text):,} chars) + Google Search grounding")

    else:
        text_block = (
            "(No PDF text available — using Google Patents as the sole source)\n"
        )
        step1_instruction = (
            "Search Google Patents for \"{patent_number}\" ({jurisdiction}).\n"
            "Pull the FULL claim listing verbatim.\n"
            "If unavailable, try the INPADOC family equivalent.\n"
            "If still unavailable, return the CLAIMS_UNAVAILABLE error JSON."
        ).format(patent_number=patent_number, jurisdiction=jurisdiction)
        text_source = "search_only"
        print(f"[Step 7] ⚠ No ChromaDB chunks — Google Search grounding only")

    prompt = _DECOMPOSE_PROMPT.format(
        patent_number     = patent_number,
        jurisdiction      = jurisdiction,
        drug_name         = drug_name,
        filing_date       = filing_date,
        blocking_category = blocking_category,
        source_file       = source_file,
        text_block        = text_block,
        step1_instruction = step1_instruction,
        text_source       = text_source,
    )

    # ── Call Gemini — always with Google Search grounding ─────
    config = types.GenerateContentConfig(
        tools       = [types.Tool(google_search=types.GoogleSearch())],
        temperature = 0.0,
    )

    try:
        response = await _get_gemini_client().aio.models.generate_content(
            model    = MODEL,
            contents = prompt,
            config   = config,
        )

        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        if not raw:
            print(f"[Step 7] Empty response for {patent_number}")
            return None

        result = json.loads(raw)

        if "error" in result:
            print(f"[Step 7] ⚠  {result['error']}")
            return result

        # Attach charting metadata
        result["charting_priority"]  = item.get("charting_priority")
        result["patent_expiry_year"] = item.get("patent_expiry_year")

        n_claims = len(result.get("independent_claims", []))
        n_lims   = sum(
            len(c.get("limitations", []))
            for c in result.get("independent_claims", [])
        )
        src = result.get("text_source", text_source)
        print(f"[Step 7] ✓ {patent_number} — {n_claims} independent claim(s), "
              f"{n_lims} limitation(s) | source: {src}")
        return result

    except json.JSONDecodeError as e:
        print(f"[Step 7] JSON parse error for {patent_number}: {e}")
        print(f"         Raw (first 500): {raw[:500]}")
        return None
    except Exception as e:
        print(f"[Step 7] Gemini error for {patent_number}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Markdown table renderer
# ─────────────────────────────────────────────────────────────

def _render_table(result: dict) -> str:
    if "error" in result:
        return f"\n## ⚠ {result['patent_number']} — {result['error']}\n"

    lines = []
    patent_num = result.get("patent_number", "?")
    drug       = result.get("drug_name", "?")
    expiry     = result.get("patent_expiry_year", "?")
    priority   = result.get("charting_priority", "?")
    src        = result.get("text_source", "?")

    lines.append(
        f"\n## Charting Priority #{priority} — {patent_num} ({result.get('jurisdiction','?')})"
    )
    lines.append(
        f"**Drug:** {drug} | **Priority date:** {result.get('priority_date','?')} | "
        f"**Expiry:** {expiry} | **Category:** {result.get('blocking_category','?')} | "
        f"**Text source:** {src}"
    )
    lines.append("")
    lines.append("| Claim | Limitation ID | Limitation (verbatim) | Type | Prior Art (Step 8) |")
    lines.append("|-------|--------------|----------------------|------|-------------------|")

    for claim in result.get("independent_claims", []):
        cn = claim.get("claim_number", "?")
        for lim in claim.get("limitations", []):
            lid   = lim.get("limitation_id", "")
            text  = lim.get("limitation_text_verbatim", "").replace("|", "\\|").replace("\n", " ")
            ltype = lim.get("limitation_type", "")
            flags = "; ".join(lim.get("flags", []))
            flag_note = f" ⚠ {flags}" if flags else ""
            lines.append(f"| {cn} | {lid} | {text}{flag_note} | {ltype} | — |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def _write_patent_output(drug_name: str, result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_drug   = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    safe_patent = re.sub(r"[^a-zA-Z0-9_-]", "_", result.get("patent_number", "unknown"))

    json_path = output_dir / f"{safe_drug}_{safe_patent}_claim_skeleton.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  → Claim skeleton  : {json_path}")

    md_path = output_dir / f"{safe_drug}_{safe_patent}_charting_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_table(result))
    print(f"  → Charting table  : {md_path}")


def _write_drug_summary(drug_name: str, all_results: list[dict], output_dir: Path) -> None:
    safe_drug = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    md_path   = output_dir / f"{safe_drug}_all_charting_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Step 7 Charting Tables — {drug_name}\n")
        f.write(f"**{len(all_results)} patent(s) decomposed**\n")
        for r in sorted(all_results, key=lambda x: x.get("charting_priority") or 999):
            f.write(_render_table(r))
            f.write("\n\n---\n")
    print(f"\n  → Combined tables : {md_path}")


# ─────────────────────────────────────────────────────────────
# Main async runner
# ─────────────────────────────────────────────────────────────

async def process_drug(
    drug_name:       str,
    patent_filter:   Optional[str]  = None,
    priority_filter: Optional[int]  = None,
    rerun_step6:     bool           = False,
    output_dir:      Path           = STEP7_OUTPUT_DIR,
) -> list[dict]:
    queue = _ensure_step6(drug_name, force=rerun_step6)
    if not queue:
        return []

    if patent_filter:
        queue = [i for i in queue if i["patent_number"].upper() == patent_filter.upper()]
        if not queue:
            print(f"[Step 7] Patent '{patent_filter}' not in charting queue for '{drug_name}'.")
            return []

    if priority_filter is not None:
        queue = [i for i in queue if i.get("charting_priority") == priority_filter]
        if not queue:
            print(f"[Step 7] No patent at priority {priority_filter} for '{drug_name}'.")
            return []

    print(f"[Step 7] Processing {len(queue)} patent(s) for '{drug_name}' "
          f"(up to {_WORKERS} workers)...")

    sem = asyncio.Semaphore(_WORKERS)

    async def _bounded(item):
        async with sem:
            return await decompose_patent(item)

    items   = sorted(queue, key=lambda x: x.get("charting_priority") or 999)
    raw     = await asyncio.gather(*[_bounded(item) for item in items],
                                   return_exceptions=True)

    results = []
    for item, result in zip(items, raw):
        if isinstance(result, Exception):
            print(f"[Step 7] ⚠  {item['patent_number']} raised: {result}")
        elif result:
            _write_patent_output(drug_name, result, output_dir)
            results.append(result)
        else:
            print(f"[Step 7] ⚠  Skipping {item['patent_number']} — decomposition failed.")

    if results:
        _write_drug_summary(drug_name, results, output_dir)

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7 – Patent Claim Decomposition (ChromaDB → Gemini, fallback: Google Search)"
    )
    parser.add_argument("--drug",        "-d", required=True)
    parser.add_argument("--patent",      "-p", default=None,
                        help="Process only this patent number")
    parser.add_argument("--priority",    type=int, default=None,
                        help="Process only this charting_priority")
    parser.add_argument("--rerun_step6", action="store_true",
                        help="Force re-run step6 before step7")
    parser.add_argument("--output_dir",  default=str(STEP7_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 7] Drug        : {args.drug}")
    print(f"[Step 7] Model       : {MODEL}")
    print(f"[Step 7] ChromaDB    : {CHROMA_DB_PATH}")
    print(f"[Step 7] Patent      : {args.patent or 'all in queue'}")
    print(f"[Step 7] Priority    : {args.priority or 'all'}")
    print(f"[Step 7] Output dir  : {output_dir.resolve()}")

    results = asyncio.run(process_drug(
        drug_name       = args.drug,
        patent_filter   = args.patent,
        priority_filter = args.priority,
        rerun_step6     = args.rerun_step6,
        output_dir      = output_dir,
    ))

    src_counts = {}
    for r in results:
        src = r.get("text_source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    print(f"\n[Step 7] Done. {len(results)} patent(s) decomposed.")
    for src, count in src_counts.items():
        print(f"         {count} via {src}")

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
