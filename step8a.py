"""
step8a.py — Invalidity Prior-Art Hunter
========================================
One chart skeleton at a time, one limitation at a time.
Finds published disclosures dated BEFORE the priority date that
read on each limitation — cited to the exact passage.

Data sources:
  - ChromaDB patent chunks  (indexed full text from the PDF pipeline)
  - Analysis cache          (blocking analyser results per patent)
  - Google Search grounding (PubMed, ClinicalTrials.gov, Google Patents, medRxiv)

Model: gemini-2.5-flash-preview-05-20 with Google Search grounding

Usage:
    python step8a.py --drug Axitinib                       # all patents in step7 output
    python step8a.py --drug Axitinib --patent US10123456   # one patent
    python step8a.py --drug Axitinib --rerun_step7         # force re-run step7 first
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

from llm_client import generate, parse_json_response, get_model_name, is_claude, is_gemini

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

STEP7_OUTPUT_DIR   = Path(os.getenv("STEP7_OUTPUT_DIR",     Path(__file__).parent / "step7_output"))
STEP8A_OUTPUT_DIR  = Path(os.getenv("STEP8A_OUTPUT_DIR",    Path(__file__).parent / "step8a_output"))
CHROMA_DB_PATH     = str(Path(__file__).parent / "chroma_patent_db")
ANALYSIS_CACHE_DIR = Path(os.getenv("ANALYSIS_CACHE_DIR",   Path(__file__).parent / "analysis_cache"))

_WORKERS = int(os.getenv("PIPELINE_WORKERS", "6"))

# ─────────────────────────────────────────────────────────────
# ChromaDB — lazy init
# ─────────────────────────────────────────────────────────────

_chroma_client = None

def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _chroma_client


def _sanitize_collection_name(drug_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", drug_name.strip())
    safe = re.sub(r"[_\-]{2,}", "_", safe)
    safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
    safe = safe.ljust(3, "x")[:55]
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
    return f"patents_{safe}"


def _get_chunks_from_chroma(drug_name: str, filename: str) -> Optional[str]:
    """Retrieve and reassemble full patent text from ChromaDB chunks."""
    try:
        chroma     = _get_chroma()
        col_name   = _sanitize_collection_name(drug_name)
        collection = chroma.get_collection(name=col_name)
        result     = collection.get(
            where={"filename": {"$eq": filename}},
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        chunks = [
            (meta.get("chunk_index", -1), doc)
            for doc, meta in zip(result["documents"], result["metadatas"])
            if meta.get("chunk_index", -1) != -1 and doc != "__index_complete__"
        ]
        if not chunks:
            return None
        chunks.sort(key=lambda x: x[0])
        return "\n".join(t for _, t in chunks)
    except Exception as e:
        print(f"[ChromaDB] Error: {e}")
        return None


def _find_filename_in_chroma(drug_name: str, patent_number: str) -> Optional[str]:
    """Fuzzy-match a patent number to a stored filename in ChromaDB."""
    try:
        chroma     = _get_chroma()
        col_name   = _sanitize_collection_name(drug_name)
        collection = chroma.get_collection(name=col_name)
        all_meta   = collection.get(include=["metadatas"])
        if not all_meta["ids"]:
            return None
        patent_norm = patent_number.replace("-", "").replace(" ", "").upper()
        for meta in all_meta["metadatas"]:
            fn = meta.get("filename", "")
            fn_norm = Path(fn).stem.replace("-", "").replace(" ", "").upper()
            if patent_norm in fn_norm or fn_norm in patent_norm:
                return fn
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Analysis cache reader
# ─────────────────────────────────────────────────────────────

def _load_analysis_cache(drug_name: str, patent_number: str) -> Optional[dict]:
    """Load the blocking analysis cache for a patent (if it exists)."""
    safe_drug = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    cache_dir = ANALYSIS_CACHE_DIR / safe_drug
    if not cache_dir.exists():
        return None
    # Try exact patent number match first, then scan all files
    pn_norm = patent_number.replace("-", "").replace(" ", "").upper()
    for f in cache_dir.glob("*.json"):
        fn_norm = f.stem.replace("-", "").replace(" ", "").replace("_", "").upper()
        if pn_norm in fn_norm or fn_norm in pn_norm:
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


# ─────────────────────────────────────────────────────────────
# Step 7 loader
# ─────────────────────────────────────────────────────────────

def _load_step7_skeletons(drug_name: str, patent_filter: Optional[str] = None) -> list[dict]:
    """Load all claim skeleton JSONs from step7 output for a drug."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    pattern = f"{safe}_*_claim_skeleton.json"
    files = sorted(STEP7_OUTPUT_DIR.glob(pattern))

    if not files:
        print(f"[Step 8a] No step7 skeletons found for '{drug_name}' in {STEP7_OUTPUT_DIR}")
        return []

    skeletons = []
    for f in files:
        try:
            skel = json.loads(f.read_text(encoding="utf-8"))
            if "error" in skel:
                print(f"[Step 8a] Skipping {f.name} — has error flag: {skel['error']}")
                continue
            if patent_filter:
                if skel.get("patent_number", "").upper() != patent_filter.upper():
                    continue
            skeletons.append(skel)
        except Exception as e:
            print(f"[Step 8a] Failed to read {f.name}: {e}")

    print(f"[Step 8a] Loaded {len(skeletons)} skeleton(s) for '{drug_name}'")
    return skeletons


async def _ensure_step7(drug_name: str, force: bool = False) -> None:
    """Run step7 if no skeletons exist for drug."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    existing = list(STEP7_OUTPUT_DIR.glob(f"{safe}_*_claim_skeleton.json"))
    if existing and not force:
        return
    print(f"[Step 8a] Step 7 output not found — running step7 for '{drug_name}'...")
    try:
        from step7 import process_drug
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("step7", Path(__file__).parent / "step7.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        process_drug = mod.process_drug
    await process_drug(drug_name=drug_name)


# ─────────────────────────────────────────────────────────────
# Source routing
# ─────────────────────────────────────────────────────────────

_SOURCE_ROUTES = {
    "compound_structure": ["Google Patents", "PubMed"],
    "salt_form":          ["Google Patents", "PubMed"],
    "excipient":          ["PubMed", "Google Patents"],
    "concentration":      ["PubMed", "Google Patents"],
    "pH":                 ["PubMed", "Google Patents"],
    "dosing":             ["ClinicalTrials.gov", "PubMed", "medRxiv"],
    "method_step":        ["ClinicalTrials.gov", "PubMed", "medRxiv"],
    "device_feature":     ["Google Patents"],
    "process_step":       ["Google Patents", "PubMed"],
}


# ─────────────────────────────────────────────────────────────
# Prompt — ONE limitation per call, multiple prior arts each
# ─────────────────────────────────────────────────────────────

_PRIOR_ART_PROMPT = """\
You are an invalidity prior-art hunter for patent claim charting.
You find published disclosures dated STRICTLY BEFORE the priority date
that READ ON the ONE limitation below — cited to the exact passage.

CRITICAL REQUIREMENT: You MUST search ALL routed sources and return EVERY
qualifying reference found. Do NOT return just one reference and stop.
The evidence array must contain ALL references that read on this limitation.
Target at minimum 3 references. There is no upper limit.

PATENT CONTEXT
==============
Patent number : {patent_number}
Jurisdiction  : {jurisdiction}
Drug          : {drug_name}
Priority date : {priority_date}  (HARD date bound)
Claim         : {claim_number}

{context_block}

LIMITATION TO SEARCH
====================
ID   : {limitation_id}
Type : {limitation_type}
Text : {limitation_text}
Drug : {drug_name}
{flags_block}
Sources to search: {sources}

SEARCH STRATEGY (follow this order for each source)
====================================================
For EACH source listed above, run at least 3 queries:
  Query A: "{drug_name}" + key technical terms from the limitation text
  Query B: drug synonyms / INN / brand names + limitation feature
  Query C: "{patent_number}" on Google Patents → open the patent page →
           read the "Cited by" and "References cited" sections →
           check EACH cited/citing document for this limitation's feature.
           These are the highest-quality prior art sources.
  Query D: "{drug_name}" + "{patent_number}" + limitation feature terms
  Query E: structural/functional/terminological variants of the limitation alone

IMPORTANT: The patent's own citations (found via Query C on Google Patents)
are the most productive search path. Examiners and inventors have already
identified the closest prior art — start there.

HARD DATE REJECTION RULE
=========================
You MUST check the publication date of EVERY reference BEFORE including it.
If the publication date is ON or AFTER {priority_date}: DO NOT INCLUDE IT.
There are ZERO exceptions. A reference published after the priority date is
NOT prior art and must never appear in the evidence array.

If you are uncertain about a reference's date, DO NOT INCLUDE IT.

A reference within 12 months BEFORE {priority_date} → include it BUT
flag "GRACE_PERIOD - admissibility case-by-case; counsel to confirm"
and set grace_flag: true.

GLOBAL RULES
=============
1. DATE BOUND (ABSOLUTE): Any reference with publication_date >= {priority_date}
   is INVALID prior art and MUST NOT appear in the evidence array.
   Check every reference's date. No exceptions. If date is unknown, exclude it.
2. READ-ON TEST: the passage must disclose the limitation's SPECIFIC feature,
   not merely the same topic. Reject topical-only matches.
3. ANCHORING: every passage MUST carry a pinpoint locus:
   - PubMed      : PMID + section/paragraph
   - ClinicalTrials.gov: NCT id + section
   - Google Patents: patent number + column:line or claim/paragraph
   - medRxiv     : DOI + section
   No locus -> do not include.
4. CITATION URL: for every reference, include the full URL where the document
   can be accessed (e.g. https://patents.google.com/patent/US6123456,
   https://pubmed.ncbi.nlm.nih.gov/12345678/).
5. If only a pharmacopoeia/handbook/supplier disclosure would read on an
   excipient/concentration/pH limitation, flag "OUT_OF_CORPUS".
6. Budget: MAX 8 query reformulations. Use structural, functional AND
   terminological variants.
7. Never fabricate PMIDs, NCT ids, patent numbers, DOIs, or URLs.
8. MULTIPLE REFERENCES REQUIRED: After finding the first qualifying reference,
   CONTINUE searching for more. Search each routed source independently.
   Return every reference with confidence_score >= 0.5. Do not stop early.

OUTPUT FORMAT
=============
CRITICAL: The "evidence" array MUST contain ALL qualifying references you found.
Do NOT stop after one. Search exhaustively across all routed sources and return
EVERY reference that has a passage reading on this limitation.
Minimum target: 3 references per limitation (if they exist in the literature).
Maximum: no limit — include all qualifying references found.

Return ONLY a single valid JSON object — no markdown fences, no prose.

{{
  "patent_number": "{patent_number}",
  "priority_date": "{priority_date}",
  "claim_number": {claim_number},
  "limitation_id": "{limitation_id}",
  "limitation_text_verbatim": "{limitation_text_escaped}",
  "evidence": [
    {{
      "reference_id": "Smith_US6123456_1999",
      "source": "Google Patents",
      "publication_date": "1999-03-15",
      "pre_priority": true,
      "grace_flag": false,
      "locus": "Col. 4, lines 22-35",
      "passage_verbatim": "An injectable aqueous solution comprising axitinib at a concentration of 1-10 mg/mL...",
      "citation_url": "https://patents.google.com/patent/US6123456",
      "reads_on_rationale": "Discloses identical compound in aqueous formulation at overlapping concentration range",
      "confidence_score": 0.92,
      "confidence_rationale": "Exact compound structure match with identical formulation parameters"
    }},
    {{
      "reference_id": "Jones_PMID12345678_2001",
      "source": "PubMed",
      "publication_date": "2001-06-15",
      "pre_priority": true,
      "grace_flag": false,
      "locus": "Results section, paragraph 3",
      "passage_verbatim": "The compound was formulated as an aqueous solution at pH 6.5...",
      "citation_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
      "reads_on_rationale": "Describes aqueous formulation of the same drug class with matching pH range",
      "confidence_score": 0.78,
      "confidence_rationale": "Same drug class and formulation type but different specific compound"
    }},
    {{
      "reference_id": "Chen_EP0987654_2000",
      "source": "Google Patents",
      "publication_date": "2000-11-20",
      "pre_priority": true,
      "grace_flag": false,
      "locus": "Claim 1, lines 1-8",
      "passage_verbatim": "...",
      "citation_url": "https://patents.google.com/patent/EP0987654",
      "reads_on_rationale": "...",
      "confidence_score": 0.71,
      "confidence_rationale": "..."
    }}
  ],
  "limitation_status": "covered",
  "flags": []
}}

confidence_score: 0.0 to 1.0 where:
  0.9-1.0 = exact feature match, verbatim or near-verbatim disclosure
  0.7-0.89 = strong read-on, same feature with minor differences
  0.5-0.69 = moderate, partial overlap requiring interpretation
  below 0.5 = weak, topical only — do NOT include

confidence_rationale: one sentence explaining WHY the confidence is at that level.

If NO qualifying art exists: set limitation_status="uncovered", evidence=[],
and add a flag explaining the gap.
"""


# ─────────────────────────────────────────────────────────────
# Core per-limitation processor — with timeout + retry
# ─────────────────────────────────────────────────────────────

_TIMEOUT_SECS  = int(os.getenv("LLM_TIMEOUT", "120"))
_MAX_RETRIES   = int(os.getenv("LLM_RETRIES", "2"))


async def _search_one_limitation(
    patent_number:  str,
    jurisdiction:   str,
    drug_name:      str,
    priority_date:  str,
    claim_number:   int,
    lim:            dict,
    context_block:  str,
) -> Optional[dict]:
    """One LLM call per limitation with timeout + retry."""
    lid   = lim.get("limitation_id", "?")
    ltext = lim.get("limitation_text_verbatim", "")
    ltype = lim.get("limitation_type", "unknown")
    flags = lim.get("flags", [])

    sources     = ", ".join(_SOURCE_ROUTES.get(ltype, ["Google Patents", "PubMed"]))
    flags_block = f"Flags: {', '.join(flags)}" if flags else ""
    ltext_escaped = ltext.replace('"', '\\"')

    prompt = _PRIOR_ART_PROMPT.format(
        patent_number        = patent_number,
        jurisdiction         = jurisdiction,
        drug_name            = drug_name,
        priority_date        = priority_date,
        claim_number         = claim_number,
        limitation_id        = lid,
        limitation_type      = ltype,
        limitation_text      = ltext,
        limitation_text_escaped = ltext_escaped,
        sources              = sources,
        flags_block          = flags_block,
        context_block        = context_block,
    )

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            raw = await asyncio.wait_for(
                generate(
                    prompt            = prompt,
                    use_web_search    = True,
                    temperature       = 0.0,
                    max_output_tokens = 65536,
                ),
                timeout=_TIMEOUT_SECS,
            )
            if not raw:
                print(f"[Step 8a]   {lid}: empty response (attempt {attempt})")
                last_error = "empty response"
                continue

            result = parse_json_response(raw)
            if result is None:
                print(f"[Step 8a]   {lid}: JSON parse failed (attempt {attempt})")
                last_error = "JSON parse failed"
                continue

            if isinstance(result, list):
                result = result[0] if result else None
            if result:
                status = result.get("limitation_status", "?")
                n_ev   = len(result.get("evidence", []))
                print(f"[Step 8a]   {lid}: {status} ({n_ev} evidence) | model: {get_model_name()}")
            return result

        except asyncio.TimeoutError:
            print(f"[Step 8a]   {lid}: TIMEOUT after {_TIMEOUT_SECS}s (attempt {attempt})")
            last_error = f"timeout after {_TIMEOUT_SECS}s"
            continue
        except Exception as e:
            print(f"[Step 8a]   {lid}: error - {e} (attempt {attempt})")
            last_error = str(e)
            if attempt <= _MAX_RETRIES:
                await asyncio.sleep(2 * attempt)
            continue

    print(f"[Step 8a]   {lid}: FAILED after {_MAX_RETRIES + 1} attempts")
    return {
        "patent_number": patent_number, "priority_date": priority_date,
        "claim_number": claim_number, "limitation_id": lid,
        "limitation_text_verbatim": ltext, "evidence": [],
        "limitation_status": "uncovered",
        "flags": [f"SEARCH_FAILED - {last_error}; manual search required"],
    }


async def _search_prior_art_for_claim(
    skeleton:       dict,
    claim:          dict,
    patent_text:    Optional[str],
    analysis_cache: Optional[dict],
) -> list[dict]:
    """Search prior art for all limitations of one claim (parallel within claim)."""
    patent_number = skeleton["patent_number"]
    jurisdiction  = skeleton.get("jurisdiction", "")
    drug_name     = skeleton.get("drug_name", "")
    priority_date = skeleton.get("priority_date", "Unknown")
    claim_number  = claim.get("claim_number", "?")
    limitations   = claim.get("limitations", [])

    if not limitations:
        return []

    context_parts = []
    if patent_text:
        excerpt = patent_text[:80_000] if len(patent_text) > 80_000 else patent_text
        context_parts.append(f"INDEXED PATENT TEXT (from ChromaDB)\n{'='*50}\n{excerpt}\n")
    if analysis_cache:
        cache_summary = {k: analysis_cache.get(k) for k in [
            "claim_category", "tag", "blocking_category", "reason",
            "step2_elements_present", "step3_evidence_summary", "step5_reason",
        ] if analysis_cache.get(k)}
        if cache_summary:
            context_parts.append(f"BLOCKING ANALYSIS CACHE\n{'='*50}\n{json.dumps(cache_summary, indent=2)}\n")
    context_block = "\n".join(context_parts) or "(No cached context)\n"

    print(f"[Step 8a] Claim {claim_number}: {len(limitations)} limitation(s) (up to {_WORKERS} parallel)")

    sem = asyncio.Semaphore(_WORKERS)

    async def _bounded_lim(lim):
        async with sem:
            return await _search_one_limitation(
                patent_number, jurisdiction, drug_name, priority_date,
                claim_number, lim, context_block,
            )

    raw_results = await asyncio.gather(
        *[_bounded_lim(lim) for lim in limitations], return_exceptions=True,
    )

    results = []
    for lim, res in zip(limitations, raw_results):
        lid = lim.get("limitation_id", "?")
        if isinstance(res, Exception):
            print(f"[Step 8a]   {lid}: raised {res}")
            results.append({"patent_number": patent_number, "priority_date": priority_date,
                "claim_number": claim_number, "limitation_id": lid,
                "limitation_text_verbatim": lim.get("limitation_text_verbatim", ""),
                "evidence": [], "limitation_status": "uncovered",
                "flags": [f"EXCEPTION - {res}"]})
        elif res is not None:
            results.append(res)
        else:
            results.append({"patent_number": patent_number, "priority_date": priority_date,
                "claim_number": claim_number, "limitation_id": lid,
                "limitation_text_verbatim": lim.get("limitation_text_verbatim", ""),
                "evidence": [], "limitation_status": "uncovered",
                "flags": ["NO_RESPONSE"]})

    covered = sum(1 for r in results if r.get("limitation_status") == "covered")
    print(f"[Step 8a] * Claim {claim_number}: {covered}/{len(results)} covered")
    return results


# ─────────────────────────────────────────────────────────────
# Per-patent orchestrator — saves immediately after completing
# ─────────────────────────────────────────────────────────────

async def process_patent(skeleton: dict, drug_name: str, output_dir: Path) -> dict:
    """Process all claims for one patent. Saves to disk immediately."""
    patent_number = skeleton["patent_number"]
    source_file   = skeleton.get("source_file", "")

    print(f"\n[Step 8a] {'='*50}")
    print(f"[Step 8a] Patent: {patent_number} | Drug: {drug_name}")
    print(f"[Step 8a] {'='*50}")

    patent_text = None
    if source_file:
        patent_text = _get_chunks_from_chroma(drug_name, source_file)
    if patent_text is None:
        matched = _find_filename_in_chroma(drug_name, patent_number)
        if matched:
            patent_text = _get_chunks_from_chroma(drug_name, matched)
    if patent_text:
        print(f"[Step 8a] ChromaDB: {len(patent_text):,} chars")

    analysis_cache = _load_analysis_cache(drug_name, patent_number)
    if analysis_cache:
        print(f"[Step 8a] Analysis cache: loaded")

    claims = skeleton.get("independent_claims", [])
    all_limitation_results = []

    for claim in claims:
        lim_results = await _search_prior_art_for_claim(
            skeleton, claim, patent_text, analysis_cache
        )
        all_limitation_results.extend(lim_results)

    output = {
        "patent_number": patent_number,
        "drug_name": drug_name,
        "priority_date": skeleton.get("priority_date", "Unknown"),
        "jurisdiction": skeleton.get("jurisdiction", ""),
        "source_file": source_file,
        "limitation_results": all_limitation_results,
    }

    # Save immediately after this patent
    _write_output(drug_name, output, output_dir)
    covered = sum(1 for lr in all_limitation_results if lr.get("limitation_status") == "covered")
    print(f"[Step 8a] SAVED {patent_number}: {covered}/{len(all_limitation_results)} covered\n")

    return output


# ─────────────────────────────────────────────────────────────
# Main runner — SEQUENTIAL per patent, with skip logic
# ─────────────────────────────────────────────────────────────

async def process_drug(
    drug_name:     str,
    patent_filter: Optional[str] = None,
    rerun_step7:   bool          = False,
    rerun:         bool          = False,
    output_dir:    Path          = STEP8A_OUTPUT_DIR,
) -> list[dict]:
    await _ensure_step7(drug_name, force=rerun_step7)

    skeletons = _load_step7_skeletons(drug_name, patent_filter=patent_filter)
    if not skeletons:
        print(f"[Step 8a] No skeletons for '{drug_name}'.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_drug = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)

    results = []
    for i, skel in enumerate(skeletons, 1):
        pn          = skel["patent_number"]
        safe_patent = re.sub(r"[^a-zA-Z0-9_-]", "_", pn)
        out_path    = output_dir / f"{safe_drug}_{safe_patent}_prior_art.json"

        # Skip if already done
        if not rerun and out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                n_lim    = len(existing.get("limitation_results", []))
                print(f"[Step 8a] [{i}/{len(skeletons)}] SKIP {pn} — already done ({n_lim} lim(s))")
                results.append(existing)
                continue
            except Exception:
                pass  # corrupted — re-run

        print(f"[Step 8a] [{i}/{len(skeletons)}] Processing {pn}...")
        try:
            result = await process_patent(skel, drug_name, output_dir)
            results.append(result)
        except Exception as e:
            print(f"[Step 8a] [{i}/{len(skeletons)}] FAILED {pn}: {e}")
            # Continue to next patent — don't lose all progress

    # Combined JSON
    if results:
        combined_path = output_dir / f"{safe_drug}_all_prior_art.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Combined JSON: {combined_path}")

    covered = sum(1 for r in results for lr in r.get("limitation_results", [])
                  if lr.get("limitation_status") == "covered")
    total   = sum(len(r.get("limitation_results", [])) for r in results)
    print(f"\n[Step 8a] Done. {len(results)} patent(s), {covered}/{total} limitations covered.")
    return results


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def _write_output(drug_name: str, result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_drug   = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    safe_patent = re.sub(r"[^a-zA-Z0-9_-]", "_", result.get("patent_number", "unknown"))

    json_path = output_dir / f"{safe_drug}_{safe_patent}_prior_art.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  → Prior art JSON  : {json_path}")

    # Summary table (markdown)
    md_path = output_dir / f"{safe_drug}_{safe_patent}_prior_art_summary.md"
    lines   = [
        f"# Step 8a Prior Art — {result['patent_number']}",
        f"**Drug:** {drug_name} | **Priority date:** {result.get('priority_date', '?')}",
        "",
        "| Claim | Lim ID | Type | Status | References | Flags |",
        "|-------|--------|------|--------|------------|-------|",
    ]
    for lr in result.get("limitation_results", []):
        cn      = lr.get("claim_number", "?")
        lid     = lr.get("limitation_id", "?")
        # Infer type from limitation text or leave as stored
        ltype   = ""
        status  = lr.get("limitation_status", "?")
        refs    = ", ".join(e.get("reference_id", "?") for e in lr.get("evidence", []))
        flags   = "; ".join(lr.get("flags", []))
        lines.append(f"| {cn} | {lid} | {ltype} | {status} | {refs or '—'} | {flags or '—'} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → Summary table   : {md_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8a – Prior-Art Hunter (sequential per patent, parallel per limitation)"
    )
    parser.add_argument("--drug",        "-d", required=True)
    parser.add_argument("--patent",      "-p", default=None)
    parser.add_argument("--rerun",       action="store_true",
                        help="Re-search all patents even if output exists")
    parser.add_argument("--rerun_step7", action="store_true")
    parser.add_argument("--output_dir",  default=str(STEP8A_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 8a] Drug       : {args.drug}")
    print(f"[Step 8a] Model      : {get_model_name()}")
    print(f"[Step 8a] Timeout    : {_TIMEOUT_SECS}s | Retries: {_MAX_RETRIES}")
    print(f"[Step 8a] Workers    : {_WORKERS} (parallel per limitation)")
    print(f"[Step 8a] Output     : {output_dir.resolve()}")

    results = asyncio.run(process_drug(
        drug_name     = args.drug,
        patent_filter = args.patent,
        rerun_step7   = args.rerun_step7,
        rerun         = args.rerun,
        output_dir    = output_dir,
    ))


if __name__ == "__main__":
    main()
