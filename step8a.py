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
# Core per-limitation processor — one LLM call per limitation
# ─────────────────────────────────────────────────────────────

async def _search_one_limitation(
    patent_number:  str,
    jurisdiction:   str,
    drug_name:      str,
    priority_date:  str,
    claim_number:   int,
    lim:            dict,
    context_block:  str,
) -> Optional[dict]:
    """One LLM call for one limitation. Returns the result dict or None."""
    lid   = lim.get("limitation_id", "?")
    ltext = lim.get("limitation_text_verbatim", "")
    ltype = lim.get("limitation_type", "unknown")
    flags = lim.get("flags", [])

    sources    = ", ".join(_SOURCE_ROUTES.get(ltype, ["Google Patents", "PubMed"]))
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

    try:
        raw = await generate(
            prompt         = prompt,
            use_web_search = True,
            temperature    = 0.0,
            max_output_tokens = 65536,
        )

        if not raw:
            print(f"[Step 8a]   {lid}: empty response")
            return None

        result = parse_json_response(raw)
        if result is None:
            print(f"[Step 8a]   {lid}: JSON parse failed — marked uncovered")
            return {
                "patent_number":            patent_number,
                "priority_date":            priority_date,
                "claim_number":             claim_number,
                "limitation_id":            lid,
                "limitation_text_verbatim": ltext,
                "evidence":                 [],
                "limitation_status":        "uncovered",
                "flags":                    ["PARSE_ERROR - LLM response truncated"],
            }

        if isinstance(result, list):
            result = result[0] if result else None

        if result:
            status = result.get("limitation_status", "?")
            n_ev   = len(result.get("evidence", []))
            print(f"[Step 8a]   {lid}: {status} ({n_ev} evidence) | model: {get_model_name()}")

        return result

    except Exception as e:
        print(f"[Step 8a]   {lid}: LLM error — {e}")
        return None


async def _search_prior_art_for_claim(
    skeleton:       dict,
    claim:          dict,
    patent_text:    Optional[str],
    analysis_cache: Optional[dict],
) -> list[dict]:
    """
    Run the prior-art search for all limitations of one independent claim.
    Fires ONE Gemini call per limitation (parallelised up to _WORKERS)
    to avoid JSON truncation from large multi-limitation responses.
    """
    patent_number = skeleton["patent_number"]
    jurisdiction  = skeleton.get("jurisdiction", "")
    drug_name     = skeleton.get("drug_name", "")
    priority_date = skeleton.get("priority_date", "Unknown")
    claim_number  = claim.get("claim_number", "?")
    limitations   = claim.get("limitations", [])

    if not limitations:
        print(f"[Step 8a] No limitations for claim {claim_number} of {patent_number}")
        return []

    # ── Build shared context block once (reused across all limitation calls)
    context_parts = []
    if patent_text:
        excerpt = patent_text[:80_000] if len(patent_text) > 80_000 else patent_text
        context_parts.append(
            "INDEXED PATENT TEXT (from ChromaDB — for context)\n"
            "===================================================\n"
            f"{excerpt}\n"
        )
    if analysis_cache:
        cache_summary = {
            k: analysis_cache.get(k)
            for k in [
                "claim_category", "tag", "blocking_category", "reason",
                "step2_elements_present", "step3_evidence_summary",
                "step5_reason",
            ]
            if analysis_cache.get(k)
        }
        if cache_summary:
            context_parts.append(
                "BLOCKING ANALYSIS CACHE\n"
                "=======================\n"
                f"{json.dumps(cache_summary, indent=2)}\n"
            )
    context_block = "\n".join(context_parts) if context_parts else "(No cached context)\n"

    print(f"[Step 8a] Claim {claim_number} of {patent_number}: "
          f"searching {len(limitations)} limitation(s) individually "
          f"(up to {_WORKERS} in parallel)...")

    # ── One call per limitation, parallelised ─────────────────
    sem = asyncio.Semaphore(_WORKERS)

    async def _bounded_lim(lim):
        async with sem:
            return await _search_one_limitation(
                patent_number, jurisdiction, drug_name, priority_date,
                claim_number, lim, context_block,
            )

    raw_results = await asyncio.gather(
        *[_bounded_lim(lim) for lim in limitations],
        return_exceptions=True,
    )

    results = []
    for lim, res in zip(limitations, raw_results):
        lid = lim.get("limitation_id", "?")
        if isinstance(res, Exception):
            print(f"[Step 8a]   {lid}: raised {res}")
        elif res is not None:
            results.append(res)
        else:
            # Gemini returned nothing — mark uncovered
            results.append({
                "patent_number":           patent_number,
                "priority_date":           priority_date,
                "claim_number":            claim_number,
                "limitation_id":           lid,
                "limitation_text_verbatim": lim.get("limitation_text_verbatim", ""),
                "evidence":                [],
                "limitation_status":       "uncovered",
                "flags":                   ["NO_RESPONSE - Gemini returned empty; manual search required"],
            })

    covered = sum(1 for r in results if r.get("limitation_status") == "covered")
    print(f"[Step 8a] ✓ Claim {claim_number}: {covered}/{len(results)} limitations covered")
    return results


# ─────────────────────────────────────────────────────────────
# Per-patent orchestrator
# ─────────────────────────────────────────────────────────────

async def process_patent(skeleton: dict) -> dict:
    """
    Process all independent claims for one patent skeleton.
    Returns the full step8a output for this patent.
    """
    patent_number = skeleton["patent_number"]
    drug_name     = skeleton.get("drug_name", "")
    source_file   = skeleton.get("source_file", "")

    print(f"\n[Step 8a] {'═'*50}")
    print(f"[Step 8a] Patent: {patent_number} | Drug: {drug_name}")
    print(f"[Step 8a] {'═'*50}")

    # ── Load ChromaDB text ────────────────────────────────────
    patent_text = None
    if source_file:
        patent_text = _get_chunks_from_chroma(drug_name, source_file)
    if patent_text is None:
        matched = _find_filename_in_chroma(drug_name, patent_number)
        if matched:
            patent_text = _get_chunks_from_chroma(drug_name, matched)
    if patent_text:
        print(f"[Step 8a] ✓ ChromaDB text: {len(patent_text):,} chars")
    else:
        print(f"[Step 8a] ⚠ No ChromaDB text available")

    # ── Load analysis cache ───────────────────────────────────
    analysis_cache = _load_analysis_cache(drug_name, patent_number)
    if analysis_cache:
        print(f"[Step 8a] ✓ Analysis cache loaded (tag: {analysis_cache.get('tag', '?')})")
    else:
        print(f"[Step 8a] ⚠ No analysis cache")

    # ── Process each independent claim in parallel ────────────
    claims = skeleton.get("independent_claims", [])
    all_limitation_results: list[dict] = []

    if claims:
        sem = asyncio.Semaphore(_WORKERS)

        async def _bounded_claim(claim):
            async with sem:
                return await _search_prior_art_for_claim(
                    skeleton, claim, patent_text, analysis_cache
                )

        claim_results = await asyncio.gather(
            *[_bounded_claim(claim) for claim in claims],
            return_exceptions=True,
        )
        for claim, res in zip(claims, claim_results):
            if isinstance(res, Exception):
                print(f"[Step 8a] ⚠  Claim {claim.get('claim_number')} raised: {res}")
            else:
                all_limitation_results.extend(res)

    output = {
        "patent_number": patent_number,
        "drug_name":     drug_name,
        "priority_date": skeleton.get("priority_date", "Unknown"),
        "jurisdiction":  skeleton.get("jurisdiction", ""),
        "source_file":   source_file,
        "limitation_results": all_limitation_results,
    }

    return output


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
# Main async runner
# ─────────────────────────────────────────────────────────────

async def process_drug(
    drug_name:     str,
    patent_filter: Optional[str] = None,
    rerun_step7:   bool          = False,
    output_dir:    Path          = STEP8A_OUTPUT_DIR,
) -> list[dict]:
    await _ensure_step7(drug_name, force=rerun_step7)

    skeletons = _load_step7_skeletons(drug_name, patent_filter=patent_filter)
    if not skeletons:
        print(f"[Step 8a] No skeletons to process for '{drug_name}'.")
        return []

    print(f"[Step 8a] Processing {len(skeletons)} patent(s) for '{drug_name}' "
          f"(up to {_WORKERS} workers)...")

    sem = asyncio.Semaphore(_WORKERS)

    async def _bounded(skel):
        async with sem:
            return await process_patent(skel)

    raw = await asyncio.gather(
        *[_bounded(skel) for skel in skeletons],
        return_exceptions=True,
    )

    results = []
    for skel, result in zip(skeletons, raw):
        if isinstance(result, Exception):
            print(f"[Step 8a] ⚠  {skel.get('patent_number')} raised: {result}")
        else:
            _write_output(drug_name, result, output_dir)
            results.append(result)

    # Combined summary
    if results:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        combined_path = output_dir / f"{safe}_all_prior_art.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  → Combined JSON   : {combined_path}")

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8a – Invalidity Prior-Art Hunter (ChromaDB + cache + Google Search)"
    )
    parser.add_argument("--drug",        "-d", required=True)
    parser.add_argument("--patent",      "-p", default=None)
    parser.add_argument("--rerun_step7", action="store_true")
    parser.add_argument("--output_dir",  default=str(STEP8A_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 8a] Drug       : {args.drug}")
    print(f"[Step 8a] Model      : {get_model_name()}")
    print(f"[Step 8a] ChromaDB   : {CHROMA_DB_PATH}")
    print(f"[Step 8a] Cache      : {ANALYSIS_CACHE_DIR}")
    print(f"[Step 8a] Output     : {output_dir.resolve()}")

    results = asyncio.run(process_drug(
        drug_name     = args.drug,
        patent_filter = args.patent,
        rerun_step7   = args.rerun_step7,
        output_dir    = output_dir,
    ))

    covered = sum(
        1 for r in results
        for lr in r.get("limitation_results", [])
        if lr.get("limitation_status") == "covered"
    )
    total = sum(len(r.get("limitation_results", [])) for r in results)
    print(f"\n[Step 8a] Done. {len(results)} patent(s), {covered}/{total} limitations covered.")


if __name__ == "__main__":
    main()
