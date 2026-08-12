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

from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

STEP7_OUTPUT_DIR   = Path(os.getenv("STEP7_OUTPUT_DIR",     Path(__file__).parent / "step7_output"))
STEP8A_OUTPUT_DIR  = Path(os.getenv("STEP8A_OUTPUT_DIR",    Path(__file__).parent / "step8a_output"))
CHROMA_DB_PATH     = str(Path(__file__).parent / "chroma_patent_db")
ANALYSIS_CACHE_DIR = Path(os.getenv("ANALYSIS_CACHE_DIR",   Path(__file__).parent / "analysis_cache"))

# ─────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────

_api_key      = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=_api_key)
MODEL         = "gemini-2.5-flash-preview-05-20"

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


def _ensure_step7(drug_name: str, force: bool = False) -> None:
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
    asyncio.get_event_loop().run_until_complete(process_drug(drug_name=drug_name))


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
# Gemini prompt
# ─────────────────────────────────────────────────────────────

_PRIOR_ART_PROMPT = """\
You are an invalidity prior-art hunter for patent claim charting.
You find published disclosures dated STRICTLY BEFORE the priority date
that READ ON each limitation — cited to the exact passage.

PATENT CONTEXT
==============
Patent number : {patent_number}
Jurisdiction  : {jurisdiction}
Drug          : {drug_name}
Priority date : {priority_date}  (HARD date bound — only references published
                                   STRICTLY BEFORE this date qualify)

{context_block}

CLAIM {claim_number} — LIMITATIONS TO SEARCH
=============================================
{limitations_block}

SOURCE ROUTING
==============
{routing_block}

GLOBAL RULES
=============
1. DATE BOUND (hard): a reference qualifies ONLY if its public disclosure date
   is STRICTLY BEFORE {priority_date}. Anything within the 12 months before
   {priority_date} → GRACE bucket: flag "GRACE_PERIOD - admissibility
   case-by-case; counsel to confirm".
2. READ-ON TEST: the passage must disclose the limitation's SPECIFIC feature,
   not merely the same topic. Reject topical-only matches.
3. ANCHORING: every passage MUST have a pinpoint locus:
   - PubMed: PMID + section/paragraph
   - ClinicalTrials.gov: NCT id + section
   - Google Patents: patent number + column:line or claim/paragraph
   - medRxiv: DOI + section
   No locus → do not include.
4. If only a pharmacopoeia/handbook/supplier disclosure would read on an
   excipient/concentration/pH limitation, flag "OUT_OF_CORPUS - paid/CAS/
   pharmacopoeia source required".
5. Budget: MAX 8 query reformulations per limitation. Use structural,
   functional, AND terminological variants (prior art rarely uses claim
   wording). Widen to adjacent sources or examiner-cited art if needed.
6. Never fabricate PMIDs, NCT ids, patent numbers, or DOIs. If unsure, omit.
7. Prefer the fewest strong references per limitation.

OUTPUT FORMAT
=============
Return ONLY a valid JSON array — no markdown fences, no prose outside JSON.
One object per limitation:

[
  {{
    "patent_number": "{patent_number}",
    "priority_date": "{priority_date}",
    "claim_number": {claim_number},
    "limitation_id": "1.P",
    "limitation_text_verbatim": "...",
    "evidence": [
      {{
        "reference_id": "Ma_US6123456_1999",
        "source": "Google Patents",
        "publication_date": "1999-03-15",
        "pre_priority": true,
        "grace_flag": false,
        "locus": "Col. 4, lines 22-35",
        "passage_verbatim": "...",
        "reads_on_rationale": "Discloses the identical aqueous formulation...",
        "confidence": "high"
      }}
    ],
    "limitation_status": "covered",
    "flags": []
  }}
]

If a limitation has NO qualifying art, set limitation_status="uncovered" and
evidence=[], with a flag explaining the gap (e.g. "OUT_OF_CORPUS").
"""


# ─────────────────────────────────────────────────────────────
# Core per-claim processor
# ─────────────────────────────────────────────────────────────

async def _search_prior_art_for_claim(
    skeleton:       dict,
    claim:          dict,
    patent_text:    Optional[str],
    analysis_cache: Optional[dict],
) -> list[dict]:
    """
    Run the prior-art search for all limitations of one independent claim.
    Returns a list of per-limitation result dicts.
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

    # ── Build context block (ChromaDB text + analysis cache) ──
    context_parts = []
    if patent_text:
        # Include a truncated excerpt for context (not the full text — prompt is already large)
        excerpt = patent_text[:80_000] if len(patent_text) > 80_000 else patent_text
        context_parts.append(
            "INDEXED PATENT TEXT (from ChromaDB — for context, not the sole search source)\n"
            "=============================================================================\n"
            f"{excerpt}\n"
        )
    if analysis_cache:
        # Include key analysis fields: claim category, blocking reason, step2 elements
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
                "BLOCKING ANALYSIS CACHE (from the pipeline's classification)\n"
                "=============================================================\n"
                f"{json.dumps(cache_summary, indent=2)}\n"
            )

    context_block = "\n".join(context_parts) if context_parts else "(No cached context available)\n"

    # ── Build limitations block ────────────────────────────────
    lim_lines = []
    for lim in limitations:
        lid   = lim.get("limitation_id", "?")
        text  = lim.get("limitation_text_verbatim", "")
        ltype = lim.get("limitation_type", "unknown")
        flags = lim.get("flags", [])
        flag_str = f"  [FLAGS: {', '.join(flags)}]" if flags else ""
        lim_lines.append(f"  {lid} ({ltype}): {text}{flag_str}")

    # ── Build routing block ────────────────────────────────────
    route_lines = []
    seen_types = set()
    for lim in limitations:
        ltype = lim.get("limitation_type", "unknown")
        if ltype in seen_types:
            continue
        seen_types.add(ltype)
        sources = _SOURCE_ROUTES.get(ltype, ["Google Patents", "PubMed"])
        route_lines.append(f"  {ltype} → {', '.join(sources)}")

    prompt = _PRIOR_ART_PROMPT.format(
        patent_number   = patent_number,
        jurisdiction    = jurisdiction,
        drug_name       = drug_name,
        priority_date   = priority_date,
        claim_number    = claim_number,
        context_block   = context_block,
        limitations_block = "\n".join(lim_lines),
        routing_block   = "\n".join(route_lines),
    )

    print(f"[Step 8a] Searching prior art for {patent_number} Claim {claim_number} "
          f"({len(limitations)} limitation(s))...")

    try:
        response = await gemini_client.aio.models.generate_content(
            model   = MODEL,
            contents = prompt,
            config  = types.GenerateContentConfig(
                tools       = [types.Tool(google_search=types.GoogleSearch())],
                temperature = 0.0,
            ),
        )

        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        if not raw:
            print(f"[Step 8a] Empty response for claim {claim_number}")
            return []

        results = json.loads(raw)
        if isinstance(results, dict):
            results = [results]

        covered = sum(1 for r in results if r.get("limitation_status") == "covered")
        total   = len(results)
        print(f"[Step 8a] ✓ Claim {claim_number}: {covered}/{total} limitations covered")

        return results

    except json.JSONDecodeError as e:
        print(f"[Step 8a] JSON parse error for claim {claim_number}: {e}")
        print(f"          Raw (first 500): {raw[:500]}")
        return []
    except Exception as e:
        print(f"[Step 8a] Gemini error for claim {claim_number}: {e}")
        return []


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

    # ── Process each independent claim ────────────────────────
    claims = skeleton.get("independent_claims", [])
    all_limitation_results: list[dict] = []

    for claim in claims:
        cn = claim.get("claim_number", "?")
        lim_results = await _search_prior_art_for_claim(
            skeleton, claim, patent_text, analysis_cache
        )
        all_limitation_results.extend(lim_results)

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
    _ensure_step7(drug_name, force=rerun_step7)

    skeletons = _load_step7_skeletons(drug_name, patent_filter=patent_filter)
    if not skeletons:
        print(f"[Step 8a] No skeletons to process for '{drug_name}'.")
        return []

    results = []
    for skel in skeletons:
        result = await process_patent(skel)
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
    print(f"[Step 8a] Model      : {MODEL}")
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
