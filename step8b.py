"""
step8b.py — Claim-Chart Combination Optimiser
===============================================
One independent claim at a time.

Two-phase approach:
  Phase 1 (deterministic): Build coverage matrix, identify §102 anticipation
           grounds, run greedy set cover for initial §103 candidates.
  Phase 2 (LLM-assisted):  Gemini evaluates the coverage matrix to find
           alternative §103 combinations, rank grounds by strength, provide
           combination rationales, and recommend gap-closure strategies.

Data sources:
  - Step 8a output (per-limitation evidence)
  - Step 7 output  (claim skeletons)
  - ChromaDB       (patent chunks — for technical context)
  - Analysis cache  (blocking analyser metadata)

Model: gemini-2.5-flash-preview-05-20

Usage:
    python step8b.py --drug Axitinib                       # all patents
    python step8b.py --drug Axitinib --patent US10123456   # one patent
    python step8b.py --drug Axitinib --rerun_step8a        # re-run step8a first
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from llm_client import generate, parse_json_response, get_model_name, is_claude, is_gemini

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

STEP7_OUTPUT_DIR   = Path(os.getenv("STEP7_OUTPUT_DIR",   Path(__file__).parent / "step7_output"))
STEP8A_OUTPUT_DIR  = Path(os.getenv("STEP8A_OUTPUT_DIR",  Path(__file__).parent / "step8a_output"))
STEP8B_OUTPUT_DIR  = Path(os.getenv("STEP8B_OUTPUT_DIR",  Path(__file__).parent / "step8b_output"))
CHROMA_DB_PATH     = str(Path(__file__).parent / "chroma_patent_db")
ANALYSIS_CACHE_DIR = Path(os.getenv("ANALYSIS_CACHE_DIR", Path(__file__).parent / "analysis_cache"))

# ─────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────

_WORKERS       = int(os.getenv("PIPELINE_WORKERS", "6"))

# ─────────────────────────────────────────────────────────────
# ChromaDB + analysis cache helpers
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
    except Exception:
        return None


def _find_filename_in_chroma(drug_name: str, patent_number: str) -> Optional[str]:
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


def _load_analysis_cache(drug_name: str, patent_number: str) -> Optional[dict]:
    safe_drug = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    cache_dir = ANALYSIS_CACHE_DIR / safe_drug
    if not cache_dir.exists():
        return None
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
# Step 8a / Step 7 loaders
# ─────────────────────────────────────────────────────────────

def _load_step8a(drug_name: str, patent_filter: Optional[str] = None) -> list[dict]:
    safe    = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    pattern = f"{safe}_*_prior_art.json"
    files   = sorted(STEP8A_OUTPUT_DIR.glob(pattern))
    files   = [f for f in files if "_all_prior_art" not in f.name]
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if patent_filter and data.get("patent_number", "").upper() != patent_filter.upper():
                continue
            results.append(data)
        except Exception as e:
            print(f"[Step 8b] Failed to read {f.name}: {e}")
    return results


def _load_step7_skeleton(drug_name: str, patent_number: str) -> Optional[dict]:
    safe    = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    safe_pn = re.sub(r"[^a-zA-Z0-9_-]", "_", patent_number)
    path    = STEP7_OUTPUT_DIR / f"{safe}_{safe_pn}_claim_skeleton.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


async def _ensure_step8a(drug_name: str, force: bool = False) -> None:
    safe     = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    existing = list(STEP8A_OUTPUT_DIR.glob(f"{safe}_*_prior_art.json"))
    existing = [f for f in existing if "_all_prior_art" not in f.name]
    if existing and not force:
        return
    print(f"[Step 8b] Step 8a output not found — running step8a for '{drug_name}'...")
    try:
        from step8a import process_drug
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("step8a", Path(__file__).parent / "step8a.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        process_drug = mod.process_drug
    await process_drug(drug_name=drug_name)


# ─────────────────────────────────────────────────────────────
# Phase 1: Deterministic coverage matrix + ground identification
# ─────────────────────────────────────────────────────────────

def _build_coverage_matrix(
    claim_number:       int,
    limitation_results: list[dict],
) -> dict:
    """
    Build the raw coverage matrix and identify deterministic grounds.
    Returns intermediate data for Phase 2 (LLM refinement).
    """
    claim_lims = [
        lr for lr in limitation_results
        if lr.get("claim_number") == claim_number
    ]

    all_lim_ids = [lr["limitation_id"] for lr in claim_lims]
    total       = len(all_lim_ids)

    if total == 0:
        return {"total": 0, "all_lim_ids": [], "ref_coverage": {},
                "grace_coverage": {}, "lim_texts": {}, "lim_flags": {},
                "covered_lims": set(), "claim_lims": []}

    ref_coverage:   dict[str, set] = defaultdict(set)
    grace_coverage: dict[str, set] = defaultdict(set)
    lim_texts:      dict[str, str] = {}
    lim_flags:      dict[str, list] = {}
    covered_lims:   set = set()

    # Also collect reference metadata for LLM context
    ref_meta: dict[str, dict] = {}

    for lr in claim_lims:
        lid   = lr["limitation_id"]
        ltext = lr.get("limitation_text_verbatim", "")
        lim_texts[lid] = ltext
        lim_flags[lid] = lr.get("flags", [])

        for ev in lr.get("evidence", []):
            ref_id = ev.get("reference_id", "")
            if not ref_id:
                continue

            if ref_id not in ref_meta:
                ref_meta[ref_id] = {
                    "source":           ev.get("source", ""),
                    "publication_date": ev.get("publication_date", ""),
                    "pre_priority":     ev.get("pre_priority", False),
                    "grace_flag":       ev.get("grace_flag", False),
                }

            if ev.get("grace_flag"):
                grace_coverage[ref_id].add(lid)
            elif ev.get("pre_priority"):
                ref_coverage[ref_id].add(lid)
                covered_lims.add(lid)

    return {
        "total":          total,
        "all_lim_ids":    all_lim_ids,
        "ref_coverage":   {k: sorted(v) for k, v in ref_coverage.items()},
        "grace_coverage": {k: sorted(v) for k, v in grace_coverage.items()},
        "lim_texts":      lim_texts,
        "lim_flags":      lim_flags,
        "covered_lims":   covered_lims,
        "ref_meta":       ref_meta,
        "claim_lims":     claim_lims,
    }


def _deterministic_grounds(
    patent_number: str,
    claim_number:  int,
    matrix:        dict,
) -> dict:
    """
    Phase 1: deterministic §102 + greedy §103.
    Returns the claim grounds dict.
    """
    total       = matrix["total"]
    all_lim_ids = matrix["all_lim_ids"]
    ref_cov     = {k: set(v) for k, v in matrix["ref_coverage"].items()}
    grace_cov   = {k: set(v) for k, v in matrix["grace_coverage"].items()}
    lim_texts   = matrix["lim_texts"]
    lim_flags   = matrix["lim_flags"]
    covered_lims = matrix["covered_lims"]

    if total == 0:
        return {
            "patent_number": patent_number, "claim_number": claim_number,
            "total_limitations": 0, "grounds": [],
            "gap_limitations": [], "grace_only_limitations": [],
        }

    grounds    = []
    ground_idx = 0

    # §102: single reference covers ALL
    for ref_id, covered in ref_cov.items():
        if covered >= set(all_lim_ids):
            ground_idx += 1
            grounds.append({
                "ground_id":       f"G{ground_idx}",
                "basis":           "102",
                "references":      [ref_id],
                "reference_count": 1,
                "covered_lims":    sorted(covered),
                "coverage_pct":    100.0,
            })

    # §103: greedy set cover
    if not any(g["basis"] == "102" for g in grounds) and ref_cov:
        combo_refs    = []
        combo_covered = set()
        pool          = dict(ref_cov)

        while set(all_lim_ids) - combo_covered and pool:
            best_ref = max(pool, key=lambda r: len(pool[r] - combo_covered))
            gain     = pool[best_ref] - combo_covered
            if not gain:
                break
            combo_refs.append(best_ref)
            combo_covered |= gain
            del pool[best_ref]

        if combo_refs:
            ground_idx += 1
            pct = round(len(combo_covered) / total * 100, 1)
            grounds.append({
                "ground_id":       f"G{ground_idx}",
                "basis":           "103",
                "references":      combo_refs,
                "reference_count": len(combo_refs),
                "covered_lims":    sorted(combo_covered),
                "coverage_pct":    pct,
            })

    # Gap list + grace annex
    all_covered = set()
    for g in grounds:
        all_covered |= set(g["covered_lims"])

    gap_lims = [
        {"limitation_id": lid, "limitation_text": lim_texts.get(lid, ""), "flags": lim_flags.get(lid, [])}
        for lid in all_lim_ids if lid not in all_covered
    ]

    grace_only = []
    for lid in all_lim_ids:
        if lid not in all_covered:
            for ref_id, g_lims in grace_cov.items():
                if lid in g_lims:
                    grace_only.append({"limitation_id": lid, "reference_id": ref_id})

    return {
        "patent_number":          patent_number,
        "claim_number":           claim_number,
        "total_limitations":      total,
        "grounds":                grounds,
        "gap_limitations":        gap_lims,
        "grace_only_limitations": grace_only,
    }


# ─────────────────────────────────────────────────────────────
# Phase 2: LLM-assisted refinement
# ─────────────────────────────────────────────────────────────

_LLM_OPTIMISE_PROMPT = """\
You are an invalidity claim-chart combination optimiser for patent litigation.
You do NO new prior-art searching. You work only with the evidence already found.

PATENT: {patent_number}
DRUG: {drug_name}
CLAIM: {claim_number}
PRIORITY DATE: {priority_date}

{context_block}

COVERAGE MATRIX (rows = limitations, cols = references)
========================================================
{matrix_text}

DETERMINISTIC RESULTS (Phase 1)
================================
§102 grounds: {n_102}
§103 grounds: {n_103} (greedy set cover)
Gaps: {n_gaps} limitation(s) uncovered
Grace-only: {n_grace}

Phase 1 grounds:
{grounds_json}

Gap limitations:
{gaps_json}

YOUR TASKS
==========
1. ALTERNATIVE §103 COMBINATIONS: Using the coverage matrix, find up to 3
   alternative reference combinations (beyond the greedy set cover) that cover
   the same or more limitations. Rank them by:
   - Fewer references = stronger
   - Same field of art = stronger
   - Earlier publication dates = stronger
   Only propose combinations where EVERY reference has a qualifying (non-grace,
   pre-priority) passage with a locus for the limitations it covers.

2. COMBINATION RATIONALE: For each §103 ground (Phase 1 + your alternatives),
   provide a brief (1-2 sentence) technical rationale explaining why a person
   of ordinary skill in the art (POSITA) would look to combine these references.
   This is a FACTUAL note (same field, same problem, cross-referenced), NOT a
   legal argument.

3. GROUND STRENGTH RANKING: Rank ALL grounds (§102 + all §103) from strongest
   to weakest. Explain the ranking in one sentence per ground.

4. GAP RECOMMENDATIONS: For each uncovered limitation, recommend the type of
   source most likely to cover it (pharmacopoeia/handbook, CAS registry,
   clinical trial archive, foreign patent, supplier datasheet, etc.)

OUTPUT: single valid JSON, no markdown fences.

{{
  "patent_number": "{patent_number}",
  "claim_number": {claim_number},
  "alternative_103_grounds": [
    {{
      "ground_id": "G_ALT_1",
      "references": ["ref1", "ref2"],
      "covered_lims": ["1.a", "1.b"],
      "coverage_pct": 85.0,
      "combination_rationale": "Both references address formulation stability..."
    }}
  ],
  "combination_rationales": {{
    "G1": "Single reference anticipates all limitations...",
    "G2": "Ref A provides the compound while Ref B provides the formulation..."
  }},
  "strength_ranking": [
    {{ "ground_id": "G1", "rank": 1, "reason": "Complete anticipation..." }}
  ],
  "gap_recommendations": [
    {{ "limitation_id": "1.c", "recommended_source": "USP monograph / pharmacopoeia",
       "rationale": "Excipient concentration ranges are typically published in..." }}
  ]
}}
"""


async def _llm_refine_grounds(
    patent_number:  str,
    drug_name:      str,
    claim_number:   int,
    priority_date:  str,
    matrix:         dict,
    det_grounds:    dict,
    patent_text:    Optional[str],
    analysis_cache: Optional[dict],
) -> Optional[dict]:
    """
    Phase 2: Use Gemini to find alternative combinations, rationales,
    and gap recommendations.
    """
    total       = matrix["total"]
    ref_cov     = matrix["ref_coverage"]
    lim_texts   = matrix["lim_texts"]
    ref_meta    = matrix.get("ref_meta", {})

    if total == 0:
        return None

    # Build matrix text
    all_lim_ids = matrix["all_lim_ids"]
    all_ref_ids = sorted(ref_cov.keys())

    if not all_ref_ids:
        print(f"[Step 8b LLM] No references to optimise for claim {claim_number}")
        return None

    matrix_lines = ["Limitation | " + " | ".join(all_ref_ids)]
    matrix_lines.append("-" * 40)
    for lid in all_lim_ids:
        cells = []
        for ref_id in all_ref_ids:
            if lid in ref_cov.get(ref_id, []):
                cells.append("✓")
            else:
                cells.append("·")
        ltext_short = lim_texts.get(lid, "")[:60]
        matrix_lines.append(f"{lid} ({ltext_short}...) | " + " | ".join(cells))

    # Build context block
    context_parts = []
    if patent_text:
        excerpt = patent_text[:40_000] if len(patent_text) > 40_000 else patent_text
        context_parts.append(
            "PATENT TEXT EXCERPT (from ChromaDB)\n"
            "====================================\n"
            f"{excerpt}\n"
        )
    if analysis_cache:
        cache_summary = {
            k: analysis_cache.get(k)
            for k in ["claim_category", "tag", "reason", "step3_evidence_summary"]
            if analysis_cache.get(k)
        }
        if cache_summary:
            context_parts.append(
                "ANALYSIS CACHE\n"
                "===============\n"
                f"{json.dumps(cache_summary, indent=2)}\n"
            )
    context_block = "\n".join(context_parts) if context_parts else "(No additional context)\n"

    prompt = _LLM_OPTIMISE_PROMPT.format(
        patent_number = patent_number,
        drug_name     = drug_name,
        claim_number  = claim_number,
        priority_date = priority_date,
        context_block = context_block,
        matrix_text   = "\n".join(matrix_lines),
        n_102         = sum(1 for g in det_grounds["grounds"] if g["basis"] == "102"),
        n_103         = sum(1 for g in det_grounds["grounds"] if g["basis"] == "103"),
        n_gaps        = len(det_grounds["gap_limitations"]),
        n_grace       = len(det_grounds["grace_only_limitations"]),
        grounds_json  = json.dumps(det_grounds["grounds"], indent=2),
        gaps_json     = json.dumps(det_grounds["gap_limitations"], indent=2),
    )

    print(f"[Step 8b LLM] Optimising grounds for {patent_number} Claim {claim_number} "
          f"| model: {get_model_name()}...")

    try:
        raw = await generate(
            prompt         = prompt,
            use_web_search = True,
            temperature    = 0.1,
            max_output_tokens = 65536,
        )

        if not raw:
            print(f"[Step 8b LLM] Empty response for claim {claim_number}")
            return None

        result = parse_json_response(raw)
        if result is None:
            print(f"[Step 8b LLM] JSON parse failed for claim {claim_number}")
            return None

        print(f"[Step 8b LLM] ✓ Claim {claim_number}: "
              f"{len(result.get('alternative_103_grounds', []))} alternative(s), "
              f"{len(result.get('strength_ranking', []))} ranked ground(s)")
        return result

    except Exception as e:
        print(f"[Step 8b LLM] LLM error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Combined per-claim processor
# ─────────────────────────────────────────────────────────────

async def _build_claim_grounds(
    patent_number:      str,
    drug_name:          str,
    claim_number:       int,
    priority_date:      str,
    limitation_results: list[dict],
    patent_text:        Optional[str],
    analysis_cache:     Optional[dict],
) -> dict:
    """
    Phase 1 (deterministic) + Phase 2 (LLM) for one claim.
    Merges LLM refinements into the deterministic output.
    """
    # Phase 1
    matrix     = _build_coverage_matrix(claim_number, limitation_results)
    det_result = _deterministic_grounds(patent_number, claim_number, matrix)

    # Phase 2
    llm_result = await _llm_refine_grounds(
        patent_number, drug_name, claim_number, priority_date,
        matrix, det_result, patent_text, analysis_cache,
    )

    # Merge LLM refinements into the deterministic output
    if llm_result:
        # Add alternative §103 grounds
        alt_grounds = llm_result.get("alternative_103_grounds", [])
        existing_idx = len(det_result["grounds"])
        for i, alt in enumerate(alt_grounds):
            existing_idx += 1
            alt["ground_id"]       = alt.get("ground_id", f"G{existing_idx}")
            alt["basis"]           = "103"
            alt["reference_count"] = len(alt.get("references", []))
            det_result["grounds"].append(alt)

        # Attach rationales to grounds
        rationales = llm_result.get("combination_rationales", {})
        for g in det_result["grounds"]:
            gid = g["ground_id"]
            if gid in rationales:
                g["combination_rationale"] = rationales[gid]

        # Attach strength ranking
        det_result["strength_ranking"] = llm_result.get("strength_ranking", [])

        # Attach gap recommendations
        gap_recs = {r["limitation_id"]: r for r in llm_result.get("gap_recommendations", [])}
        for gap in det_result["gap_limitations"]:
            lid = gap["limitation_id"]
            if lid in gap_recs:
                gap["recommended_source"] = gap_recs[lid].get("recommended_source", "")
                gap["recommendation"]     = gap_recs[lid].get("rationale", "")

    return det_result


# ─────────────────────────────────────────────────────────────
# Per-patent orchestrator
# ─────────────────────────────────────────────────────────────

async def process_patent(
    prior_art_data: dict,
    skeleton:       Optional[dict],
) -> dict:
    """Process all claims for one patent."""
    patent_number      = prior_art_data["patent_number"]
    drug_name          = prior_art_data.get("drug_name", "")
    priority_date      = prior_art_data.get("priority_date", "")
    source_file        = prior_art_data.get("source_file", "")
    limitation_results = prior_art_data.get("limitation_results", [])

    print(f"\n[Step 8b] {'═'*50}")
    print(f"[Step 8b] Patent: {patent_number} | Drug: {drug_name}")

    # Load ChromaDB text
    patent_text = None
    if source_file:
        patent_text = _get_chunks_from_chroma(drug_name, source_file)
    if patent_text is None:
        matched = _find_filename_in_chroma(drug_name, patent_number)
        if matched:
            patent_text = _get_chunks_from_chroma(drug_name, matched)
    if patent_text:
        print(f"[Step 8b] ✓ ChromaDB text: {len(patent_text):,} chars")

    # Load analysis cache
    analysis_cache = _load_analysis_cache(drug_name, patent_number)
    if analysis_cache:
        print(f"[Step 8b] ✓ Analysis cache loaded")

    # Determine claim numbers
    claim_numbers = sorted(set(
        lr.get("claim_number") for lr in limitation_results
        if lr.get("claim_number") is not None
    ))
    if not claim_numbers and skeleton:
        claim_numbers = [c["claim_number"] for c in skeleton.get("independent_claims", [])]

    all_claim_grounds = []
    if claim_numbers:
        sem = asyncio.Semaphore(_WORKERS)

        async def _bounded_claim(cn):
            async with sem:
                return await _build_claim_grounds(
                    patent_number, drug_name, cn, priority_date,
                    limitation_results, patent_text, analysis_cache,
                )

        claim_results = await asyncio.gather(
            *[_bounded_claim(cn) for cn in claim_numbers],
            return_exceptions=True,
        )
        for cn, res in zip(claim_numbers, claim_results):
            if isinstance(res, Exception):
                print(f"[Step 8b] ⚠  Claim {cn} raised: {res}")
            else:
                all_claim_grounds.append(res)

    output = {
        "patent_number": patent_number,
        "drug_name":     drug_name,
        "priority_date": priority_date,
        "jurisdiction":  prior_art_data.get("jurisdiction", ""),
        "claim_grounds": all_claim_grounds,
    }

    # Print summary
    for cg in all_claim_grounds:
        cn      = cg["claim_number"]
        total   = cg["total_limitations"]
        n_102   = sum(1 for g in cg["grounds"] if g["basis"] == "102")
        n_103   = sum(1 for g in cg["grounds"] if g["basis"] == "103")
        n_gaps  = len(cg["gap_limitations"])
        n_grace = len(cg["grace_only_limitations"])
        ranked  = len(cg.get("strength_ranking", []))
        print(f"  Claim {cn}: {total} lim(s) | "
              f"102: {n_102} | 103: {n_103} | "
              f"gaps: {n_gaps} | grace: {n_grace} | ranked: {ranked}")

    return output


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def _write_output(drug_name: str, result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_drug   = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    safe_patent = re.sub(r"[^a-zA-Z0-9_-]", "_", result.get("patent_number", "unknown"))

    # JSON
    json_path = output_dir / f"{safe_drug}_{safe_patent}_grounds.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  → Grounds JSON    : {json_path}")

    # Markdown
    md_path = output_dir / f"{safe_drug}_{safe_patent}_coverage.md"
    lines   = [
        f"# Step 8b Coverage — {result['patent_number']}",
        f"**Drug:** {drug_name} | **Priority date:** {result.get('priority_date', '?')}",
    ]
    for cg in result.get("claim_grounds", []):
        cn = cg["claim_number"]
        lines.append(f"\n## Claim {cn} ({cg['total_limitations']} limitations)")

        if cg["grounds"]:
            lines.append("\n| Ground | Basis | References | Coverage | Rationale |")
            lines.append("|--------|-------|-----------|----------|-----------|")
            for g in cg["grounds"]:
                refs = ", ".join(g.get("references", []))
                rat  = g.get("combination_rationale", "—")[:80]
                cov  = f"{g.get('coverage_pct', 0)}%"
                lines.append(f"| {g['ground_id']} | §{g['basis']} | {refs} | {cov} | {rat} |")
        else:
            lines.append("\n*No grounds identified.*")

        # Strength ranking
        if cg.get("strength_ranking"):
            lines.append(f"\n### Strength Ranking")
            for sr in cg["strength_ranking"]:
                lines.append(f"{sr.get('rank', '?')}. **{sr.get('ground_id', '?')}** — {sr.get('reason', '')}")

        # Gaps with recommendations
        if cg["gap_limitations"]:
            lines.append(f"\n### Gaps ({len(cg['gap_limitations'])})")
            for gap in cg["gap_limitations"]:
                flags = f" [{', '.join(gap['flags'])}]" if gap.get("flags") else ""
                rec   = f" → *{gap.get('recommended_source', '')}*" if gap.get("recommended_source") else ""
                lines.append(f"- **{gap['limitation_id']}**: {gap.get('limitation_text', '')}{flags}{rec}")
                if gap.get("recommendation"):
                    lines.append(f"  {gap['recommendation']}")

        if cg["grace_only_limitations"]:
            lines.append(f"\n### Grace Annex ({len(cg['grace_only_limitations'])})")
            for ga in cg["grace_only_limitations"]:
                lines.append(f"- **{ga['limitation_id']}** ← {ga['reference_id']} (within 12mo of priority)")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → Coverage MD     : {md_path}")


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

async def run_for_drug(
    drug_name:     str,
    patent_filter: Optional[str] = None,
    rerun_step8a:  bool          = False,
    output_dir:    Path          = STEP8B_OUTPUT_DIR,
) -> list[dict]:
    await _ensure_step8a(drug_name, force=rerun_step8a)

    prior_art_results = _load_step8a(drug_name, patent_filter=patent_filter)
    if not prior_art_results:
        print(f"[Step 8b] No step8a data for '{drug_name}'.")
        return []

    print(f"[Step 8b] Processing {len(prior_art_results)} patent(s) for '{drug_name}' "
          f"(up to {_WORKERS} workers)...")

    sem = asyncio.Semaphore(_WORKERS)

    async def _bounded(pa):
        async with sem:
            pn       = pa["patent_number"]
            skeleton = _load_step7_skeleton(drug_name, pn)
            return await process_patent(pa, skeleton)

    raw = await asyncio.gather(
        *[_bounded(pa) for pa in prior_art_results],
        return_exceptions=True,
    )

    results = []
    for pa, result in zip(prior_art_results, raw):
        if isinstance(result, Exception):
            print(f"[Step 8b] ⚠  {pa['patent_number']} raised: {result}")
        else:
            _write_output(drug_name, result, output_dir)
            results.append(result)

    if results:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        combined_path = output_dir / f"{safe}_all_grounds.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  → Combined JSON   : {combined_path}")

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8b – Claim-Chart Combination Optimiser (deterministic + LLM refinement)"
    )
    parser.add_argument("--drug",         "-d", required=True)
    parser.add_argument("--patent",       "-p", default=None)
    parser.add_argument("--rerun_step8a", action="store_true")
    parser.add_argument("--output_dir",   default=str(STEP8B_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 8b] Drug       : {args.drug}")
    print(f"[Step 8b] Model      : {MODEL}")
    print(f"[Step 8b] ChromaDB   : {CHROMA_DB_PATH}")
    print(f"[Step 8b] Cache      : {ANALYSIS_CACHE_DIR}")
    print(f"[Step 8b] Output     : {output_dir.resolve()}")

    results = asyncio.run(run_for_drug(
        drug_name     = args.drug,
        patent_filter = args.patent,
        rerun_step8a  = args.rerun_step8a,
        output_dir    = output_dir,
    ))

    total_102 = sum(
        1 for r in results for cg in r.get("claim_grounds", [])
        for g in cg.get("grounds", []) if g["basis"] == "102"
    )
    total_103 = sum(
        1 for r in results for cg in r.get("claim_grounds", [])
        for g in cg.get("grounds", []) if g["basis"] == "103"
    )
    print(f"\n[Step 8b] Done. §102 grounds: {total_102}, §103 grounds: {total_103}")


if __name__ == "__main__":
    main()
