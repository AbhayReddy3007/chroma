"""
tools.py
─────────
Main orchestration entry point for the patent analysis pipeline.

Imports and delegates to the following modules:

  gcs_lister.py            — Step 1:  List PDFs from GCS
  indexer.py               — Steps 2–3: Index + deduplicate + date backfill
  blocking_analyser.py     — Steps 4–5: Blocking analysis + business rules
  phase_fetcher.py         — Steps 6–7: Clinical phase (BigQuery + fallback Excel)
  calculators.py           — Steps 8–12: Approval year, exclusivity, expiry,
                                          years to entry, pediatric, PTE, score
  approval_date_fetcher.py — Step 13:   Real-world approval dates (FDA/EMA/Gemini/news)
  excel_exporter.py        — Steps 14–15: Per-drug + combined Excel export

Results are cached as JSON files (results_cache/ and analysis_cache/).
When re-running, only NEW drugs or drugs with NEW patents get re-analysed.

Usage:
    import asyncio
    from tools import get_dimension_i_patent_data

    # Analyse all jurisdictions
    result = asyncio.run(get_dimension_i_patent_data("Semaglutide"))

    # Analyse only US and EU patents
    result = asyncio.run(get_dimension_i_patent_data("Semaglutide", jurisdictions=["US", "EP"]))
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Module imports ───────────────────────────────────────────────────────────

from .gcs_lister import list_drug_pdf_filenames_from_gcs, GCS_BUCKET_NAME

from .indexer import (
    get_or_create_collection,
    run_indexing,
)

from .blocking_analyser import load_formulation_excel

from .blocking_analyser import (
    invalidate_drug_cache as invalidate_blocking_cache,
)

from .phase_fetcher import (
    fetch_clinical_timeline,
    assign_patent_phases,
)

from .calculators import run_calculations

from .approval_date_fetcher import fetch_approval_dates

from .excel_exporter import export_to_excel, export_combined_excel

# ── Environment config ───────────────────────────────────────────────────────

BQ_TABLE_NAME      = os.getenv("BQ_TABLE_NAME")
BQ_PROJECT_ID      = os.getenv("BQ_PROJECT_ID")
BQ_DATASET_ID      = os.getenv("BQ_DATASET_ID")
BQ_SERVICE_ACCOUNT = os.getenv("BQ_SERVICE_ACCOUNT")

# ── Results cache (JSON file) ─────────────────────────────────────────────
#
# Structure:
#   <RESULTS_CACHE_DIR>/<drug_name>.json
#     {
#       "drug": "...",
#       "analysis_date": "...",
#       "source_files": ["file1.pdf", "file2.pdf"],
#       "patents": [ {patent_dict}, ... ]
#     }

RESULTS_CACHE_DIR = Path(
    os.getenv("RESULTS_CACHE_DIR", Path(__file__).parent / "results_cache")
)
RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _results_cache_path(drug_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    return RESULTS_CACHE_DIR / f"{safe}.json"


def _store_results(drug_name: str, patents: list, analysis_date: str, source_files: list):
    """Stores the full analysis results for a drug as a JSON file."""
    try:
        payload = {
            "drug":          drug_name,
            "analysis_date": analysis_date,
            "source_files":  sorted(source_files),
            "patents":       patents,
        }
        path = _results_cache_path(drug_name)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"[RESULTS CACHE] Stored {len(patents)} patent(s) for '{drug_name}' → {path.name}")
    except Exception as e:
        print(f"[RESULTS CACHE] Failed to store results: {e}")


def _load_cached_results(
    drug_name: str,
    current_file_count: int,
    current_files: Optional[List[str]] = None,
) -> Optional[list]:
    """
    Loads cached results for a drug if they exist and are not stale.

    Staleness detection:
      - If the EXACT SAME set of files → full cache hit (return all results)
      - If files were ADDED → returns None so the pipeline can do incremental
        analysis (per-patent cache in blocking_analyser handles the rest)
      - If files were REMOVED → returns None (full re-analysis needed)

    Returns:
        List of patent dicts if fully cached, None otherwise.
    """
    path = _results_cache_path(drug_name)
    if not path.exists():
        return None

    try:
        payload      = json.loads(path.read_text(encoding="utf-8"))
        cached_files = set(payload.get("source_files", []))
        cached_date  = payload.get("analysis_date", "")
        patents      = payload.get("patents", [])

        if not patents:
            return None

        # Per-file staleness check
        if current_files:
            current_files_set = set(current_files)

            if current_files_set == cached_files:
                # Exact match — full cache hit
                print(f"[RESULTS CACHE] Full cache hit for '{drug_name}' "
                      f"({len(patents)} patent(s), analysed: {cached_date})")
                return patents

            elif current_files_set > cached_files:
                new_files = current_files_set - cached_files
                print(f"[RESULTS CACHE] {len(new_files)} new file(s) for '{drug_name}': "
                      f"{sorted(new_files)}")
                print(f"[RESULTS CACHE] Per-patent cache will handle incremental analysis")
                return None

            else:
                removed = cached_files - current_files_set
                print(f"[RESULTS CACHE] {len(removed)} file(s) removed for '{drug_name}' "
                      f"→ full re-analysis")
                return None
        else:
            # Fallback: count-based check
            if len(cached_files) != current_file_count:
                print(f"[RESULTS CACHE] Cache stale for '{drug_name}': "
                      f"cached {len(cached_files)} vs current {current_file_count}")
                return None

            print(f"[RESULTS CACHE] Loaded {len(patents)} cached patent(s) for '{drug_name}' "
                  f"(analysed: {cached_date})")
            return patents

    except (json.JSONDecodeError, OSError) as e:
        print(f"[RESULTS CACHE] Failed to load cache for '{drug_name}': {e}")

    return None


def _filter_by_jurisdictions(patents: list, jurisdictions: Optional[List[str]]) -> list:
    """Filters patents to only the specified jurisdictions."""
    if not jurisdictions:
        return patents  # no filter = all jurisdictions

    jur_set = {j.upper() for j in jurisdictions}
    filtered = [p for p in patents if (p.get("jurisdiction") or "").upper() in jur_set]
    print(f"[FILTER] {len(patents)} patents → {len(filtered)} after jurisdiction filter {list(jur_set)}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def get_dimension_i_patent_data(
    drug_name:                 str,
    has_pediatric_exclusivity: bool = True,
    reindex:                   bool = False,
    jurisdictions:             Optional[List[str]] = None,
    bq_table_name:             Optional[str] = None,
    bq_project_id:             Optional[str] = None,
    bq_dataset_id:             Optional[str] = None,
    bq_service_account:        Optional[str] = None,
) -> dict:
    """
    Full RAG pipeline for patent analysis of a single drug.

    Args:
        drug_name:                 Drug name (must match GCS folder)
        has_pediatric_exclusivity: Default True
        reindex:                   Force re-indexing
        jurisdictions:             Filter results to these jurisdictions (e.g. ["US", "EP"]).
                                   None = all jurisdictions.
        bq_*:                      Override env vars

    Returns dict with patents list, scores, dates, etc.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[PIPELINE] Starting for: {drug_name}")
    if jurisdictions:
        print(f"[PIPELINE] Jurisdiction filter: {jurisdictions}")
    print(f"{'='*60}")

    _bq_table   = bq_table_name      or BQ_TABLE_NAME
    _bq_project = bq_project_id      or BQ_PROJECT_ID
    _bq_dataset = bq_dataset_id      or BQ_DATASET_ID
    _bq_sa      = bq_service_account or BQ_SERVICE_ACCOUNT

    # ── Step 1: List PDFs from GCS ────────────────────────────────────────────
    print(f"\n[STEP 1] Listing PDFs from GCS...")
    pdf_refs = list_drug_pdf_filenames_from_gcs(drug_name)
    print(f"[STEP 1] {len(pdf_refs)} PDF(s) found")

    if not pdf_refs:
        return {
            "drug_name":     drug_name,
            "error":         f"No PDFs found for '{drug_name}' in GCS bucket '{GCS_BUCKET_NAME}'.",
            "patents":       [],
            "source_files":  [],
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        }

    # ── Check results cache — skip analysis if already done ───────────────────
    if not reindex:
        current_files = [ref["filename"] for ref in pdf_refs]
        cached = _load_cached_results(drug_name, len(pdf_refs), current_files=current_files)
        if cached:
            # Apply jurisdiction filter if requested
            filtered = _filter_by_jurisdictions(cached, jurisdictions)

            # Recalculate scores for the filtered subset
            filtered = run_calculations(filtered)

            analysis_date = datetime.now().strftime("%Y-%m-%d")

            # Generate Excel even from cache
            excel_path          = export_to_excel(drug_name, filtered, analysis_date)
            combined_excel_path = export_combined_excel(analysis_date)

            elapsed = time.time() - t0
            print(f"\n[PIPELINE] Using cached results — {len(filtered)} patent(s) in {elapsed:.1f}s")

            return {
                "drug_name":               drug_name,
                "analysis_date":           analysis_date,
                "patents":                 filtered,
                "source_files":            [ref["filename"] for ref in pdf_refs],
                "processing_time_seconds": round(elapsed, 1),
                "clinical_timeline":       {},
                "phase_data_source":       "cached",
                "excel_path":              excel_path,
                "combined_excel_path":     combined_excel_path,
                "from_cache":              True,
            }

    # ── Steps 2–5: Pipelined indexing + analysis ─────────────────────────────
    #
    # Instead of: index ALL patents → then analyse ALL patents (sequential),
    # we pipeline so that Step 1 analysis fires per-patent the moment it is
    # confirmed ready in ChromaDB, overlapped with the clinical timeline fetch.
    #
    # Concurrency map:
    #   • run_indexing    — up to INDEXING_WORKERS (5) patents in parallel
    #   • Step 1 tasks    — fire immediately on each on_indexed callback,
    #                       throttled by _ANALYSIS_CONCURRENCY (5) inside
    #                       blocking_analyser._run_step1_only
    #   • Timeline fetch  — runs concurrently with both indexing and Step 1
    #   • Steps 2+ (Phase 2) — run after all Step 1s complete (CoM routing
    #                       needs the full patent set before it can decide
    #                       which patent is the primary CoM per jurisdiction)
    #
    print(f"\n[STEPS 2–5] Starting pipelined indexing + analysis...")
    collection = get_or_create_collection(drug_name)

    # Import per-patent Step 1 helpers from blocking_analyser
    from .blocking_analyser import (
        _run_step1_only,
        _run_steps2_plus,
        _build_com_blocking_result,
        get_drug_rows,
        load_cached_patents_bulk,
        store_patent_analysis,
        invalidate_drug_cache,
        is_non_analysable_patent,
        skipped_result,
        error_result,
        _print_summary_table,
    )
    from pathlib import Path as _Path

    # Pre-compute which filenames are analysable (same filter as run_blocking_analysis)
    all_filenames  = [ref["filename"] for ref in pdf_refs]
    analysis_files = [f for f in all_filenames if not is_non_analysable_patent(f)]
    skipped_files  = [f for f in all_filenames if     is_non_analysable_patent(f)]

    # Load per-patent cache (mirrors run_blocking_analysis cache logic)
    if not reindex:
        cached_results = load_cached_patents_bulk(drug_name, analysis_files)
        new_files      = [f for f in analysis_files if f not in cached_results]
        if cached_results:
            print(
                f"[CACHE] {len(cached_results)} patent(s) loaded from cache, "
                f"{len(new_files)} new patent(s) to analyse"
            )
        else:
            print(f"[CACHE] No cached results — analysing all {len(analysis_files)} patent(s)")
    else:
        invalidate_drug_cache(drug_name)
        cached_results = {}
        new_files      = analysis_files

    # Step 1 futures: dict[filename → asyncio.Task] for patents that need analysis
    step1_tasks: dict = {}

    async def _on_indexed(filename: str) -> None:
        """
        Callback fired by run_indexing as soon as a patent is ready in ChromaDB.
        Immediately launches Step 1 analysis for patents not already cached,
        so analysis overlaps with the remaining indexing work.
        """
        if filename in new_files and filename not in step1_tasks:
            print(f"[PIPELINE] {filename} — indexed, launching Step 1 immediately")
            step1_tasks[filename] = asyncio.ensure_future(
                _run_step1_only(filename, collection)
            )

    # Cached patents are handled directly in the results loop below —
    # they don't need Step 1 tasks since their full analysis is already stored.

    # Run indexing (fires _on_indexed per patent) and timeline fetch in parallel
    print(f"[STEPS 2–3] Indexing {len(pdf_refs)} patent(s) with up to 5 workers...")
    print(f"[STEPS 4–5 PRE] Fetching clinical timeline concurrently...")

    _, timeline = await asyncio.gather(
        run_indexing(drug_name, pdf_refs, collection, reindex=reindex, on_indexed=_on_indexed),
        fetch_clinical_timeline(
            drug_name          = drug_name,
            bq_table_name      = _bq_table,
            bq_project_id      = _bq_project,
            bq_dataset_id      = _bq_dataset,
            bq_service_account = _bq_sa,
        ),
    )

    # Build drug_phase from timeline (same logic as before)
    _GEO_TO_JUR = {
        "United States": "US", "EU": "EP", "Japan": "JP", "China": "CN",
        "India": "IN", "South Korea": "KR", "Australia": "AU", "Canada": "CA",
        "Brazil": "BR", "Mexico": "MX", "Russia": "RU",
    }
    geography_stages = timeline.get("geography_stages", {})
    drug_phase = {"US": geography_stages.get("United States"), "EP": geography_stages.get("EU")}
    for geo_name, phase in geography_stages.items():
        jur = _GEO_TO_JUR.get(geo_name)
        if jur and jur not in drug_phase:
            drug_phase[jur] = phase
    phase_summary = " | ".join(f"{k}: {v}" for k, v in sorted(drug_phase.items()) if v)
    print(f"[STEPS 4–5 PRE] Drug phase → {phase_summary}")

    # Any new patents not yet triggered (e.g. indexing failed to extract text)
    # still need a Step 1 attempt — ensure they're all launched before we await
    for filename in new_files:
        if filename not in step1_tasks:
            print(f"[PIPELINE] {filename} — not indexed successfully, attempting Step 1 anyway")
            step1_tasks[filename] = asyncio.ensure_future(
                _run_step1_only(filename, collection)
            )

    # Await all Step 1 tasks (most are already done or nearly done)
    if step1_tasks:
        print(f"[STEPS 4–5] Awaiting {len(step1_tasks)} Step 1 task(s)...")
        step1_results_list = await asyncio.gather(
            *step1_tasks.values(), return_exceptions=True
        )
        new_phase1_results = dict(zip(step1_tasks.keys(), step1_results_list))
    else:
        new_phase1_results = {}

    # ── CoM routing (identical to run_blocking_analysis logic) ───────────────
    # Build unified phase1 view across cached + new patents
    all_phase1 = []
    for filename, cached_patent in cached_results.items():
        all_phase1.append({
            "filename":      filename,
            "patent_number": cached_patent.get("patent_number", _Path(filename).stem),
            "jurisdiction":  (cached_patent.get("jurisdiction") or "").upper(),
            "is_com":        cached_patent.get("claim_category") == "Composition of Matter"
                             and cached_patent.get("tag") == "BLOCKING",
            "filing_date":   cached_patent.get("filing_date"),
            "_from_cache":   True,
        })
    for filename, result in new_phase1_results.items():
        if isinstance(result, Exception) or result is None:
            all_phase1.append({"filename": filename, "_failed": True})
        else:
            result["_from_cache"] = False
            all_phase1.append(result)

    primary_com_filenames: set = set()
    all_jurisdictions = sorted(set(
        r.get("jurisdiction") for r in all_phase1
        if isinstance(r, dict) and r.get("jurisdiction") and not r.get("_failed")
    ))
    print(f"[CoM ROUTING] Jurisdictions found: {all_jurisdictions}")
    for jurisdiction in all_jurisdictions:
        com_candidates = [
            r for r in all_phase1
            if isinstance(r, dict) and not r.get("_failed")
            and r.get("is_com") and r.get("jurisdiction") == jurisdiction
        ]
        if not com_candidates:
            continue
        com_candidates.sort(key=lambda r: r.get("filing_date") or "9999-99-99")
        primary = com_candidates[0]
        primary_com_filenames.add(primary["filename"])
        print(
            f"[CoM ROUTING] Primary CoM for {jurisdiction}: "
            f"{primary.get('patent_number', '?')} (filed: {primary.get('filing_date') or 'unknown'})"
            f" → BLOCKING (skips Steps 2+)"
        )
        for secondary in com_candidates[1:]:
            print(
                f"[CoM ROUTING] Secondary CoM for {jurisdiction}: "
                f"{secondary.get('patent_number', '?')} → sent to Steps 2+ as Formulation-class"
            )

    # ── Build results: cached + route new patents to Phase 2 ─────────────────
    drug_rows = get_drug_rows(drug_name)
    patents: list = []
    phase2_inputs: list = []

    for filename, cached_patent in cached_results.items():
        patents.append(cached_patent)
        print(f"[RESULT] {filename} → from cache ({cached_patent.get('tag', '?')})")

    for filename, result in new_phase1_results.items():
        if isinstance(result, Exception) or result is None:
            print(f"[ERROR] Phase 1 failed for {filename}: {result}")
            patents.append(error_result(filename))
            continue
        if filename in primary_com_filenames:
            com_result = _build_com_blocking_result(result)
            patents.append(com_result)
            store_patent_analysis(drug_name, filename, com_result)
            print(f"[RESULT] {filename} → NEW primary CoM BLOCKING (cached)")
        else:
            if result.get("step1", {}).get("claim_category") == "Composition of Matter":
                result["step1"]["claim_category"] = "Formulation"
                result["step1"]["is_composition_of_matter"] = False
                print(
                    f"[CoM ROUTING] {result.get('patent_number')} reclassified: "
                    f"Composition of Matter → Formulation"
                )
            phase2_inputs.append(result)

    if phase2_inputs:
        print(f"\n[PHASE 2] Running Steps 2+ on {len(phase2_inputs)} patent(s) in parallel...")
        phase2_results = await asyncio.gather(
            *[_run_steps2_plus(p, drug_name, drug_rows, drug_phase) for p in phase2_inputs],
            return_exceptions=True,
        )
        for phase1_data, result in zip(phase2_inputs, phase2_results):
            if isinstance(result, Exception):
                print(f"[ERROR] Phase 2 failed for {phase1_data['filename']}: {result}")
                patents.append(error_result(phase1_data["filename"]))
            else:
                patents.append(result)
                store_patent_analysis(drug_name, phase1_data["filename"], result)
                print(f"[RESULT] {phase1_data['filename']} → NEW {result.get('tag', '?')} (cached)")

    for filename in skipped_files:
        patents.append(skipped_result(filename))

    _print_summary_table(drug_name, patents)
    print(f"[STEPS 2–5] {len(patents)} patent(s) analysed")

    # ── Steps 6–7: Assign phase per patent (timeline already fetched above) ───
    print(f"\n[STEPS 6–7] Assigning clinical phase to patents...")
    patents = assign_patent_phases(patents, timeline)

    # ── Steps 8–12: Derived calculations (first pass) ─────────────────────────
    print(f"\n[STEPS 8–12] Running first-pass calculations...")
    patents = run_calculations(patents)

    # ── Step 13: Fetch real-world approval dates (Marketed only) ──────────────
    print(f"\n[STEP 13] Fetching real-world approval dates...")
    bq_companies = [
        c.strip() for c in
        str(timeline.get("company_name", "")).split(",") if c.strip()
    ]
    bq_brands = [
        b.strip() for b in
        str(timeline.get("brand_name", "")).split(",") if b.strip()
    ]

    us_marketed = any(
        p.get("phase_at_filing") == "Marketed"
        and (p.get("jurisdiction") or "").upper() == "US"
        for p in patents
    )
    eu_marketed = any(
        p.get("phase_at_filing") == "Marketed"
        and (p.get("jurisdiction") or "").upper() == "EP"
        for p in patents
    )
    print(f"[STEP 13] US Marketed: {us_marketed} | EU Marketed: {eu_marketed}")

    approval = await fetch_approval_dates(
        drug_name    = drug_name,
        bq_companies = bq_companies,
        bq_brands    = bq_brands,
        fetch_us     = us_marketed,
        fetch_eu     = eu_marketed,
    )

    # Attach approval dates to all patents
    for p in patents:
        p["approval_date_us"]        = approval["US"]["date"]
        p["approval_date_eu"]        = approval["EU"]["date"]
        p["approval_date_us_source"] = approval["US"]["source"]
        p["approval_date_eu_source"] = approval["EU"]["source"]

    # ── Step 13b: Recalculate with real approval dates ─────────────────────────
    print(f"\n[STEP 13b] Recalculating with real approval dates...")
    patents = run_calculations(patents)

    analysis_date = datetime.now().strftime("%Y-%m-%d")

    # ── Store ALL results in cache DB (before jurisdiction filtering) ──────────
    _store_results(drug_name, patents, analysis_date, [ref["filename"] for ref in pdf_refs])

    # ── Apply jurisdiction filter if requested ────────────────────────────────
    if jurisdictions:
        patents = _filter_by_jurisdictions(patents, jurisdictions)
        # Recalculate scores for the filtered subset
        patents = run_calculations(patents)

    # ── Step 14: Export per-drug Excel ────────────────────────────────────────
    print(f"\n[STEP 14] Exporting per-drug Excel...")
    excel_path = export_to_excel(drug_name, patents, analysis_date)

    # ── Step 15: Regenerate combined Excel ────────────────────────────────────
    print(f"\n[STEP 15] Regenerating combined Excel...")
    combined_excel_path = export_combined_excel(analysis_date)

    elapsed = time.time() - t0
    print(f"\n[PIPELINE] Done in {elapsed:.1f}s — {len(patents)} patent(s)")
    print(f"{'='*60}\n")

    return {
        "drug_name":               drug_name,
        "analysis_date":           analysis_date,
        "patents":                 patents,
        "source_files":            [ref["filename"] for ref in pdf_refs],
        "processing_time_seconds": round(elapsed, 1),
        "clinical_timeline":       timeline,
        "phase_data_source":       timeline.get("source", "unavailable"),
        "excel_path":              excel_path,
        "combined_excel_path":     combined_excel_path,
        "from_cache":              False,
    }
