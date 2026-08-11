"""
step6.py — IP Portfolio Scope Engine
=====================================
Deterministic date arithmetic and patent estate partitioning.
No LLM calls. No claim reading. No prior art search.

Usage:
    python step6.py --input Patent_classification.csv [--output_dir ./step6_output]
                    [--entry_year 2031]  # omit for AUTO

Input CSV columns (from excel_exporters.py):
    Drug Name, Patent Number, Jurisdiction, Tag, Blocking Category,
    Step 1 Claim Category, Filing Date, PTE (months), Type, Source File
    (all other columns are carried through but not used for logic)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

JURISDICTIONS = {"US", "EP"}

INCLUDE_FORECASTED = False          # Forecasted rows have no claim text

CATEGORY_PRIORITY: dict[str, int] = {
    "Composition of Matter": 1,
    "Salt/Polymorph":        2,
    "Formulation":           3,
    "Method of Treatment":   4,
    "Dosage Regimen":        5,
    "Device":                6,
    "Manufacturing Process": 7,
}

# Column name aliases — map canonical names → whatever the CSV may call them
_COL_ALIASES: dict[str, list[str]] = {
    "drug_name":       ["Drug Name", "drug_name", "Drug"],
    "patent_number":   ["Patent Number", "patent_number", "Patent"],
    "jurisdiction":    ["Jurisdiction", "jurisdiction"],
    "tag":             ["Tag", "tag"],
    "blocking_cat":    ["Blocking Category", "blocking_category"],
    "claim_category":  ["Step 1 Claim Category", "claim_category", "Claim Category"],
    "filing_date":     ["Filing Date", "filing_date"],
    "pte_months":      ["PTE (months)", "pte_months", "PTE"],
    "type_":           ["Type", "type"],
    "source_file":     ["Source File", "source_file"],
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return mapping canonical_key -> actual column name in df."""
    resolved: dict[str, str] = {}
    available = set(df.columns)
    for key, candidates in _COL_ALIASES.items():
        for c in candidates:
            if c in available:
                resolved[key] = c
                break
        else:
            resolved[key] = candidates[0]   # will be NaN on access — caught later
    return resolved


def _parse_filing_year(raw: object) -> Optional[int]:
    """Extract 4-digit year from a filing date string. Returns None if missing."""
    if raw is None or pd.isna(raw) or str(raw).strip().lower() in ("", "unknown", "n/a", "none", "null"):
        return None
    s = str(raw).strip()
    # Match any 4-digit year (handles DD-MMM-YYYY, YYYY-MM-DD, YYYY, etc.)
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return int(m.group()) if m else None


def _pte_to_years(raw: object) -> float:
    """Convert PTE (months) to years, rounded to nearest integer. Blank -> 0."""
    if raw is None or pd.isna(raw) or str(raw).strip().lower() in ("", "n/a", "none", "null"):
        return 0
    try:
        return round(float(raw) / 12)
    except (ValueError, TypeError):
        return 0


def _category_priority(cat: object) -> int:
    return CATEGORY_PRIORITY.get(str(cat).strip(), 99)


# ─────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────

def run_step6(
    df_raw: pd.DataFrame,
    target_entry_year: Optional[int] = None,   # None = AUTO
) -> dict:
    """
    Run the full 6-step scope engine on the input dataframe.

    Returns a dict keyed by drug_name, each value containing:
        summary, charting_queue (list of dicts), copy_set (list of dicts),
        flags (list of str), dropped (dict of reason -> count).
    """
    col = _resolve_columns(df_raw)

    def g(row: pd.Series, key: str) -> object:
        """Safe getter — returns '' if column absent."""
        c = col.get(key, "")
        return row.get(c, "") if c in row.index else ""

    # ── STEP 1: Filter to chartable universe ─────────────────
    dropped: dict[str, int] = {
        "not_blocking":       0,
        "not_existing":       0,
        "wrong_jurisdiction": 0,
    }

    chartable_rows = []
    for _, row in df_raw.iterrows():
        tag  = str(g(row, "tag")).strip().upper()
        typ  = str(g(row, "type_")).strip()
        jur  = str(g(row, "jurisdiction")).strip().upper()

        if tag != "BLOCKING":
            dropped["not_blocking"] += 1
            continue

        if not INCLUDE_FORECASTED and typ.lower() != "existing":
            dropped["not_existing"] += 1
            continue

        if jur not in JURISDICTIONS:
            dropped["wrong_jurisdiction"] += 1
            continue

        chartable_rows.append(row)

    df = pd.DataFrame(chartable_rows).reset_index(drop=True)

    if df.empty:
        return {
            "__global__": {
                "summary":        {"flag": "NO_BLOCKING_IN_SCOPE - confirm with counsel"},
                "charting_queue": [],
                "copy_set":       [],
                "flags":          ["NO_BLOCKING_IN_SCOPE - confirm with counsel"],
                "dropped":        dropped,
            }
        }

    # ── STEP 2: Compute per-patent expiry ────────────────────
    expiry_years   = []
    missing_filing = []

    for idx, row in df.iterrows():
        patent   = str(g(row, "patent_number")).strip()
        filing_y = _parse_filing_year(g(row, "filing_date"))
        pte_y    = _pte_to_years(g(row, "pte_months"))

        if filing_y is None:
            expiry_years.append(None)
            missing_filing.append(patent)
        else:
            expiry_years.append(filing_y + 20 + int(pte_y))

    df = df.copy()
    df["_patent_expiry_year"] = expiry_years
    df["_filing_year"]        = [_parse_filing_year(g(row, "filing_date")) for _, row in df.iterrows()]
    df["_missing_filing"]     = [p in missing_filing for p in df.apply(lambda r: str(g(r, "patent_number")).strip(), axis=1)]

    # ── Group by Drug and run Steps 3–6 ──────────────────────
    results: dict = {}

    drug_col = col.get("drug_name", "Drug Name")
    for drug_name, grp in df.groupby(drug_col, sort=True):
        flags: list[str] = []
        drug_results: dict = {}

        # MISSING_FILING_DATE flags
        mf_patents = grp.loc[grp["_missing_filing"], col["patent_number"]].tolist()
        for p in mf_patents:
            flags.append(f"MISSING_FILING_DATE:{p} - route to counsel")

        # Work only with patents that have a valid expiry
        valid = grp[~grp["_missing_filing"]].copy()
        null_set = grp[grp["_missing_filing"]].copy()

        # ── STEP 3: Floor patent / TARGET_ENTRY_YEAR ─────────
        if target_entry_year is not None:
            tey = target_entry_year
            floor_patent = None
            floor_year   = tey
        else:
            # AUTO: earliest expiry among BLOCKING patents
            if valid.empty:
                tey          = None
                floor_patent = None
                floor_year   = None
                flags.append("NO_COMPOUND_ANCHOR - floor date heuristic; escalate to counsel")
            else:
                floor_row    = valid.loc[valid["_patent_expiry_year"].idxmin()]
                floor_year   = int(floor_row["_patent_expiry_year"])
                floor_patent = str(g(floor_row, "patent_number")).strip()
                tey          = floor_year

                # Flag if floor patent is not a Compound/CoM patent
                floor_cat = str(g(floor_row, "claim_category")).strip()
                if floor_cat not in ("Composition of Matter", "Salt/Polymorph"):
                    flags.append(
                        f"NO_COMPOUND_ANCHOR - floor patent '{floor_patent}' "
                        f"is '{floor_cat}' (not CoM/Compound); floor date heuristic; escalate to counsel"
                    )

        # ── STEP 4: Controlling patent ────────────────────────
        if not valid.empty:
            ctrl_row     = valid.loc[valid["_patent_expiry_year"].idxmax()]
            ctrl_patent  = str(g(ctrl_row, "patent_number")).strip()
            ctrl_year    = int(ctrl_row["_patent_expiry_year"])
        else:
            ctrl_patent = None
            ctrl_year   = None

        # ── STEP 5: Partition on cut-line ─────────────────────
        if tey is not None and not valid.empty:
            copy_mask = valid["_patent_expiry_year"] <= tey
            copy_df   = valid[copy_mask].copy()
            work_df   = valid[~copy_mask].copy()
        else:
            copy_df = pd.DataFrame()
            work_df = valid.copy()

        if valid.empty:
            flags.append("NO_BLOCKING_IN_SCOPE - confirm with counsel")

        # ── STEP 6: Order the charting queue ──────────────────
        if not work_df.empty:
            work_df = work_df.copy()
            work_df["_cat_priority"] = work_df.apply(
                lambda r: _category_priority(g(r, "claim_category")), axis=1
            )
            work_df = work_df.sort_values(
                by=["_patent_expiry_year", "_cat_priority"],
                ascending=[True, True],
            ).reset_index(drop=True)
            work_df["charting_priority"] = range(1, len(work_df) + 1)

        # ── Build charting_queue output ───────────────────────
        charting_queue = []
        for _, row in work_df.iterrows() if not work_df.empty else []:
            patent_flags = []
            if row.get("_missing_filing"):
                patent_flags.append(f"MISSING_FILING_DATE - route to counsel")

            charting_queue.append({
                "charting_priority":   int(row["charting_priority"]),
                "drug_name":           drug_name,
                "patent_number":       str(g(row, "patent_number")).strip(),
                "jurisdiction":        str(g(row, "jurisdiction")).strip().upper(),
                "blocking_category":   str(g(row, "blocking_cat")).strip(),
                "step1_claim_category":str(g(row, "claim_category")).strip(),
                "filing_date":         str(g(row, "filing_date")).strip(),
                "patent_expiry_year":  int(row["_patent_expiry_year"]),
                "source_file":         str(g(row, "source_file")).strip(),
                "flags":               patent_flags,
            })

        # ── Build COPY_SET output ─────────────────────────────
        copy_set = []
        for _, row in copy_df.iterrows() if not copy_df.empty else []:
            copy_set.append({
                "patent_number":       str(g(row, "patent_number")).strip(),
                "blocking_category":   str(g(row, "blocking_cat")).strip(),
                "step1_claim_category":str(g(row, "claim_category")).strip(),
                "patent_expiry_year":  int(row["_patent_expiry_year"]),
                "jurisdiction":        str(g(row, "jurisdiction")).strip().upper(),
                "source_file":         str(g(row, "source_file")).strip(),
            })

        # Null-date patents logged in flags but not queued
        null_patents = []
        for _, row in null_set.iterrows() if not null_set.empty else []:
            null_patents.append({
                "patent_number": str(g(row, "patent_number")).strip(),
                "flag":          "MISSING_FILING_DATE - route to counsel",
            })

        results[str(drug_name)] = {
            "summary": {
                "drug":                    str(drug_name),
                "floor_patent":            floor_patent,
                "floor_year":              floor_year,
                "controlling_patent":      ctrl_patent,
                "controlling_patent_year": ctrl_year,
                "target_entry_year":       tey,
                "count_copy":              len(copy_df),
                "count_work":              len(work_df),
                "count_null_filing":       len(null_set),
                "dropped":                 dropped,
                "flags":                   flags,
            },
            "charting_queue":  charting_queue,
            "copy_set":        copy_set,
            "null_date_set":   null_patents,
            "flags":           flags,
        }

    return results


# ─────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────

def _print_summary(drug_name: str, result: dict) -> None:
    s = result["summary"]
    print(f"\n{'═'*70}")
    print(f"  DRUG: {drug_name}")
    print(f"{'═'*70}")

    if "flag" in s:
        print(f"  ⚠  {s['flag']}")
        return

    print(f"  Floor Patent      : {s['floor_patent'] or 'N/A'} (expiry {s['floor_year'] or 'N/A'})")
    print(f"  Controlling Patent: {s['controlling_patent'] or 'N/A'} (expiry {s['controlling_patent_year'] or 'N/A'})")
    print(f"  Target Entry Year : {s['target_entry_year'] or 'N/A'}")
    print(f"  COPY set          : {s['count_copy']} patent(s)  [expired / no action]")
    print(f"  WORK set          : {s['count_work']} patent(s)  [charting queue]")
    print(f"  Missing dates     : {s['count_null_filing']} patent(s)  [routed to counsel]")

    d = s.get("dropped", {})
    print(f"  Dropped — not blocking     : {d.get('not_blocking', 0)}")
    print(f"  Dropped — not existing     : {d.get('not_existing', 0)}")
    print(f"  Dropped — wrong jurisdiction: {d.get('wrong_jurisdiction', 0)}")

    if result["flags"]:
        print(f"\n  ⚠  FLAGS:")
        for f in result["flags"]:
            print(f"     • {f}")


def _write_outputs(drug_name: str, result: dict, output_dir: Path) -> None:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)

    # JSON: charting queue
    queue_path = output_dir / f"{safe}_charting_queue.json"
    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(result["charting_queue"], f, indent=2)
    print(f"  → Charting queue : {queue_path}")

    # CSV: COPY_SET
    if result["copy_set"]:
        copy_path = output_dir / f"{safe}_copy_set.csv"
        pd.DataFrame(result["copy_set"]).to_csv(copy_path, index=False)
        print(f"  → COPY set       : {copy_path}")

    # CSV: null-date patents
    if result["null_date_set"]:
        null_path = output_dir / f"{safe}_missing_dates.csv"
        pd.DataFrame(result["null_date_set"]).to_csv(null_path, index=False)
        print(f"  → Missing dates  : {null_path}")

    # JSON: full summary
    summary_path = output_dir / f"{safe}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)
    print(f"  → Summary        : {summary_path}")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6 – IP Portfolio Scope Engine (deterministic, no LLM)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to Patent_classification.csv (output of the main pipeline)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="./step6_output",
        help="Directory to write output files (default: ./step6_output)",
    )
    parser.add_argument(
        "--entry_year", "-y",
        type=int,
        default=None,
        help="Fixed TARGET_ENTRY_YEAR for all drugs (omit for AUTO)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 6] Reading: {input_path}")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    print(f"[Step 6] {len(df)} rows loaded, {df.shape[1]} columns")
    print(f"[Step 6] Jurisdictions in scope: {sorted(JURISDICTIONS)}")
    print(f"[Step 6] TARGET_ENTRY_YEAR: {args.entry_year or 'AUTO'}")
    print(f"[Step 6] INCLUDE_FORECASTED: {INCLUDE_FORECASTED}")

    results = run_step6(df, target_entry_year=args.entry_year)

    for drug_name, result in results.items():
        _print_summary(drug_name, result)
        _write_outputs(drug_name, result, output_dir)

    print(f"\n[Step 6] Done. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
