"""
step6.py — IP Portfolio Scope Engine
=====================================
Deterministic date arithmetic and patent estate partitioning.
No LLM calls. No claim reading. No prior art search.

Input:  Per-drug Excel files from patent_exports/ (output of excel_exporters.py)
        File pattern: patent_exports/{Drug}_{YYYYMMDD}.xlsx
        (picks the most recent file for each drug if multiple dates exist)

Output: step6_output/{Drug}_charting_queue.json
        step6_output/{Drug}_copy_set.csv
        step6_output/{Drug}_missing_dates.csv
        step6_output/{Drug}_summary.json

Usage:
    python step6.py                                          # all drugs in patent_exports/
    python step6.py --drugs Axitinib Minocycline            # specific drugs
    python step6.py --drugs Axitinib --entry_year 2031      # fixed entry year
    python step6.py --excel_dir ./my_exports --output_dir ./out
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
# Paths — match excel_exporters.py exactly
# ─────────────────────────────────────────────────────────────

import os
EXCEL_OUTPUT_DIR = Path(os.getenv("EXCEL_OUTPUT_DIR", Path(__file__).parent / "patent_exports"))
STEP6_OUTPUT_DIR = Path(os.getenv("STEP6_OUTPUT_DIR", Path(__file__).parent / "step6_output"))

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

JURISDICTIONS      = {"US", "EP"}
INCLUDE_FORECASTED = False

CATEGORY_PRIORITY: dict[str, int] = {
    "Composition of Matter": 1,
    "Salt/Polymorph":        2,
    "Formulation":           3,
    "Method of Treatment":   4,
    "Dosage Regimen":        5,
    "Device":                6,
    "Manufacturing Process": 7,
}

# Map canonical keys -> possible CSV/Excel column names
_COL_ALIASES: dict[str, list[str]] = {
    "drug_name":      ["Drug Name",               "drug_name",      "Drug"],
    "patent_number":  ["Patent Number",            "patent_number",  "Patent"],
    "jurisdiction":   ["Jurisdiction",             "jurisdiction"],
    "tag":            ["Tag",                      "tag"],
    "blocking_cat":   ["Blocking Category",        "blocking_category"],
    "claim_category": ["Step 1 Claim Category",    "claim_category", "Claim Category"],
    "filing_date":    ["Filing Date",              "filing_date"],
    "pte_months":     ["PTE (months)",             "pte_months",     "PTE"],
    "type_":          ["Type",                     "type"],
    "source_file":    ["Source File",              "source_file"],
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    available = set(df.columns)
    for key, candidates in _COL_ALIASES.items():
        for c in candidates:
            if c in available:
                resolved[key] = c
                break
        else:
            resolved[key] = candidates[0]
    return resolved


def _parse_filing_year(raw: object) -> Optional[int]:
    if raw is None or pd.isna(raw) or str(raw).strip().lower() in ("", "unknown", "n/a", "none", "null"):
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(raw).strip())
    return int(m.group()) if m else None


def _pte_to_years(raw: object) -> int:
    if raw is None or pd.isna(raw) or str(raw).strip().lower() in ("", "n/a", "none", "null"):
        return 0
    try:
        return round(float(raw) / 12)
    except (ValueError, TypeError):
        return 0


def _category_priority(cat: object) -> int:
    return CATEGORY_PRIORITY.get(str(cat).strip(), 99)


def _find_excel_for_drug(drug_name: str, excel_dir: Path) -> Optional[Path]:
    """
    Find the most recent Excel file for a drug in excel_dir.
    File pattern: {safe_drug}_{YYYYMMDD}.xlsx
    safe_drug = re.sub(r'[^a-zA-Z0-9_-]', '_', drug_name)
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    matches = sorted(excel_dir.glob(f"{safe}_*.xlsx"), reverse=True)  # newest first
    if matches:
        return matches[0]
    # Fuzzy fallback: normalise both sides
    norm = drug_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    for f in sorted(excel_dir.glob("*.xlsx"), reverse=True):
        f_norm = f.stem.lower().replace(" ", "").replace("-", "").replace("_", "")
        if f_norm.startswith(norm):
            return f
    return None


def _load_drug_excel(drug_name: str, excel_dir: Path) -> Optional[pd.DataFrame]:
    path = _find_excel_for_drug(drug_name, excel_dir)
    if path is None:
        print(f"[Step 6] No Excel found for '{drug_name}' in {excel_dir}")
        return None
    print(f"[Step 6] Loading: {path.name}")
    df = pd.read_excel(path, sheet_name="Patents", dtype=str)
    df = df.fillna("")
    return df


# ─────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────

def run_step6(
    df_raw: pd.DataFrame,
    target_entry_year: Optional[int] = None,
) -> dict:
    """
    Run the full 6-step scope engine on a dataframe (one or many drugs).

    Returns dict keyed by drug_name:
        summary, charting_queue, copy_set, null_date_set, flags, dropped
    """
    col = _resolve_columns(df_raw)

    def g(row: pd.Series, key: str) -> object:
        c = col.get(key, "")
        return row.get(c, "") if c in row.index else ""

    # ── STEP 1: Filter to chartable universe ─────────────────
    dropped_global: dict[str, int] = {
        "not_blocking":        0,
        "not_existing":        0,
        "wrong_jurisdiction":  0,
    }

    chartable_rows = []
    for _, row in df_raw.iterrows():
        tag = str(g(row, "tag")).strip().upper()
        typ = str(g(row, "type_")).strip()
        jur = str(g(row, "jurisdiction")).strip().upper()

        if tag != "BLOCKING":
            dropped_global["not_blocking"] += 1
            continue
        if not INCLUDE_FORECASTED and typ.lower() != "existing":
            dropped_global["not_existing"] += 1
            continue
        if jur not in JURISDICTIONS:
            dropped_global["wrong_jurisdiction"] += 1
            continue
        chartable_rows.append(row)

    df = pd.DataFrame(chartable_rows).reset_index(drop=True)

    if df.empty:
        return {
            "__global__": {
                "summary":       {"flag": "NO_BLOCKING_IN_SCOPE - confirm with counsel"},
                "charting_queue": [],
                "copy_set":       [],
                "null_date_set":  [],
                "flags":          ["NO_BLOCKING_IN_SCOPE - confirm with counsel"],
                "dropped":        dropped_global,
            }
        }

    # ── STEP 2: Compute per-patent expiry ─────────────────────
    def _expiry(row):
        y = _parse_filing_year(g(row, "filing_date"))
        return (y + 20 + _pte_to_years(g(row, "pte_months"))) if y is not None else None

    df = df.copy()
    df["_expiry"]          = df.apply(_expiry, axis=1)
    df["_filing_year"]     = df.apply(lambda r: _parse_filing_year(g(r, "filing_date")), axis=1)
    df["_missing_filing"]  = df["_expiry"].isna()

    # ── Group by Drug, run Steps 3–6 ──────────────────────────
    results: dict = {}
    drug_col = col.get("drug_name", "Drug Name")

    for drug_name, grp in df.groupby(drug_col, sort=True):
        flags: list[str] = []

        mf_patents = grp.loc[grp["_missing_filing"], col["patent_number"]].tolist()
        for p in mf_patents:
            flags.append(f"MISSING_FILING_DATE:{p} - route to counsel")

        valid    = grp[~grp["_missing_filing"]].copy()
        null_set = grp[grp["_missing_filing"]].copy()

        # ── STEP 3: Floor / TARGET_ENTRY_YEAR ────────────────
        if target_entry_year is not None:
            tey          = target_entry_year
            floor_patent = None
            floor_year   = tey
        elif valid.empty:
            tey          = None
            floor_patent = None
            floor_year   = None
            flags.append("NO_COMPOUND_ANCHOR - floor date heuristic; escalate to counsel")
        else:
            floor_row    = valid.loc[valid["_expiry"].idxmin()]
            floor_year   = int(floor_row["_expiry"])
            floor_patent = str(g(floor_row, "patent_number")).strip()
            tey          = floor_year
            floor_cat    = str(g(floor_row, "claim_category")).strip()
            if floor_cat not in ("Composition of Matter", "Salt/Polymorph"):
                flags.append(
                    f"NO_COMPOUND_ANCHOR - floor patent '{floor_patent}' "
                    f"is '{floor_cat}' (not CoM/Compound); floor date heuristic; escalate to counsel"
                )

        # ── STEP 4: Controlling patent ────────────────────────
        if not valid.empty:
            ctrl_row    = valid.loc[valid["_expiry"].idxmax()]
            ctrl_patent = str(g(ctrl_row, "patent_number")).strip()
            ctrl_year   = int(ctrl_row["_expiry"])
        else:
            ctrl_patent = None
            ctrl_year   = None

        # ── STEP 5: Partition ─────────────────────────────────
        if tey is not None and not valid.empty:
            copy_df = valid[valid["_expiry"] <= tey].copy()
            work_df = valid[valid["_expiry"] >  tey].copy()
        else:
            copy_df = pd.DataFrame()
            work_df = valid.copy()

        if valid.empty:
            flags.append("NO_BLOCKING_IN_SCOPE - confirm with counsel")

        # ── STEP 6: Order queue ───────────────────────────────
        if not work_df.empty:
            work_df["_cat_pri"] = work_df.apply(
                lambda r: _category_priority(g(r, "claim_category")), axis=1
            )
            work_df = work_df.sort_values(
                ["_expiry", "_cat_pri"], ascending=[True, True]
            ).reset_index(drop=True)
            work_df["charting_priority"] = range(1, len(work_df) + 1)

        # ── Build outputs ─────────────────────────────────────
        charting_queue = []
        for _, row in (work_df.iterrows() if not work_df.empty else []):
            charting_queue.append({
                "charting_priority":    int(row["charting_priority"]),
                "drug_name":            str(drug_name),
                "patent_number":        str(g(row, "patent_number")).strip(),
                "jurisdiction":         str(g(row, "jurisdiction")).strip().upper(),
                "blocking_category":    str(g(row, "blocking_cat")).strip(),
                "step1_claim_category": str(g(row, "claim_category")).strip(),
                "filing_date":          str(g(row, "filing_date")).strip(),
                "patent_expiry_year":   int(row["_expiry"]),
                "source_file":          str(g(row, "source_file")).strip(),
                "flags":                [],
            })

        copy_set = []
        for _, row in (copy_df.iterrows() if not copy_df.empty else []):
            copy_set.append({
                "patent_number":        str(g(row, "patent_number")).strip(),
                "blocking_category":    str(g(row, "blocking_cat")).strip(),
                "step1_claim_category": str(g(row, "claim_category")).strip(),
                "patent_expiry_year":   int(row["_expiry"]),
                "jurisdiction":         str(g(row, "jurisdiction")).strip().upper(),
                "source_file":          str(g(row, "source_file")).strip(),
            })

        null_date_set = []
        for _, row in (null_set.iterrows() if not null_set.empty else []):
            null_date_set.append({
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
                "dropped":                 dropped_global,
                "flags":                   flags,
            },
            "charting_queue": charting_queue,
            "copy_set":        copy_set,
            "null_date_set":   null_date_set,
            "flags":           flags,
        }

    return results


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def write_outputs(drug_name: str, result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)

    queue_path = output_dir / f"{safe}_charting_queue.json"
    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(result["charting_queue"], f, indent=2)
    print(f"  → Charting queue  : {queue_path}")

    if result["copy_set"]:
        copy_path = output_dir / f"{safe}_copy_set.csv"
        pd.DataFrame(result["copy_set"]).to_csv(copy_path, index=False)
        print(f"  → COPY set        : {copy_path}")

    if result["null_date_set"]:
        null_path = output_dir / f"{safe}_missing_dates.csv"
        pd.DataFrame(result["null_date_set"]).to_csv(null_path, index=False)
        print(f"  → Missing dates   : {null_path}")

    summary_path = output_dir / f"{safe}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)
    print(f"  → Summary         : {summary_path}")


def print_summary(drug_name: str, result: dict) -> None:
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
    print(f"  Dropped — not BLOCKING      : {d.get('not_blocking', 0)}")
    print(f"  Dropped — not Existing      : {d.get('not_existing', 0)}")
    print(f"  Dropped — wrong jurisdiction: {d.get('wrong_jurisdiction', 0)}")
    if result["flags"]:
        print(f"\n  ⚠  FLAGS:")
        for fl in result["flags"]:
            print(f"     • {fl}")


def run_for_drug(
    drug_name: str,
    excel_dir: Path,
    output_dir: Path,
    target_entry_year: Optional[int] = None,
) -> Optional[dict]:
    """
    Load the Excel for drug_name, run step 6, write outputs.
    Returns the result dict for this drug (or None if Excel not found).
    """
    df = _load_drug_excel(drug_name, excel_dir)
    if df is None:
        return None

    # Inject Type = "Existing" for all rows (as per prompt assumption)
    col = _resolve_columns(df)
    type_col = col.get("type_", "Type")
    if type_col not in df.columns:
        df[type_col] = "Existing"
    else:
        df[type_col] = df[type_col].replace("", "Existing").fillna("Existing")
        # Only override completely blank/missing cells, leave explicit values
        df[type_col] = df[type_col].apply(
            lambda v: "Existing" if str(v).strip() == "" else v
        )

    results = run_step6(df, target_entry_year=target_entry_year)
    drug_result = results.get(drug_name) or (list(results.values())[0] if results else None)

    if drug_result is None:
        print(f"[Step 6] No results produced for '{drug_name}'")
        return None

    print_summary(drug_name, drug_result)
    write_outputs(drug_name, drug_result, output_dir)
    return drug_result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6 – IP Portfolio Scope Engine (deterministic, no LLM)"
    )
    parser.add_argument(
        "--drugs", nargs="+", default=None,
        help="Drug names to process (default: all drugs found in excel_dir)",
    )
    parser.add_argument(
        "--excel_dir", default=str(EXCEL_OUTPUT_DIR),
        help=f"Directory containing per-drug Excel exports (default: {EXCEL_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir", default=str(STEP6_OUTPUT_DIR),
        help=f"Directory for step6 outputs (default: {STEP6_OUTPUT_DIR}, auto-created)",
    )
    parser.add_argument(
        "--entry_year", type=int, default=None,
        help="Fixed TARGET_ENTRY_YEAR for all drugs (omit for AUTO)",
    )
    args = parser.parse_args()

    excel_dir  = Path(args.excel_dir)
    output_dir = Path(args.output_dir)

    if not excel_dir.exists():
        print(f"ERROR: excel_dir not found: {excel_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover drugs from filenames if not specified
    if args.drugs:
        drug_names = args.drugs
    else:
        seen: dict[str, Path] = {}
        for f in sorted(excel_dir.glob("*.xlsx")):
            # Strip trailing _YYYYMMDD to get drug name
            stem = re.sub(r"_\d{8}$", "", f.stem)
            safe_stem = stem
            if safe_stem not in seen:
                seen[safe_stem] = f
        drug_names = list(seen.keys())
        print(f"[Step 6] Auto-discovered drugs: {drug_names}")

    if not drug_names:
        print(f"[Step 6] No drugs found in {excel_dir}. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"[Step 6] Processing {len(drug_names)} drug(s): {drug_names}")
    print(f"[Step 6] Jurisdictions in scope: {sorted(JURISDICTIONS)}")
    print(f"[Step 6] TARGET_ENTRY_YEAR: {args.entry_year or 'AUTO'}")
    print(f"[Step 6] INCLUDE_FORECASTED: {INCLUDE_FORECASTED}")
    print(f"[Step 6] Output dir: {output_dir.resolve()}")

    for drug_name in drug_names:
        run_for_drug(
            drug_name          = drug_name,
            excel_dir          = excel_dir,
            output_dir         = output_dir,
            target_entry_year  = args.entry_year,
        )

    print(f"\n[Step 6] Done. Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
