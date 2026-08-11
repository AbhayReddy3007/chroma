"""
phase_processor.py
──────────────────
Reads an Excel where:
  - Sheet 1 (Summary) is ignored
  - Each subsequent sheet is named after a drug/molecule
  - Each sheet contains that drug's clinical trial phase data

For each molecule:
  1. Normalises Phase — sub-phases like 3a, 3b → 3; roman numerals → arabic.
  2. For each normalised phase, picks the row with the earliest Start Date.
  3. Drops original Phase 1/2, calculates them from Phase 3 (−3yrs, −4yrs).
  4. Fetches real approval dates via approval_date_fetcher and adds Marketed rows.

Usage:
    python phase_processor.py --input path/to/input.xlsx --output path/to/output.xlsx

    # Or import and use programmatically:
    from phase_processor import process_phase_excel
    result_df = process_phase_excel("path/to/input.xlsx")
"""

import asyncio
import re
import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Make local modules importable ─────────────────────────────────────────────
_here   = Path(__file__).resolve().parent
_parent = _here.parent
_pkg    = _here.name

for _p in [str(_here), str(_parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib
try:
    _fetcher = importlib.import_module(f"{_pkg}.approval_date_fetcher")
    fetch_approval_dates = _fetcher.fetch_approval_dates
    _APPROVAL_AVAILABLE = True
except Exception as _e:
    print(f"[WARNING] approval_date_fetcher unavailable: {_e}")
    _APPROVAL_AVAILABLE = False
    async def fetch_approval_dates(*a, **kw):
        return {"US": {"date": None}, "EU": {"date": None}}

try:
    _gcs = importlib.import_module(f"{_pkg}.gcs_lister")
    _get_gcs_client    = _gcs.get_gcs_client
    _GCS_BUCKET_NAME   = _gcs.GCS_BUCKET_NAME
    _GCS_PATENTS_PREFIX = _gcs.GCS_PATENTS_PREFIX
    _gcs_normalize     = _gcs.normalize
    _GCS_AVAILABLE     = True
except Exception as _e:
    print(f"[WARNING] gcs_lister unavailable: {_e}")
    _GCS_AVAILABLE = False


def _list_gcs_drug_names() -> list:
    """Lists all drug folder names from GCS bucket."""
    if not _GCS_AVAILABLE or not _GCS_BUCKET_NAME:
        print("[GCS] GCS not available — cannot filter by GCS drugs")
        return []
    try:
        client = _get_gcs_client()
        prefix = _GCS_PATENTS_PREFIX.rstrip("/") + "/"
        blobs = list(client.list_blobs(_GCS_BUCKET_NAME, prefix=prefix, delimiter="/"))

        # Get folder names from prefixes
        prefix_depth = len(prefix.split("/")) - 1
        all_blobs = list(client.list_blobs(_GCS_BUCKET_NAME, prefix=prefix))
        drug_folders = set()
        for blob in all_blobs:
            parts = blob.name.split("/")
            if len(parts) > prefix_depth + 1:
                folder = parts[prefix_depth].strip()
                if folder:
                    drug_folders.add(folder)

        drugs = sorted(drug_folders)
        print(f"[GCS] Found {len(drugs)} drug folders: {drugs}")
        return drugs
    except Exception as e:
        print(f"[GCS] Failed to list drug folders: {e}")
        return []


# ─────────────────────────────────────────────
# Drug name aliases (Excel name → GCS name)
# ─────────────────────────────────────────────
# Maps known alternate names in the input Excel to the canonical
# GCS folder name for matching purposes.

_DRUG_ALIASES = {
    "aleniglipron l-arginine": "aleniglipron",
    "aleniglipron l arginine": "aleniglipron",
}


def _canonicalise_drug_name(name: str) -> str:
    """Returns the canonical GCS drug name, resolving known aliases."""
    lower = name.strip().lower()
    if lower in _DRUG_ALIASES:
        canonical = _DRUG_ALIASES[lower]
        print(f"[ALIAS] '{name}' → '{canonical}'")
        return canonical
    # Also try normalised match
    norm = lower.replace(" ", "").replace("-", "").replace("_", "")
    for alias, canonical in _DRUG_ALIASES.items():
        alias_norm = alias.replace(" ", "").replace("-", "").replace("_", "")
        if alias_norm == norm:
            print(f"[ALIAS] '{name}' → '{canonical}' (normalised)")
            return canonical
    return name.strip()


# ─────────────────────────────────────────────
# Phase normalisation
# ─────────────────────────────────────────────

def normalise_phase(raw: str) -> Optional[str]:
    """
    Normalises a phase string to a canonical label.

    Examples:
        "3a"           → "Phase 3"
        "3b"           → "Phase 3"
        "Phase IIa"    → "Phase 2"
        "Phase III"    → "Phase 3"
        "Phase |||"    → "Phase 3"   (pipe chars treated as roman numerals)
        "Phase |||b"   → "Phase 3"   (pipe chars with sub-phase letter)
        "Phase ||a"    → "Phase 2"
        "Phase |"      → "Phase 1"
        "2/3"          → "Phase 2"   (takes the first number)
        "1"            → "Phase 1"
        "Marketed"     → "Marketed"
        "Preclinical"  → "Preclinical"
    """
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None

    # Replace pipe characters used as roman numerals (e.g. "|||" → "III", "||" → "II", "|" → "I")
    # Must be done before lowercasing so we can substitute with uppercase roman numerals.
    # Replace longest sequences first to avoid partial substitution (||| before ||, || before |).
    s = re.sub(r'\|\|\|', 'III', s)
    s = re.sub(r'\|\|',   'II',  s)
    s = re.sub(r'\|',     'I',   s)

    low = s.lower()

    # Direct matches
    if low in ("marketed", "launched", "approved", "registered"):
        return "Marketed"
    if low in ("pre-registration", "preregistration", "pre registration", "nda/bla", "nda", "bla"):
        return "Pre-registration"
    if low in ("preclinical", "pre-clinical", "discovery"):
        return "Preclinical"

    # Strip "phase" prefix for further parsing
    cleaned = re.sub(r"^phase[\s\-]*", "", low).strip()

    # Roman numerals (I, II, III, IV, with optional sub-phase letter)
    _roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    roman_match = re.match(r"^(i{1,3}v?)\s*[a-z]?\s*$", cleaned, re.IGNORECASE)
    if roman_match:
        num = _roman.get(roman_match.group(1).lower())
        if num:
            return f"Phase {num}"

    # Arabic numerals with optional sub-phase letter (e.g. "3a", "3b", "2/3")
    num_match = re.match(r"^(\d)", cleaned)
    if num_match:
        return f"Phase {num_match.group(1)}"

    return None


# ─────────────────────────────────────────────
# Live approval date fetching
# ─────────────────────────────────────────────

async def _fetch_approval_for_molecules(molecules: list) -> dict:
    """
    Fetches approval dates for a list of molecule names using
    approval_date_fetcher.fetch_approval_dates (FDA API → EMA → Gemini → news).

    Returns: {molecule_name: {"US": date_or_None, "EU": date_or_None}}
    """
    results = {}
    for mol in molecules:
        print(f"\n[APPROVAL] Fetching approval date for '{mol}'...")
        try:
            approval = await fetch_approval_dates(
                drug_name    = mol,
                bq_companies = [],
                bq_brands    = [],
                fetch_us     = True,
                fetch_eu     = True,
            )
            us_date = approval.get("US", {}).get("date")
            eu_date = approval.get("EU", {}).get("date")
            results[mol] = {"US": us_date, "EU": eu_date}
            print(f"[APPROVAL] '{mol}' → US: {us_date or 'Not found'} | EU: {eu_date or 'Not found'}")
        except Exception as e:
            print(f"[APPROVAL] Failed for '{mol}': {e}")
            results[mol] = {"US": None, "EU": None}
    return results


# ─────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────

def process_phase_excel(input_path: str) -> pd.DataFrame:
    """
    Reads the input Excel (skipping the first summary sheet), treats each
    subsequent sheet as a separate drug, uses the sheet name as the Molecule
    name, normalises phases, and for each molecule + phase returns the row
    with the earliest Start Date.

    Steps:
      1. Skip sheet 1 (summary); load all remaining sheets, injecting
         sheet name as the 'Molecule' column.
      2. Normalise phases (3a→3, III→3, etc.)
      3. Keep earliest Start Date per molecule per phase
      4. Drop original Phase 1/2 from input
      5. Calculate Phase 2 = Phase 3 - 3 years, Phase 1 = Phase 3 - 4 years
      6. Fetch live approval dates and add Marketed rows

    Expected columns in each drug sheet (case-insensitive, flexible naming):
        Phase
        Start Date / Start Year

    All other columns are preserved in the output.

    Returns:
        DataFrame with all original columns + "Molecule" + "Normalised Phase"
        + "Approval Date", filtered to earliest Start Date per molecule per
        normalised phase.
    """
    xl = pd.ExcelFile(input_path)
    sheet_names = xl.sheet_names[1:]  # skip first sheet (summary)

    if not sheet_names:
        raise ValueError("No drug sheets found after the summary sheet.")

    print(f"Found {len(sheet_names)} drug sheet(s): {sheet_names}")

    dfs = []
    for sheet in sheet_names:
        sheet_df = pd.read_excel(xl, sheet_name=sheet)
        sheet_df.columns = [c.strip() for c in sheet_df.columns]
        # Use sheet name as the molecule/drug name
        sheet_df["Molecule"] = sheet.strip()
        dfs.append(sheet_df)
        print(f"  Loaded sheet '{sheet}': {len(sheet_df)} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # ── Resolve column names ──────────────────────────────────────────────
    col_map = {c.strip().lower(): c for c in df.columns}

    # Molecule column is always "Molecule" (injected above)
    mol_col = "Molecule"

    phase_col = next(
        (col_map[k] for k in col_map if k == "phase"),
        None,
    )
    date_col = next(
        (col_map[k] for k in col_map if k in ("start date", "startdate", "start_date", "start year", "startyear", "start_year")),
        None,
    )

    if not phase_col:
        raise ValueError(f"No Phase column found. Available: {list(df.columns)}")
    if not date_col:
        raise ValueError(f"No Start Date/Start Year column found. Available: {list(df.columns)}")

    print(f"Using → Molecule='{mol_col}' | Phase='{phase_col}' | Date='{date_col}'")

    # ── Normalise phase ───────────────────────────────────────────────────
    df["Normalised Phase"] = df[phase_col].apply(normalise_phase)

    before = len(df)
    df = df.dropna(subset=["Normalised Phase"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable phase values")

    # ── Parse dates / years ─────────────────────────────────────────────
    # The input may have full dates (2008-03-06) or just years (2008).
    # Convert years to Jan 1 of that year for sorting and calculation.
    def _parse_date_or_year(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", ""):
            return pd.NaT
        # Try as year first (integer or float like 2008.0)
        try:
            year = int(float(s))
            if 1900 <= year <= 2100:
                return pd.Timestamp(year=year, month=1, day=1)
        except (ValueError, TypeError):
            pass
        # Try as full date
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return parsed

    df["_parsed_date"] = df[date_col].apply(_parse_date_or_year)

    no_date = df["_parsed_date"].isna().sum()
    if no_date:
        print(f"Warning: {no_date} rows have unparseable dates/years — they will be deprioritised")

    # ── For each molecule + normalised phase, keep earliest Start Date ────
    df = df.sort_values("_parsed_date", na_position="last")
    result = df.groupby([mol_col, "Normalised Phase"], sort=False).first().reset_index()

    # Clean up
    result = result.drop(columns=["_parsed_date"])

    # ── Drop original Phase 1, Phase 2, and Phase 4 rows from input ────────
    before_filter = len(result)
    result = result[~result["Normalised Phase"].isin(["Phase 1", "Phase 2", "Phase 4"])]
    dropped_phases = before_filter - len(result)
    if dropped_phases:
        print(f"Dropped {dropped_phases} original Phase 1/Phase 2/Phase 4 rows")

    # ── Calculate Phase 2 and Phase 1 Start Dates from Phase 3 ───────────
    # Phase 2 Start Date = Phase 3 Start Date - 3 years
    # Phase 1 Start Date = Phase 3 Start Date - 4 years
    from dateutil.relativedelta import relativedelta

    phase3_rows = result[result["Normalised Phase"] == "Phase 3"].copy()
    new_rows = []

    for _, row in phase3_rows.iterrows():
        p3_raw = row[date_col]
        p3_date = _parse_date_or_year(p3_raw)
        if pd.isna(p3_date):
            continue

        mol_name = row[mol_col]

        # Check if the original value was a year-only
        is_year_only = False
        try:
            year_val = int(float(str(p3_raw).strip()))
            if 1900 <= year_val <= 2100:
                is_year_only = True
        except (ValueError, TypeError):
            pass

        p2_date = p3_date - relativedelta(years=3)
        p1_date = p3_date - relativedelta(years=4)

        # Build Phase 2 row
        p2_row = row.copy()
        p2_row["Normalised Phase"] = "Phase 2"
        p2_row[date_col] = p2_date.year if is_year_only else p2_date
        p2_row[phase_col] = "Phase 2 (calculated)"
        new_rows.append(p2_row)

        # Build Phase 1 row
        p1_row = row.copy()
        p1_row["Normalised Phase"] = "Phase 1"
        p1_row[date_col] = p1_date.year if is_year_only else p1_date
        p1_row[phase_col] = "Phase 1 (calculated)"
        new_rows.append(p1_row)

        if is_year_only:
            print(f"  {mol_name}: Phase 3 = {p3_date.year} → Phase 2 = {p2_date.year} → Phase 1 = {p1_date.year}")
        else:
            print(f"  {mol_name}: Phase 3 = {p3_date.strftime('%d-%m-%Y')} → Phase 2 = {p2_date.strftime('%d-%m-%Y')} → Phase 1 = {p1_date.strftime('%d-%m-%Y')}")

    if new_rows:
        result = pd.concat([result, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"Added {len(new_rows)} calculated Phase 1/Phase 2 rows")
    else:
        print("No Phase 3 rows found — cannot calculate Phase 1/Phase 2")

    # ── Fetch live approval dates ONLY for marketed drugs ───────────────────
    # Determine which drugs are marketed from the original input data
    if _APPROVAL_AVAILABLE:
        # Check original df for any row where phase normalises to "Marketed"
        all_marketed = set()
        for _, row in df.iterrows():
            if normalise_phase(str(row.get(phase_col, ""))) == "Marketed":
                mol = str(row.get(mol_col, "")).strip()
                if mol and mol.lower() not in ("nan", "none", ""):
                    all_marketed.add(mol)

        # Also check the processed result
        for mol in result[result["Normalised Phase"] == "Marketed"][mol_col].unique():
            all_marketed.add(mol)

        marketed_molecules = sorted(all_marketed)

        if marketed_molecules:
            print(f"\n[APPROVAL] Marketed drugs detected: {marketed_molecules}")
            print(f"[APPROVAL] Fetching approval dates for {len(marketed_molecules)} marketed molecule(s)...")

            def _run_async(coro):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError
                    return loop.run_until_complete(coro)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(coro)

            approval_map = _run_async(_fetch_approval_for_molecules(marketed_molecules))

            # Add Approval Date column to all rows (empty by default)
            result["Approval Date"] = ""

            marketed_rows = []
            for mol_name, dates in approval_map.items():
                # Pick the earliest approval date across US and EU
                parsed_dates = []
                for geo in ("US", "EU"):
                    raw = dates.get(geo)
                    if raw:
                        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
                        if pd.notna(parsed):
                            parsed_dates.append(parsed)

                if not parsed_dates:
                    continue

                earliest = min(parsed_dates)

                # Fill in approval date on existing Marketed rows
                existing_marketed = result[
                    (result[mol_col] == mol_name) &
                    (result["Normalised Phase"] == "Marketed")
                ]
                if not existing_marketed.empty:
                    result.loc[existing_marketed.index, "Approval Date"] = earliest
                    print(f"  {mol_name}: Marketed row exists — added approval date {earliest.strftime('%d-%m-%Y')}")
                else:
                    # Build a Marketed row
                    marketed_row = {col: "" for col in result.columns}
                    marketed_row[mol_col] = mol_name
                    marketed_row["Normalised Phase"] = "Marketed"
                    marketed_row[phase_col] = "Marketed"
                    marketed_row[date_col] = earliest
                    marketed_row["Approval Date"] = earliest
                    marketed_rows.append(marketed_row)
                    print(f"  {mol_name}: Marketed = {earliest.strftime('%d-%m-%Y')}")

            if marketed_rows:
                result = pd.concat([result, pd.DataFrame(marketed_rows)], ignore_index=True)
                print(f"Added {len(marketed_rows)} Marketed rows from live approval dates")
        else:
            print("\n[APPROVAL] No marketed drugs found — skipping approval date fetching")
            result["Approval Date"] = ""
    else:
        print("[APPROVAL] approval_date_fetcher not available — skipping Marketed rows")
        result["Approval Date"] = ""

    # ── Filter to only drugs present in GCS ───────────────────────────────
    gcs_drugs = _list_gcs_drug_names()
    if gcs_drugs:
        before_gcs = len(result)
        # Normalise GCS drug names for matching
        gcs_norm = {d.strip().lower().replace(" ", "").replace("-", "").replace("_", "") for d in gcs_drugs}

        def _mol_in_gcs(mol_name):
            # First try the canonical name (resolves aliases)
            canonical = _canonicalise_drug_name(str(mol_name))
            norm = canonical.lower().replace(" ", "").replace("-", "").replace("_", "")
            if norm in gcs_norm:
                return True
            # Also try the original name directly
            orig_norm = str(mol_name).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
            return orig_norm in gcs_norm

        result = result[result[mol_col].apply(_mol_in_gcs)].reset_index(drop=True)
        after_gcs = len(result)
        dropped_gcs = before_gcs - after_gcs
        if dropped_gcs:
            print(f"\n[GCS FILTER] Dropped {dropped_gcs} rows for drugs not in GCS")
        print(f"[GCS FILTER] Kept {after_gcs} rows for {result[mol_col].nunique()} drug(s) present in GCS")
    else:
        print("[GCS FILTER] Could not list GCS drugs — keeping all molecules")

    # Sort output: molecule, then phase order
    _phase_order = {
        "Preclinical": 0, "Phase 1": 1, "Phase 2": 2,
        "Phase 3": 3, "Pre-registration": 4, "Marketed": 5,
    }
    result["_sort"] = result["Normalised Phase"].map(_phase_order).fillna(99)
    result = result.sort_values([mol_col, "_sort"]).drop(columns=["_sort"]).reset_index(drop=True)

    print(f"\nResult: {len(result)} rows ({result[mol_col].nunique()} molecules)")
    return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process phase Excel — normalise phases, calculate Phase 1/2, fetch approval dates.")
    parser.add_argument("--input",  required=True, help="Path to input Excel file")
    parser.add_argument("--output", default=None,   help="Path to output Excel file (default: input_processed.xlsx)")
    args = parser.parse_args()

    result = process_phase_excel(args.input)

    # ── Format date columns as date-only strings (no 00:00:00) ────────────
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].dt.strftime("%d-%m-%Y").replace("NaT", "")

    out_path = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_processed"))

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Phases")
        ws = writer.sheets["Phases"]
        for col_cells in ws.columns:
            length = max(
                len(str(cell.value)) if cell.value else 0
                for cell in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(length + 4, 60)

    print(f"\nSaved to: {out_path}")
