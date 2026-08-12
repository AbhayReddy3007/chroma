"""
export_charts.py — Export step9 all_final_charts JSON to Excel
===============================================================
Adds a "Filed By" column (from BigQuery patent_discovery table) beside
each Patent No. column across all sheets.

Assignee lookup priority:
  1. eval_actual_assignee
  2. assignee
  3. Retry with underscores stripped (US_12534530_B2 → US12534530B2)
  4. "Unknown" if all fail

Usage:
    python export_charts.py --drug Axitinib
    python export_charts.py --drug Axitinib --input path/to/custom.json
    python export_charts.py --drug Axitinib --output my_charts.xlsx
    python export_charts.py --drug Axitinib --no_bq   # skip BQ, leave Filed By blank
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# Load .env (explicitly from the script's own directory first,
# then fall back to python-dotenv's normal cwd-based search).
# This matters because GCS_SERVICE_ACCOUNT is read from here —
# if the .env isn't found, the BigQuery client silently falls
# back to ADC (application default credentials), which is the
# most common cause of "Could not create BigQuery client".
# ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv, find_dotenv

    _env_path = SCRIPT_DIR / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)
        print(f"[ENV] Loaded .env from {_env_path}")
    else:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=True)
            print(f"[ENV] Loaded .env from {found}")
        else:
            print(f"[ENV] No .env file found (looked in {SCRIPT_DIR} and cwd). "
                  f"GCS_SERVICE_ACCOUNT must be set as a real environment variable instead.")
except ImportError:
    print("[ENV] python-dotenv not installed (`pip install python-dotenv`) — "
          "a .env file will NOT be read. Set GCS_SERVICE_ACCOUNT as an actual "
          "environment variable, or install python-dotenv.")

import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

STEP9_OUTPUT_DIR = SCRIPT_DIR / "step9_output"
BQ_TABLE         = os.getenv("BQ_PATENT_TABLE",
                             "prj-portfolio-ai-dev.portfolio_data.patent_discovery")


# ─────────────────────────────────────────────────────────────
# BigQuery assignee lookup
# ─────────────────────────────────────────────────────────────

def _strip_underscores(patent_number: str) -> str:
    """US_12534530_B2  →  US12534530B2"""
    return patent_number.replace("_", "")


def _make_bq_client():
    """
    Build a BigQuery client, preferring GCS_SERVICE_ACCOUNT (path to a
    service-account JSON key) over application default credentials.

    Returns (client, error_message). If client is None, error_message
    explains exactly why.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        return None, ("google-cloud-bigquery is not installed. "
                       "Run: pip install google-cloud-bigquery google-auth")

    sa_path = os.getenv("GCS_SERVICE_ACCOUNT")

    if sa_path:
        sa_path = os.path.expanduser(sa_path.strip())
        # Resolve relative paths against the script directory, not the cwd,
        # so it works the same whether you run this from the project root
        # or from inside step9_output/ etc.
        candidate = Path(sa_path)
        if not candidate.is_absolute():
            candidate = (SCRIPT_DIR / candidate).resolve()

        if not candidate.exists():
            return None, (f"GCS_SERVICE_ACCOUNT is set to '{sa_path}' but no file "
                           f"exists there (resolved to '{candidate}'). Check the path "
                           f"in your .env — it should point at the service-account "
                           f"JSON key file, e.g. GCS_SERVICE_ACCOUNT=./keys/sa.json")

        try:
            key_data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as e:
            return None, (f"GCS_SERVICE_ACCOUNT points at '{candidate}' but it isn't "
                           f"valid JSON ({e}). Make sure it's the downloaded service "
                           f"account key file, not something else.")

        missing = [k for k in ("type", "project_id", "private_key", "client_email")
                   if k not in key_data]
        if missing:
            return None, (f"'{candidate}' doesn't look like a service-account key "
                           f"(missing fields: {missing}). Download a fresh key from "
                           f"GCP Console → IAM & Admin → Service Accounts → Keys.")

        try:
            from google.oauth2 import service_account
        except ImportError:
            return None, "google-auth is not installed. Run: pip install google-auth"

        try:
            credentials = service_account.Credentials.from_service_account_file(
                str(candidate),
                scopes=["https://www.googleapis.com/auth/bigquery.readonly"],
            )
            client = bigquery.Client(credentials=credentials,
                                     project=credentials.project_id)
            print(f"[BQ] Authenticated via GCS_SERVICE_ACCOUNT: {candidate} "
                  f"(project={credentials.project_id})")
            return client, None
        except Exception as e:
            return None, (f"Failed to build a client from '{candidate}': {e}")

    # No GCS_SERVICE_ACCOUNT set at all — fall back to ADC, but say so loudly.
    print("[BQ] GCS_SERVICE_ACCOUNT not set — falling back to application default "
          "credentials (gcloud auth application-default login). If that's not what "
          "you intended, add GCS_SERVICE_ACCOUNT=/path/to/key.json to your .env.")
    try:
        client = bigquery.Client()
        return client, None
    except Exception as e:
        return None, (f"No GCS_SERVICE_ACCOUNT set and application default "
                       f"credentials failed too: {e}")


def fetch_assignees(patent_numbers: list[str]) -> dict[str, str]:
    """
    Query BigQuery for all patent numbers in one batched call.
    Returns {patent_number: assignee_string}.

    Tries each patent number as-is first; if no row found, retries
    with underscores stripped.
    """
    if not patent_numbers:
        return {}

    client, err = _make_bq_client()
    if client is None:
        print(f"[BQ] Could not create BigQuery client: {err}")
        print("[BQ] Filed By will show 'Unknown' for all patents. "
              "Use --no_bq to skip this step silently next time.")
        return {p: "Unknown" for p in patent_numbers}

    # Build two sets: original and stripped variants
    originals = list(dict.fromkeys(patent_numbers))   # deduplicated, order preserved
    stripped  = {_strip_underscores(p): p for p in originals}  # stripped → original

    # Combined set for a single IN-clause query
    all_variants = list(dict.fromkeys(originals + list(stripped.keys())))
    placeholders = ", ".join(f"'{p}'" for p in all_variants)

    query = f"""
        SELECT patent_number, eval_actual_assignee, assignee
        FROM `{BQ_TABLE}`
        WHERE patent_number IN ({placeholders})
    """

    print(f"[BQ] Querying assignees for {len(originals)} patent(s)...")
    try:
        rows = list(client.query(query).result())
    except Exception as e:
        print(f"[BQ] Query failed: {e}")
        traceback.print_exc()
        print("[BQ] Filed By will show 'Unknown' for all patents.")
        return {p: "Unknown" for p in patent_numbers}

    # Index results: patent_number → best assignee value
    bq_index: dict[str, str] = {}
    for row in rows:
        pn    = row.patent_number or ""
        value = row.eval_actual_assignee or row.assignee or ""
        if pn and value:
            bq_index[pn] = value

    print(f"[BQ] Got {len(bq_index)} result(s) from BigQuery.")

    # Map each original patent number to its assignee
    result: dict[str, str] = {}
    for orig in originals:
        if orig in bq_index:
            result[orig] = bq_index[orig]
            continue
        # Try stripped variant
        s = _strip_underscores(orig)
        if s != orig and s in bq_index:
            result[orig] = bq_index[s]
            print(f"[BQ]   {orig} → matched via stripped form '{s}'")
            continue
        # Not found
        print(f"[BQ]   {orig} → no assignee found (Filed By: Unknown)")
        result[orig] = "Unknown"

    return result


# ─────────────────────────────────────────────────────────────
# Excel helpers
# ─────────────────────────────────────────────────────────────

def _border():
    thin = Side(style="thin", color="000000")
    return Border(left=thin, right=thin, top=thin, bottom=thin)

def _cell(ws, row, col, value, bold=False, wrap=True, font_size=10):
    c = ws.cell(row=row, column=col, value=str(value) if value is not None else "")
    c.font      = Font(name="Arial", bold=bold, size=font_size)
    c.alignment = Alignment(wrap_text=wrap, vertical="top")
    c.border    = _border()
    return c

def _header_row(ws, headers, row, font_size=10):
    for col, h in enumerate(headers, 1):
        _cell(ws, row, col, h, bold=True, font_size=font_size)
    return row + 1

def _col_widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ─────────────────────────────────────────────────────────────
# Excel builder
# ─────────────────────────────────────────────────────────────

def build_excel(
    drug_name:  str,
    all_charts: list,
    output_path: Path,
    assignees:  dict[str, str],
) -> None:

    def filed_by(patent: str) -> str:
        return assignees.get(patent, "Unknown")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # ── Sheet 1: Claim Charts ─────────────────────────────────
    # Columns: Patent No. | Filed By | Claim | Ground | Basis |
    #          Limitation (verbatim) | Limitation ID |
    #          Prior Art Passage(s) | Reference(s) | Locus / Citation
    ws1 = wb.create_sheet("Claim Charts")
    _col_widths(ws1, [14, 30, 8, 10, 8, 60, 14, 80, 35, 40])
    r = _header_row(ws1, [
        "Patent No.", "Filed By", "Claim", "Ground", "Basis",
        "Limitation (verbatim)", "Limitation ID",
        "Prior Art Passage(s)", "Reference(s)", "Locus / Citation",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        fb     = filed_by(patent)
        for chart in chart_data.get("charts", []):
            cn        = chart["claim_number"]
            ground_id = chart["ground_id"]
            basis     = chart["basis"]

            # Ground header row (span all 10 columns)
            ws1.merge_cells(start_row=r, start_column=1, end_row=r, end_column=10)
            c = ws1.cell(row=r, column=1, value=chart.get("basis_line", ""))
            c.font      = Font(name="Arial", bold=True, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="center")
            c.border    = _border()
            ws1.row_dimensions[r].height = 20
            r += 1

            for row_data in chart.get("rows", []):
                lid   = row_data.get("limitation_id", "")
                ltext = row_data.get("limitation_text", "")

                passages_parts, ref_parts, locus_parts = [], [], []
                for cd in row_data.get("cells", []):
                    ref_id = cd.get("reference_id", "")
                    for p in cd.get("passages", []):
                        passage = p.get("passage_verbatim", "")
                        locus   = p.get("locus", "")
                        if passage and "Not disclosed" not in passage:
                            label = f"[{ref_id}] " if len(row_data["cells"]) > 1 else ""
                            passages_parts.append(f'{label}"{passage}"')
                            ref_parts.append(ref_id)
                            locus_parts.append(f"{ref_id} at {locus}" if locus else ref_id)

                passage_text = "\n\n".join(passages_parts) or "Not disclosed — see Gap List"
                ref_text     = "; ".join(dict.fromkeys(ref_parts)) or "—"
                locus_text   = "\n".join(locus_parts) or "—"

                _cell(ws1, r, 1,  patent)
                _cell(ws1, r, 2,  fb)
                _cell(ws1, r, 3,  cn)
                _cell(ws1, r, 4,  ground_id)
                _cell(ws1, r, 5,  f"§{basis}")
                _cell(ws1, r, 6,  ltext)
                _cell(ws1, r, 7,  lid)
                _cell(ws1, r, 8,  passage_text)
                _cell(ws1, r, 9,  ref_text)
                _cell(ws1, r, 10, locus_text)
                ws1.row_dimensions[r].height = max(40, min(120, 15 * (passage_text.count("\n") + 2)))
                r += 1

            r += 1  # blank row between grounds

    ws1.freeze_panes = "A2"

    # ── Sheet 2: Coverage Summary ─────────────────────────────
    ws2 = wb.create_sheet("Coverage Summary")
    _col_widths(ws2, [14, 30, 8, 8, 45, 15, 12, 50, 50])
    r = _header_row(ws2, [
        "Patent No.", "Filed By", "Claim", "Ground", "Reference(s)",
        "Covered / Total", "Coverage %", "Uncovered Lim IDs", "Combination Rationale",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        fb     = filed_by(patent)
        for cs in chart_data.get("coverage_summary", []):
            _cell(ws2, r, 1, patent)
            _cell(ws2, r, 2, fb)
            _cell(ws2, r, 3, cs.get("claim", ""))
            _cell(ws2, r, 4, cs.get("ground", ""))
            _cell(ws2, r, 5, ", ".join(cs.get("references", [])))
            _cell(ws2, r, 6, cs.get("covered_total", ""))
            _cell(ws2, r, 7, f"{cs.get('coverage_pct', 0)}%")
            _cell(ws2, r, 8, ", ".join(cs.get("uncovered_lim_ids", [])) or "—")
            _cell(ws2, r, 9, cs.get("combination_rationale", "—"))
            ws2.row_dimensions[r].height = 30
            r += 1

    ws2.freeze_panes = "A2"

    # ── Sheet 3: Reference List ───────────────────────────────
    ws3 = wb.create_sheet("Reference List")
    _col_widths(ws3, [14, 30, 25, 60, 15, 14, 25])
    r = _header_row(ws3, [
        "Patent No.", "Filed By", "Short Name", "Full Citation",
        "Publication Date", "Pre-Priority?", "Access",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        fb     = filed_by(patent)
        for ref in chart_data.get("reference_list", []):
            _cell(ws3, r, 1, patent)
            _cell(ws3, r, 2, fb)
            _cell(ws3, r, 3, ref.get("short_name", ""), bold=True)
            _cell(ws3, r, 4, ref.get("full_citation", ""))
            _cell(ws3, r, 5, ref.get("publication_date", ""))
            _cell(ws3, r, 6, "Yes" if ref.get("pre_priority") else "No")
            _cell(ws3, r, 7, ref.get("access", "publicly accessible"))
            ws3.row_dimensions[r].height = 25
            r += 1

    ws3.freeze_panes = "A2"

    # ── Sheet 4: Gap List ─────────────────────────────────────
    ws4 = wb.create_sheet("Gap List")
    _col_widths(ws4, [14, 30, 8, 14, 60, 30, 30, 55])
    r = _header_row(ws4, [
        "Patent No.", "Filed By", "Claim", "Limitation ID",
        "Limitation (verbatim)", "Flags", "Recommended Source", "Recommendation",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        fb     = filed_by(patent)
        for gap in chart_data.get("gap_list", []):
            _cell(ws4, r, 1, patent)
            _cell(ws4, r, 2, fb)
            _cell(ws4, r, 3, gap.get("claim_number", ""))
            _cell(ws4, r, 4, gap.get("limitation_id", ""), bold=True)
            _cell(ws4, r, 5, gap.get("limitation_text", ""))
            _cell(ws4, r, 6, "; ".join(gap.get("flags", [])) or "—")
            _cell(ws4, r, 7, gap.get("recommended_source", "—"))
            _cell(ws4, r, 8, gap.get("recommendation", "—"))
            ws4.row_dimensions[r].height = 35
            r += 1

    if r == 2:
        ws4.cell(row=2, column=1, value="No gaps — all limitations covered.")

    ws4.freeze_panes = "A2"

    # ── Sheet 5: Grace Annex ──────────────────────────────────
    ws5 = wb.create_sheet("Grace Annex")
    _col_widths(ws5, [14, 30, 8, 14, 35, 60])
    r = _header_row(ws5, [
        "Patent No.", "Filed By", "Claim", "Limitation ID",
        "Reference (Grace Period)", "Note",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        fb     = filed_by(patent)
        for ga in chart_data.get("grace_annex", []):
            _cell(ws5, r, 1, patent)
            _cell(ws5, r, 2, fb)
            _cell(ws5, r, 3, ga.get("claim_number", ""))
            _cell(ws5, r, 4, ga.get("limitation_id", ""), bold=True)
            _cell(ws5, r, 5, ga.get("reference_id", ""))
            _cell(ws5, r, 6,
                  "Published within 12 months of priority — "
                  "admissibility case-by-case; counsel to confirm")
            ws5.row_dimensions[r].height = 30
            r += 1

    if r == 2:
        ws5.cell(row=2, column=1, value="No grace-period-only limitations.")

    ws5.freeze_panes = "A2"

    wb.save(str(output_path))
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export step9 all_final_charts JSON to Excel with Filed By column"
    )
    parser.add_argument("--drug",   "-d", required=True,  help="Drug name")
    parser.add_argument("--input",  "-i", default=None,   help="Path to JSON file (optional)")
    parser.add_argument("--output", "-o", default=None,   help="Output .xlsx path (optional)")
    parser.add_argument("--no_bq",        action="store_true",
                        help="Skip BigQuery lookup; Filed By will show 'Unknown'")
    args = parser.parse_args()

    # Locate input JSON
    if args.input:
        json_path = Path(args.input)
    else:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.drug)
        json_path = STEP9_OUTPUT_DIR / f"{safe}_all_final_charts.json"

    if not json_path.exists():
        print(f"ERROR: JSON not found: {json_path}", file=sys.stderr)
        print("Run step9.py first, or pass --input with the correct path.")
        sys.exit(1)

    all_charts = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(all_charts, dict):
        all_charts = [all_charts]

    print(f"Loaded {len(all_charts)} patent(s) from {json_path.name}")

    # Collect all unique patent numbers across all charts
    patent_numbers = list(dict.fromkeys(
        c.get("patent", "") for c in all_charts if c.get("patent")
    ))
    print(f"Patents: {patent_numbers}")

    # Fetch assignees from BigQuery
    assignees: dict[str, str] = {}
    if not args.no_bq:
        assignees = fetch_assignees(patent_numbers)
    else:
        print("[BQ] Skipped (--no_bq)")

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.drug)
        out_path = json_path.parent / f"{safe}_claim_charts.xlsx"

    build_excel(args.drug, all_charts, out_path, assignees)


if __name__ == "__main__":
    main()
