"""
export_charts.py — Export step9 all_final_charts JSON to Excel
===============================================================
Usage:
    python export_charts.py --drug Axitinib
    python export_charts.py --drug Axitinib --input path/to/custom.json
    python export_charts.py --drug Axitinib --output my_charts.xlsx
"""

import argparse
import json
import re
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

STEP9_OUTPUT_DIR = Path(__file__).parent / "step9_output"


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
        c = _cell(ws, row, col, h, bold=True, font_size=font_size)
    return row + 1

def _col_widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def build_excel(drug_name: str, all_charts: list, output_path: Path) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # ── Sheet 1: Claim Charts ─────────────────────────────────
    ws1 = wb.create_sheet("Claim Charts")
    _col_widths(ws1, [14, 8, 10, 8, 60, 14, 80, 35, 40])
    r = _header_row(ws1, [
        "Patent No.", "Claim", "Ground", "Basis",
        "Limitation (verbatim)", "Limitation ID",
        "Prior Art Passage(s)", "Reference(s)", "Locus / Citation",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        for chart in chart_data.get("charts", []):
            cn        = chart["claim_number"]
            ground_id = chart["ground_id"]
            basis     = chart["basis"]
            refs      = ", ".join(chart.get("references", []))

            # Ground header (merged)
            ws1.merge_cells(start_row=r, start_column=1, end_row=r, end_column=9)
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

                _cell(ws1, r, 1, patent)
                _cell(ws1, r, 2, cn)
                _cell(ws1, r, 3, ground_id)
                _cell(ws1, r, 4, f"§{basis}")
                _cell(ws1, r, 5, ltext)
                _cell(ws1, r, 6, lid)
                _cell(ws1, r, 7, passage_text)
                _cell(ws1, r, 8, ref_text)
                _cell(ws1, r, 9, locus_text)
                ws1.row_dimensions[r].height = max(40, min(120, 15 * (passage_text.count("\n") + 2)))
                r += 1

            r += 1  # blank row between grounds

    ws1.freeze_panes = "A2"

    # ── Sheet 2: Coverage Summary ─────────────────────────────
    ws2 = wb.create_sheet("Coverage Summary")
    _col_widths(ws2, [14, 8, 8, 45, 15, 12, 50, 50])
    r = _header_row(ws2, [
        "Patent No.", "Claim", "Ground", "Reference(s)",
        "Covered / Total", "Coverage %", "Uncovered Lim IDs", "Combination Rationale",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        for cs in chart_data.get("coverage_summary", []):
            _cell(ws2, r, 1, patent)
            _cell(ws2, r, 2, cs.get("claim", ""))
            _cell(ws2, r, 3, cs.get("ground", ""))
            _cell(ws2, r, 4, ", ".join(cs.get("references", [])))
            _cell(ws2, r, 5, cs.get("covered_total", ""))
            _cell(ws2, r, 6, f"{cs.get('coverage_pct', 0)}%")
            _cell(ws2, r, 7, ", ".join(cs.get("uncovered_lim_ids", [])) or "—")
            _cell(ws2, r, 8, cs.get("combination_rationale", "—"))
            ws2.row_dimensions[r].height = 30
            r += 1

    ws2.freeze_panes = "A2"

    # ── Sheet 3: Reference List ───────────────────────────────
    ws3 = wb.create_sheet("Reference List")
    _col_widths(ws3, [14, 25, 60, 15, 14, 25])
    r = _header_row(ws3, [
        "Patent No.", "Short Name", "Full Citation",
        "Publication Date", "Pre-Priority?", "Access",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        for ref in chart_data.get("reference_list", []):
            _cell(ws3, r, 1, patent)
            _cell(ws3, r, 2, ref.get("short_name", ""), bold=True)
            _cell(ws3, r, 3, ref.get("full_citation", ""))
            _cell(ws3, r, 4, ref.get("publication_date", ""))
            _cell(ws3, r, 5, "Yes" if ref.get("pre_priority") else "No")
            _cell(ws3, r, 6, ref.get("access", "publicly accessible"))
            ws3.row_dimensions[r].height = 25
            r += 1

    ws3.freeze_panes = "A2"

    # ── Sheet 4: Gap List ─────────────────────────────────────
    ws4 = wb.create_sheet("Gap List")
    _col_widths(ws4, [14, 8, 14, 60, 30, 30, 55])
    r = _header_row(ws4, [
        "Patent No.", "Claim", "Limitation ID", "Limitation (verbatim)",
        "Flags", "Recommended Source", "Recommendation",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        for gap in chart_data.get("gap_list", []):
            _cell(ws4, r, 1, patent)
            _cell(ws4, r, 2, gap.get("claim_number", ""))
            _cell(ws4, r, 3, gap.get("limitation_id", ""), bold=True)
            _cell(ws4, r, 4, gap.get("limitation_text", ""))
            _cell(ws4, r, 5, "; ".join(gap.get("flags", [])) or "—")
            _cell(ws4, r, 6, gap.get("recommended_source", "—"))
            _cell(ws4, r, 7, gap.get("recommendation", "—"))
            ws4.row_dimensions[r].height = 35
            r += 1

    if r == 2:
        ws4.cell(row=2, column=1, value="No gaps — all limitations covered.")

    ws4.freeze_panes = "A2"

    # ── Sheet 5: Grace Annex ──────────────────────────────────
    ws5 = wb.create_sheet("Grace Annex")
    _col_widths(ws5, [14, 8, 14, 35, 60])
    r = _header_row(ws5, [
        "Patent No.", "Claim", "Limitation ID",
        "Reference (Grace Period)", "Note",
    ], 1)

    for chart_data in all_charts:
        patent = chart_data.get("patent", "")
        for ga in chart_data.get("grace_annex", []):
            _cell(ws5, r, 1, patent)
            _cell(ws5, r, 2, ga.get("claim_number", ""))
            _cell(ws5, r, 3, ga.get("limitation_id", ""), bold=True)
            _cell(ws5, r, 4, ga.get("reference_id", ""))
            _cell(ws5, r, 5,
                "Published within 12 months of priority — "
                "admissibility case-by-case; counsel to confirm")
            ws5.row_dimensions[r].height = 30
            r += 1

    if r == 2:
        ws5.cell(row=2, column=1, value="No grace-period-only limitations.")

    ws5.freeze_panes = "A2"

    wb.save(str(output_path))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export step9 all_final_charts JSON to Excel"
    )
    parser.add_argument("--drug",   "-d", required=True,  help="Drug name")
    parser.add_argument("--input",  "-i", default=None,   help="Path to JSON file (optional)")
    parser.add_argument("--output", "-o", default=None,   help="Output .xlsx path (optional)")
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
        all_charts = [all_charts]   # single-patent file

    print(f"Loaded {len(all_charts)} patent(s) from {json_path.name}")

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.drug)
        out_path = json_path.parent / f"{safe}_claim_charts.xlsx"

    build_excel(args.drug, all_charts, out_path)


if __name__ == "__main__":
    main()
