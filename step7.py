"""
step7.py — Patent Claim Decomposition Engine
=============================================
Reads ONE patent at a time from step6's charting queue.
Retrieves full claim text via Gemini + Google Search grounding,
decomposes independent claims into litigation-style limitations,
and emits a JSON skeleton chart + a human-readable table.

If step6 output is not available for a drug, step6 is run first automatically.

Model: gemini-2.5-flash-preview  (with Google Search grounding)

Usage:
    # Process all patents in the charting queue for a drug
    python step7.py --drug Axitinib

    # Process one specific patent
    python step7.py --drug Axitinib --patent US10123456

    # Process priority 1 patent only
    python step7.py --drug Axitinib --priority 1

    # Force re-run step6 before step7
    python step7.py --drug Axitinib --rerun_step6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────
# Paths — must match step6.py
# ─────────────────────────────────────────────────────────────

EXCEL_OUTPUT_DIR = Path(os.getenv("EXCEL_OUTPUT_DIR", Path(__file__).parent / "patent_exports"))
STEP6_OUTPUT_DIR = Path(os.getenv("STEP6_OUTPUT_DIR", Path(__file__).parent / "step6_output"))
STEP7_OUTPUT_DIR = Path(os.getenv("STEP7_OUTPUT_DIR", Path(__file__).parent / "step7_output"))

# ─────────────────────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────────────────────

_api_key      = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=_api_key)
MODEL         = "gemini-2.5-flash-preview-05-20"

# ─────────────────────────────────────────────────────────────
# Step 6 bootstrap
# ─────────────────────────────────────────────────────────────

def _ensure_step6(drug_name: str, force: bool = False) -> Optional[list]:
    """
    Return the charting queue for drug_name from step6 output.
    If the queue file is missing (or force=True), run step6 first.
    Returns None if step6 cannot produce output.
    """
    safe       = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    queue_path = STEP6_OUTPUT_DIR / f"{safe}_charting_queue.json"

    if force or not queue_path.exists():
        print(f"[Step 7] Step 6 output not found for '{drug_name}' — running step6 first...")
        # Import and run step6 inline
        try:
            from step6 import run_for_drug
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "step6", Path(__file__).parent / "step6.py"
            )
            step6_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(step6_mod)
            run_for_drug = step6_mod.run_for_drug

        STEP6_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result = run_for_drug(
            drug_name  = drug_name,
            excel_dir  = EXCEL_OUTPUT_DIR,
            output_dir = STEP6_OUTPUT_DIR,
        )
        if result is None:
            print(f"[Step 7] Step 6 failed for '{drug_name}'. Cannot continue.")
            return None

    if not queue_path.exists():
        print(f"[Step 7] Charting queue still not found after step6 run: {queue_path}")
        return None

    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)

    print(f"[Step 7] Loaded charting queue: {len(queue)} patent(s) for '{drug_name}'")
    return queue


# ─────────────────────────────────────────────────────────────
# Gemini claim retrieval + decomposition
# ─────────────────────────────────────────────────────────────

_CLAIMS_PROMPT = """\
You are a patent claim decomposition engine for IP litigation analysis.
You do NOT assess prior art here. You read claims only.

TASK
====
Patent number : {patent_number}
Jurisdiction  : {jurisdiction}
Drug          : {drug_name}
Filing date   : {filing_date}
Category      : {blocking_category}

STEP 1 — Retrieve claims
Search Google Patents for patent number "{patent_number}" ({jurisdiction}).
Pull the FULL claim listing verbatim from the patent document.
If unavailable on Google Patents, try the INPADOC family equivalent.
If still unavailable, return exactly:
  {{"error": "CLAIMS_UNAVAILABLE - route to counsel", "patent_number": "{patent_number}"}}
and stop.

STEP 2 — Keep independent claims only
Independent claims do NOT contain phrases like "claim 1" or "as recited in".
List each independent claim number and its FULL verbatim text.

STEP 3 — Decompose each independent claim into limitations
Use litigation-style labels:
  [<n>.P]        = preamble (everything before "comprising:" / "consisting of:")
  [<n>.a], [<n>.b], ...  = each element/wherein/said/at-least clause

Rules:
- ONE testable technical feature per limitation.
- Split compound clauses (e.g. "X at pH 5.5-7.5 AND 5-10 mg/mL") into
  SEPARATE limitations.
- Quote claim words VERBATIM. Do not paraphrase.
- Preserve antecedent basis exactly as written.
- Tag each limitation with ONE limitation_type from:
    compound_structure | salt_form | excipient | concentration |
    pH | device_feature | dosing | method_step | process_step
- If the claim is a Markush / genus claim, decompose the independent scaffold
  and add flag "MARKUSH - structural search required in Step 8" to that limitation.

OUTPUT FORMAT
=============
Return ONLY a single valid JSON object — no markdown fences, no prose outside JSON.

{{
  "drug_name": "{drug_name}",
  "patent_number": "{patent_number}",
  "jurisdiction": "{jurisdiction}",
  "priority_date": "{filing_date}",
  "blocking_category": "{blocking_category}",
  "source_file": "{source_file}",
  "independent_claims": [
    {{
      "claim_number": 1,
      "claim_text_verbatim": "...",
      "limitations": [
        {{
          "limitation_id": "1.P",
          "limitation_text_verbatim": "...",
          "limitation_type": "compound_structure",
          "flags": []
        }},
        {{
          "limitation_id": "1.a",
          "limitation_text_verbatim": "...",
          "limitation_type": "concentration",
          "flags": []
        }}
      ]
    }}
  ]
}}
"""


async def decompose_patent(item: dict) -> Optional[dict]:
    """
    Run the full claim decomposition for one charting queue item.
    Uses Gemini 2.5 Flash Preview with Google Search grounding.
    Returns the parsed JSON result dict, or None on failure.
    """
    patent_number    = item["patent_number"]
    jurisdiction     = item["jurisdiction"]
    drug_name        = item["drug_name"]
    filing_date      = item.get("filing_date", "Unknown")
    blocking_category = item.get("blocking_category", "")
    source_file      = item.get("source_file", "")

    prompt = _CLAIMS_PROMPT.format(
        patent_number     = patent_number,
        jurisdiction      = jurisdiction,
        drug_name         = drug_name,
        filing_date       = filing_date,
        blocking_category = blocking_category,
        source_file       = source_file,
    )

    print(f"\n[Step 7] Decomposing {patent_number} ({jurisdiction}) for {drug_name}...")

    try:
        response = await gemini_client.aio.models.generate_content(
            model    = MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                tools       = [types.Tool(google_search=types.GoogleSearch())],
                temperature = 0.0,   # deterministic — we want verbatim claims
            ),
        )

        raw = (response.text or "").strip()

        # Strip markdown fences if model wraps output despite instruction
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        if not raw:
            print(f"[Step 7] Empty response for {patent_number}")
            return None

        result = json.loads(raw)

        # Propagate charting metadata
        result["charting_priority"] = item.get("charting_priority")
        result["patent_expiry_year"] = item.get("patent_expiry_year")

        return result

    except json.JSONDecodeError as e:
        print(f"[Step 7] JSON parse error for {patent_number}: {e}")
        print(f"         Raw response (first 500 chars): {raw[:500]}")
        return None
    except Exception as e:
        print(f"[Step 7] Gemini error for {patent_number}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Markdown table renderer
# ─────────────────────────────────────────────────────────────

def _render_table(result: dict) -> str:
    lines = []
    patent_num = result.get("patent_number", "?")
    drug       = result.get("drug_name", "?")
    expiry     = result.get("patent_expiry_year", "?")
    priority   = result.get("charting_priority", "?")

    lines.append(f"\n## Charting Priority #{priority} — {patent_num} ({result.get('jurisdiction','?')})")
    lines.append(f"**Drug:** {drug} | **Priority date:** {result.get('priority_date','?')} "
                 f"| **Expiry:** {expiry} | **Category:** {result.get('blocking_category','?')}")
    lines.append("")
    lines.append("| Claim | Limitation ID | Limitation (verbatim) | Type | Prior Art (Step 8) |")
    lines.append("|-------|--------------|----------------------|------|-------------------|")

    for claim in result.get("independent_claims", []):
        cn = claim.get("claim_number", "?")
        for lim in claim.get("limitations", []):
            lid   = lim.get("limitation_id", "")
            text  = lim.get("limitation_text_verbatim", "").replace("|", "\\|").replace("\n", " ")
            ltype = lim.get("limitation_type", "")
            flags = "; ".join(lim.get("flags", []))
            flag_note = f" ⚠ {flags}" if flags else ""
            lines.append(f"| {cn} | {lid} | {text}{flag_note} | {ltype} | — |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def _write_patent_output(drug_name: str, result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_drug   = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    safe_patent = re.sub(r"[^a-zA-Z0-9_-]", "_", result.get("patent_number", "unknown"))

    # JSON skeleton
    json_path = output_dir / f"{safe_drug}_{safe_patent}_claim_skeleton.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  → Claim skeleton  : {json_path}")

    # Markdown table
    md_path = output_dir / f"{safe_drug}_{safe_patent}_charting_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_table(result))
    print(f"  → Charting table  : {md_path}")


def _write_drug_summary(drug_name: str, all_results: list[dict], output_dir: Path) -> None:
    """Write a combined charting table for all patents of a drug."""
    safe_drug = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    md_path   = output_dir / f"{safe_drug}_all_charting_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Step 7 Charting Tables — {drug_name}\n")
        f.write(f"**{len(all_results)} patent(s) decomposed**\n")
        for r in sorted(all_results, key=lambda x: x.get("charting_priority") or 999):
            f.write(_render_table(r))
            f.write("\n\n---\n")
    print(f"\n  → Combined tables : {md_path}")


# ─────────────────────────────────────────────────────────────
# Main async runner
# ─────────────────────────────────────────────────────────────

async def process_drug(
    drug_name:     str,
    patent_filter: Optional[str]  = None,   # specific patent number
    priority_filter: Optional[int] = None,  # specific charting priority
    rerun_step6:   bool            = False,
    output_dir:    Path            = STEP7_OUTPUT_DIR,
) -> list[dict]:
    """
    Process all (or filtered) patents in the charting queue for drug_name.
    Returns list of decomposition result dicts.
    """
    queue = _ensure_step6(drug_name, force=rerun_step6)
    if not queue:
        return []

    # Apply filters
    if patent_filter:
        queue = [i for i in queue if i["patent_number"].upper() == patent_filter.upper()]
        if not queue:
            print(f"[Step 7] Patent '{patent_filter}' not found in charting queue for '{drug_name}'.")
            return []

    if priority_filter is not None:
        queue = [i for i in queue if i.get("charting_priority") == priority_filter]
        if not queue:
            print(f"[Step 7] No patent with priority {priority_filter} for '{drug_name}'.")
            return []

    print(f"[Step 7] Processing {len(queue)} patent(s) for '{drug_name}'...")

    # Run decompositions — sequentially to respect Gemini rate limits
    # (each call already does a web search grounding round-trip)
    results = []
    for item in sorted(queue, key=lambda x: x.get("charting_priority") or 999):
        result = await decompose_patent(item)
        if result:
            _write_patent_output(drug_name, result, output_dir)
            results.append(result)
        else:
            print(f"[Step 7] ⚠  Skipping {item['patent_number']} — decomposition failed.")

    if results:
        _write_drug_summary(drug_name, results, output_dir)

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7 – Patent Claim Decomposition Engine (Gemini + Google Search)"
    )
    parser.add_argument(
        "--drug", "-d", required=True,
        help="Drug name (must match step6 output, e.g. 'Axitinib')",
    )
    parser.add_argument(
        "--patent", "-p", default=None,
        help="Process only this patent number (e.g. US10123456)",
    )
    parser.add_argument(
        "--priority", type=int, default=None,
        help="Process only the patent with this charting_priority (e.g. 1)",
    )
    parser.add_argument(
        "--rerun_step6", action="store_true",
        help="Force re-run step6 even if its output already exists",
    )
    parser.add_argument(
        "--output_dir", default=str(STEP7_OUTPUT_DIR),
        help=f"Output directory (default: {STEP7_OUTPUT_DIR}, auto-created)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 7] Drug        : {args.drug}")
    print(f"[Step 7] Model       : {MODEL}")
    print(f"[Step 7] Patent      : {args.patent or 'all in queue'}")
    print(f"[Step 7] Priority    : {args.priority or 'all'}")
    print(f"[Step 7] Output dir  : {output_dir.resolve()}")

    results = asyncio.run(process_drug(
        drug_name       = args.drug,
        patent_filter   = args.patent,
        priority_filter = args.priority,
        rerun_step6     = args.rerun_step6,
        output_dir      = output_dir,
    ))

    print(f"\n[Step 7] Done. {len(results)} patent(s) decomposed.")
    if not results:
        print("[Step 7] ⚠  No patents decomposed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
