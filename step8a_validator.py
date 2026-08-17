"""
step8a_validator.py — Prior Art Validator
==========================================
Reads step8a output (per-limitation prior art JSONs) and independently
validates each piece of evidence — re-fetching and re-reading the cited
source, checking the passage is real, confirming the date pre-dates priority,
and assigning a fresh confidence score 0.0 to 1.0.

Uses the same model as the rest of the pipeline (LLM_MODEL in .env),
with web search to actually verify the cited passage exists at the URL.

Output: validated JSON files in step8a_validated_output/
        markdown summary per drug showing which evidence passed/failed

Usage:
    python step8a_validator.py --drug Axitinib
    python step8a_validator.py --drug Axitinib --patent US10123456
    python step8a_validator.py --drug Axitinib --min_score 0.7
    python step8a_validator.py --drug Axitinib --model gemini-3.1-pro
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

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from llm_client import generate, parse_json_response, get_model_name

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

STEP8A_OUTPUT_DIR    = Path(os.getenv("STEP8A_OUTPUT_DIR",
                             Path(__file__).parent / "step8a_output"))
VALIDATED_OUTPUT_DIR = Path(os.getenv("STEP8A_VALIDATED_DIR",
                             Path(__file__).parent / "step8a_validated_output"))
_WORKERS = int(os.getenv("PIPELINE_WORKERS", "6"))

# ─────────────────────────────────────────────────────────────
# Validation prompt
# ─────────────────────────────────────────────────────────────

_VALIDATE_PROMPT = """\
You are an expert patent invalidity analyst performing quality control on
prior art citations. Your job is to independently verify each piece of
evidence and assign an honest confidence score.

PATENT BEING CHALLENGED
=======================
Patent number  : {patent_number}
Priority date  : {priority_date}
Claim          : {claim_number}
Drug           : {drug_name}

LIMITATION BEING MAPPED
=======================
ID   : {limitation_id}
Text : {limitation_text}

EVIDENCE TO VALIDATE
====================
Reference ID      : {reference_id}
Source            : {source}
Claimed pub date  : {publication_date}
Locus             : {locus}
Citation URL      : {citation_url}
Passage claimed   : "{passage_verbatim}"
Original rationale: {reads_on_rationale}
Original score    : {original_score}

VALIDATION TASKS
================
1. VERIFY THE SOURCE EXISTS: Search for "{reference_id}" and visit {citation_url}.
   Confirm the document exists and is publicly accessible.

2. VERIFY THE DATE: Confirm the document was published BEFORE {priority_date}.
   If the date is within 12 months before priority, flag GRACE_PERIOD.

3. VERIFY THE PASSAGE: Check whether the claimed passage actually appears in
   the cited document at the cited locus. Quote what you find at that location.

4. VERIFY THE READ-ON: Does the verified passage actually disclose the specific
   feature described in the limitation text for {drug_name}?
   Not just the same topic — the specific claimed feature.

5. SCORE: Assign a confidence_score 0.0 to 1.0 based on:
   - 0.9-1.0: Passage verified verbatim, exact feature match, date confirmed pre-priority
   - 0.7-0.89: Passage found with minor wording differences, strong feature match
   - 0.5-0.69: Passage partially confirmed, feature overlap but not exact
   - 0.3-0.49: Document exists but passage not found at locus, or only topical match
   - 0.0-0.29: Document not found, date wrong, or passage does not read on limitation
               → mark verdict: "REJECTED"

OUTPUT FORMAT
=============
Return ONLY a single valid JSON object — no markdown fences.

{{
  "reference_id":         "{reference_id}",
  "patent_number":        "{patent_number}",
  "claim_number":         {claim_number},
  "limitation_id":        "{limitation_id}",
  "drug_name":            "{drug_name}",
  "validation_checks": {{
    "source_exists":       true,
    "date_confirmed":      true,
    "date_pre_priority":   true,
    "passage_verified":    true,
    "reads_on_confirmed":  true
  }},
  "verified_passage":     "actual text found at the locus (may differ slightly from claimed)",
  "verified_date":        "YYYY-MM-DD",
  "verified_url":         "{citation_url}",
  "confidence_score":     0.85,
  "confidence_rationale": "One sentence: why this score, what was confirmed and what wasn't",
  "verdict":              "VALIDATED",
  "flags":                [],
  "validator_notes":      "Any discrepancies between claimed and found passage, date corrections, etc."
}}

verdict must be one of: "VALIDATED" | "PARTIALLY_VALIDATED" | "REJECTED"
flags may include: "GRACE_PERIOD" | "PASSAGE_NOT_FOUND" | "DATE_WRONG" |
                   "URL_INACCESSIBLE" | "TOPICAL_ONLY" | "FABRICATED_REFERENCE"
"""


# ─────────────────────────────────────────────────────────────
# Per-evidence validator
# ─────────────────────────────────────────────────────────────

async def _validate_one_evidence(
    patent_number:  str,
    priority_date:  str,
    claim_number:   int,
    limitation_id:  str,
    limitation_text: str,
    drug_name:      str,
    ev:             dict,
) -> dict:
    """Validate a single evidence item. Returns enriched evidence dict."""
    ref_id   = ev.get("reference_id", "")
    source   = ev.get("source", "")
    pub_date = ev.get("publication_date", "")
    locus    = ev.get("locus", "")
    url      = ev.get("citation_url", "")
    passage  = ev.get("passage_verbatim", "")
    rationale = ev.get("reads_on_rationale", "")
    orig_score = ev.get("confidence_score", "N/A")

    prompt = _VALIDATE_PROMPT.format(
        patent_number      = patent_number,
        priority_date      = priority_date,
        claim_number       = claim_number,
        limitation_id      = limitation_id,
        limitation_text    = limitation_text,
        drug_name          = drug_name,
        reference_id       = ref_id,
        source             = source,
        publication_date   = pub_date,
        locus              = locus,
        citation_url       = url or "not provided",
        passage_verbatim   = passage,
        reads_on_rationale = rationale,
        original_score     = orig_score,
    )

    try:
        raw = await generate(
            prompt            = prompt,
            use_web_search    = True,
            temperature       = 0.0,
            max_output_tokens = 8192,
        )

        if not raw:
            print(f"  [VALIDATOR] {ref_id}: empty response")
            return _failed_validation(ev, "EMPTY_RESPONSE")

        result = parse_json_response(raw)
        if result is None:
            print(f"  [VALIDATOR] {ref_id}: JSON parse failed")
            return _failed_validation(ev, "PARSE_ERROR")

        score   = result.get("confidence_score", 0.0)
        verdict = result.get("verdict", "UNKNOWN")
        print(f"  [VALIDATOR] {ref_id}: {verdict} | score: {score:.2f} "
              f"(was: {orig_score}) | model: {get_model_name()}")

        # Merge validation result back onto the original evidence item
        merged = {**ev}
        merged["validated"]              = True
        merged["confidence_score"]       = score          # overwrite with validated score
        merged["confidence_rationale"]   = result.get("confidence_rationale", "")
        merged["verdict"]                = verdict
        merged["validation_checks"]      = result.get("validation_checks", {})
        merged["verified_passage"]       = result.get("verified_passage", "")
        merged["verified_date"]          = result.get("verified_date", pub_date)
        merged["verified_url"]           = result.get("verified_url", url)
        merged["validator_notes"]        = result.get("validator_notes", "")
        merged["validation_flags"]       = result.get("flags", [])
        return merged

    except Exception as e:
        print(f"  [VALIDATOR] {ref_id}: LLM error — {e}")
        return _failed_validation(ev, f"LLM_ERROR: {e}")


def _failed_validation(ev: dict, reason: str) -> dict:
    """Return evidence dict marked as validation failed."""
    return {
        **ev,
        "validated":            False,
        "confidence_score":     0.0,
        "confidence_rationale": f"Validation failed: {reason}",
        "verdict":              "REJECTED",
        "validation_checks":    {},
        "verified_passage":     "",
        "verified_date":        "",
        "verified_url":         "",
        "validator_notes":      reason,
        "validation_flags":     ["VALIDATION_FAILED"],
    }


# ─────────────────────────────────────────────────────────────
# Per-limitation validator
# ─────────────────────────────────────────────────────────────

async def _validate_limitation_result(
    lr:       dict,
    drug_name: str,
) -> dict:
    """Validate all evidence items for one limitation."""
    patent_number   = lr.get("patent_number", "")
    priority_date   = lr.get("priority_date", "")
    claim_number    = lr.get("claim_number", 0)
    limitation_id   = lr.get("limitation_id", "")
    limitation_text = lr.get("limitation_text_verbatim", "")
    evidence        = lr.get("evidence", [])

    if not evidence:
        return {**lr, "evidence": [], "validation_summary": "no_evidence"}

    print(f"  [VALIDATOR] Claim {claim_number}, {limitation_id}: "
          f"validating {len(evidence)} evidence item(s)...")

    sem = asyncio.Semaphore(_WORKERS)

    async def _bounded(ev):
        async with sem:
            return await _validate_one_evidence(
                patent_number, priority_date, claim_number,
                limitation_id, limitation_text, drug_name, ev,
            )

    validated_evidence = await asyncio.gather(
        *[_bounded(ev) for ev in evidence],
        return_exceptions=True,
    )

    results = []
    for ev, res in zip(evidence, validated_evidence):
        if isinstance(res, Exception):
            print(f"  [VALIDATOR] {ev.get('reference_id', '?')} raised: {res}")
            results.append(_failed_validation(ev, str(res)))
        else:
            results.append(res)

    # Compute summary stats
    validated_count   = sum(1 for r in results if r.get("verdict") == "VALIDATED")
    partial_count     = sum(1 for r in results if r.get("verdict") == "PARTIALLY_VALIDATED")
    rejected_count    = sum(1 for r in results if r.get("verdict") == "REJECTED")
    avg_score         = (
        sum(r.get("confidence_score", 0.0) for r in results) / len(results)
        if results else 0.0
    )

    # Re-derive limitation status from validated results
    strong_evidence = [r for r in results if r.get("confidence_score", 0) >= 0.5
                       and r.get("verdict") != "REJECTED"]
    new_status = "covered" if strong_evidence else "uncovered"

    return {
        **lr,
        "evidence":           results,
        "limitation_status":  new_status,
        "validation_summary": {
            "total":     len(results),
            "validated": validated_count,
            "partial":   partial_count,
            "rejected":  rejected_count,
            "avg_score": round(avg_score, 3),
            "strong_evidence_count": len(strong_evidence),
        },
    }


# ─────────────────────────────────────────────────────────────
# Per-patent validator
# ─────────────────────────────────────────────────────────────

async def validate_patent(
    prior_art_data: dict,
    min_score:      float = 0.0,
) -> dict:
    """Validate all limitations for one patent."""
    patent_number = prior_art_data.get("patent_number", "")
    drug_name     = prior_art_data.get("drug_name", "")
    limitation_results = prior_art_data.get("limitation_results", [])

    print(f"\n[VALIDATOR] {'═'*50}")
    print(f"[VALIDATOR] Patent: {patent_number} | Drug: {drug_name}")
    print(f"[VALIDATOR] {len(limitation_results)} limitation(s) to validate")
    print(f"[VALIDATOR] {'═'*50}")

    validated_lrs = []
    for lr in limitation_results:
        validated_lr = await _validate_limitation_result(lr, drug_name)

        # Filter out evidence below min_score
        if min_score > 0:
            validated_lr["evidence"] = [
                e for e in validated_lr["evidence"]
                if e.get("confidence_score", 0) >= min_score
            ]

        validated_lrs.append(validated_lr)

    # Overall stats
    total_ev     = sum(len(lr.get("evidence", [])) for lr in validated_lrs)
    accepted_ev  = sum(
        1 for lr in validated_lrs
        for e in lr.get("evidence", [])
        if e.get("verdict") != "REJECTED"
    )
    covered      = sum(1 for lr in validated_lrs if lr.get("limitation_status") == "covered")
    total_lims   = len(validated_lrs)

    print(f"\n[VALIDATOR] ✓ {patent_number}: "
          f"{covered}/{total_lims} limitations covered after validation | "
          f"{accepted_ev}/{total_ev} evidence items accepted")

    return {
        **prior_art_data,
        "limitation_results": validated_lrs,
        "validation_overall": {
            "model":          get_model_name(),
            "min_score":      min_score,
            "total_evidence": total_ev,
            "accepted":       accepted_ev,
            "rejected":       total_ev - accepted_ev,
            "limitations_covered": covered,
            "limitations_total":   total_lims,
        },
    }


# ─────────────────────────────────────────────────────────────
# Loaders
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
            print(f"[VALIDATOR] Failed to read {f.name}: {e}")
    return results


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def _write_output(drug_name: str, result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_drug   = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    safe_patent = re.sub(r"[^a-zA-Z0-9_-]", "_", result.get("patent_number", "unknown"))

    # JSON
    json_path = output_dir / f"{safe_drug}_{safe_patent}_validated.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  → Validated JSON  : {json_path}")

    # Markdown summary
    md_path = output_dir / f"{safe_drug}_{safe_patent}_validation_report.md"
    _write_markdown(drug_name, result, md_path)
    print(f"  → Validation report: {md_path}")


def _write_markdown(drug_name: str, result: dict, path: Path) -> None:
    ov      = result.get("validation_overall", {})
    patent  = result.get("patent_number", "")
    drug    = result.get("drug_name", drug_name)
    lines   = [
        f"# Validation Report — {patent}",
        f"**Drug:** {drug} | **Model:** {ov.get('model', '?')} | "
        f"**Min score:** {ov.get('min_score', 0)}",
        f"",
        f"## Overall",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Limitations covered | {ov.get('limitations_covered', 0)} / {ov.get('limitations_total', 0)} |",
        f"| Evidence accepted   | {ov.get('accepted', 0)} / {ov.get('total_evidence', 0)} |",
        f"| Evidence rejected   | {ov.get('rejected', 0)} |",
        f"",
        f"## Per-Limitation Results",
        f"",
        f"| Claim | Lim ID | Status | Evidence | Avg Score | Validated | Rejected |",
        f"|-------|--------|--------|----------|-----------|-----------|---------|",
    ]

    for lr in result.get("limitation_results", []):
        cn   = lr.get("claim_number", "?")
        lid  = lr.get("limitation_id", "?")
        stat = lr.get("limitation_status", "?")
        vs   = lr.get("validation_summary", {})
        lines.append(
            f"| {cn} | {lid} | {stat} | {vs.get('total', 0)} | "
            f"{vs.get('avg_score', 0):.2f} | {vs.get('validated', 0)} | "
            f"{vs.get('rejected', 0)} |"
        )

    lines.append("")
    lines.append("## Evidence Detail")

    for lr in result.get("limitation_results", []):
        cn   = lr.get("claim_number", "?")
        lid  = lr.get("limitation_id", "?")
        ltext = lr.get("limitation_text_verbatim", "")[:80]
        lines.append(f"\n### Claim {cn}, {lid}")
        lines.append(f"*{ltext}...*")
        lines.append("")
        lines.append("| Reference | Verdict | Score | Flags |")
        lines.append("|-----------|---------|-------|-------|")
        for ev in lr.get("evidence", []):
            ref     = ev.get("reference_id", "?")
            verdict = ev.get("verdict", "?")
            score   = ev.get("confidence_score", 0)
            flags   = "; ".join(ev.get("validation_flags", []) + ev.get("flags", []))
            url     = ev.get("verified_url") or ev.get("citation_url", "")
            ref_link = f"[{ref}]({url})" if url else ref
            lines.append(f"| {ref_link} | {verdict} | {score:.2f} | {flags or '—'} |")
            if ev.get("validator_notes"):
                lines.append(f"  > *{ev['validator_notes']}*")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

async def run_validation(
    drug_name:     str,
    patent_filter: Optional[str] = None,
    min_score:     float         = 0.0,
    output_dir:    Path          = VALIDATED_OUTPUT_DIR,
) -> list[dict]:
    prior_art_list = _load_step8a(drug_name, patent_filter=patent_filter)
    if not prior_art_list:
        print(f"[VALIDATOR] No step8a output found for '{drug_name}' in {STEP8A_OUTPUT_DIR}")
        print("            Run step8a.py first.")
        return []

    print(f"[VALIDATOR] Validating {len(prior_art_list)} patent(s) for '{drug_name}'")
    print(f"[VALIDATOR] Model: {get_model_name()}")
    print(f"[VALIDATOR] Min score threshold: {min_score}")

    results = []
    for pa in prior_art_list:
        result = await validate_patent(pa, min_score=min_score)
        _write_output(drug_name, result, output_dir)
        results.append(result)

    # Combined JSON
    if results:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        combined_path = output_dir / f"{safe}_all_validated.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  → Combined JSON   : {combined_path}")

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prior Art Validator — independently verifies step8a evidence and re-scores 0-1"
    )
    parser.add_argument("--drug",      "-d", required=True,
                        help="Drug name (must match step8a output)")
    parser.add_argument("--patent",    "-p", default=None,
                        help="Validate only this patent number (optional)")
    parser.add_argument("--model",     "-m", default=None,
                        help="LLM model (overrides LLM_MODEL in .env)")
    parser.add_argument("--min_score",       default=0.0, type=float,
                        help="Minimum confidence score to keep evidence (default: 0.0 = keep all)")
    parser.add_argument("--output_dir",      default=str(VALIDATED_OUTPUT_DIR))
    args = parser.parse_args()

    if args.model:
        os.environ["LLM_MODEL"] = args.model.strip()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Prior Art Validator")
    print("=" * 60)
    print(f"  Drug       : {args.drug}")
    print(f"  Model      : {get_model_name()}")
    print(f"  Patent     : {args.patent or 'all'}")
    print(f"  Min score  : {args.min_score}")
    print(f"  Output dir : {output_dir.resolve()}")
    print("=" * 60)

    results = asyncio.run(run_validation(
        drug_name     = args.drug,
        patent_filter = args.patent,
        min_score     = args.min_score,
        output_dir    = output_dir,
    ))

    total_accepted = sum(r.get("validation_overall", {}).get("accepted", 0) for r in results)
    total_rejected = sum(r.get("validation_overall", {}).get("rejected", 0) for r in results)
    print(f"\n[VALIDATOR] Done. {total_accepted} accepted, {total_rejected} rejected.")

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
