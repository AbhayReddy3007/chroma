"""
run_pipeline.py — Full Invalidity Pipeline
===========================================
Runs all steps in order for one or more drugs:
  Step 6  → Scope (filter blocking patents, compute expiry)
  Step 7  → Claim decomposition (ChromaDB + Google Search)
  Step 8a → Prior art search (per limitation, drug name included)
  Step 8a Validator → Verify and re-score prior art
  Step 8b → Combination optimiser (102/103 grounds)
  Step 9  → Assemble final charts (Word + Excel)

Usage:
    python run_pipeline.py --drug Axitinib
    python run_pipeline.py --drug Axitinib Minocycline
    python run_pipeline.py --drug Axitinib --model gemini-3.1-pro-preview
    python run_pipeline.py --drug Axitinib --model claude-sonnet-4-6
    python run_pipeline.py --drug Axitinib --rerun_all
    python run_pipeline.py --drug Axitinib --min_score 0.6
    python run_pipeline.py --drug Axitinib --skip_validator
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────
# Module loader — imports step files as modules from same folder
# ─────────────────────────────────────────────────────────────

def _load_module(name: str) -> object:
    path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────
# Per-drug pipeline
# ─────────────────────────────────────────────────────────────

async def run_drug(
    drug_name:      str,
    rerun_all:      bool  = False,
    skip_validator: bool  = False,
    min_score:      float = 0.0,
    patent_filter:  str   = None,
) -> bool:
    """Run the full pipeline for one drug. Returns True if successful."""

    print(f"\n{'='*65}")
    print(f"  Drug: {drug_name}")
    print(f"{'='*65}")

    from llm_client import get_model_name

    steps = [
        "Step 6  — Scope",
        "Step 7  — Claim decomposition",
        "Step 8a — Prior art search",
        "Step 8a — Validator" if not skip_validator else "Step 8a — Validator (SKIPPED)",
        "Step 8b — Combination optimiser",
        "Step 9  — Chart assembly",
    ]
    for s in steps:
        print(f"  {'✓' if 'SKIPPED' not in s else '○'} {s}")
    print(f"  Model : {get_model_name()}")
    print(f"  Rerun : {rerun_all}")
    if not skip_validator:
        print(f"  Min score (validator): {min_score}")
    print(f"{'='*65}\n")

    t_start = time.time()

    # ── Step 6 ────────────────────────────────────────────────
    print(f"\n[Pipeline] ── STEP 6: Scope ──────────────────────────")
    try:
        step6 = _load_module("step6")
        STEP6_OUT = Path(__file__).parent / "step6_output"
        safe      = __import__("re").sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        queue_exists = list(STEP6_OUT.glob(f"{safe}_charting_queue.json"))
        if rerun_all or not queue_exists:
            result6 = step6.run_for_drug(
                drug_name          = drug_name,
                excel_dir          = Path(os.getenv("EXCEL_OUTPUT_DIR",
                                          Path(__file__).parent / "patent_exports")),
                output_dir         = STEP6_OUT,
                target_entry_year  = None,
            )
            if result6 is None:
                print(f"[Pipeline] ✗ Step 6 failed for '{drug_name}'")
                return False
        else:
            print(f"[Pipeline] ✓ Step 6 output exists — skipping (use --rerun_all to re-run)")
    except Exception as e:
        print(f"[Pipeline] ✗ Step 6 error: {e}")
        return False

    # ── Step 7 ────────────────────────────────────────────────
    print(f"\n[Pipeline] ── STEP 7: Claim decomposition ───────────")
    try:
        step7 = _load_module("step7")
        await step7.process_drug(
            drug_name   = drug_name,
            rerun_step6 = False,   # step6 already done above
            rerun       = rerun_all,
        )
    except Exception as e:
        print(f"[Pipeline] ✗ Step 7 error: {e}")
        return False

    # ── Step 8a ───────────────────────────────────────────────
    print(f"\n[Pipeline] ── STEP 8a: Prior art search ─────────────")
    try:
        step8a = _load_module("step8a")
        STEP8A_OUT = Path(__file__).parent / "step8a_output"
        safe_drug  = __import__("re").sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        pa_exists  = [f for f in STEP8A_OUT.glob(f"{safe_drug}_*_prior_art.json")
                      if "_all_prior_art" not in f.name]
        if rerun_all or not pa_exists:
            await step8a.process_drug(
                drug_name     = drug_name,
                patent_filter = patent_filter,
                rerun_step7   = False,
            )
        else:
            print(f"[Pipeline] ✓ Step 8a output exists — skipping (use --rerun_all to re-run)")
    except Exception as e:
        print(f"[Pipeline] ✗ Step 8a error: {e}")
        return False

    # ── Step 8a Validator ────────────────────────────────────
    if not skip_validator:
        print(f"\n[Pipeline] ── STEP 8a VALIDATOR: Verify prior art ──")
        try:
            validator = _load_module("step8a_validator")
            await validator.run_validation(
                drug_name     = drug_name,
                patent_filter = patent_filter,
                min_score     = min_score,
            )
        except Exception as e:
            print(f"[Pipeline] ✗ Validator error: {e}")
            # Non-fatal — continue to 8b/9 with unvalidated prior art
            print(f"[Pipeline] ⚠  Continuing without validation results")
    else:
        print(f"\n[Pipeline] ── STEP 8a VALIDATOR: SKIPPED ────────────")

    # ── Step 8b ───────────────────────────────────────────────
    print(f"\n[Pipeline] ── STEP 8b: Combination optimiser ────────")
    try:
        step8b = _load_module("step8b")
        STEP8B_OUT = Path(__file__).parent / "step8b_output"
        gf_exists  = [f for f in STEP8B_OUT.glob(f"{safe_drug}_*_grounds.json")
                      if "_all_grounds" not in f.name]
        if rerun_all or not gf_exists:
            await step8b.run_for_drug(
                drug_name     = drug_name,
                patent_filter = patent_filter,
                rerun_step8a  = False,
            )
        else:
            print(f"[Pipeline] ✓ Step 8b output exists — skipping (use --rerun_all to re-run)")
    except Exception as e:
        print(f"[Pipeline] ✗ Step 8b error: {e}")
        return False

    # ── Step 9 ────────────────────────────────────────────────
    print(f"\n[Pipeline] ── STEP 9: Chart assembly ────────────────")
    try:
        step9 = _load_module("step9")
        STEP9_OUT = Path(__file__).parent / "step9_output"
        STEP9_OUT.mkdir(parents=True, exist_ok=True)
        results = await step9.run_for_drug(
            drug_name     = drug_name,
            patent_filter = patent_filter,
            rerun_all     = rerun_all,
            output_dir    = STEP9_OUT,
        )
        if not results:
            print(f"[Pipeline] ✗ Step 9 produced no charts for '{drug_name}'")
            return False
    except Exception as e:
        print(f"[Pipeline] ✗ Step 9 error: {e}")
        return False

    elapsed = time.time() - t_start
    print(f"\n[Pipeline] ✓ '{drug_name}' complete in {elapsed:.1f}s")
    return True


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main_async(args) -> None:
    # Set model before anything is imported
    if args.model:
        os.environ["LLM_MODEL"] = args.model.strip()

    from llm_client import get_model_name

    print("=" * 65)
    print("  IP Invalidity Pipeline — Steps 6 → 7 → 8a → Val → 8b → 9")
    print("=" * 65)
    print(f"  Drugs      : {', '.join(args.drug)}")
    print(f"  Model      : {get_model_name()}")
    print(f"  Rerun all  : {args.rerun_all}")
    print(f"  Validator  : {'SKIP' if args.skip_validator else 'YES'}")
    if not args.skip_validator:
        print(f"  Min score  : {args.min_score}")
    print("=" * 65)

    successes = []
    failures  = []

    for drug in args.drug:
        ok = await run_drug(
            drug_name      = drug,
            rerun_all      = args.rerun_all,
            skip_validator = args.skip_validator,
            min_score      = args.min_score,
            patent_filter  = args.patent,
        )
        (successes if ok else failures).append(drug)

    print(f"\n{'='*65}")
    print(f"  Pipeline complete")
    print(f"  Success : {', '.join(successes) or 'none'}")
    if failures:
        print(f"  Failed  : {', '.join(failures)}")
    print(f"{'='*65}")

    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full invalidity pipeline: Steps 6 → 7 → 8a → Validator → 8b → 9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --drug Axitinib
  python run_pipeline.py --drug Axitinib Minocycline
  python run_pipeline.py --drug Axitinib --model gemini-3.1-pro-preview
  python run_pipeline.py --drug Axitinib --model claude-sonnet-4-6
  python run_pipeline.py --drug Axitinib --rerun_all
  python run_pipeline.py --drug Axitinib --skip_validator
  python run_pipeline.py --drug Axitinib --min_score 0.6
        """
    )
    parser.add_argument("--drug",           "-d", nargs="+", required=True,
                        help="One or more drug names")
    parser.add_argument("--model",          "-m", default=None,
                        help="LLM model (overrides LLM_MODEL in .env)")
    parser.add_argument("--patent",         "-p", default=None,
                        help="Process only this patent number (optional)")
    parser.add_argument("--rerun_all",      action="store_true",
                        help="Re-run all steps even if output already exists")
    parser.add_argument("--skip_validator", action="store_true",
                        help="Skip the step 8a validator")
    parser.add_argument("--min_score",      type=float, default=0.0,
                        help="Validator min confidence score to keep evidence (default: 0.0)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
