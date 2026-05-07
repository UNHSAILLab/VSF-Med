"""Phase 4 pilot driver. Runs one or more model_ids over an eval_cases.jsonl.

Usage:
    # 10-case sanity run on medgemma_4b_it
    python scripts/run_pilot.py --models medgemma_4b_it --case-limit 10

    # Full pilot run on the 4 locally-cached models
    python scripts/run_pilot.py --models chexone_4b_instruct medgemma_4b_it medgemma_27b_it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runner import BatchRunner, run_model_on_cases  # noqa: E402
from models.wrappers import build_wrapper, list_registered  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-cases",
                    default=str(ROOT / "data/processed/eval_cases.pilot.jsonl"))
    ap.add_argument("--responses",
                    default=str(ROOT / "data/results/pilot_responses.jsonl"))
    ap.add_argument("--models", nargs="+", required=True,
                    help=f"Registered: {list_registered()}")
    ap.add_argument("--case-limit", type=int, default=None,
                    help="Cap number of base cases (each yields up to 8 condition rows).")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of condition_ids to run (default: all 8).")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    summaries = []
    for model_id in args.models:
        wrapper = build_wrapper(model_id)
        runner = BatchRunner(
            eval_cases_path=Path(args.eval_cases),
            responses_path=Path(args.responses),
            case_limit=args.case_limit,
            conditions=args.conditions,
            resume=not args.no_resume,
        )
        summary = runner.run(wrapper)
        summaries.append(summary)
        print(json.dumps(summary), file=sys.stderr, flush=True)

    print(json.dumps({"runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
