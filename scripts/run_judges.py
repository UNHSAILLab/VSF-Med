"""Phase 6 judge scoring driver.

Usage:
    # Score the full pilot with 3 judges (requires API keys)
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...
    python scripts/run_judges.py

    # Smaller smoke run (5 responses)
    python scripts/run_judges.py --response-limit 5

    # Single-judge run for cost / sanity check
    python scripts/run_judges.py --judges anthropic --response-limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judges import compute_reliability, score_responses  # noqa: E402


PROVIDER_DEFAULTS = {
    "anthropic": {"snapshot": "claude-opus-4-7"},
    "openai":    {"snapshot": "gpt-5.2"},
    "google":    {"snapshot": "gemini-3.0-pro"},
}


def build_judges(providers, snapshots: dict):
    judges = []
    for prov in providers:
        snap = snapshots.get(prov) or PROVIDER_DEFAULTS[prov]["snapshot"]
        if prov == "anthropic":
            from judges.anthropic_judge import AnthropicJudge
            judges.append(AnthropicJudge(snapshot=snap))
        elif prov == "openai":
            from judges.openai_judge import OpenAIJudge
            judges.append(OpenAIJudge(snapshot=snap))
        elif prov == "google":
            from judges.gemini_judge import GeminiJudge
            judges.append(GeminiJudge(snapshot=snap))
        else:
            raise ValueError(f"Unknown provider: {prov}")
    return judges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses-glob",
                    default=str(ROOT / "data/results/pilot.*.jsonl"),
                    help="Glob for response files")
    ap.add_argument("--base-cases", default=str(ROOT / "data/processed/base_cases.csv"))
    ap.add_argument("--eval-cases", default=str(ROOT / "data/processed/eval_cases.jsonl"))
    ap.add_argument("--scores-out",
                    default=str(ROOT / "data/results/pilot_judge_scores.jsonl"))
    ap.add_argument("--judges", nargs="+", default=["anthropic", "openai", "google"],
                    choices=["anthropic", "openai", "google"])
    ap.add_argument("--snapshot-anthropic", default=PROVIDER_DEFAULTS["anthropic"]["snapshot"])
    ap.add_argument("--snapshot-openai",    default=PROVIDER_DEFAULTS["openai"]["snapshot"])
    ap.add_argument("--snapshot-google",    default=PROVIDER_DEFAULTS["google"]["snapshot"])
    ap.add_argument("--response-limit", type=int, default=None,
                    help="Cap number of responses scored (across all input files).")
    ap.add_argument("--reliability-only", action="store_true",
                    help="Skip API calls, just compute reliability on existing scores file.")
    args = ap.parse_args()

    snapshots = {
        "anthropic": args.snapshot_anthropic,
        "openai": args.snapshot_openai,
        "google": args.snapshot_google,
    }
    scores_path = Path(args.scores_out)

    if not args.reliability_only:
        # Auto-load repo-local .env if no SDK keys are in env yet
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v

        # Verify required keys are present before any spend
        missing = []
        if "anthropic" in args.judges and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_KEY")):
            missing.append("ANTHROPIC_API_KEY")
        if "openai" in args.judges and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY")):
            missing.append("OPENAI_API_KEY (or OPEN_AI_KEY)")
        if "google" in args.judges and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
        if missing:
            print(f"ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
            return 2

        responses_paths = sorted(Path().glob(args.responses_glob))
        if not responses_paths:
            print(f"ERROR: no files match {args.responses_glob}", file=sys.stderr)
            return 2
        print(f"Response files: {[p.name for p in responses_paths]}", file=sys.stderr)

        judges = build_judges(args.judges, snapshots)
        print(f"Judges: {[j.judge_model_id for j in judges]}", file=sys.stderr)

        summary = score_responses(
            responses_paths=responses_paths,
            base_cases_csv=Path(args.base_cases),
            eval_cases_jsonl=Path(args.eval_cases),
            scores_path=scores_path,
            judges=judges,
            response_limit=args.response_limit,
        )
        print(json.dumps(summary, indent=2))

    # Compute reliability if scores file exists
    if scores_path.exists():
        report = compute_reliability(scores_path)
        out = scores_path.with_suffix(".reliability.json")
        out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReliability report → {out}", file=sys.stderr)
        print(f"  n_responses: {report.n_responses}", file=sys.stderr)
        print(f"  judges: {report.judges}", file=sys.stderr)
        print(f"  Krippendorff α (pooled): {report.krippendorff_alpha_pooled:.3f}", file=sys.stderr)
        print(f"  Per-dim α:", file=sys.stderr)
        for dim, a in report.krippendorff_alpha_per_dim.items():
            print(f"    {dim:<35} {a:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
