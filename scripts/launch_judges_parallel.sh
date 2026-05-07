#!/usr/bin/env bash
# Launch 3 judges in parallel — one process per judge, each writes to its
# own per-judge JSONL. Merging happens later via simple cat for analysis.
# Resume is automatic (each driver inspects its own scores file).

set -euo pipefail
cd "$(dirname "$0")/.."

set -a; [ -f .env ] && source .env; set +a

mkdir -p data/results/judge_logs

RESPONSES_GLOB="data/results/pilot.*.jsonl"
SCORES=data/results

declare -a JOBS=(
  "anthropic  claude-haiku-4-5-20251001  pilot_judge_scores.haiku45.jsonl"
  "openai     gpt-4o-mini                pilot_judge_scores.gpt4o_mini.jsonl"
  "google     gemini-2.5-flash           pilot_judge_scores.gemini25_flash.jsonl"
)

for job in "${JOBS[@]}"; do
  read -r PROVIDER SNAP OUT <<<"$job"
  LOG="$SCORES/judge_logs/${PROVIDER}_${SNAP}.log"
  echo "Launching $PROVIDER ($SNAP) → $SCORES/$OUT"
  case "$PROVIDER" in
    anthropic) SNAP_FLAG="--snapshot-anthropic" ;;
    openai)    SNAP_FLAG="--snapshot-openai" ;;
    google)    SNAP_FLAG="--snapshot-google" ;;
  esac
  nohup python scripts/run_judges.py \
    --judges "$PROVIDER" \
    "$SNAP_FLAG" "$SNAP" \
    --responses-glob "$RESPONSES_GLOB" \
    --scores-out "$SCORES/$OUT" \
    > "$LOG" 2>&1 &
  echo "  pid=$!"
done

echo ""
echo "All 3 judges launched. Monitor with:"
echo "  wc -l $SCORES/pilot_judge_scores.*.jsonl"
echo "  tail -f $SCORES/judge_logs/*.log"
