#!/usr/bin/env bash
# Launch the 3 frontier API targets (Claude Haiku 4.5, GPT-5.4-mini,
# Gemini 3 Flash) on the 200-case pilot. No GPU needed — pure API calls.
# Resumable via the BatchRunner's existing (case_id, condition_id) skip logic.

set -euo pipefail
cd "$(dirname "$0")/.."

set -a; [ -f .env ] && source .env; set +a

mkdir -p data/results/pilot_logs

EVAL=data/processed/eval_cases.pilot.jsonl
RESULTS=data/results

declare -a JOBS=(
  "claude_haiku_4_5  pilot.claude_haiku_4_5.jsonl"
  "gpt_5_4_mini      pilot.gpt_5_4_mini.jsonl"
  "gemini_3_flash    pilot.gemini_3_flash.jsonl"
)

for job in "${JOBS[@]}"; do
  read -r MODEL OUT <<<"$job"
  LOG="$RESULTS/pilot_logs/${MODEL}.log"
  echo "Launching $MODEL → $RESULTS/$OUT"
  nohup python scripts/run_pilot.py \
    --eval-cases "$EVAL" \
    --responses "$RESULTS/$OUT" \
    --models "$MODEL" \
    > "$LOG" 2>&1 &
  echo "  pid=$!"
done

echo ""
echo "All 3 frontier targets launched. Monitor with:"
echo "  wc -l $RESULTS/pilot.{claude_haiku_4_5,gpt_5_4_mini,gemini_3_flash}.jsonl"
echo "  tail -f $RESULTS/pilot_logs/{claude_haiku_4_5,gpt_5_4_mini,gemini_3_flash}.log"
