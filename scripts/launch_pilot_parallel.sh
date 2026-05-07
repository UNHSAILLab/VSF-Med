#!/usr/bin/env bash
# Launch the 4 cached local model wrappers in parallel, one per GPU.
# Each process is pinned to a single GPU via CUDA_VISIBLE_DEVICES and writes
# to a model-specific JSONL so processes don't contend on the same file.
# Re-running is safe — the BatchRunner resume logic skips completed (case, condition).

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/results/pilot_logs

EVAL=data/processed/eval_cases.pilot.jsonl
RESULTS=data/results

# (gpu, model_id, responses_filename)
declare -a JOBS=(
  "1 chexone_4b_reasoning   pilot.chexone_4b_reasoning.jsonl"
  "2 chexone_4b_instruct    pilot.chexone_4b_instruct.jsonl"
  "3 medgemma_4b_it         pilot.medgemma_4b_it.jsonl"
  "4 medgemma_27b_it        pilot.medgemma_27b_it.jsonl"
)

for job in "${JOBS[@]}"; do
  read -r GPU MODEL OUT <<<"$job"
  LOG="$RESULTS/pilot_logs/${MODEL}.log"
  echo "Launching $MODEL on GPU $GPU → $RESULTS/$OUT (log: $LOG)"
  CUDA_VISIBLE_DEVICES="$GPU" \
    nohup python scripts/run_pilot.py \
      --eval-cases "$EVAL" \
      --responses "$RESULTS/$OUT" \
      --models "$MODEL" \
      > "$LOG" 2>&1 &
  echo "  pid=$!"
done

echo ""
echo "All 4 jobs launched in background. Monitor with:"
echo "  tail -f $RESULTS/pilot_logs/*.log"
echo "  wc -l $RESULTS/pilot.*.jsonl"
echo ""
echo "Each model will produce up to 1600 responses (200 cases × 8 conditions)."
echo "Resume is automatic on re-run."
