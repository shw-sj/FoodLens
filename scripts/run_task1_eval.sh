#!/usr/bin/env bash
# Greedy + beam inference and evaluation for one task-1 checkpoint.
#
#   bash scripts/run_task1_eval.sh
#   bash scripts/run_task1_eval.sh checkpoints/task1/vit_transformer/best.pt test 5 1.0
#
# Args: [checkpoint] [split] [beam_size] [length_penalty]
# Env:  PYTHON, DEVICE (e.g. cuda:0), NUM_WORKERS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

CKPT="${1:-checkpoints/task1/vit_transformer/best.pt}"
SPLIT="${2:-test}"
BEAM_SIZE="${3:-5}"
LENGTH_PENALTY="${4:-1.0}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 2
fi

MODEL_NAME="$(basename "$(dirname "$CKPT")")"
OUT_DIR="outputs/task1/predictions"
METRIC_DIR="outputs/task1/metrics"
mkdir -p "$OUT_DIR" "$METRIC_DIR"

GREEDY_CSV="$OUT_DIR/${MODEL_NAME}_${SPLIT}_greedy.csv"
BEAM_CSV="$OUT_DIR/${MODEL_NAME}_${SPLIT}_beam${BEAM_SIZE}_lp${LENGTH_PENALTY}.csv"

DEVICE_ARGS=()
if [ -n "$DEVICE" ]; then
    DEVICE_ARGS=(--device "$DEVICE")
fi

echo "[run_task1_eval] greedy: $GREEDY_CSV"
"$PYTHON" -m src.task1_caption.infer \
    --checkpoint "$CKPT" --split "$SPLIT" --method greedy \
    --num-workers "$NUM_WORKERS" ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
    --output "$GREEDY_CSV"

echo "[run_task1_eval] beam=$BEAM_SIZE lp=$LENGTH_PENALTY: $BEAM_CSV"
"$PYTHON" -m src.task1_caption.infer \
    --checkpoint "$CKPT" --split "$SPLIT" --method beam \
    --beam-size "$BEAM_SIZE" --length-penalty "$LENGTH_PENALTY" \
    --num-workers "$NUM_WORKERS" ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
    --output "$BEAM_CSV"

echo "[run_task1_eval] evaluating"
"$PYTHON" -m src.task1_caption.evaluate \
    --pred "greedy=$GREEDY_CSV" \
    --pred "beam${BEAM_SIZE}_lp${LENGTH_PENALTY}=$BEAM_CSV" \
    --split "$SPLIT" \
    --out "$METRIC_DIR/${MODEL_NAME}_${SPLIT}_greedy_vs_beam.json" \
    --markdown "$METRIC_DIR/${MODEL_NAME}_${SPLIT}_greedy_vs_beam.md"

echo "[run_task1_eval] done"
