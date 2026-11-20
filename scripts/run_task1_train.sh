#!/usr/bin/env bash
# Train one task-1 model.
#
#   bash scripts/run_task1_train.sh resnet18_gru
#   bash scripts/run_task1_train.sh vit_transformer
#
# Any extra arguments are passed to src/task1_caption/train.py, e.g.
#   bash scripts/run_task1_train.sh resnet18_gru --limit-train 32 --limit-val 16 --epochs 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

if [ "$#" -lt 1 ]; then
    echo "usage: bash scripts/run_task1_train.sh {resnet18_gru|vit_transformer} [train args...]" >&2
    exit 2
fi

MODEL="$1"
shift
case "$MODEL" in
    resnet18_gru|vit_transformer) ;;
    *)
        echo "unknown model '$MODEL'; expected resnet18_gru or vit_transformer" >&2
        exit 2
        ;;
esac

CONFIG="configs/task1_${MODEL}.yaml"
PYTHON="${PYTHON:-python}"

echo "[run_task1_train] model=$MODEL config=$CONFIG"
exec "$PYTHON" -m src.task1_caption.train --config "$CONFIG" "$@"
