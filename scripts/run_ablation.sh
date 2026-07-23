#!/bin/bash
# DAGeR: Ablation study on one dataset
# Usage: bash scripts/run_ablation.sh [DATA_DIR] [NAME]
set -e

DATA_DIR="${1:-data/W-I-S1}"
NAME="${2:-W-I-S1}"
DIM=128
MAX_K=16
EPOCHS=30
BS=256

# full: complete model (no --ablation flag)
# no_bridge: remove the multi-constraint virtual link refining (anchor bridges)
# emb_only: embedding-only track (no structural track in the hybrid injector)
for ABL in full no_bridge emb_only; do
    OUT_DIR="outputs_abl_${ABL}"
    echo ">>> Ablation: $ABL -> $OUT_DIR"
    if [ "$ABL" = "full" ]; then
        python DAGeR_main.py \
            --data_dir "$DATA_DIR" --name "$NAME" --output_dir "$OUT_DIR" \
            --dim $DIM --max_k $MAX_K --epochs $EPOCHS --batch_size $BS \
            --scorer transe
    else
        python DAGeR_main.py \
            --data_dir "$DATA_DIR" --name "$NAME" --output_dir "$OUT_DIR" \
            --dim $DIM --max_k $MAX_K --epochs $EPOCHS --batch_size $BS \
            --scorer transe --ablation "$ABL"
    fi
done
