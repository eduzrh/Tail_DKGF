#!/bin/bash
# DAGR: Batch run all encoders on W-I-S1 (plug-and-play benchmark)
# Usage: bash scripts/run_all.sh   (from the repo root)
set -e

DATA_DIR="data/W-I-S1"
NAME="W-I-S1"
DIM=128
MAX_K=16
EPOCHS=30
BS=256

echo "============================================"
echo "DAGR: Plug-and-Play Encoder Benchmark"
echo "Dataset: $NAME | Epochs: $EPOCHS | Dim: $DIM"
echo "============================================"

for SCORER in transe transh gnn exefuse; do
    OUT_DIR="outputs_${SCORER}"
    echo ""
    echo ">>> Running: --scorer $SCORER -> $OUT_DIR"
    python DAGR_main.py \
        --data_dir "$DATA_DIR" \
        --name "$NAME" \
        --output_dir "$OUT_DIR" \
        --dim $DIM --max_k $MAX_K --epochs $EPOCHS --batch_size $BS \
        --scorer "$SCORER"
    echo "<<< Done: $SCORER"
done

echo ""
echo "Results summary:"
for SCORER in transe transh gnn exefuse; do
    OUT_DIR="outputs_${SCORER}"
    if [ -f "$OUT_DIR/results_${NAME}.txt" ]; then
        echo "  [$SCORER]"
        head -2 "$OUT_DIR/results_${NAME}.txt"
    fi
done
