#!/bin/bash
# DAGeR: Quick smoke test (5 epochs, ~3 min on GPU)
# Run this first to verify everything works!
# Usage: bash scripts/run_quick_test.sh   (from the repo root)
set -e

python DAGeR_main.py \
    --data_dir data/W-I-S1 \
    --name W-I-S1 \
    --output_dir outputs_smoke_test \
    --dim 64 --max_k 8 --epochs 5 --batch_size 256 \
    --scorer transe

echo ""
echo "Smoke test passed! Ready for full training."
echo "Next: bash scripts/run_all.sh"
