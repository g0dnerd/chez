#!/usr/bin/env bash
# FT weight-decay sweep for the FT-512 (v10) net.
# v10 with ft_weight_decay=0 saturated: only 9.3% of FT outputs in CReLU [0,127],
# -130 Elo vs v9-256. Sweep FT regularization and gate on occupancy before SPRT.
set -u

DATA=data/selfplay_10k_40M.bin
SUMMARY=sweep_summary.log
: > "$SUMMARY"

for wd in 0.01 0.05 0.1; do
    out="data/net_v10_ft512_wd${wd}.nnue"
    log="train_v10_wd${wd}.log"
    echo "[$(date +%H:%M:%S)] training ft_weight_decay=${wd} -> ${out}"
    ./zig-out/bin/train_nnue \
        --data "$DATA" \
        --export "$out" \
        --epochs 50 --lr 0.005 --lambda 0.90 --sigmoid_divisor 1101 \
        --ft_weight_decay "$wd" \
        --checkpoint_interval 10 \
        > "$log" 2>&1

    {
        echo "================ ft_weight_decay=${wd} ================"
        grep -aoE "best val_loss=[0-9.]+" "$log" | tail -1
        ./zig-out/bin/nnue-inspect --net "$out" 2>&1 \
            | grep -E "ft_biases|ft_weights|occupancy|range=|Evals|startpos|^  r|^  rnbq|^  4k|^  8/"
        echo
    } >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] done ft_weight_decay=${wd}"
done

echo "[$(date +%H:%M:%S)] sweep complete. summary in ${SUMMARY}"
