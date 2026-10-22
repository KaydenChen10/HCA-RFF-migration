#!/bin/bash
# LoRa 对照实验：baseline 与 HCA 并行分片训练（每片 ~245s，自动断点续训）
cd "$(dirname "$0")/LoRa" || exit 1

COMMON="--train_rx n210_1 --train_day 1 --test_rx rtl_6 --test_day 1 2 3 4 \
  --candidates 128 256 512 1024 --per_class 200 --epochs 100 \
  --patience_early_stop 15 --batch_size 128 --max_seconds 220"

PIDS=()
if grep -q "Average:" logs/chunk_base.log 2>/dev/null; then
  echo "baseline 已完成，跳过训练"
else
  OMP_NUM_THREADS=12 python run_CIS.py $COMMON --backbone rescISnet \
    >> logs/chunk_base.log 2>&1 &
  PIDS+=($!)
fi
if grep -q "Average:" logs/chunk_hca.log 2>/dev/null; then
  echo "HCA 已完成，跳过训练"
else
  OMP_NUM_THREADS=12 python run_CIS.py $COMMON --backbone rescISnet_hca --hca_bands 32 \
    >> logs/chunk_hca.log 2>&1 &
  PIDS+=($!)
fi
if ((${#PIDS[@]})); then
  wait "${PIDS[@]}"
fi

echo "=== baseline 进度 ==="; grep -cE "^Train Epoch" logs/chunk_base.log 2>/dev/null
tail -2 logs/chunk_base.log
echo "=== HCA 进度 ==="; grep -cE "^Train Epoch" logs/chunk_hca.log 2>/dev/null
tail -2 logs/chunk_hca.log
echo "=== 完成标记 ==="
grep -l "Average:" logs/chunk_base.log 2>/dev/null && echo BASE-DONE
grep -l "Average:" logs/chunk_hca.log 2>/dev/null && echo HCA-DONE
