#!/bin/bash
# 全量实验启动（多 seed、双 GPU 并行、断点续跑）。
# 用法（服务器后台）: setsid bash scripts/run_grid.sh > results/grid.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p results/logs results/final_marks

run_list() {
  gpu=$1; list=$2
  while IFS= read -r cfg; do
    [ -z "$cfg" ] && continue
    name=$(basename "$cfg" .yaml)
    if [ -f "results/final_marks/${name}.done" ]; then
      echo "[gpu${gpu}] SKIP ${name}"
      continue
    fi
    echo "[gpu${gpu}] START ${name} $(date +%F_%T)"
    if CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train_fl.py --config "$cfg" \
        >> "results/logs/${name}.log" 2>&1; then
      touch "results/final_marks/${name}.done"
      echo "[gpu${gpu}] DONE ${name} $(date +%F_%T)"
    else
      echo "[gpu${gpu}] FAIL ${name} $(date +%F_%T)  (see results/logs/${name}.log)"
    fi
  done < "$list"
}

run_list 0 configs/generated/gpu0.list &
P0=$!
run_list 1 configs/generated/gpu1.list &
P1=$!
wait $P0 $P1
echo "GRID_DONE"
