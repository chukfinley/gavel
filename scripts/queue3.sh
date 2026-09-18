#!/usr/bin/env bash
# The two stages that failed in the first queue, with settings that fit.
set -u
cd /home/user/git/typedec
PY=".venv/bin/python"
export MEMGUARD_ALLOW_MB=22000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
OJ=/home/user/git/openjev
mkdir -p results logs
stamp () { date '+%H:%M:%S'; }

measure () {
  local name=$1 ckpt=$2 long=${3:-2048}
  echo "[$(stamp)] measuring $name"
  $PY scripts/calibrate.py --checkpoint "$ckpt" --dev data/dev_strat_v2.jsonl --rows 3000 \
      > "logs/${name}_calib.log" 2>&1
  local cal="${ckpt%.pt}-calibrated.pt"
  $PY scripts/eval_openjev.py --checkpoint "$cal" --batch-size 4 --max-length 256 \
      --fixtures authored144=$OJ/benchmarks/data/authored144.jsonl \
                 wanli256=$OJ/_fixtures/wanli256.jsonl \
      --out "results/${name}_fixtures.json" > "logs/${name}_fixtures.log" 2>&1
  $PY scripts/eval_typesafe.py --checkpoint "$cal" --max-length "$long" \
      --out "results/${name}_typesafe.json" > "logs/${name}_typesafe.log" 2>&1
  for set in general quiz multilingual tools browser; do
    [ -f "data/test_${set}.jsonl" ] || continue
    $PY scripts/eval_general.py --checkpoint "$cal" --test "data/test_${set}.jsonl" \
        --batch-size 4 --max-length 512 --out "results/${name}_${set}.json" \
        > "logs/${name}_${set}.log" 2>&1
  done
}

echo "[$(stamp)] stage 2b: Qwen3.5-0.8B as backbone, the size AlexWortega published"
$PY scripts/train.py --backbone Qwen/Qwen3.5-0.8B --grad-checkpoint --adam8bit \
  --train data/train_v4.jsonl --dev data/dev_strat_v2.jsonl \
  --out runs/v7b-qwen08 --steps 6000 --decision-batch 2 --anchor-batch 4 \
  --max-length 192 --lr 1e-5 --eval-every 2000 --eval-rows 1500 > logs/v7b.log 2>&1
measure v7b-qwen08 runs/v7b-qwen08/best.pt 1024

echo "[$(stamp)] stage 3b: ModernBERT-large with a smaller footprint"
$PY scripts/train.py --backbone answerdotai/ModernBERT-large --grad-checkpoint --adam8bit \
  --train data/train_v4.jsonl --dev data/dev_strat_v2.jsonl --long data/long_v2.jsonl \
  --long-every 8 --long-batch 1 --long-max-length 512 \
  --out runs/v8b-large --steps 14000 --decision-batch 3 --anchor-batch 6 \
  --max-length 192 --lr 2e-5 --eval-every 3000 --eval-rows 3000 > logs/v8b.log 2>&1
measure v8b-large runs/v8b-large/best.pt

echo "[$(stamp)] queue3 finished"
