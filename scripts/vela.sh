#!/usr/bin/env bash
# The 32k-context encoder, trained in two stages.
#
# Stage one uses only the sources this project started with, so the result is
# comparable to every earlier run and a broken setup shows up before ten hours
# of compute are spent. Stage two continues from that checkpoint with
# everything else: long documents, code, safety policies, tickets, languages.
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
PY=".venv/bin/python"
export MEMGUARD_ALLOW_MB=22000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
OJ=${OJ:-/root/run/openjev}
BACKBONE=llm-semantic-router/Vela-1.0-Encoder-307M
mkdir -p results logs
stamp () { date '+%m-%d %H:%M:%S'; }

measure () {
  local name=$1 ckpt=$2
  [ -f "$ckpt" ] || { echo "[$(stamp)] $name: no checkpoint"; return; }
  echo "[$(stamp)] measuring $name"
  $PY scripts/calibrate.py --checkpoint "$ckpt" --dev data/dev_strat_v2.jsonl --rows 2600 \
      --max-length 256 > "logs/${name}_calib.log" 2>&1
  local cal="${ckpt%.pt}-calibrated.pt"
  [ -f "$cal" ] || cal="$ckpt"
  for set in general quiz multilingual tools browser moderation more; do
    [ -f "data/test_${set}.jsonl" ] || continue
    $PY scripts/eval_general.py --checkpoint "$cal" --test "data/test_${set}.jsonl" \
        --batch-size 4 --max-length 512 --out "results/${name}_${set}.json" \
        > "logs/${name}_${set}.log" 2>&1
  done
  if [ -d "$OJ" ]; then
    $PY scripts/eval_openjev.py --checkpoint "$cal" --batch-size 4 --max-length 256 \
        --fixtures authored144=$OJ/benchmarks/data/authored144.jsonl \
        --out "results/${name}_fixtures.json" > "logs/${name}_fixtures.log" 2>&1
  fi
  $PY scripts/eval_router.py --checkpoint "$cal" --rows 1000 \
      --out "results/${name}_router.json" > "logs/${name}_router.log" 2>&1
  $PY scripts/report.py > /dev/null 2>&1
}

echo "[$(stamp)] === stage 1: the original sources only ==="
$PY scripts/train.py --backbone "$BACKBONE" --grad-checkpoint --adam8bit \
  --train data/train.jsonl --dev data/dev_strat_v2.jsonl \
  --out runs/vela-stage1 --steps 10000 --augment 0.7 \
  --decision-batch 4 --anchor-batch 8 --max-length 256 --lr 2e-5 \
  --eval-every 2500 --eval-rows 2600 > logs/vela-stage1.log 2>&1
measure vela-stage1 runs/vela-stage1/best.pt

echo "[$(stamp)] === stage 2: everything else, continued from stage 1 ==="
$PY scripts/train.py --backbone "$BACKBONE" --grad-checkpoint --adam8bit \
  --init-from runs/vela-stage1/best.pt \
  --train data/train_v6.jsonl --dev data/dev_strat_v2.jsonl \
  --long data/long_v2.jsonl --long-every 4 --long-batch 1 --long-max-length 8192 \
  --out runs/vela-stage2 --steps 24000 --augment 0.7 \
  --decision-batch 4 --anchor-batch 8 --max-length 256 --lr 1.5e-5 \
  --eval-every 3000 --eval-rows 2600 > logs/vela-stage2.log 2>&1
measure vela-stage2 runs/vela-stage2/best.pt

echo "[$(stamp)] vela finished"
