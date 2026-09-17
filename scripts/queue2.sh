#!/usr/bin/env bash
# Runs after the first queue: the agent sector (tool selection).
set -u
cd /home/user/git/typedec
PY=".venv/bin/python"
export MEMGUARD_ALLOW_MB=22000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p results logs

until grep -q "queue finished" logs_queue.log 2>/dev/null; do sleep 120; done
echo "[$(date '+%H:%M:%S')] stage 6: the agent sector (tools and browser actions)"

# Where the model stood before this sector was trained.
$PY scripts/eval_general.py --checkpoint runs/v6-all/best-calibrated.pt \
  --test data/test_tools.jsonl --batch-size 8 --max-length 512 \
  --out results/v6-base_tools_before.json > logs/v6_tools_before.log 2>&1

$PY scripts/train.py --backbone answerdotai/ModernBERT-base --init-from runs/v6-all/best.pt \
  --train data/agent.jsonl --replay data/train_v4.jsonl --replay-share 0.4 \
  --dev data/dev_strat_v2.jsonl --out runs/v11-tools \
  --steps 6000 --decision-batch 6 --anchor-batch 12 --max-length 320 --lr 1.5e-5 \
  --eval-every 2000 --eval-rows 3000 > logs/v11.log 2>&1

$PY scripts/calibrate.py --checkpoint runs/v11-tools/best.pt --dev data/dev_strat_v2.jsonl \
  --rows 3000 > logs/v11_calib.log 2>&1
for set in tools browser general quiz multilingual; do
  [ -f "data/test_${set}.jsonl" ] || continue
  $PY scripts/eval_general.py --checkpoint runs/v11-tools/best-calibrated.pt \
    --test "data/test_${set}.jsonl" --batch-size 8 --max-length 512 \
    --out "results/v11-tools_${set}.json" > "logs/v11_${set}.log" 2>&1
done
$PY scripts/eval_router.py --checkpoint runs/v11-tools/best-calibrated.pt --rows 1500 \
  --out results/v11-tools_router.json > logs/v11_router.log 2>&1
echo "[$(date '+%H:%M:%S')] queue2 finished"
