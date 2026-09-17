#!/usr/bin/env bash
# Unattended queue: train, calibrate and measure several variants in order.
# Each stage writes its own log and results file, thus a failure of one stage
# does not lose the others.
set -u
cd /home/user/git/typedec
PY=".venv/bin/python"
export MEMGUARD_ALLOW_MB=22000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
OJ=/home/user/git/openjev
mkdir -p results logs

stamp () { date '+%H:%M:%S'; }

measure () {                       # measure <name> <checkpoint> <long-ctx>
  local name=$1 ckpt=$2 long=${3:-2048}
  echo "[$(stamp)] measuring $name" >&2
  $PY scripts/calibrate.py --checkpoint "$ckpt" --dev data/dev_strat_v2.jsonl \
      --rows 3000 > "logs/${name}_calib.log" 2>&1
  local cal="${ckpt%.pt}-calibrated.pt"
  $PY scripts/eval_openjev.py --checkpoint "$cal" --batch-size 8 --max-length 256 \
      --fixtures authored144=$OJ/benchmarks/data/authored144.jsonl \
                 wanli256=$OJ/_fixtures/wanli256.jsonl \
      --out "results/${name}_fixtures.json" > "logs/${name}_fixtures.log" 2>&1
  $PY scripts/eval_typesafe.py --checkpoint "$cal" --max-length "$long" \
      --out "results/${name}_typesafe.json" > "logs/${name}_typesafe.log" 2>&1
  for set in general quiz multilingual; do
    local file="data/test_${set}.jsonl"
    [ -f "$file" ] || continue
    $PY scripts/eval_general.py --checkpoint "$cal" --test "$file" --batch-size 8 \
        --max-length 512 --out "results/${name}_${set}.json" > "logs/${name}_${set}.log" 2>&1
  done
  $PY scripts/eval_router.py --checkpoint "$cal" --rows 1500 \
      --out "results/${name}_router.json" > "logs/${name}_router.log" 2>&1
}

echo "[$(stamp)] stage 1: base on the full mix"
$PY scripts/train.py --backbone answerdotai/ModernBERT-base \
  --train data/train_v4.jsonl --dev data/dev_strat_v2.jsonl --long data/long_v2.jsonl \
  --long-every 5 --long-batch 1 --long-max-length 896 --out runs/v6-all \
  --steps 30000 --decision-batch 6 --anchor-batch 12 --max-length 224 --lr 3e-5 \
  --eval-every 3000 --eval-rows 3000 > logs/v6.log 2>&1
measure v6-base runs/v6-all/best.pt

echo "[$(stamp)] stage 2: a decoder backbone, for comparison only"
$PY scripts/train.py --backbone Qwen/Qwen3.5-0.6B --grad-checkpoint \
  --train data/train_v4.jsonl --dev data/dev_strat_v2.jsonl --long data/long_v2.jsonl \
  --long-every 8 --long-batch 1 --long-max-length 512 --out runs/v7-qwen06 \
  --steps 8000 --decision-batch 2 --anchor-batch 4 --max-length 224 --lr 1e-5 \
  --eval-every 2000 --eval-rows 1500 > logs/v7.log 2>&1
measure v7-qwen06 runs/v7-qwen06/best.pt 1024

echo "[$(stamp)] stage 3: the larger encoder"
$PY scripts/train.py --backbone answerdotai/ModernBERT-large \
  --train data/train_v4.jsonl --dev data/dev_strat_v2.jsonl --long data/long_v2.jsonl \
  --long-every 6 --long-batch 1 --long-max-length 640 --out runs/v8-large \
  --steps 14000 --decision-batch 3 --anchor-batch 6 --max-length 224 --lr 2e-5 \
  --eval-every 2000 --eval-rows 3000 > logs/v8.log 2>&1
measure v8-large runs/v8-large/best.pt

echo "[$(stamp)] stage 4: specialised branches from the base model"
$PY - <<'PYEOF' > logs/split.log 2>&1
import sys, random
sys.path.insert(0, "src")
from typedec.schema import read_jsonl, write_jsonl
rows = list(read_jsonl("data/train_v4.jsonl"))
route = {"router-difficulty", "router-tier", "banking77", "clinc", "ag-news", "dbpedia",
         "tweet-offensive", "tweet-hate", "sms-spam", "synth-routing", "synth-action",
         "mnli", "wanli-train", "anli-r3"}
doc = {"boolq", "quality", "race", "synth-packet", "synth-incident", "synth-evidence",
       "sciq-passage", "mnli", "wanli-train", "anli-r3"}
write_jsonl("data/train_route.jsonl", [r for r in rows if r.source in route])
write_jsonl("data/train_doc.jsonl", [r for r in rows if r.source in doc
                                     or r.source.startswith("abstain")])
print("route", sum(1 for r in rows if r.source in route))
PYEOF

$PY scripts/train.py --backbone answerdotai/ModernBERT-base --init-from runs/v6-all/best.pt \
  --train data/train_route.jsonl --dev data/dev_strat_v2.jsonl --out runs/v9-route \
  --steps 6000 --decision-batch 6 --anchor-batch 12 --max-length 224 --lr 1.5e-5 \
  --eval-every 2000 --eval-rows 3000 > logs/v9.log 2>&1
measure v9-route runs/v9-route/best.pt

$PY scripts/train.py --backbone answerdotai/ModernBERT-base --init-from runs/v6-all/best.pt \
  --train data/train_doc.jsonl --dev data/dev_strat_v2.jsonl --long data/long_v2.jsonl \
  --long-every 3 --long-batch 1 --long-max-length 896 --out runs/v10-doc \
  --steps 6000 --decision-batch 4 --anchor-batch 8 --max-length 224 --lr 1.5e-5 \
  --eval-every 2000 --eval-rows 3000 > logs/v10.log 2>&1
measure v10-doc runs/v10-doc/best.pt

echo "[$(stamp)] queue finished"
