#!/usr/bin/env bash
# Keeps the card busy without supervision. Every job logs on its own and a
# failure of one job does not stop the rest.
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
PY=".venv/bin/python"
export MEMGUARD_ALLOW_MB=22000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
OJ=/home/user/git/openjev
mkdir -p results logs
stamp () { date '+%m-%d %H:%M:%S'; }

# A rented card usually has more memory than the one this was written on.
# These can be raised from the environment instead of editing the jobs.
DB=${DECISION_BATCH:-6}          # decisions per step
AB=${ANCHOR_BATCH:-12}           # inference pairs per step
LB=${LONG_BATCH:-1}              # long documents per step
EB=${EVAL_BATCH:-4}
LONG_CTX=${LONG_CTX:-2048}       # how far the long curriculum reaches

measure () {                        # measure <name> <checkpoint> [typesafe-ctx]
  local name=$1 ckpt=$2 long=${3:-2048}
  [ -f "$ckpt" ] || { echo "[$(stamp)] $name: no checkpoint, skipped"; return; }
  echo "[$(stamp)] measuring $name"
  $PY scripts/calibrate.py --checkpoint "$ckpt" --dev data/dev_strat_v2.jsonl --rows 3000 \
      > "logs/${name}_calib.log" 2>&1
  local cal="${ckpt%.pt}-calibrated.pt"
  [ -f "$cal" ] || cal="$ckpt"
  $PY scripts/eval_openjev.py --checkpoint "$cal" --batch-size "$EB" --max-length 256 \
      --fixtures authored144=$OJ/benchmarks/data/authored144.jsonl \
                 wanli256=$OJ/_fixtures/wanli256.jsonl \
      --out "results/${name}_fixtures.json" > "logs/${name}_fixtures.log" 2>&1
  $PY scripts/eval_typesafe.py --checkpoint "$cal" --max-length "$long" \
      --out "results/${name}_typesafe.json" > "logs/${name}_typesafe.log" 2>&1
  for set in general quiz multilingual tools browser moderation; do
    [ -f "data/test_${set}.jsonl" ] || continue
    $PY scripts/eval_general.py --checkpoint "$cal" --test "data/test_${set}.jsonl" \
        --batch-size "$EB" --max-length 512 --out "results/${name}_${set}.json" \
        > "logs/${name}_${set}.log" 2>&1
  done
  $PY scripts/eval_router.py --checkpoint "$cal" --rows 1500 \
      --out "results/${name}_router.json" > "logs/${name}_router.log" 2>&1
  $PY scripts/report.py > /dev/null 2>&1
}

job () {                            # job <name> <args...>
  local name=$1; shift
  [ -f "runs/${name}/done" ] && { echo "[$(stamp)] $name already done"; return; }
  echo "[$(stamp)] === $name ==="
  $PY scripts/train.py --out "runs/${name}" "$@" > "logs/${name}.log" 2>&1
  touch "runs/${name}/done" 2>/dev/null
  measure "$name" "runs/${name}/best.pt"
}

COMMON="--train data/train_v6.jsonl --dev data/dev_strat_v2.jsonl --long data/long_v2.jsonl"

# 1. A decoder backbone at the size the other project published.
job v13-qwen08 --train data/train_v6.jsonl --dev data/dev_strat_v2.jsonl \
  --backbone Qwen/Qwen3.5-0.8B --grad-checkpoint --adam8bit \
  --steps 6000 --decision-batch "$((DB / 3))" --anchor-batch "$((AB / 3))" --max-length 192 --lr 1e-5 \
  --eval-every 2000 --eval-rows 1500

# 2. The larger encoder.
job v14-large $COMMON --backbone answerdotai/ModernBERT-large --grad-checkpoint --adam8bit \
  --long-every 8 --long-batch "$LB" --long-max-length "$((${LONG_CTX:-2048} / 2))" --steps 16000 \
  --decision-batch "$((DB / 2))" --anchor-batch "$((AB / 2))" --max-length 192 --lr 2e-5 \
  --eval-every 4000 --eval-rows 3000

# 3. The main model on everything, longer than before.
job v12-base $COMMON --backbone answerdotai/ModernBERT-base --augment 0.7 \
  --long-every 5 --long-batch "$LB" --long-max-length "${LONG_CTX:-2048}" --steps 40000 \
  --decision-batch "$DB" --anchor-batch "$AB" --max-length 224 --lr 3e-5 \
  --eval-every 4000 --eval-rows 3000

# 4. Specialised branches from the main model.
job v17-route --init-from runs/v12-base/best.pt --backbone answerdotai/ModernBERT-base \
  --train data/train_route.jsonl --dev data/dev_strat_v2.jsonl \
  --steps 8000 --decision-batch "$DB" --anchor-batch "$AB" --max-length 224 --lr 1.5e-5 \
  --eval-every 4000 --eval-rows 3000
job v18-doc --init-from runs/v12-base/best.pt --backbone answerdotai/ModernBERT-base \
  --train data/train_doc.jsonl --dev data/dev_strat_v2.jsonl --long data/long_v2.jsonl \
  --long-every 3 --long-batch "$LB" --long-max-length "${LONG_CTX:-2048}" --steps 8000 \
  --decision-batch "$((DB * 2 / 3))" --anchor-batch "$((AB * 2 / 3))" --max-length 224 --lr 1.5e-5 \
  --eval-every 4000 --eval-rows 3000
job v19-agent --init-from runs/v12-base/best.pt --backbone answerdotai/ModernBERT-base \
  --train data/agent.jsonl --replay data/train_v6.jsonl --replay-share 0.4 \
  --dev data/dev_strat_v2.jsonl --steps 8000 --decision-batch "$DB" --anchor-batch "$AB" \
  --max-length 320 --lr 1.5e-5 --eval-every 4000 --eval-rows 3000

# 5. Settings search, the first thing to drop, short runs, so that the long run uses the better values.
for alpha in 0.3 0.7; do
  job "v15-alpha${alpha}" $COMMON --backbone answerdotai/ModernBERT-base \
    --long-every 5 --long-batch "$LB" --long-max-length "${LONG_CTX:-2048}" --steps 9000 \
    --decision-batch "$DB" --anchor-batch "$AB" --max-length 224 --lr 3e-5 \
    --source-alpha "$alpha" --eval-every 3000 --eval-rows 3000
done
for brier in 0.0 1.0; do
  job "v16-brier${brier}" $COMMON --backbone answerdotai/ModernBERT-base \
    --long-every 5 --long-batch "$LB" --long-max-length "${LONG_CTX:-2048}" --steps 9000 \
    --decision-batch "$DB" --anchor-batch "$AB" --max-length 224 --lr 3e-5 \
    --brier-weight "$brier" --eval-every 3000 --eval-rows 3000
done

# The same model without augmentation, to measure what it is worth.
job v20-noaug $COMMON --backbone answerdotai/ModernBERT-base --augment 0.0 \
  --long-every 5 --long-batch "$LB" --long-max-length "${LONG_CTX:-2048}" --steps 9000 \
  --decision-batch "$DB" --anchor-batch "$AB" --max-length 224 --lr 3e-5 \
  --eval-every 3000 --eval-rows 3000
job v21-aug $COMMON --backbone answerdotai/ModernBERT-base --augment 0.7 \
  --long-every 5 --long-batch "$LB" --long-max-length "${LONG_CTX:-2048}" --steps 9000 \
  --decision-batch "$DB" --anchor-batch "$AB" --max-length 224 --lr 3e-5 \
  --eval-every 3000 --eval-rows 3000

# A genuine 32768-token encoder, found by searching for anything bidirectional
# above ModernBERT's 8192. It is ModernBERT underneath, so only the name changes.
# Multilingual pretraining comes with it, which the language strata should show.
job v22-vela32k --train data/train_v6.jsonl --dev data/dev_strat_v2.jsonl \
  --long data/long_v2.jsonl --backbone llm-semantic-router/Vela-1.0-Encoder-307M \
  --grad-checkpoint --adam8bit --augment 0.7 \
  --long-every 4 --long-batch 1 --long-max-length 8192 --steps 12000 \
  --decision-batch "$((DB / 4))" --anchor-batch "$((AB / 4))" --max-length 256 --lr 2e-5 \
  --eval-every 3000 --eval-rows 3000

echo "[$(stamp)] marathon finished"
