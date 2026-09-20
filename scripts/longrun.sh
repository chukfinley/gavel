#!/usr/bin/env bash
# The full-day run: the 32k encoder, every source, everything learned so far.
#
# The backbone stays at 32768 tokens on purpose. It is the one property no
# other open model in this class has — Von is 512, Laya 1024, the DeBERTa
# reproduction 512 — and users in the field ask for 30k-token decisions.
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
PY=".venv/bin/python"
export MEMGUARD_ALLOW_MB=22000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
BACKBONE=${BACKBONE:-llm-semantic-router/Vela-1.0-Encoder-307M}
STEPS=${STEPS:-60000}
# The values the 32k encoder was trained at before. A larger card can raise
# them from the environment instead of editing this job.
DB=${DECISION_BATCH:-4}
AB=${ANCHOR_BATCH:-8}
LB=${LONG_BATCH:-1}
mkdir -p results logs
stamp () { date '+%m-%d %H:%M:%S'; }

echo "[$(stamp)] assembling"
$PY scripts/build_scales.py >> logs/build.log 2>&1
$PY scripts/assemble.py | tail -1

echo "[$(stamp)] training $STEPS steps on $BACKBONE"
$PY scripts/train.py --backbone "$BACKBONE" --grad-checkpoint --adam8bit \
  ${INIT:+--init-from "$INIT"} \
  --train data/train_v6.jsonl --dev data/dev_strat_v2.jsonl \
  --long data/long_v2.jsonl --long-every 4 --long-batch "$LB" --long-max-length 8192 \
  --out runs/longrun --steps "$STEPS" --augment 0.7 --source-alpha 0.4 \
  --decision-batch "$DB" --anchor-batch "$AB" --max-length 512 --lr 1.5e-5 \
  --eval-every 5000 --eval-rows 2600 > logs/longrun.log 2>&1

echo "[$(stamp)] calibrating"
$PY scripts/calibrate.py --checkpoint runs/longrun/best.pt --dev data/dev_strat_v2.jsonl \
  --rows 2600 --max-length 512 > logs/longrun_calib.log 2>&1

echo "[$(stamp)] measuring"
$PY scripts/eval_cbench.py --checkpoint runs/longrun/best-calibrated.pt \
  --max-length 512 --out results/longrun_cbench.json 2>&1 | tail -12
for s in general quiz multilingual tools browser moderation more semrouter kotoba; do
  [ -f "data/test_${s}.jsonl" ] && $PY scripts/eval_general.py \
    --checkpoint runs/longrun/best-calibrated.pt --test "data/test_${s}.jsonl" \
    --batch-size 4 --max-length 1024 --out "results/longrun_${s}.json" \
    > "logs/longrun_${s}.log" 2>&1
done
$PY scripts/eval_router.py --checkpoint runs/longrun/best-calibrated.pt --rows 1500 \
  --out results/longrun_router.json > logs/longrun_router.log 2>&1

echo "[$(stamp)] publishing"
$PY scripts/export_hf.py --checkpoint runs/longrun/best-calibrated.pt --out export/gavel-vela-32k
$PY - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path="export/gavel-vela-32k", repo_id="chukfinley/gavel-vela-32k",
                  repo_type="model", commit_message="long run: scales, criteria wording, full mix")
for folder in ("results", "logs"):
    api.upload_folder(folder_path=folder, path_in_repo=folder,
                      repo_id="chukfinley/gavel-runs", repo_type="dataset",
                      commit_message="long run")
print("published")
PYEOF
echo "[$(stamp)] LONGRUN_DONE"
