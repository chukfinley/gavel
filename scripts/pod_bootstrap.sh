#!/usr/bin/env bash
# Runs inside a rented pod. Builds everything from public sources, trains, and
# publishes results back to Hugging Face, so no data crosses the home uplink.
#
# Required environment: HF_TOKEN (write). Optional: GAVEL_REPO, RESULTS_REPO,
# MARATHON_ARGS.
set -u
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_PROGRESS_BARS=1
REPO=${GAVEL_REPO:-https://github.com/chukfinley/gavel.git}
RESULTS=${RESULTS_REPO:-chukfinley/gavel-runs}
WORK=/workspace/gavel

log () { echo "[$(date -u '+%F %T')] $*" | tee -a /workspace/bootstrap.log; }

log "pod ${RUNPOD_POD_ID:-unknown} starting"
log "installing tools"
apt-get update -qq && apt-get install -y -qq git curl build-essential >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"

log "cloning $REPO"
git clone --depth 1 "$REPO" "$WORK" >/dev/null 2>&1
cd "$WORK" || exit 1
# The rented image already ships a working torch against the installed driver.
# Downloading another 2.5 GB copy is the slowest step of the whole start, so the
# environment is built on top of what is there.
uv venv --system-site-packages >/dev/null 2>&1
PY=.venv/bin/python
$PY -c "import torch" 2>/dev/null || \
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -q "transformers>=4.48" "datasets>=3.0" scikit-learn tqdm pandas \
                  accelerate bitsandbytes huggingface_hub
log "torch: $($PY -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"
# A rented machine regularly comes up with a working `nvidia-smi` and a torch
# that cannot see the card anyway: the image ships a cu130 build and some
# community hosts do not carry the matching runtime. Installing the
# conservative cu124 build into the virtual environment fixes it, and a newer
# driver runs an older runtime happily. Without this check the run would fall
# back to the CPU and spend a day of rent for nothing.
cuda_ok () { $PY -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; }
if ! cuda_ok; then
  log "torch cannot see the card; trying the cu124 build"
  (nvidia-smi 2>&1 | sed -n '1,10p' || echo "nvidia-smi not present") >> /workspace/bootstrap.log
  uv pip install -q --reinstall torch --index-url https://download.pytorch.org/whl/cu124 \
    >> /workspace/bootstrap.log 2>&1
  log "after reinstall: $($PY -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"
fi
if ! cuda_ok; then
  log "FATAL this pod cannot run CUDA. Terminate it and start another."
  $PY - <<'PYEOF' 2>/dev/null
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("RESULTS_REPO", "chukfinley/gavel-runs")
api.create_repo(repo, repo_type="dataset", exist_ok=True)
api.upload_file(path_or_fileobj="/workspace/bootstrap.log", path_in_repo="bootstrap.log",
                repo_id=repo, repo_type="dataset", commit_message="this pod cannot run CUDA")
PYEOF
  # Exiting makes the container restart in a loop and clone again every
  # thirty seconds, so the pod idles instead and waits to be terminated.
  sleep infinity
fi

# Publish logs and results every few minutes, so the run can be watched from
# outside without a shell on this machine.
cat > /workspace/publish.sh <<'PUB'
#!/usr/bin/env bash
cd /workspace/gavel || exit 0
while true; do
  .venv/bin/python - <<'PYEOF' >/dev/null 2>&1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("RESULTS_REPO", "chukfinley/gavel-runs")
api.create_repo(repo, repo_type="dataset", exist_ok=True)
for folder, prefix in [("results", "results"), ("logs", "logs")]:
    if os.path.isdir(folder):
        api.upload_folder(folder_path=folder, path_in_repo=prefix, repo_id=repo,
                          repo_type="dataset", commit_message="progress")
for name in ("COMPARISON.md", "/workspace/bootstrap.log", "/workspace/build.log"):
    if os.path.isfile(name):
        api.upload_file(path_or_fileobj=name, path_in_repo=os.path.basename(name),
                        repo_id=repo, repo_type="dataset", commit_message="progress")
PYEOF
  sleep 120
done
PUB
chmod +x /workspace/publish.sh
nohup /workspace/publish.sh >/dev/null 2>&1 &

# The JevBench items are a git clone, fetched once so the evaluation does
# not depend on the network hours later.
git clone -q --depth 1 https://github.com/fstandhartinger/jevbench \
  /workspace/jevbench >/dev/null 2>&1 || log "jevbench clone failed"

log "building datasets from public sources"
$PY scripts/build_data.py        --per-source 40000 --out data   >> /workspace/build.log 2>&1
$PY scripts/build_business.py                                    >> /workspace/build.log 2>&1
$PY scripts/build_long.py --filler-lines 10                      >> /workspace/build.log 2>&1
$PY scripts/build_abstain.py                                     >> /workspace/build.log 2>&1
$PY scripts/build_router.py                                      >> /workspace/build.log 2>&1
$PY scripts/build_quiz.py                                        >> /workspace/build.log 2>&1
$PY scripts/build_knowledge.py                                   >> /workspace/build.log 2>&1
$PY scripts/build_multilingual.py                                >> /workspace/build.log 2>&1
$PY scripts/build_tools.py --limit 30000                         >> /workspace/build.log 2>&1
$PY scripts/build_browser.py                                     >> /workspace/build.log 2>&1
$PY scripts/build_moderation.py                                  >> /workspace/build.log 2>&1
$PY scripts/build_more.py                                        >> /workspace/build.log 2>&1
$PY scripts/build_semrouter.py                                   >> /workspace/build.log 2>&1
$PY scripts/build_kotoba.py                                      >> /workspace/build.log 2>&1
$PY scripts/build_domains.py                                     >> /workspace/build.log 2>&1
$PY scripts/build_games.py                                       >> /workspace/build.log 2>&1
$PY scripts/build_testset.py --per-source 400                    >> /workspace/build.log 2>&1
$PY scripts/build_devstrat.py                                    >> /workspace/build.log 2>&1
for name in train business router abstain_short quiz knowledge multilingual tools \
            browser moderation more semrouter kotoba domains games; do
  [ -s "data/${name}.jsonl" ] || log "WARNING data/${name}.jsonl is missing or empty"
done


# The long run assembles its own mixes (it builds the ordered scales first),
# trains, calibrates, measures against nine test sets plus the independent
# benchmark, and publishes on its own. The batch sizes are the ones the 32k
# encoder was trained at before, left overridable so a larger card can be used
# without editing the job.
export BACKBONE=${BACKBONE:-llm-semantic-router/Vela-1.0-Encoder-307M}
export STEPS=${STEPS:-60000}
export DECISION_BATCH=${DECISION_BATCH:-4}
export ANCHOR_BATCH=${ANCHOR_BATCH:-8}
export LONG_BATCH=${LONG_BATCH:-1}
export SPAN_STEPS=${SPAN_STEPS:-20000}
export SPAN_BATCH=${SPAN_BATCH:-8}
log "starting the long run: $STEPS steps on $BACKBONE"
./scripts/longrun.sh >> /workspace/bootstrap.log 2>&1
log "long run exit: $?"

# The long run publishes its own model, results and logs. Anything else that
# produced a checkpoint is published here, so nothing is lost with the pod.
log "publishing any other finished models"
for run in runs/*/best-calibrated.pt; do
  [ -f "$run" ] || continue
  name=$(basename "$(dirname "$run")")
  [ "$name" = "longrun" ] && continue
  $PY scripts/export_hf.py --checkpoint "$run" --out "export/$name" >> /workspace/bootstrap.log 2>&1
  $PY - "$name" <<'PYEOF' >> /workspace/bootstrap.log 2>&1
import os, sys
from huggingface_hub import HfApi
name = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"chukfinley/gavel-{name}"
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=f"export/{name}", repo_id=repo, repo_type="model",
                  commit_message="trained on a rented consumer card")
print("published", repo)
PYEOF
done
log "done, the pod can be stopped"
