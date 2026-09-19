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

log "installing tools"
apt-get update -qq && apt-get install -y -qq git curl build-essential >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"

log "cloning $REPO"
git clone --depth 1 "$REPO" "$WORK" >/dev/null 2>&1
cd "$WORK" || exit 1
uv venv --python 3.12 >/dev/null 2>&1
uv pip install -q torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -q "transformers>=4.48" "datasets>=3.0" scikit-learn tqdm pandas \
                  accelerate bitsandbytes huggingface_hub
PY=.venv/bin/python

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
$PY scripts/build_testset.py --per-source 400                    >> /workspace/build.log 2>&1
$PY scripts/build_devstrat.py                                    >> /workspace/build.log 2>&1

log "assembling the training mixes"
$PY - <<'PYEOF' >> /workspace/build.log 2>&1
import sys, random
sys.path.insert(0, "src")
from typedec.schema import read_jsonl, write_jsonl
rng = random.Random(11)
parts = ["data/train.jsonl", "data/business.jsonl", "data/router.jsonl",
         "data/abstain_short.jsonl", "data/quiz.jsonl", "data/knowledge.jsonl",
         "data/multilingual.jsonl", "data/tools.jsonl", "data/browser.jsonl",
         "data/moderation.jsonl"]
rows = []
for path in parts:
    try:
        rows += [r for r in read_jsonl(path) if not r.source.endswith("-hi")]
    except Exception as error:
        print("missing", path, error)
rng.shuffle(rows)
write_jsonl("data/train_v6.jsonl", rows)
long_rows = list(read_jsonl("data/long.jsonl")) + list(read_jsonl("data/abstain_long.jsonl"))
rng.shuffle(long_rows)
write_jsonl("data/long_v2.jsonl", long_rows)
agent = list(read_jsonl("data/tools.jsonl")) + list(read_jsonl("data/browser.jsonl"))
rng.shuffle(agent); write_jsonl("data/agent.jsonl", agent)
route = {"router-difficulty", "router-tier", "banking77", "ag-news", "dbpedia",
         "tweet-offensive", "tweet-hate", "sms-spam", "synth-routing", "synth-action",
         "mnli", "wanli-train", "anli-r3"}
doc = {"boolq", "quality", "race", "synth-packet", "synth-incident", "synth-evidence",
       "sciq-passage", "mnli", "wanli-train", "anli-r3"}
write_jsonl("data/train_route.jsonl", [r for r in rows if r.source in route])
write_jsonl("data/train_doc.jsonl", [r for r in rows if r.source in doc
                                     or r.source.startswith("abstain")])
print("train_v6", len(rows))
PYEOF
log "training rows: $(wc -l < data/train_v6.jsonl)"


# A 24 GB card takes roughly three times the batch of the 12 GB card this was
# written on, which is where the rented time is saved.
export DECISION_BATCH=${DECISION_BATCH:-16}
export ANCHOR_BATCH=${ANCHOR_BATCH:-32}
export LONG_BATCH=${LONG_BATCH:-2}
export EVAL_BATCH=${EVAL_BATCH:-16}
export LONG_CTX=${LONG_CTX:-2048}
log "starting the marathon"
./scripts/marathon.sh >> /workspace/bootstrap.log 2>&1

log "publishing the finished models"
for run in runs/*/best-calibrated.pt; do
  [ -f "$run" ] || continue
  name=$(basename "$(dirname "$run")")
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
