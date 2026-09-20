# Start here

You are taking over a project mid-flight. Read these three files first, in
order, and do not re-derive what they already record:

1. `HANDOVER.md` — what exists, what is published, what is half finished.
2. `LEARNINGS.md` — the independent benchmark, what every competing project
   did, and the seven things this project measured for itself.
2b. `FIELD.md` — the competition as read from their own repositories on
   2026-09-20: the benchmark's second suite, who publishes training code and
   data, and why Von's 0.923 is not a number to chase.
3. `PLAN.md` — what to build next, in order, with what it should buy.
4. `README.md` — the model and how it is used.

## The one-sentence version

`gavel` is a decision model, not a language model: a state and a list of
options go in, one typed answer and a calibrated probability come out, nothing
is generated. Each option becomes a hypothesis about the state and a three-way
entailment head scores it.

## What the user wants next

Run `scripts/longrun.sh` on a rented RunPod machine and report the numbers.
Everything is wired already: the pod clones this repository, rebuilds all
fourteen datasets from public sources, trains, calibrates, measures against
nine test sets plus the independent `jabr/classifier-benchmark`, and uploads
the model and every result to Hugging Face by itself.

```bash
source ~/.config/gavel/keys.env          # not in this repository, on purpose
python scripts/runpod_launch.py start --gpu "NVIDIA GeForce RTX 3090"
python scripts/runpod_launch.py status   # cost per hour
python scripts/runpod_launch.py stop     # when done, or the credit keeps burning
```

The RunPod account holds about $10.64 and a 3090 costs $0.22/h, so roughly 48
hours are available. The long run is about 60000 steps, six to eight hours.

Two things that cost time last round: the pod REST endpoint rejects Python's
default user agent with a 403 (`runpod_launch.py` already sends a curl one),
and community 4090s are often unavailable, so pass several `gpuTypeIds` or take
the 3090.

## Hard constraints from the user

* **The 32768-token backbone stays** (`llm-semantic-router/Vela-1.0-Encoder-307M`).
  It is the differentiator: Von has 512 tokens, Laya 1024, the DeBERTa
  reproduction 512, and users in the field ask for 30k-token decisions.
  Distilling *from* another model is allowed; starting from its checkpoint is
  not.
* No local GPU: the workstation's driver was upgraded without a reboot, so
  `nvidia-driver-610` (610.57.04) and the loaded kernel module (610.43.02)
  disagree. Everything runs on the rented pod until the machine restarts.
* Write notes into the repository, never into assistant memory.
* German, short answers, no moralising. Name risks only when they cost money.

## Where this stands against the field

`jabr/classifier-benchmark`, identical rows: Jev 0.974, Von 1.0.1 0.923,
GLiNER2 0.795, **gavel 0.654**, Laya 0.615. Our two worst tasks are both
ordered scales (0.333 and 0.444), and `scripts/build_scales.py` was written to
fix exactly that; the long run is the first that includes it.

Where this model already wins: ten languages (XNLI German 0.845 against Laya's
0.731 average outside English), tool selection 0.990, prompt injection 0.980,
safety policies 0.935, SciQ with a passage 0.980, and the 32k context.

## The open work, in order

1. Run the long run and see what ordered scales and the criteria wording buy.
2. Finish `scripts/build_grounded.py` — it stalled at 5000 of 40000 rows
   because the BM25 index is pure Python. Use `rank_bm25`. Retrieval already
   lifted MMLU from 0.293 to 0.367 with no training at all.
3. Test the unreadable-input failure: feed a language the model has never seen
   and check whether confidence drops. Laya measured 0.000 accuracy at 0.952
   confidence on their English checkpoint — if ours behaves the same, the
   abstain mechanism is worthless exactly where it is needed.
4. Distil from `MoritzLaurer/bge-m3-zeroshot-v2.0` as a teacher.
5. Answer several questions in one pass — the throughput argument, and the one
   piece of the competing designs not yet taken.

## The four traps that cost hours last round

1. **`dockerStartCmd` replaces the image entrypoint and with it the ssh
   daemon.** Two ways to run, pick one deliberately. *Without* it: create
   the pod, ssh in, start the work in tmux, inspect at will. *With* it, as
   `runpod_launch.py` does: `pod_bootstrap.sh` runs unattended and publishes
   its logs, results and models to the Hub every two minutes on its own —
   three pods trained that way on 2026-09-20 — but there is no shell, so a
   pod that misbehaves can only be read from the Hub and terminated.
2. **The REST endpoint rejects Python's default user agent** with 403 and
   Cloudflare error 1010. `scripts/runpod_launch.py` sends a curl one; keep it.
3. **`/workspace` holds 20 GB, the root filesystem 120.** Work under `/root`,
   or the dataset build runs out of space. `pod_bootstrap.sh` is rewritten with
   `sed 's|/workspace|/root/run|g'` for exactly this.
4. **Set `HF_HUB_DISABLE_XET=1`.** The xet backend stalls on these machines and
   leaves zero-byte `.incomplete` files.

Also: the image already ships a working torch. `uv venv --system-site-packages`
saves downloading another 2.5 GB copy, about ten minutes per pod start.

The configuration that worked:

```json
{"computeType": "GPU",
 "gpuTypeIds": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090",
                "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 5090"],
 "gpuTypePriority": "availability", "gpuCount": 1,
 "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
 "containerDiskInGb": 120, "cloudType": "COMMUNITY", "ports": ["22/tcp"]}
```

On CUDA: torch 2.6.0+cu124 reported `cuda.is_available() == True` on the
3090 in one session; in another, eight Community hosts in a row showed a card
in `nvidia-smi` and no device for torch, on the image's own torch and on a
fresh cu124 install alike. Both happened. `runpod_launch.py ensure` rents
until a host passes, which is the only fix that covers both cases.

On `--system-site-packages`: it saves the torch download, and it inherits
the image's torchvision, which is built against the image's torch and breaks
the transformers import (`operator torchvision::nms does not exist`, reported
as a missing ModernBert class) the moment a different torch lands in the
venv. The unattended bootstrap builds a clean venv for that reason; the
download took about two minutes on these hosts.
