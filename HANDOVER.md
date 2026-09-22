# Where this stands, 2026-09-20

Read `CLAUDE.md` first for the rules, `FIELD.md` for the competition,
`PLAN.md` for what to build next. This file is the state.

## What exists

* **Code**: <https://github.com/chukfinley/gavel>
* **Models**: <https://huggingface.co/chukfinley/gavel-base> (150 M, 8k) and
  <https://huggingface.co/chukfinley/gavel-vela-32k> (308 M, 32k,
  multilingual — the one in use)
* **Every result file and log**:
  <https://huggingface.co/datasets/chukfinley/gavel-runs>

## What changed today, and why it matters

**The architecture is now one sequence per decision.** `src/gavel/span.py`
reads `[CLS] [STATE] state [Q] question [OPT] a [OPT] b` once and scores every
option from the same hidden states, with a softmax inside each question's
group. The old pair scorer in `src/gavel/api.py` ran the state through the
encoder once per option; measured, that is 109 ms for two options and 377 ms
for eight at 64 tokens of state, on this CPU.

The first attempt at a single sequence failed years of debugging ago and
`src/gavel/model.py` still records why: cross entropy sat at ln(number of
options) forever. The cause was the pooling, not the idea. A marker token
carries no relation between a question and an option, so no gradient
separates them. Pooling the option span and the question span and feeding
`[question; option; question*option]` gives the head that relation, which is
what kotoba's working head does. `tests/test_span.py::test_the_head_can_learn`
is the proof: ln(3) to under 0.1 in sixty steps.

**Two measurement bugs were found, both of which flattered or punished us
wrongly.** `eval_cbench.py` read a Score question's levels as a mapping when
they are a list, fell back to three bare digits, and so scored the
five-level `incident_severity` with three options while hiding every level
description from the model. Our published 0.654 on that benchmark is not a
valid number. `eval_jevbench.py` had the same bug on its first run.

**Von's 0.923 is not a number to chase.** Its training repository generates
the benchmark's own v1 test cases, character for character — see `FIELD.md`.

**The benchmark landscape is three suites, not one**: the classifier
benchmark now has a v2 of 49 tasks and 869 cases, JevBench has 231 public
items in three tiers with a leaderboard, and Bespoke's Nimble suite has 13
subsets. Only the first two are wired up.

## Measured today

| | |
|---|---|
| JevBench easy / standard / hard | **0.958 / 0.514 / 0.387**, weighted 0.545 |
| the only other encoder there, open-jev-deberta 435 M | 1.000 / 0.431 / 0.378, weighted 0.524 |
| unreadable input: confidence drop | 0.691 to 0.536 at worst, accuracy at chance |
| latency, 64 tokens, 2 / 5 / 8 options | 109 / 240 / 377 ms, CPU, 8 threads |
| latency, 256 tokens | 301 / 730 / 1297 ms |
| latency, 1024 tokens, 2 options | 1369 ms |

Published for comparison, per case on Apple MPS: Laya 46 ms, Von 55 ms,
GLiNER2 73 ms, hosted Jev about 330 ms.

## The pipeline now

`scripts/longrun.sh` does everything in one go and the pod runs it by itself:

1. builds the ordered scales, the named-criteria rows, the claim rows, the
   thirteen new domains and the game decisions, then assembles the mixes
2. trains the pair scorer, 60000 steps
3. calibrates it
4. **distils it into the span scorer**, 20000 steps, pair model as teacher
5. measures both on the classifier benchmark v1+v2, JevBench and nine test
   sets, plus the latency bench
6. publishes the model, the span head, every result and every log

New data since yesterday: `build_domains.py` (13 sources — CoLA, PII, LEDGAR,
WikiQA, HelpSteer2, FEVER, LIAR2, CLINC with out-of-scope, civil comments,
GoEmotions, Yahoo topics, Amazon stars, financial headlines),
`build_games.py` (4716 snake and maze decisions from NanoJev's CC0 corpus),
`build_criteria.py` (79730 named-criteria rows and 25574 claim rows).

## Renting a machine

```bash
python scripts/runpod_launch.py ensure       # rents until one can run CUDA
python scripts/runpod_launch.py status
python scripts/runpod_launch.py stop
```

`ensure` exists because five community hosts in a row came up with a working
`nvidia-smi` and a torch that saw no device, on both the cu130 build the
image ships and the cu124 build the bootstrap installs. The bootstrap now
detects that, publishes its log and idles rather than exiting, because
exiting made the container restart every thirty seconds.

## Two traps that cost an afternoon, so they are written down

**A pod can have a working GPU and still be useless.** Eight machines in a
row showed a card in `nvidia-smi`, with `libcuda.so.1` present and loading,
and `torch.cuda.is_available()` still False. It is a broken host, not
something to debug: `runpod_launch.py ensure` rents until one works. The
request that does work is in `CLAUDE.md`; 120 GB of container disk and one
entry in `gpuTypeIds` are part of it.

**`Could not import module 'ModernBertForSequenceClassification'` is a lie.**
The real error underneath is `operator torchvision::nms does not exist`.
transformers imports `image_utils`, which imports torchvision, and the pod
image ships a torchvision built against its own torch. The moment a
different torch lands in the venv, that import raises and transformers
reports a missing model class. It works on the developer machine because
torchvision is not installed there at all, and transformers skips it when
absent. The pod builds a clean venv now — no system packages, torch
2.6.0+cu124, transformers 5.17.0, no torchvision.

Both failures are now caught before the eight-minute dataset build rather
than after it.

## What is half finished

* Two GPU runs completed on 2026-09-20; the results table is in `FIELD.md`
  ("Two recipes and two heads"). The published model is the baseline-recipe
  pair model with the locally distilled span head beside it.
* ~~`build_grounded.py` stalls on a pure-Python BM25~~ fixed: the weights are
  precomputed into a sparse matrix, 120000 passages index in 8 s and a search
  takes 0.9 ms, so the full 40000 rows take under a minute.
* Bespoke's Nimble suite is not wired up. Its 13 subsets rebuild from
  manifests in `bespokelabsai/nimble`.
* `src/gavel/longdoc.py` works but has never been trained for or measured at
  scale.
* ~~The unreadable-input test~~ ran and passed: accuracy falls to chance on
  five unseen scripts and confidence falls by 0.154 with it. `LEARNINGS.md`
  item 7.

* The local `data/train_v6.jsonl` was the old 566k mix; only the pods
  had the 1.09M one. On 2026-09-21 `_scratch/build_mix_local.sh` rebuilds
  it here (same chain as `longrun.sh` lines 27-38) and touches
  `data/.mix_v6_full`. `_scratch/wait_gpu_then_span.sh` then distils the
  span head from Hub `main` for 40000 steps on the 3060 as `base-span-40k`,
  as soon as the card has 10 GB free and that marker exists. It never
  kills another process; ComfyUI held the card when it was started.
* Dev overlap, first pass on the old mix: 8367 training rows share state
  and question with a dev row, 7672 of them `abstain-missing` (empty state,
  generic question, so the key is weak). Redo with options in the key on
  the new mix before drawing a conclusion for the v2 gap.

* **WebUI** (2026-09-21): `scripts/webui.sh` serves `src/gavel/webui.py` on
  port 8030. Scoreboard from `results/` next to the published numbers,
  playground (pair or span, bundled questions), benchmark and demo jobs as
  subprocesses of the pod scripts, training tab with the 40k distillation
  button. The page is `src/gavel/web/index.html`, no build step.
* **`decide_many` on the span head is broken as trained.** Three snake
  safety questions in one sequence: 0.065 right; the same three one by
  one: 0.95. `train_span.py` never packs several questions behind one
  state, so the head has never seen a second `[Q]` group. Fix: build
  training sequences with 1–4 questions that share a state (games,
  jevbench-style rows, the criteria wordings) before trusting the bundled
  path. Until then the WebUI's bundle switch is a demonstration of the bug.
* **Snake is answered from the prior.** Pair and span both say "Yes" to
  every safety question (150/150 on training rows, gold 94 % yes), so the
  demo's 10.8 food eaten is the food heuristic alone; `yes_rate` in
  `results/demo_snake_*.json` makes that visible. The 1356 snake rows are
  too few and too one-sided to teach collision reasoning.
* **Web navigation** works: span head 6/8 sites at 34 ms per page, pair
  4/8 at 771 ms (40 links each pass through the pair model one by one).

* **Jev as a teacher, 2026-09-21.** OpenRouter serves `typesafe/jev-1.13`
  on `POST /api/alpha/decisions` with TypeSafe's native body (`state`,
  `questions{type,instructions,criteria}`), 0.042 $ per million input
  tokens, output free, `usage.cost` in every reply. `scripts/label_with_jev.py`
  sends our rows (several questions per state in one request), logs every
  request and raw reply to `logs/jev_requests.jsonl`, and writes
  `data/jev/<name>.jsonl` with Jev's probability per option. The whole dev
  set cost 0.064 $. Both trainers take `--teacher-file` and add
  KL(Jev || student) on the rows it covers; the augmentations carry the
  distribution through `drop_distractors` and `negate` (`src/gavel/teacher.py`,
  `tests/test_teacher.py`). Key lives in the gitignored `.env`.
* **Jev on our dev set** (`results/jev_dev.json`, 18 strata): 0.799 mean
  against 0.690 for the baseline pair model (`results/base-pair_dev.json`).
  Jev wins every knowledge stratum by 30–50 points (mmlu 0.91 vs 0.39,
  arc-easy 0.995 vs 0.52, openbookqa 0.94 vs 0.43): it knows facts, a 307M
  encoder does not. We win abstain, claims, criteria, domains, moderation,
  packet, router and scales. Distilling Jev's probabilities helps where
  the task is reading the state; it cannot give the encoder world knowledge.
* **Scales rows are half unreadable by design.** `build_scales.py` line 104
  writes a bare level number as the option for half the options, so a
  0–3 scale reads "0: calm and factual", "1", "2", "3". Jev scores 0.33
  there, our model 0.615 by memorising the generator. Either always write
  the level text, or define the scale in the question. Decide before the
  next mix.

* **Jev beside us, live (2026-09-21).** `src/gavel/jevapi.py` puts the
  closed model behind `decide`/`decide_many` over OpenRouter's decisions
  endpoint; the WebUI's playground and live tab run every request on both,
  ours left, Jev right. On amazon.de, goal "Microsoft Surface Pro 12 inch
  Snapdragon X Plus 16 GB 256 GB": Jev reaches the product in 3–4 hops
  (Computer → Tablets → Microsoft → the item, 98 %) for 0.003 $; our pair
  model reads 92 links as 92 sequences, takes 14 s per page, and picks
  "Lieferung verfolgen" at 3 %. That gap is the work. Two levers, both
  wired: Jev's probabilities as a KL target (`--teacher-file`, labels in
  `data/jev/`), and Jev's own navigation decisions as training rows
  (`scripts/harvest_webnav.py` → `data/traces/webnav.jsonl` →
  `scripts/build_webnav.py` → source `webnav` in the mix). The span head
  is the right shape for 100-link pages (one pass); it needs the packed
  training (`--pack`) before its bundled answers can be trusted.
* `_scratch/jev_distil_chain.sh` continues the baseline pair model with
  the Jev labels (10k steps, batch 4, 3060) as `runs/jev-pair`, then dev,
  JevBench and cbench land in `results/jev-pair_*.json`.

* **`runs/jev-pair` (2026-09-21, 10k steps, 79k Jev-labelled rows as KL
  target, teacher share 0.5, lr 1e-5): no movement.** Dev 0.695 vs 0.690,
  JevBench 1.000/0.694/0.342 vs 0.979/0.694/0.369, cbench 0.623 vs 0.629.
  Knowledge strata up a little (knowledge-held 0.575→0.650, openbookqa
  0.425→0.480, wanli 0.650→0.710), the rest noise. Soft labels on 9 % of
  the rows do not carry facts into a 307M encoder. Results in
  `results/jev-pair_*.json`.
* **`_scratch/local_full_chain.sh` (started 2026-09-22 13:15 local):** the
  mix with every new source (webnav from Jev's navigation, balanced snake
  from the simulator, the Jev-4b distillation tickets, Enron spam, 22k
  products, Jev's task answers), pair model continued 30k steps with the
  Jev KL target, then the span head with packed questions, 20k steps.
  About 7 h + 4 h on the 3060. `runs/full-pair`, `runs/full-span`,
  `results/full-*`. `_scratch/night.log` has the stamps.
* **Tasks tab and sources (2026-09-22):** email triage (Enron, spam gold)
  and product → 33 categories (gold), both models per item, agreement and
  accuracy in the UI; `scripts/run_task.py`, `src/gavel/tasks.py`. Jev's
  answers on all 824 items are harvested into `data/traces/tasks.jsonl`
  and folded in by `build_tasktraces.py`. `build_jevdistill.py` turns
  MagaBitmex/jev-4b-distill-data (7200 questions with Jev's distributions)
  into a source plus two test sets, `test_jevdistill` and
  `test_jevdistill_hard`.

* **Local `full-pair` run, 2026-09-22, stopped by the owner at step 15400
  of 30000 (SIGTERM).** Dev 0.682 → 0.687 → 0.693 at 5k/10k/15k against
  0.690 for the baseline; `runs/full-pair/best.pt` is the step-15000
  checkpoint on the 966k-row mix with the Jev KL target. New rule in
  `CLAUDE.md`: nothing trains on the workstation any more. The same
  recipe goes to a pod through `scripts/longrun.sh`, which now pulls the
  Jev labels and the harvested traces from the `chukfinley/gavel-runs`
  dataset (`data/`), continues from the Hub pair model, and distils the
  span head with packed questions.

* **`build_domains.py` "Aborted (core dumped)" on pods is harmless.** On
  2026-09-22 the pod log shows `wrote 147593 training rows and 3250 test
  rows` and *then* the abort: it happens at interpreter exit (a native
  destructor in arrow or tokenizers under Python 3.11), after every file
  is written. `longrun.sh` does not stop on it. Do not chase it.

* **Pod run `pod-full`, 2026-09-22 (RTX 3090, 1.35 $):** pair model
  continued from Hub main on the 1.14M-row mix with the Jev KL target, 30k
  steps, batch 8/16, lr 2e-5. Dev 0.687 → 0.684 → 0.686 → 0.695 → 0.700
  → **0.702** (20 strata, now including the products, emails and task
  held-outs). JevBench 0.979 / 0.681 / 0.324 (baseline 0.979 / 0.694 /
  0.369), cbench v2 0.610, combined 0.619 (baseline 0.619 / 0.629). Better
  on our own strata, worse on the two outside suites, hard tier −4.5.
  Results in `results/pod-full_*.json`; the model is Hub main `2e9a4a65`,
  the baseline stays at revision `1023a938`.
* **The span distillation on that pod died of OOM at step 1400**: packed
  batches (four questions per sequence, batch 16) sent ~320 pairs through
  the teacher in one forward. `teacher_distribution` now runs in blocks of
  64, `longrun.sh` records the real exit code (a `$(stamp)` in the echo had
  reset `$?` to 0), and `SKIP_TRAIN=1 PAIR_REVISION=<sha>` runs only
  calibration, measurement and the span head from a published pair model.
  The Jev-4b dataset upstream is gone (404); our copy lives in
  `chukfinley/gavel-runs/data/jev4b/` and the builder falls back to it.

## The one thing to decide

Whether the product is short states on a CPU or long documents on a GPU.
Both are real, they do not fit in one claim, and `FIELD.md` has the
arithmetic. The long-document position is the one nobody else holds openly.
