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

* **No GPU run has completed with the current code.** Everything above was
  built and tested on CPU; the training numbers are from yesterday's
  architecture. A run started at 08:38 on 2026-09-20 with the backbone
  loading correctly for the first time.
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

## The one thing to decide

Whether the product is short states on a CPU or long documents on a GPU.
Both are real, they do not fit in one claim, and `FIELD.md` has the
arithmetic. The long-document position is the one nobody else holds openly.
