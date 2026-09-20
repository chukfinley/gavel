# Rules for working in this repository

## Write it down here, never in assistant memory

Assistant memory lives on one machine. The moment a session runs somewhere
else, it is gone. Anything that should survive goes into a file here and gets
committed: `START_HERE.md`, `HANDOVER.md`, `LEARNINGS.md`, `FIELD.md`,
`PLAN.md`, or this file. A note that is not in the repository is lost.

Read in this order when taking over: `START_HERE.md`, `HANDOVER.md`,
`LEARNINGS.md`, `FIELD.md`, `PLAN.md`.

## The competing repositories are maintained, so check them, do not remember them

Every project in this class is under active development. `wfzyx/von`,
`Mapika/decider`, `jaredpalmer/kev`, `C-Tianyu/NanoJev`,
`kotoba-lang/typed-decisions`, `jabr/classifier-benchmark` and
`fstandhartinger/jevbench` all had commits on 2026-09-20, the same day
`FIELD.md` was written. Their numbers, their suites and their claims move.

So before stating anything about them: fetch the repository or the model card
again, and date the claim. Two things already changed underneath us — the
benchmark grew a second suite of 869 cases that we were not measuring, and
Von's training repository started generating the benchmark's own test cases.
Both were invisible from a summary written a day earlier.

## The benchmarks we report on

| suite | what it is | where |
|---|---|---|
| `jabr/classifier-benchmark` | v1 78 cases, v2 869 cases, gold labels | `scripts/eval_cbench.py` |
| JevBench public items | easy 48, standard 72, hard 111, has a leaderboard | `scripts/eval_jevbench.py` |
| Bespoke Nimble public suite | 13 subsets, 3880 records, Jev wire format | not wired up yet |
| our own strata | nine test files plus the router | `scripts/eval_general.py` |

Rules for reporting: never pool sources, always give the denominator, and say
which suite a number is from. **Do not chase v1** of the classifier benchmark
— Von's `training/prepare_decision_dataset.py` generates its cases verbatim,
so the score there means nothing. v2, combined, and JevBench hard are the
numbers that separate systems.

## What is fixed about the model

* **The backbone stays `llm-semantic-router/Vela-1.0-Encoder-307M`**, 32768
  tokens. It is the product: the smallest model in its class and the only
  open one that reads a long document. Distilling *from* another model is
  fine; starting from its checkpoint is not.
* **One sequence per decision, not one per option.** The state is read once
  and every option is scored from the same hidden states (`src/gavel/span.py`).
  The pair-per-option scorer in `src/gavel/api.py` stays as the accuracy
  reference and the teacher, but it is not the product: it made a five-option
  decision cost five readings of the state, measured at 240 ms against 109 ms
  on 64 tokens.
* Several questions about one state ride in the same sequence. That is the
  throughput argument and the last piece of the competing designs we had not
  taken.

## Measuring

* Measure before claiming. `scripts/bench_latency.py` for speed,
  `scripts/eval_jevbench.py` and `scripts/eval_cbench.py` for accuracy. Both
  write JSON into `results/`, which is published to the Hub.
* A low score is a measurement bug until proven otherwise. Two of ours were:
  a truncated evaluation that looked like a below-chance result, and a reader
  that offered three options on a five-level question.
* `ruff check src scripts tests` and `pytest` before committing.

## Rented machines

* `python scripts/runpod_launch.py ensure` rents until one machine can
  actually run CUDA, then leaves it running. Community hosts regularly come
  up with a working `nvidia-smi` and a torch that sees no device; four in a
  row did on 2026-09-20. Do not swap them by hand.
* The pod clones this repository from GitHub, so **push before starting one**.
* `scripts/runpod_launch.py stop` when a run is done. Credit burns per hour.
* The workstation GPU is unusable until it reboots: the driver package was
  upgraded while the old kernel module is still loaded. Everything heavy runs
  on a rented card; CPU is fine for latency measurements and small evals.

## Commits

Commit as `chukfinley <77645077+chukfinley@users.noreply.github.com>`, which
is already the global git configuration, so do not override it. No session
links, no co-author trailers, no tool metadata in the message — only what
describes the commit.
