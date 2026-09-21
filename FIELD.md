# The field, as measured on 2026-09-20

Everything here was read from the project's own repository, model card or
published result file on 2026-09-20. Where a number is someone else's claim it
says so. This file replaces guessing about the competition; update it, do not
re-derive it.

## There are three external suites, not one

`jabr/classifier-benchmark` is where Von, GLiNER2 and Laya report. It is not
where the rest of the field reports.

**JevBench** (<https://benchmarkheaven.com/jev-models>, harness at
`fstandhartinger/jevbench`) ranks Jev-class systems over 534 decisions in
four tiers; 231 items are public — easy 48, standard 72, hard 111. Published
accuracy on those public items, from decider's README:

| system | easy | standard | hard |
|---|---:|---:|---:|
| GPT-5.6 Luna, low reasoning | 1.000 | 0.972 | **0.964** |
| Jev 1.13.0 (closed) | 1.000 | 0.986 | 0.730 |
| djev (Maisa) | 1.000 | 0.986 | 0.676 |
| OpenJev (DiffusionGemma 26B-A4B) | 1.000 | 0.972 | 0.640 |
| SemIf (Qwen3.5-4B) | 1.000 | 0.986 | 0.613 |
| decider-2b v10 (1.9 B) | 1.000 | 0.847 | 0.459 |
| Bespoke Nimble 9B | 1.000 | 0.931 | 0.369 |
| open-jev-deberta-v3-large (435 M) | 1.000 | **0.431** | 0.378 |
| gavel-vela-32k (308 M), pair scorer, old data | 0.958 | 0.514 | 0.387 |
| **gavel, pair scorer, 2026-09-20 mix (1.09 M rows)** | **0.979** | **0.694** | 0.369 |

Weighted over all 231 items: Jev 0.866, decider 0.692, Nimble 0.675,
**gavel new 0.597**, gavel old 0.545, open-jev-deberta 0.524.

The 2026-09-20 run — same recipe, the mix grown from 566 k to 1.09 M rows
with ordered scales, named criteria, claims, thirteen domains, games,
specialist routing and retrieved evidence — moved the standard tier by 18
points. Per family: ordinal 0.750 → **1.000**, extraction 0.417 → **0.917**,
routing 0.167 → **0.583** (the routing builder did what it was written for),
intent 0.667 and adequacy 0.500 unchanged, policy 0.583 → 0.500. On the
hard tier, long_policy 0.263 → **0.526** — the long-document families
doubled — while the small reasoning families (tradeoff, judge_hard,
probability, 5-17 items each) fell, for 0.387 → 0.369 overall. Latency
99 ms per item on the 3090. Dev mean over 20 strata 0.703, calibrated ECE
0.034.

On the classifier benchmark, measured for the first time with a correct
reader: **v1 0.744, v2 0.619, combined 0.629** (micro). That puts us above
Laya (0.585 / 0.587) and below Von (0.666 / 0.687) and GLiNER2 (0.688 /
0.697) on v2 and combined. Seven points to the two of them is the gap that
remains.

Measured here on 2026-09-20 with `scripts/eval_jevbench.py`, on the published
checkpoint, before any of the new data or the span head.

**We beat the only other encoder on that board**, on both tiers that separate
anything and overall, at 308 M against its 435 M. On the hard tier we are
also above Bespoke Nimble, which is 9 B.

**And the long-context claim does not survive contact with that tier.** Our
own per-family numbers:

    tradeoff 0.667 · judge_hard 0.588 · adversarial 0.500 · probability 0.500 ·
    temporal_numeric 0.400 · routing_hard 0.400 · multi_hop 0.278 ·
    long_policy 0.263 · trap 0.125

`long_policy` is the family with 3746-token states — the one this model was
supposed to own because Von at 512 and kotoba at 256 cannot read it at all.
We score 5 of 19 there, second worst of any family. That is the same finding
as `LEARNINGS.md` item 6, arrived at independently: a 32 k window is not
32 k of reasoning.

So the long context is worth selling as **reading a long document without
truncating it** — extraction, classification, checking one criterion across
30 000 tokens, where the alternative is a silently cut input. It is not worth
selling as multi-step reasoning over a long contract. The first is real and
measured; the second is not ours.

**Bespoke's public suite** (`bespokelabsai/nimble`, 666 stars) is 13
human-labelled subsets, 3880 records in Jev's wire format, rebuildable
byte-for-byte from manifests in the repository. Jev 1.13.0 and Nimble-9B are
measured on it, and decider reports 0.706 macro over it. Not wired up here
yet.

## Two recipes and two heads, 2026-09-20/21

Both pair models trained on the same 1.09 M-row mix. The baseline is the
original recipe (60000 steps, batch 4 + 8 anchors, lr 1.5e-5, sdpa); the
fast one is the reviewed recipe (30000 steps, batch 8 + 16, lr 2e-5,
flash-attention, no filler slots, one pass). Each was then distilled into
the one-sequence span head on the workstation's 3060 (10000 steps, batch 8,
teacher = its own pair model).

| model | easy | standard | hard | weighted | cbench v2 | ms/item |
|---|---:|---:|---:|---:|---:|---:|
| pair, baseline recipe | 0.979 | **0.694** | **0.369** | **0.597** | **0.619** | 99 (3090) |
| pair, fast recipe | 0.979 | 0.681 | 0.324 | 0.571 | 0.596 | 74 (3060) |
| span head from baseline | 0.958 | 0.528 | 0.342 | 0.528 | — | **29** (3060) |
| span head from fast | 0.979 | 0.472 | 0.342 | 0.515 | — | 30 (3060) |
| open-jev-deberta (435 M) | 1.000 | 0.431 | 0.378 | 0.524 | — | — |

What it says. **The baseline recipe is the better model**: 2.6 points ahead
on weighted JevBench, 2.3 on v2, and its hard-tier long_policy at 0.526
against the fast recipe's 0.211. Halving the steps and doubling the batch
bought half the pod time and cost that. The fast recipe's dev mean was
0.694 against 0.703, so the dev set predicted the direction but not the
size.

**The span head is 3x faster and not yet as good.** From the baseline it
holds 0.528 weighted at 29 ms per item — still above the only other
encoder on the board — but loses 17 points on the standard tier to its
teacher, mostly on extraction (0.92 → 0.58) and routing (0.58 → 0.25). Its
dev mean was still rising at 10000 steps (0.595 → 0.623 → 0.654), so this
is under-trained, not a ceiling. The acceptance test in `PLAN.md` — within
a point of the pair model — is not met; the pair model stays the product,
the span head ships beside it as `span-head.pt`.

Both pods' own stage two had trained to NaN (a padded-slot bug in the KL
term, fixed) and their evaluations had crashed on fp32 under
flash-attention (fixed); every span number above is from the local runs.

## The benchmark has two suites now

`jabr/classifier-benchmark` is still the only place these systems are scored on
identical rows, but it has grown a second suite and the other projects report
on it.

| suite | tasks | cases | Jev | GLiNER2 | Von 1.0.1 | Laya |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 8 | 78 | 0.974 | 0.795 | 0.923 | 0.615 |
| v2 | 49 | 869 | 0.964 | 0.688 | 0.666 | 0.585 |
| combined | 57 | 947 | **0.965** | 0.697 | 0.687 | 0.587 |

Micro accuracy. v2 is new tasks in domains none of these models was built for:
content moderation, code review, commit messages, allergens, hazmat shipping,
phishing, fair housing, veterinary triage, grammar. The drop from v1 to v2 is
the interesting number: Jev −1.0 points, Laya −3.1, GLiNER2 −10.7, **Von
−25.7**.

Our own v1 score of 0.654 was measured with a broken reader (see below), so it
is not comparable with anything, including our own earlier runs.

## Why Von's 0.923 is not what it looks like

`wfzyx/von` ships its training code, which is more than most do. Reading it
shows where the v1 score comes from. `training/prepare_decision_dataset.py`,
added 2026-09-19, generates the benchmark's own v1 tasks:

    def generate_department_triage_cases(n: int = 8000):
        criteria = {"billing": "Refunds, payments, invoices, subscriptions,
                                or unexpected charges", ...}

That criteria string is character-for-character the benchmark's
`support_department` question. The generated customer messages are the
benchmark's own cases: "I was charged twice on my card this month for the same
subscription, how do I get that fixed?", "The iOS app crashes immediately
every time I try to open a report.", "Do you offer a discount if we buy 50
seats for our team?". The frustration generator repeats the same trick with
"Quick question: does the exporter support CSV?" and "This is ABSURD. Your
garbage app destroyed three hours of work. Fix it NOW.", under the benchmark's
own instruction "How frustrated does the customer appear?" and its exact three
levels.

Two things have to be kept apart, and the public record does not settle which
applies to the measured checkpoint:

* The **model card for `wfzyx/von-1.0`** names only NLI sources: ANLI R1-3,
  WANLI, MultiNLI, SNLI, 250 000 class-balanced rows, loss CE + 0.5 × Brier,
  temperature 1.1692. On that recipe the v1 score would be honest.
* The **generator in the repository** is headed "for Von-1.1" and was
  committed the evening of 2026-09-19. The next commit, 2026-09-20 04:25,
  reads "release Option-Marker joint attention backend with 93.5 % macro
  bench".

So: the published 0.923 may predate the generator, but any Von number at or
above 0.93 from 2026-09-20 onward was produced by a model whose training set
contains the test cases. This is also the simplest explanation for a 25.7
point collapse on v2 — the generator covers v1 and nothing else.

Practical consequence for us: **do not chase v1.** Report v2 and combined.

## Our own measurement was broken

A `Score` question carries its levels as a *list* of descriptions and the
number of levels is the length of that list. `eval_cbench.py` tested for a
dictionary, found none, and fell back to three bare digits. So
`incident_severity`, which has five levels, was scored with three options —
every case labelled 3 or 4 was wrong by construction — and every level
description was hidden from the model, which saw "0", "1", "2". The 0.333 on
frustration and 0.444 on incident severity measured our reader, not the model.
Fixed 2026-09-20; all 947 gold labels are now offered.

## Two weight classes, and only one of them is ours

Everyone in this field scores options in a single forward pass and decodes
nothing: kev uses a pointer head, decider reads the letter logits, NanoJev
calls it "zero output-token decoding", LightJev takes a candidate softmax.
Not generating is the price of entry, not a difference.

What separates them is the backbone. An encoder of 300-435 M runs on a CPU and
costs milliseconds; a 2 B or 8 B decoder needs a GPU and 17 GB in bf16 to hold
the weights. Comparing across that line says nothing useful, so the table is
split.

### Our class: encoders, 300-435 M

| project | base | size | context | combined | weights | training code | training data |
|---|---|---:|---:|---:|---|---|---|
| **Jev** (TypeSafe) | undisclosed, encoder-style | — | 32768 | **0.965** | hosted only | no | no |
| **GLiNER2** (fastino) | DeBERTa-ish | 300 M | — | 0.697 | yes | no | no |
| **Von 1.0.1** (wfzyx) | ModernBERT | 395 M | 512 | 0.687 | yes | **yes** | sources named, generator in repo |
| **Laya** (convaiinnovations) | own | 421 M | 512 / 1024 | 0.587 | yes | no | no |
| **open-jev-deberta** (kotoba) | DeBERTa-v3-large | 435 M | 512 (state 256) | not run | yes | **yes** | **yes, 3 public sets** |
| **gavel** (us) | Vela-1.0-Encoder | **307 M** | **32768** | to be measured | yes | yes | yes, rebuilt from public sources |

This is the table that matters. We are the smallest model in it and the only
one above 1024 tokens of context apart from the closed one.

### The other class: a decoder LLM underneath

Useful to read, not to be measured against. Their numbers come from their own
suites, not from `jabr/classifier-benchmark`.

| project | base | size | their claim | training code | training data |
|---|---|---:|---|---|---|
| **kev** (jaredpalmer) | Qwen3-8B-Base + LoRA r=16 | 8 B | 0.863 in-distribution against Jev 0.845; 0.796 out-of-domain against 0.857 | **yes** | **yes, suites as JSONL in repo** |
| **decider** (Mapika) | Qwen3.5-2B-Base | 2 B | 32768 context, ~95 public sets | **yes** | **yes, registry in repo** |
| **NanoJev** (C-Tianyu) | Qwen3-0.6B | 0.6 B | games and navigation, not business rows | **yes** | **yes, on the Hub** |
| **LightJev** (rongxinzy) | Qwen3-0.6B | 0.6 B | synthetic domain, 3200 training questions | **yes** | **yes, CC0, hashed** |

Stars, for a sense of attention: NanoJev 1108, Laya 1803, kev 634, decider 102,
Von 49, LightJev 2, kotoba 1.

## What each one actually trained on

* **Von** — ANLI R1/R2/R3, WANLI, MultiNLI, SNLI (a 1.2 M pool cut to 250 k
  balanced), plus the generated operational tasks described above, plus
  `mteb/banking77` and `dair-ai/emotion`. Loss CE + 0.5 × Brier, post-hoc
  temperature. Trained on AWS spot with a launcher in the repository.
* **kev** — thirteen public sets: banking77, boolq, ag_news, multi_nli, sst5,
  yelp_review_full, trec, dbpedia_14, amazon_reviews_multi, imdb, ai2_arc,
  openbookqa, commonsense_qa. Plus programmatic policy pairs: 896 records over
  nine template families including four ordinal Score threshold families, and
  1680 records from 60 random rule structures with negation. LoRA r=16 plus a
  pointer head, cross-entropy over the option distribution, lr 5e-5, two
  epochs, 70 minutes on one H100.
* **decider** — about 95 public decision sets. Visible in `decider/data`:
  ai2_arc, openbookqa, prosocial-dialog, qasc, reward-bench, sciq, social_i_qa,
  winogrande, super_glue, dolly-15k, bitext customer support, mmlu, tweet_eval,
  liar2, clinc_oos, yahoo_answers, fever_gold_evidence, emotion, race, quality,
  ag_news, dbpedia_14, glaive-function-calling-v2, boolq, civil_comments,
  go_emotions, paws, xstory_cloze, arena-human-preference-55k, toxic-chat,
  ms_marco, wiki_qa, banking77, HelpSteer2, HelpSteer3, glue, multi_nli,
  medmcqa, PubMedQA, imdb, shp, snli, sst2, commonsense_qa, truthful_qa,
  measuring-hate-speech, sms_spam, piqa, twitter-financial-news-sentiment.
  Eight supervised stages plus teacher-written questions and contrastive
  teacher labels; `scripts/train.sh full` reproduces it.
* **kotoba** — three sets only: banking77, sst5, boolq, turned into
  choice/score/noul by hand-written question templates. Public gold labels
  only, no teacher. Their measured lesson: 0.854 on question wordings seen in
  training against 0.690 on unseen wording.
* **NanoJev** — its own programmatic corpus, published as
  `C-Tianyu/NanoJev-Data`. Mazes, navigation, arcade games, workflow decisions.
  A research direction, not a business classifier.
* **LightJev** — 3200 training questions converted from NanoJev stage1, CC0,
  every hash recorded. Honest about being a synthetic-domain prototype.

## Size is ours, speed is a choice between two products

Small is measured and real: 308 M against Von's 395 M, Laya's 421 M,
kotoba's 435 M, and against 2 B or 8 B in the other class.

Speed was measured on 2026-09-20 with `scripts/bench_latency.py`, on the
workstation CPU with 8 threads, median of five calls after a warm-up, the
published `chukfinley/gavel-vela-32k`:

| state | 2 options | 5 options | 8 options |
|---:|---:|---:|---:|
| 64 tokens | 109 ms | 240 ms | 377 ms |
| 256 tokens | 301 ms | 730 ms | 1297 ms |
| 1024 tokens | 1369 ms | 3097 ms | — |

Two things are visible and they point in different directions.

**The option loop is real.** `decide` is a single call to the model, but it
builds `[state] * len(options)`, so the batch holds the state once per option.
On a CPU nothing hides that: each extra option costs about 45 ms at 64 tokens
and about 165 ms at 256. A single-sequence span head, of the kind kotoba got
to train, removes exactly this factor.

**The option loop is not the binding limit.** 308 M parameters over 1024
tokens is roughly 0.6 TFLOP per pass, and a desktop CPU delivers 200-400
GFLOPS in practice. One to three seconds is the floor for a long state no
matter how the options are arranged. Removing the loop turns 3097 ms into
about 1400 ms, not into 100 ms.

So "runs beside your application on a CPU, under 100 ms" and "reads a
30 000-token document" are two different products:

* **Short states on a CPU.** Subject lines, chat messages, commands, tweets —
  64 to 128 tokens. Under 100 ms is reachable, and since these questions
  usually carry five to eight categories, the span head is worth building:
  it is the difference between 377 ms and roughly 120 ms.
* **Long documents.** This is the position nobody else holds openly, and Jev
  only holds it as a hosted service. It is a GPU deployment, where the batch
  runs in parallel and the option count is nearly free.

Claiming both at once does not survive anyone re-measuring it.

## What is worth taking

1. **kev's evaluation discipline.** Development partitions select, a locked
   test partition is read at most once per candidate, every number carries a
   suite hash and a git commit, and `PLAN.md` lists the corrections they made
   to their own earlier claims. We already report per stratum; the locked
   partition and the hashes we do not have.
2. **kev's finding on learning rate.** Fine-tuning erodes the base model's own
   ability, and lr is the control: at 4 B, the default 2e-4 trained MMLU down
   from 0.688 to 0.60-0.66, and 5e-5 recovered most of it. We run 1.5e-5, which
   is on the safe side of that.
3. **Programmatic policy pairs with ordinal thresholds** (kev) — nine template
   families, rule structures with negation anywhere. This is the shape our
   ordered-scale work is missing: not more levels, but *rules over* levels.
4. **decider's breadth.** About 95 public sets against our roughly 30. Their
   registry is a shopping list we can read directly.
5. **Teacher-written questions** (decider) — the questions are generated, the
   labels stay public gold. Cheaper than distilling logits and it attacks the
   wording-generalisation gap kotoba measured.

## What this means for our position

The 32 k context is still unique in our class. `Mapika/decider-2b` also takes
32768 tokens, but it needs a 2 B decoder to do it, which is a different
product: a GPU service, not something that runs beside an application on a
CPU. Against the models a buyer would actually compare us with — Von at 512,
Laya at 1024, kotoba at 512 with the state cut to 256, GLiNER2 with no long
context at all — we are the only open model that reads a 30 k-token document,
and the smallest model in the class while doing it.

The open ground is v2. Von collapses there because it trained the v1 tasks;
GLiNER2 drops 10.7 points because it is keyword-driven; Laya is low
everywhere. A model trained on a wide public mix should lose less across a
domain shift than any of them. That is the number to aim at, and it is the one
nobody except Jev currently holds.

## What the Jev demos in the field actually do (Theo, "Jev is incredible", 2026-09)

The demos that made the model known are not vision and not text
generation. Every one of them hands the model *structured program state
as text* plus a fixed set of options, and reads back one choice:

* iOS simulator and website navigation: the accessibility tree or the
  page HTML is the state, the tappable elements or links are the options.
  "Once it has the ability to see" is explicitly named as missing.
* Minecraft and board games: the game state as data, the legal moves as
  options. "Not an image because it doesn't have vision."
* Classification of 32311 chat messages, incident routing with eleven
  states, a "smart if statement" between agent steps.
* Numbers quoted: 70–500 ms per decision, P95 240 ms at 38 decisions/s,
  4 cents per million input tokens, output free, 32k context.

That is the same contract as `Gavel.decide`. Two demos of it live in this
repo now: `scripts/demo_snake.py` (safety questions per legal move, the
same wording as the training rows) and `scripts/demo_webnav.py` (HTML in,
link out, four hops). Their numbers are in `results/demo_*.json`.

Latency on an RTX 3060 (`results/latency_cuda.json`, 2026-09-21), median
of seven, pair model versus span head, and the span head with four
questions on one state:

| state tokens | options | pair ms | span ms | span, 4 questions |
|---|---|---|---|---|
| 64 | 8 | 20.6 | 16.4 | 17.3 |
| 256 | 8 | 56.8 | 18.1 | 17.8 |
| 1024 | 8 | 196.8 | 33.0 | 33.0 |
| 4096 | 2 | 255.6 | 105.2 | 107.0 |
| 4096 | 8 | 1263.1 | 107.3 | 115.2 |

The span head's cost is the state, not the options or the questions. That
is the property the one-sequence model was built for; the accuracy gap to
the pair model is the open question the 40k-step distillation answers.

## Bespoke-Nimble-9B (added 2026-09-21)

A LoRA adapter (165 MiB) on Qwen3.5-9B: booleans, enums and rubric
score levels read off the allowed answer tokens, no free text, prompt
capped at 2048 tokens, Apache 2.0. Decoder weight class, so it sits with
kev and decider, not with us. The card names no benchmark numbers and no
training sets; the suite is the 13 manifests in `bespokelabsai/nimble`,
still not wired here. Note the 2048-token cap against our 32768.
