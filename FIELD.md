# The field, as measured on 2026-09-20

Everything here was read from the project's own repository, model card or
published result file on 2026-09-20. Where a number is someone else's claim it
says so. This file replaces guessing about the competition; update it, do not
re-derive it.

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

## Everyone who has shipped something

| project | base | size | context | benchmark | weights | training code | training data |
|---|---|---:|---:|---|---|---|---|
| **Jev** (TypeSafe) | undisclosed | — | 32768 | 0.965 combined | hosted only | no | no |
| **Von 1.0.1** (wfzyx) | ModernBERT | 395 M | 512 | 0.687 combined | yes | **yes** | sources named, generator in repo |
| **GLiNER2** (fastino) | DeBERTa-ish | 300 M | — | 0.697 combined | yes | no | no |
| **Laya** (convaiinnovations) | own | 421 M | 512 / 1024 | 0.587 combined | yes | no | no |
| **kev** (jaredpalmer) | Qwen3-8B-Base + LoRA | 8 B | — | not run | adapter | **yes** | **yes, suites as JSONL in repo** |
| **decider** (Mapika) | Qwen3.5-2B-Base | 2 B | **32768** | not run | yes | **yes** | **yes, ~95 public sets + registry** |
| **NanoJev** (C-Tianyu) | Qwen3-0.6B | 0.6 B | — | not run | yes | **yes** | **yes, on the Hub** |
| **LightJev** (rongxinzy) | Qwen3-0.6B | 0.6 B | 256 | not run | yes | **yes** | **yes, CC0, hashed** |
| **open-jev-deberta** (kotoba) | DeBERTa-v3-large | 435 M | 512 (state 256) | not run | yes | **yes** | **yes, 3 public sets** |
| **cua-s1-forms** (trycua) | — | small | — | not run | yes | partly | no |
| **gavel** (us) | Vela-1.0-Encoder | 307 M | **32768** | to be re-measured | yes | yes | yes, rebuilt from public sources |

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

The 32 k context is no longer unique: `Mapika/decider-2b` accepts 32768 tokens
of state and questions. It does so with a 2 B decoder. We do it with a 307 M
encoder, so the claim narrows from "the only one" to "the smallest one, by
6.5x, and the only encoder" — still worth having, and still true against Von
(512), Laya (1024) and kotoba (512, state cut to 256).

The open ground is v2. Von collapses there because it trained the v1 tasks;
GLiNER2 drops 10.7 points because it is keyword-driven; Laya is low
everywhere. A model trained on a wide public mix should lose less across a
domain shift than any of them. That is the number to aim at, and it is the one
nobody except Jev currently holds.
