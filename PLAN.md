# What to build next, in order, and why

Written 2026-09-20 after reading every competing repository (`FIELD.md`) and
measuring our own latency (`scripts/bench_latency.py`). Each item says what
changes, what it should buy, and how we will know.

## How the others get their speed

Nobody in this field is fast because of a clever kernel. They are fast for
three reasons, in this order:

1. **One sequence, not one per option.** Von puts every option in the same
   sequence behind marker tokens and reads the state once. kotoba does the
   same with `[CLS] [STATE] state [Q] instructions [OPT] a [OPT] b`, scoring
   each option from `[mean(question tokens); mean(option tokens); product]`
   and taking a softmax inside each question's group. decider writes the
   options into a prompt as `(A) … (B) …` and reads the letter logits. We are
   the only one that runs the state through the encoder once per option.
2. **They read very little.** Von 512 tokens, Laya 512/1024, kotoba 512 with
   the state cut to 256. Speed at 55 ms is partly speed and partly refusing
   to read a long document. Our 32 768 is why we are slower and it is also
   the thing they cannot do.
3. **Many questions on one reading.** kotoba answers ten questions about one
   state in 28 ms end-to-end on an H100 — 518 questions per second at batch
   8 — because the state is encoded once for all of them. This is the
   throughput argument TypeSafe sells, and it is worth more than the option
   loop: a support ticket usually needs intent *and* urgency *and* language
   *and* a refund check, which is four readings for us and one for them.

decider adds engineering on top — shape-bucketed CUDA graphs, FP8 weights, a
batching server, and a path that answers one question with 151 options at 19x
the full-forward rate. That is worth copying later, not now.

## Measured on 2026-09-20, before any of this

JevBench public items, pair scorer, `chukfinley/gavel-vela-32k`:

| tier | ours | open-jev-deberta (435 M) | decider-2b | Jev |
|---|---:|---:|---:|---:|
| easy (48) | 0.958 | 1.000 | 1.000 | 1.000 |
| standard (72) | **0.514** | 0.431 | 0.847 | 0.986 |
| hard (111) | running | 0.378 | 0.459 | 0.730 |

We already beat the only other encoder on that board on the standard tier,
at 308 M against its 435 M. Per family, standard tier:

    ordinal 0.750 · intent 0.667 · policy 0.583 · adequacy 0.500 ·
    extraction 0.417 · routing 0.167

Ordinal was 0.333 before the evaluator was fixed, so that hole was the
reader, not the model. Routing at 2 of 12 is the real one, and item 2b below
is the answer to it.

## 1. The span head, taught by the model we already have

**Change.** A second head on the same Vela-32k backbone that reads
`[CLS] [STATE] state [Q] question [OPT] option [OPT] option … [SEP]` and
scores every option in one pass, pooled over the option span rather than off
a single marker token. Then extend it to several `[Q]` groups in one
sequence, with a softmax inside each group.

**Why it should work this time.** Our own note says a fresh marker-token head
does not train while a span head does, and kotoba shipped a working span
head. The failure was the pooling, not the idea. And we no longer have to
train it from hard labels: the cross-encoder we already trained is a correct,
calibrated teacher for exactly this task, so the span head learns from its
probability distributions. Distilling a model into a cheaper shape of itself
is a much easier problem than learning the task from scratch, and it is the
same trick the field uses on us.

**What it buys.** At 64 tokens, eight options: about 377 ms today, about
120 ms after. Multi-question turns four readings of a ticket into one. The
cross-encoder stays as the accuracy reference and for the long-document path.

**How we know.** Same rows, same machine, `scripts/bench_latency.py` plus the
benchmark: the span head must stay within one point of the cross-encoder on
v2 while cutting the 8-option time by at least half. If it loses more than
that, it does not ship and the cross-encoder remains the product.

## 2. Datasets for the domains v2 actually tests

v2 is 49 tasks and 869 cases, and it is where every open model falls over.
It is not harder, it is *elsewhere*. Our mix has roughly 30 sources and none
of these domains:

| v2 domain | public source to build from |
|---|---|
| grammar errors | `nyu-mll/glue` CoLA, JFLEG, W&I+LOCNESS |
| commit messages | CommitBench, commitpack |
| code review comments | CodeReviewer, `google/code_x_glue_cc_code_refinement` |
| PII detection | `ai4privacy/pii-masking-200k` |
| SQL injection | `b-mc2/sql-create-context` negatives, SQLi corpora |
| phishing | `ealvaradob/phishing-dataset` |
| formality | GYAFC, Pavlick formality scores |
| reading level | OneStopEnglish, CLEAR corpus |
| spoilers | `ucsd/goodreads-spoilers`, TV Tropes |
| recipes, allergens, diet | RecipeNLG, `food.com` recipes |
| voice assistant intents | SLURP, `AmazonScience/massive` (we have MASSIVE) |
| symptom and vet triage | `openlifescienceai/medmcqa` (have), symptom checkers |
| contract clauses | CUAD, LEDGAR |
| insurance and claims | — none found, generate from rules |
| weather alerts, returns, delivery | — none found, generate from rules |

And the sets decider uses that we simply do not have, which are cheap because
they are one loader each: `allenai/social_i_qa`, `allenai/reward-bench`,
`clinc/clinc_oos`, `copenlu/fever_gold_evidence`,
`google-research-datasets/go_emotions`, `google/civil_comments`,
`truthfulqa/truthful_qa`, `ucberkeley-dlab/measuring-hate-speech`,
`ybisk/piqa`, `stanfordnlp/shp`, `nvidia/HelpSteer2`, `juletxara/xstory_cloze`,
`zeroshot/twitter-financial-news-sentiment`, `chengxuphd/liar2`,
`microsoft/wiki_qa`, `allenai/openbookqa`, `allenai/ai2_arc`, `CogComp/trec`.

**How we know.** v2 micro, per task, with its denominator. The target is to
beat GLiNER2's 0.688 and Von's 0.666 there, which is the only honest
comparison left after what Von's generator does to v1.

## 2b. Routing as a rule over overlapping categories — done, untested

`build_routing.py` exists now. The JevBench routing items are short requests
over six named specialists where the tie-break lives in the instruction
("file edits with test execution use the agent, even if it is code-related").
Our mix only had tier routing, which is a different task. The new rows carry
overlapping categories and a rule that separates them; the categories and
requests are written for the file, not copied from any suite.

Untested until a GPU run finishes. The number to watch is the routing family
on JevBench standard, currently 0.167.

## 3. Rules over ordered levels, not more levels

kev trains 896 programmatic policy records over nine template families,
including four ordinal Score threshold families, plus 1680 records from 60
random rule structures with negation anywhere. They report both-correct 0.85
to 1.0 on trained structures and 0.5 to 0.6 on unseen ones at 4 B, which is
honest and still far better than nothing.

Our `build_scales.py` teaches what a level *is*. It does not teach "severity
is at least 3 when data is lost and no workaround exists", which is what an
incident-severity question actually asks. That is a rule over a scale, and it
is generated, not collected.

## 4. Several questions per state, end to end

Once the span head takes multiple `[Q]` groups, the API has to as well:
`decide_many(state, questions)` returning one distribution per question. This
is the only piece of everyone else's design we have never taken, and after
item 1 it is nearly free.

## 5. The cheap ones, kept on the list

* **Retrieved evidence.** `build_grounded.py` still stalls on a pure-Python
  BM25; `rank_bm25` fixes it. Retrieval alone lifted MMLU 0.293 → 0.367 on a
  model that had never seen an evidence block.
* **The unreadable-input test.** Feed a script the model has never seen and
  check that confidence drops. Laya measured 0.000 accuracy at 0.952
  confidence on their English checkpoint. If ours does the same, the abstain
  mechanism is worthless where it matters.
* **A locked test partition**, kev-style: development selects, the locked
  split is read once per candidate, every number carries a suite hash and a
  commit. We report per stratum already; we do not have the lock.
* **bge-m3 as a teacher**, not as a backbone. It has 8192 tokens, so it
  cannot replace Vela, but it is trained on millions of label-as-hypothesis
  pairs and can label ours.

## What we are not doing

* **Not chasing v1.** Von's training repository generates its cases; the
  score there means nothing now.
* **Not changing the backbone.** Vela-1.0-Encoder-307M at 32 768 tokens is
  the product. Nothing else open in this size class reads that far.
* **Not competing with kev or decider on accuracy.** 8 B and 2 B decoders are
  a different deployment and a different price.
