# What the field has already worked out, and what it cost us to relearn

> Read `FIELD.md` first for the current state of the competition. The table
> below is the v1 suite only, and the benchmark now has a v2 suite of 49 tasks
> and 869 cases on which the other projects also report. Our 0.654 in that
> table was produced by a reader that mis-parsed ordered scales; it is not a
> valid number.

Everything here was measured, or read from a source that measured it. Where a
number comes from someone else it is marked.

## The independent benchmark

`jabr/classifier-benchmark` (8 tasks, 78 cases, gold labels, synthetic but
cross-checked by several models) is the only place where these systems are
scored on identical rows.

| model | size | micro | macro | context |
|---|---|---:|---:|---:|
| Jev (TypeSafe, hosted) | undisclosed | 0.974 | 0.972 | 32768 |
| Von 1.0.1 | 395 M | 0.923 | 0.930 | 512 |
| GLiNER2 | 300 M | 0.795 | 0.785 | — |
| **gavel-vela-32k** | 307 M | **0.654** | **0.648** | **32768** |
| Laya | 421 M | 0.615 | 0.619 | 512 / 1024 |

Our per-task result says where the gap is:

| task | ours |
|---|---:|
| review_sentiment (score, 5) | 1.000 |
| urgency (noul) | 0.750 |
| support_department (choice, 5) | 0.733 |
| email_intent (choice, 5) | 0.700 |
| secret_leak (noul) | 0.625 |
| refund_eligible (noul) | 0.600 |
| incident_severity (score, 5) | 0.444 |
| frustration_level (score, 3) | 0.333 |

Ordered scales are the hole. Almost every row this model trained on is a
`choice`; ordered levels appear only as review stars. "How frustrated is this
customer, 0 to 2" was never trained, and it shows.

The second cause is format. Their questions carry a *named criterion per
option* (`"billing": "Refunds, payments, invoices…"`). Von and Laya train on
that shape; we glue the two together into one option string at inference.

## What the other projects did, and what is worth taking

* **Jev (TypeSafe, closed).** Encoder-style representation, parallel
  classification head, three primitives (Choice up to 255, Score 2-10, Noul),
  trained with reinforcement learning on proper scoring rules. Their own
  workflow benchmark puts them at 67.8 % against 74.1 % for the best frontier
  model — and the reference answers there *are* frontier models, so that
  number measures agreement, not truth.
* **OpenJev (TheoLeeCJ).** Frozen Qwen3.5-4B, reads the logits of the tokens
  "A", "B", "C". No training at all, and it lands within four points of Jev on
  the aligned TypeSafe rows. Taken from them: the metric contract — never pool
  sources, report every stratum with its denominator.
* **openjev (AlexWortega).** The same entailment framing as ours on a 4 B
  decoder. Their published MMLU rerank numbers (4 B 0.474, 0.8 B 0.350) show
  the framing itself caps knowledge tasks: the same 4 B model scores 0.688 with
  letter logits.
* **open-jev-deberta (kotoba-lang).** Measured the gap that matters for a
  decision model: 0.854 on question types it trained on against 0.690 on
  unseen wording. Taken from them: question augmentation, and their corpus.
  Their report also confirms our own failure — "a fresh marker-token head does
  not learn; the span head does".
* **Laya (convaiinnovations).** 100+ languages, a router that picks the
  checkpoint from the script of the input *before* the forward pass, because
  an English checkpoint fed Khmer scores 0.000 accuracy at 0.952 confidence.
  Calibration does not warn you when the model cannot read the input at all.
  We have never tested that failure mode.
* **Von (wfzyx).** 395 M, runs on CPU, single-pass option-marker attention.
  Its author states only one of six planned epochs is done. Context 512.
* **MoritzLaurer's zeroshot collection.** The prior art for this entire
  approach, years older than any of the above: `deberta-v3-large-zeroshot-v2.0`,
  `bge-m3-zeroshot-v2.0` (multilingual, 8192), `mDeBERTa-v3-base-xnli-2mil7`
  (862 k downloads). These are already trained on millions of
  label-as-hypothesis pairs. Starting a run from one of them instead of from a
  raw masked-language checkpoint is the largest untried lever we have.

## What this project measured for itself

1. **A marker head over options in one sequence does not train.** Cross entropy
   sits at ln(number of options) for thousands of steps; span pooling and the
   standard multiple-choice head fail the same way. The entailment framing
   trains from the first hundred steps. Two other projects hit the same wall.
2. **Selecting on pooled accuracy hides a trade.** Pooled development accuracy
   rose 0.76 → 0.80 while the WANLI fixture fell 0.697 → 0.612. Selection is on
   the mean over strata now.
3. **Check the length distribution before believing a low score.** The TypeSafe
   fixture has a median state of 2576 tokens: 0.353 at 256 tokens, 0.471 at
   1024. The same mistake in reverse produced a "below chance" LongBench
   result that was only a truncated evaluation.
4. **The gap is reading, not size.** SciQ with its passage 0.980; MedMCQA
   without one 0.282. Retrieval alone lifted MMLU from 0.293 to 0.367 on a
   model that had never seen an evidence block.
5. **More steps on the same data stop helping.** 30000 extra steps moved the
   mean by 0.001 and only shifted weight between strata.
6. **A 32 k window is not 32 k of reasoning.** The window is real and the
   YaRN extension holds, but LongBench-v2 sits at 0.30 where chance is 0.25.
   Finding one line in a structured document works (0.815); multi-hop
   reasoning over 16 k tokens of prose does not.
7. **Truncation was not the cause of the weak sectors.** Code defect detection
   at 256 vs 1024 tokens: 0.587 vs 0.600. Those tasks are simply hard.

## What to do next, in order

1. **Start from a zero-shot NLI checkpoint**, not from a raw encoder. This is
   the cheapest large gain available and it is a one-line change of
   `--backbone`.
2. **Train ordered scales as their own task type**, with named criteria per
   level, in the shape the benchmark uses. Our worst two tasks are both this.
3. **Train with retrieved evidence** (`scripts/build_grounded.py`, half
   finished; it needs `rank_bm25` instead of the pure-Python index).
4. **Test the unreadable-input failure mode.** Feed the model a language it has
   never seen and check whether confidence drops. If it does not, the abstain
   mechanism is worthless exactly where it is needed.
5. **Answer several questions in one pass.** The throughput argument, and the
   one piece of everyone else's design we have not taken.
