# Where this stands, 2026-09-19

## What exists

* **Code**: <https://github.com/chukfinley/gavel>
* **Models**: <https://huggingface.co/chukfinley/gavel-base> (150 M, 8k context)
  and <https://huggingface.co/chukfinley/gavel-vela-32k> (307 M, 32k context,
  multilingual — the better one)
* **Every result file and log**: <https://huggingface.co/datasets/chukfinley/gavel-runs>

## The model in one paragraph

A state and a list of options go in, one typed answer and a calibrated
probability come out, and nothing is generated. Each option is turned into a
hypothesis about the state and scored by a three-way entailment head; a softmax
over the options is the decision. Because every option is scored in its own
pass, the order of the options cannot change the result, and because the option
text is part of the input, a new category needs no retraining.

## What was measured

| | ModernBERT-base 150 M | Vela 307 M, 32k |
|---|---:|---:|
| Development strata (mean of 13) | 0.648 | 0.647 |
| General test set | 0.574 | 0.569 |
| Ten languages | 0.559 | **0.781** |
| Question banks | 0.682 | **0.756** |
| Tool selection | **0.993** | 0.990 |
| Jailbreak / fact check | — | 0.806 |
| Browser actions | **0.677** | 0.625 |
| Calibration error after fitting | 0.024 | 0.047 |

Against the published references: WANLI 0.714 beats OpenJev's frozen
Qwen3.5-4B (0.637). Prompt injection 0.980, Nvidia safety policies 0.935,
SciQ with passage 0.980. The business fixture from TypeSafe stays at 0.409
against their published 0.845, and LongBench-v2 sits at 0.30 where chance is
0.25.

## What was learned, in order of how much it cost to learn it

1. **A marker head over options in one sequence does not train.** Cross entropy
   stays at ln(number of options) for thousands of steps. Span pooling and the
   standard multiple-choice head fail the same way. The entailment framing
   trains from the first hundred steps. The kotoba group measured the same
   thing independently.
2. **Selecting on pooled accuracy hides a trade.** Run 3 raised pooled
   development accuracy from 0.76 to 0.80 while WANLI fell 0.697 to 0.612.
   Selection is on the mean over strata now.
3. **Truncation looks like incompetence.** The business fixture has a median
   state of 2576 tokens; measuring it at 256 gave 0.353, at 1024 gave 0.471.
   Always check the length distribution before believing a low score.
4. **The gap is reading, not size.** SciQ with its passage: 0.980. MedMCQA
   without one: 0.282. Retrieval lifted MMLU from 0.293 to 0.367 with no
   training at all.
5. **More steps on the same data stop helping.** Stage 3 added 30000 steps and
   moved the mean by 0.001; it only shifted weight between strata.

## What is half finished

* `scripts/build_grounded.py` was running when the credit ran out, at 5000 of
  40000 rows. Pure-Python BM25 over 120000 passages is too slow; it needs
  `rank_bm25` or an inverted index in a database before it is rerun.
* Stage 4 (`/root/stage4.sh` on the pod, lost with it) was to train on those
  grounded rows at 512 tokens. That is the first thing to do next.
* `src/typedec/longdoc.py` decides over documents longer than the window by
  scoring overlapping pieces. It works (10379 tokens, nine windows, correct
  answer) but was never trained for, and never measured at scale.

## The three levers that are worth the next day of work

1. **Train with retrieved evidence.** The model has never seen an evidence
   block, so it has no reason to trust one. Finish `build_grounded.py`, train
   stage 4, and measure MMLU and MedMCQA with and without retrieval.
2. **Distil from the 4 B.** Qwen3.5-4B reaches 0.708 on the general test where
   this model reaches 0.580. Training on its probability distributions rather
   than on hard labels usually recovers half of such a gap.
3. **Answer several questions in one pass.** The kotoba head reads one state
   and many questions at once; this model needs one pass per option. That is
   the throughput argument Jev sells, and it is the one piece of their design
   not taken.

## Repeating this

```bash
export RUNPOD_API_KEY=... HF_TOKEN=...
python scripts/runpod_launch.py start --gpu "NVIDIA GeForce RTX 3090"
```

The pod clones this repository, rebuilds every dataset from public sources,
trains, and publishes logs, results and finished models to Hugging Face by
itself. Nothing large travels over the home connection. One run of everything
measured here cost about two dollars on a rented consumer card.

Local note: the workstation's GPU is unusable until it reboots — the
`nvidia-driver-610` package was upgraded to 610.57.04 while the loaded kernel
module is still 610.43.02.
