# typedec — typed decisions without a generation loop

A decision model that answers "which of these options" in **one forward pass**.
No tokens are generated. The answer space arrives with the request, thus the
options can change on every call.

## Why not an LLM

TypeSafe's Jev is described as an encoder-style representation with a parallel
classification head over three primitives — `Choice` (up to 255 options),
`Score` (2 to 10 tiers) and `Noul` (binary) — trained with reinforcement
learning on proper scoring rules (Brier) for calibrated probabilities.

[OpenJev](https://github.com/TheoLeeCJ/openjev) reproduces the *interface* with
a frozen Qwen3.5-4B: it reads the logits of the tokens "A", "B", "C" after one
forward pass. That works, but it uses a 4-billion-parameter decoder, trained
for autoregression, to pick between two options.

This project trains the shape of the task directly:

```
[CLS] Question: <criterion>
      Situation: <state>
      Options: [OPT] first option  [OPT] second option  ... [SEP]
```

The hidden state at each `[OPT]` marker goes through one shared scoring head,
which gives one logit for each option, then a softmax. One pass, any number of
options, option text read at run time.

## Design decisions

* **Backbone:** ModernBERT (8192 native context, RoPE, alternating local and
  global attention). A second run uses a small decoder with its language-model
  head removed, to measure whether world knowledge is what the encoder lacks.
* **Loss:** cross entropy plus Brier. Cross entropy alone gives a sharp argmax
  and a meaningless confidence. The Brier term is a proper scoring rule, thus
  the probability can be used as a threshold.
* **Option order is shuffled in every batch.** OpenJev reports position
  sensitivity as a known failure; this removes the cause during training.
* **No fixed classes.** Intent data is converted with a *sampled* option set
  for each row, thus the same utterance appears with different candidates and
  the model must read the option text.

## Data

Public and human-labelled: MNLI, WANLI (train split only), ANLI-R3, BoolQ,
ARC, CommonsenseQA, Banking77 (sampled option sets), Yelp (ordered tiers).
166875 train rows, 3405 dev rows.

The 256 WANLI rows that OpenJev evaluates on come from the WANLI **test**
split. Only the train split is used here, thus that number stays clean.

## Evaluation

Two levels:

1. `scripts/eval_openjev.py` — the four OpenJev fixtures, with their published
   Qwen3.5-4B numbers as the line to beat: authored144 0.813, WANLI256 0.637,
   TypeSafe102 0.845, judgment grid 0.806.
2. `scripts/eval_general.py` — a wider held-out test set, 3600 rows in nine
   strata (ANLI-R1/R2, SNLI, BoolQ, ARC-Easy, OpenBookQA, CLINC, SST-5, MMLU).
   The OpenJev fixtures are 144, 256, 102 and 36 rows; a three-point difference
   there is four rows. Accuracy is never pooled across strata.

## Run

```bash
uv venv && uv pip install -e .
python scripts/build_data.py --per-source 40000 --out data
python scripts/build_testset.py --per-source 400
python scripts/train.py --backbone answerdotai/ModernBERT-base --out runs/mbert-base
python scripts/eval_general.py --checkpoint runs/mbert-base/best.pt
```
