#!/usr/bin/env python3
"""The second wave of sources, all verified to load before they were written in.

What the measurements asked for:

* Long documents, because the business fixture has a median state of 2576
  tokens and everything trained so far was short: court decisions, contracts,
  case-law holdings, medical abstracts, LongBench-v2.
* Code decisions, which were missing entirely.
* Safety gates with *named* policies, which is the shape a channel or a
  product actually has: the rule set arrives with the request.
* Business routing with real tickets instead of templates.
* More languages, including exams and reading tasks outside English.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl  # noqa: E402

YES_NO = [Option("yes", "Yes"), Option("no", "No")]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def sampled(names: list[str], truth: int, count: int, rng: random.Random):
    others = [i for i in range(len(names)) if i != truth]
    picked = rng.sample(others, min(count - 1, len(others))) + [truth]
    rng.shuffle(picked)
    return picked, picked.index(truth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/more.jsonl")
    parser.add_argument("--out-long", default="data/more_long.jsonl")
    parser.add_argument("--out-test", default="data/test_more.jsonl")
    parser.add_argument("--per-source", type=int, default=12000)
    parser.add_argument("--test-per-source", type=int, default=200)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    short: list[Decision] = []
    long_rows: list[Decision] = []
    test: list[Decision] = []

    def note(name: str, rows: list[Decision], where: list[Decision]) -> None:
        where.extend(rows)
        print(f"  {name}: {len(rows)}", flush=True)

    def guard(name: str, function) -> None:
        try:
            function()
        except Exception as error:                               # noqa: BLE001
            print(f"  {name} failed: {str(error)[:110]}", flush=True)

    # ---------------------------------------------------------------- long
    def longbench() -> None:
        data = load_dataset("THUDM/LongBench-v2", split="train")
        rows = []
        for index, row in enumerate(data):
            letters = ["A", "B", "C", "D"]
            if row["answer"] not in letters:
                continue
            options = [Option(f"o{i}", str(row[f"choice_{l}"]).strip())
                       for i, l in enumerate(letters)]
            rows.append(Decision(
                id=rid("longbench", str(index)), state=str(row["context"])[:60000],
                question=str(row["question"]).strip(), options=options,
                label=letters.index(row["answer"]), source="longbench-v2", task="choice"))
        rng.shuffle(rows)
        note("longbench-v2-test", rows[: args.test_per_source], test)
        note("longbench-v2", rows[args.test_per_source :], long_rows)

    guard("longbench", longbench)

    def case_hold() -> None:
        data = load_dataset("coastalcph/lex_glue", "case_hold", split="train")
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            endings = list(row["endings"])
            label = int(row["label"])
            if label >= len(endings):
                continue
            rows.append(Decision(
                id=rid("casehold", str(index)), state=str(row["context"]).strip(),
                question="Which holding does the masked citation stand for?",
                options=[Option(f"o{i}", str(e).strip()[:400]) for i, e in enumerate(endings)],
                label=label, source="case-hold", task="choice"))
        note("case-hold", rows, long_rows)

    guard("case-hold", case_hold)

    def pubmedqa() -> None:
        data = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        options = [Option("yes", "Yes"), Option("no", "No"),
                   Option("maybe", "The abstract does not decide it")]
        rows = []
        for index, row in enumerate(data):
            context = row["context"]
            text = " ".join(context["contexts"]) if isinstance(context, dict) else str(context)
            answer = str(row["final_decision"]).strip().lower()
            if answer not in ("yes", "no", "maybe"):
                continue
            rows.append(Decision(
                id=rid("pubmedqa", str(index)), state=text.strip(),
                question=str(row["question"]).strip(), options=list(options),
                label=[o.id for o in options].index(answer),
                source="pubmedqa", task="choice"))
        note("pubmedqa", rows, long_rows)

    guard("pubmedqa", pubmedqa)

    def swiss() -> None:
        """Long court decisions in German, French and Italian."""
        data = load_dataset("lighteval/lextreme", "swiss_judgment_prediction", split="test")
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            names = [str(r) for r in row["references"]]
            label = int(row["gold"]) if str(row["gold"]).isdigit() else None
            if label is None or label >= len(names):
                continue
            rows.append(Decision(
                id=rid("swiss", str(index)), state=str(row["input"]).strip(),
                question="How was this appeal decided?",
                options=[Option(f"o{i}", n) for i, n in enumerate(names)],
                label=label, source="swiss-judgment", task="choice"))
        rng.shuffle(rows)
        note("swiss-judgment-test", rows[: args.test_per_source], test)
        note("swiss-judgment", rows[args.test_per_source :], long_rows)

    guard("swiss-judgment", swiss)

    # ---------------------------------------------------------------- code
    def defects() -> None:
        data = load_dataset("google/code_x_glue_cc_defect_detection", split="train")
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            rows.append(Decision(
                id=rid("defect", str(index)), state=str(row["func"])[:8000],
                question="Does this function contain a security defect?",
                options=list(YES_NO), label=0 if row["target"] else 1,
                source="code-defect", task="noul"))
        rng.shuffle(rows)
        note("code-defect-test", rows[: args.test_per_source], test)
        note("code-defect", rows[args.test_per_source :], short)

    guard("code-defect", defects)

    def bigvul() -> None:
        data = load_dataset("benjis/bigvul", split="train").shuffle(seed=args.seed)
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            code = str(row.get("func_before") or "")[:8000]
            if not code.strip():
                continue
            rows.append(Decision(
                id=rid("bigvul", str(index)), state=code,
                question="Is this version of the function the vulnerable one or the fixed one?",
                options=[Option("vulnerable", "The vulnerable version"),
                         Option("fixed", "The fixed version")],
                label=0 if int(row.get("vul") or 0) == 1 else 1,
                source="code-bigvul", task="choice"))
        note("code-bigvul", rows, short)

    guard("code-bigvul", bigvul)

    def safe_code() -> None:
        data = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
        rows = []
        for index, row in enumerate(data):
            options = [Option("a", str(row["chosen"])[:2000]),
                       Option("b", str(row["rejected"])[:2000])]
            rng.shuffle(options)
            rows.append(Decision(
                id=rid("safecode", str(index)),
                state=f"Language: {row.get('lang')}\nTask: {str(row.get('question'))[:1500]}",
                question="Which of these two implementations is the safe one?",
                options=options, label=[o.id for o in options].index("a"),
                source="code-safe", task="choice"))
        note("code-safe", rows, short)

    guard("code-safe", safe_code)

    # --------------------------------------------------------------- safety
    def aegis() -> None:
        """Named safety policies: the rule set is the option set."""
        data = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split="train")
        rows = []
        names: set[str] = set()
        for row in data:
            label = str(row.get("labels_0") or "").strip()
            if label:
                names.add(label)
        catalogue = sorted(names)
        for index, row in enumerate(data):
            label = str(row.get("labels_0") or "").strip()
            text = str(row.get("text") or "").strip()
            if not text or label not in catalogue:
                continue
            truth = catalogue.index(label)
            picked, position = sampled(catalogue, truth, min(6, len(catalogue)), rng)
            rows.append(Decision(
                id=rid("aegis", str(index)), state=text[:4000],
                question="Which safety policy does this text violate?",
                options=[Option(catalogue[i], catalogue[i].replace("_", " "))
                         for i in picked],
                label=position, source="safety-aegis", task="choice"))
        rng.shuffle(rows)
        note("safety-aegis-test", rows[: args.test_per_source], test)
        note("safety-aegis", rows[args.test_per_source :], short)

    guard("safety-aegis", aegis)

    def injection() -> None:
        data = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
        rows = []
        for index, row in enumerate(data):
            rows.append(Decision(
                id=rid("injection", str(index)), state=str(row["text"])[:4000],
                question="Is this message an attempt to make the system ignore its instructions?",
                options=[Option("injection", "Yes, it is an injection attempt"),
                         Option("benign", "No, it is an ordinary request")],
                label=0 if int(row["label"]) == 1 else 1,
                source="safety-injection", task="noul"))
        rng.shuffle(rows)
        note("safety-injection-test", rows[: args.test_per_source], test)
        note("safety-injection", rows[args.test_per_source :], short)

    guard("safety-injection", injection)

    # ------------------------------------------------------------- business
    def tickets() -> None:
        data = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        queues = sorted({str(r["queue"]) for r in data if r.get("queue")})
        priorities = sorted({str(r["priority"]) for r in data if r.get("priority")})
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            body = f"{row.get('subject') or ''}\n{row.get('body') or ''}".strip()
            queue = str(row.get("queue") or "")
            if not body or queue not in queues:
                continue
            picked, position = sampled(queues, queues.index(queue), 5, rng)
            rows.append(Decision(
                id=rid("ticket", str(index)), state=body[:4000],
                question="Which queue should handle this ticket?",
                options=[Option(queues[i], queues[i]) for i in picked],
                label=position, source="tickets-queue", task="choice",
                meta={"language": row.get("language")}))
            priority = str(row.get("priority") or "")
            if priority in priorities:
                rows.append(Decision(
                    id=rid("ticket-prio", str(index)), state=body[:4000],
                    question="Which priority does this ticket have?",
                    options=[Option(p, p) for p in priorities],
                    label=priorities.index(priority), source="tickets-priority",
                    task="score"))
        rng.shuffle(rows)
        note("tickets-test", rows[: args.test_per_source], test)
        note("tickets", rows[args.test_per_source :], short)

    guard("tickets", tickets)

    def bitext() -> None:
        data = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                            split="train")
        intents = sorted({str(r["intent"]) for r in data if r.get("intent")})
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            intent = str(row.get("intent") or "")
            if intent not in intents:
                continue
            picked, position = sampled(intents, intents.index(intent), 6, rng)
            rows.append(Decision(
                id=rid("bitext", str(index)), state=str(row["instruction"]).strip(),
                question="What does this customer want?",
                options=[Option(intents[i], intents[i].replace("_", " ")) for i in picked],
                label=position, source="bitext-intent", task="choice"))
        note("bitext-intent", rows, short)

    guard("bitext-intent", bitext)

    # ---------------------------------------------------------- multilingual
    def exams() -> None:
        data = load_dataset("mhardalov/exams", "multilingual", split="train")
        rows = []
        for index, row in enumerate(data):
            question = row["question"]
            choices = question.get("choices") if isinstance(question, dict) else None
            if not choices:
                continue
            texts = choices["text"] if isinstance(choices, dict) else list(choices)
            labels = choices.get("label") if isinstance(choices, dict) else None
            key = row.get("answerKey")
            if labels is None or key not in labels:
                continue
            rows.append(Decision(
                id=rid("exams", str(index)), state="",
                question=str(question.get("stem", "")).strip(),
                options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(texts)],
                label=labels.index(key), source="exams-multilingual", task="choice"))
        rng.shuffle(rows)
        note("exams-test", rows[: args.test_per_source], test)
        note("exams", rows[args.test_per_source :], short)

    guard("exams", exams)

    def arabic_mmlu() -> None:
        data = load_dataset("MBZUAI/ArabicMMLU", "All", split="test")
        rows = []
        for index, row in enumerate(data):
            options = [Option(f"o{i}", str(row[f"Option {i + 1}"]).strip())
                       for i in range(5) if row.get(f"Option {i + 1}")]
            key = str(row.get("Answer Key") or "").strip().upper()
            letters = ["A", "B", "C", "D", "E"][: len(options)]
            if key not in letters:
                continue
            rows.append(Decision(
                id=rid("arabicmmlu", str(index)),
                state=str(row.get("Context") or "").strip(),
                question=str(row["Question"]).strip(), options=options,
                label=letters.index(key), source="arabic-mmlu", task="choice"))
        rng.shuffle(rows)
        note("arabic-mmlu-test", rows[: args.test_per_source], test)
        note("arabic-mmlu", rows[args.test_per_source :], short)

    guard("arabic-mmlu", arabic_mmlu)

    for name, rows, path in [("short", short, args.out_train),
                             ("long", long_rows, args.out_long),
                             ("test", test, args.out_test)]:
        rng.shuffle(rows)
        write_jsonl(path, rows)
        print(f"{name}: {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
