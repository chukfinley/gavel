#!/usr/bin/env python3
"""Ask Jev (TypeSafe, via OpenRouter) for its probabilities on our rows.

The point: Jev costs 0.042 $ per million input tokens and nothing for
output, so its answers on our whole training mix cost about the price of
one hour on a rented card. Its probabilities then teach our encoder the
way the pair model teaches the span head, but from a model that is 20
points ahead of us on the standard tier.

Rows sharing one state are sent as one request with several questions,
which is cheaper and also yields exactly the multi-question sequences the
span head was never trained on.

    .venv/bin/python scripts/label_with_jev.py --rows data/dev_strat_v2.jsonl \\
        --out data/jev/dev_strat_v2.jsonl --budget-usd 0.20

The output keeps our row id and adds `teacher`: the probability per
option, Jev's choice and confidence, and the cost so far.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "src")
from curl_cffi import requests

PRICE_PER_TOKEN = 0.042 / 1_000_000
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "typesafe/jev-1.13"


def load_key() -> str:
    for source in (os.environ.get("OPENROUTER_API_KEY"),):
        if source:
            return source
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENROUTER_API_KEY in the environment or .env")


def question_of(row: dict) -> dict:
    """Our row in Jev's question shape."""
    options = row["options"]
    if row.get("task") == "noul" and len(options) == 2:
        yes, no = options[0], options[1]
        return {"type": "noul", "instructions": row["question"],
                "criteria": {"true": yes["description"], "false": no["description"]}}
    return {"type": "choice", "instructions": row["question"],
            "criteria": {str(o["id"]): o["description"] for o in options}}


def native_body(state: str, rows: list[dict]) -> dict:
    return {"model": MODEL, "state": state,
            "questions": {row["id"]: question_of(row) for row in rows}}


def chat_body(state: str, rows: list[dict]) -> dict:
    """OpenRouter's chat shape with the native fields carried alongside.

    OpenRouter forwards unknown top-level fields to the provider for many
    endpoints; whether TypeSafe reads them is what `--probe` finds out.
    """
    body = native_body(state, rows)
    body["messages"] = [{"role": "user", "content": state}]
    return body


def parse_answers(payload: dict, rows: list[dict]) -> dict[str, dict]:
    """Jev's answers keyed by our row id, in our option order."""
    answers = payload.get("answers")
    if answers is None:  # the chat shape hides the answer in the message
        try:
            content = payload["choices"][0]["message"]["content"]
            answers = json.loads(content).get("answers", json.loads(content))
        except Exception as error:
            raise ValueError(f"no answers in response: {str(payload)[:300]}") from error
    out = {}
    for row in rows:
        answer = answers.get(row["id"])
        if not answer:
            continue
        options = row["options"]
        if answer.get("type") == "noul" or "noul" in answer:
            p = float(answer["noul"])
            probs = [p, 1.0 - p]
            choice = 0 if p >= 0.5 else 1
            confidence = max(p, 1.0 - p)
        else:
            table = answer.get("probabilities") or {}
            probs = [float(table.get(str(o["id"]), 0.0)) for o in options]
            total = sum(probs) or 1.0
            probs = [p / total for p in probs]
            choice = max(range(len(probs)), key=probs.__getitem__)
            confidence = float(answer.get("confidence", probs[choice]))
        out[row["id"]] = {"probabilities": probs, "choice": choice, "confidence": confidence}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="rows to send, 0 = all")
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--shape", choices=("native", "chat"), default="native")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-questions", type=int, default=8, help="per request")
    parser.add_argument("--probe", action="store_true", help="send one request and print it raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", default="logs/jev_requests.jsonl",
                        help="every request and raw response, one JSON line each")
    args = parser.parse_args()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()

    def log(body: dict, response, started: float, error: str | None = None) -> None:
        record = {"time": time.time(), "shape": args.shape, "ms": round((time.perf_counter() - started) * 1000, 1),
                  "request": body, "status": getattr(response, "status_code", None),
                  "response": getattr(response, "text", None), "error": error}
        with log_lock, open(args.log, "a") as handle_log:
            handle_log.write(json.dumps(record) + "\n")

    key = load_key()
    rows = [json.loads(line) for line in open(args.rows)]
    done: set[str] = set()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    rows = [r for r in rows if r["id"] not in done and r.get("options") and r.get("state") is not None]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["state"]].append(row)
    batches = []
    for state, members in groups.items():
        for start in range(0, len(members), args.max_questions):
            batches.append((state, members[start:start + args.max_questions]))
    print(f"{len(rows)} rows in {len(batches)} requests, {len(done)} already labelled", flush=True)

    session = requests.Session(impersonate="chrome")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://github.com/chukfinley/gavel", "X-Title": "gavel distillation"}
    build = native_body if args.shape == "native" else chat_body

    if args.probe:
        state, members = batches[0]
        body = build(state, members[:2])
        print(json.dumps(body, indent=1)[:1500])
        started = time.perf_counter()
        response = session.post(ENDPOINT, headers=headers, json=body, timeout=60)
        log(body, response, started)
        print("HTTP", response.status_code)
        print(response.text[:3000])
        return

    lock = threading.Lock()
    spent = {"usd": 0.0, "tokens": 0, "rows": 0, "errors": 0}
    stop = threading.Event()
    handle = open(out_path, "a")

    def work(batch):
        if stop.is_set():
            return
        state, members = batch
        body = build(state, members)
        for attempt in range(4):
            response = None
            started = time.perf_counter()
            try:
                response = session.post(ENDPOINT, headers=headers, json=body, timeout=90)
                log(body, response, started)
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                payload = response.json()
                if "error" in payload:
                    raise ValueError(str(payload["error"])[:200])
                answers = parse_answers(payload, members)
                usage = payload.get("usage") or {}
                tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                break
            except Exception as error:
                if response is None:
                    log(body, None, started, error=str(error)[:300])
                if attempt == 3:
                    with lock:
                        spent["errors"] += 1
                        if spent["errors"] <= 5:
                            print("error:", str(error)[:200], flush=True)
                    return
                time.sleep(2 ** attempt)
        with lock:
            for row in members:
                if row["id"] in answers:
                    record = {"id": row["id"], "source": row.get("source"), "label": row.get("label"),
                              "teacher": answers[row["id"]]}
                    handle.write(json.dumps(record) + "\n")
                    spent["rows"] += 1
            handle.flush()
            spent["tokens"] += tokens
            spent["usd"] += tokens * PRICE_PER_TOKEN
            if spent["usd"] >= args.budget_usd:
                stop.set()
            if spent["rows"] % 500 < len(members):
                print(f"  {spent['rows']} rows, {spent['tokens']} tokens, {spent['usd']:.4f} $, "
                      f"{spent['errors']} errors", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, batches))
    print(f"done: {spent['rows']} rows, {spent['tokens']} tokens, {spent['usd']:.4f} $, "
          f"{spent['errors']} errors{' (budget reached)' if stop.is_set() else ''}", flush=True)


if __name__ == "__main__":
    main()
