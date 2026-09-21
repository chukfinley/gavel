"""TypeSafe's Jev behind the same `decide` interface as our models.

Reached through OpenRouter's decisions endpoint with the native body.
Every request and raw reply goes to `logs/jev_requests.jsonl`, the same
file the labeller writes, so nothing sent is ever lost.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from .api import Verdict

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
_lock = threading.Lock()


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for folder in (Path.cwd(), Path(__file__).resolve().parents[2]):
        env = folder / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY=") and line.split("=", 1)[1].strip():
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no OPENROUTER_API_KEY in the environment or .env")


class JevGavel:
    """One state, typed questions, probabilities back; from the closed model."""

    device = "openrouter"

    def __init__(self, log: str | Path = "logs/jev_requests.jsonl", timeout: int = 60):
        from curl_cffi import requests
        self.session = requests.Session(impersonate="chrome")
        self.headers = {"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"}
        self.log = Path(log)
        self.timeout = timeout
        self.spent_usd = 0.0
        self.last_ms = 0.0

    def _post(self, body: dict) -> dict:
        started = time.perf_counter()
        response, error = None, None
        try:
            response = self.session.post(ENDPOINT, headers=self.headers, json=body, timeout=self.timeout)
            payload = response.json()
        except Exception as failure:
            error = str(failure)[:300]
            payload = {}
        self.last_ms = (time.perf_counter() - started) * 1000
        record = {"time": time.time(), "shape": "live", "ms": round(self.last_ms, 1), "request": body,
                  "status": getattr(response, "status_code", None),
                  "response": getattr(response, "text", None), "error": error}
        with _lock:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log, "a") as handle:
                handle.write(json.dumps(record) + "\n")
        if error or "error" in payload:
            raise RuntimeError(error or str(payload["error"])[:300])
        self.spent_usd += float((payload.get("usage") or {}).get("cost") or 0.0)
        return payload

    def decide_many(self, state: str, questions: Sequence[tuple[str, Sequence[str]]],
                    noul: bool = False) -> list[Verdict]:
        body = {"model": MODEL, "state": state or "", "questions": {}}
        for index, (question, options) in enumerate(questions):
            options = list(options)
            if noul and len(options) == 2:
                body["questions"][f"q{index}"] = {
                    "type": "noul", "instructions": question,
                    "criteria": {"true": options[0], "false": options[1]}}
            else:
                body["questions"][f"q{index}"] = {
                    "type": "choice", "instructions": question,
                    "criteria": {f"o{k}": text for k, text in enumerate(options)}}
        payload = self._post(body)
        answers = payload.get("answers") or {}
        verdicts = []
        for index, (question, options) in enumerate(questions):
            options = list(options)
            answer = answers.get(f"q{index}") or {}
            if "noul" in answer:
                p = float(answer["noul"])
                probabilities = [p, 1.0 - p]
            else:
                table = answer.get("probabilities") or {}
                probabilities = [float(table.get(f"o{k}", 0.0)) for k in range(len(options))]
                total = sum(probabilities) or 1.0
                probabilities = [p / total for p in probabilities]
            best = max(range(len(options)), key=probabilities.__getitem__)
            verdicts.append(Verdict(option=options[best], confidence=round(probabilities[best], 4),
                                    probabilities={o: round(p, 4) for o, p in zip(options, probabilities)}))
        return verdicts

    def decide(self, state: str, question: str, options: Sequence[str],
               abstain: bool = False, threshold: float = 0.0) -> Verdict:
        return self.decide_many(state, [(question, options)])[0]
