"""An HTTP service around the decision model.

    pip install "gavel[serve]"
    gavel-serve --model chukfinley/gavel-vela-32k --port 8080

    curl -s localhost:8080/decide -H 'content-type: application/json' -d '{
      "state": "The customer was charged twice for the same period.",
      "question": "Which queue should handle this?",
      "options": ["Billing support", "Account access", "Technical fault"],
      "abstain": true, "threshold": 0.6 }'

The shape follows what a caller actually needs: one state, one or many typed
questions, a distribution back for each. `threshold` marks a verdict as
abstained when the model is not sure enough, which is the point of a calibrated
probability — automate above the line, send the rest to a person.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from .api import Gavel


def build_app(model_name: str, device: str | None = None, max_length: int = 1024):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class Question(BaseModel):
        question: str
        options: list[str] = Field(min_length=2)
        abstain: bool = False
        threshold: float = 0.0

    class Request(BaseModel):
        state: str = ""
        question: str | None = None
        options: list[str] | None = None
        questions: list[Question] | None = None
        abstain: bool = False
        threshold: float = 0.0
        long: bool = False

    judge = Gavel.from_pretrained(model_name, device=device, max_length=max_length)
    app = FastAPI(title="gavel", version="0.1.0",
                  description="Typed decisions, no generation.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": model_name, "device": judge.device,
                "temperature": judge.temperature, "max_length": judge.max_length}

    @app.post("/decide")
    def decide(request: Request) -> dict[str, Any]:
        started = time.perf_counter()
        if request.questions:
            verdicts = judge.decide_many(
                request.state, [q.model_dump() for q in request.questions])
        elif request.question and request.options:
            decide_one = judge.decide_long if request.long else judge.decide
            verdicts = [decide_one(request.state, request.question, request.options)
                        if request.long else
                        judge.decide(request.state, request.question, request.options,
                                     abstain=request.abstain, threshold=request.threshold)]
        else:
            raise HTTPException(422, "give either question and options, or questions")
        return {
            "verdicts": [{"option": v.option, "confidence": v.confidence,
                          "probabilities": v.probabilities, "abstained": v.abstained,
                          **({"meta": v.meta} if v.meta else {})} for v in verdicts],
            "milliseconds": round((time.perf_counter() - started) * 1000, 2),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args.model, args.device, args.max_length),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
