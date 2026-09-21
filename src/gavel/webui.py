"""The working surface: playground, scoreboard, benchmarks, demos, training.

One page, served from this repository, no build step:

    scripts/webui.sh                # http://127.0.0.1:8030

Models are found under `export/`, `runs/` and on the Hub and loaded on
first use. Benchmarks and demos run as subprocesses of the same scripts
that the pods run, so a number shown here is a number `results/` holds.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
RESULTS, LOGS, RUNS, EXPORT = ROOT / "results", ROOT / "logs", ROOT / "runs", ROOT / "export"
JOBS_FILE = ROOT / "_scratch" / "webui_jobs.json"
HUB_MODEL = "chukfinley/gavel-vela-32k"

# What the field publishes, so the scoreboard shows the gap, not just us.
PUBLISHED = {
    "jevbench": {
        "Jev 1.13 (closed)": {"easy": 1.000, "standard": 0.986, "hard": 0.730},
        "OpenJev 26B (decoder)": {"easy": 1.000, "standard": 0.972, "hard": 0.640},
        "open-jev-deberta 435M": {"easy": 1.000, "standard": 0.431, "hard": 0.378},
    },
    "cbench": {
        "Jev (closed)": {"v1": 0.974, "v2": 0.964, "combined": 0.965},
        "Von 1.0.1 395M": {"v1": 0.923, "v2": 0.666, "combined": 0.687},
        "GLiNER2 300M": {"v1": 0.795, "v2": 0.688, "combined": 0.697},
        "Laya 421M": {"v1": 0.615, "v2": 0.585, "combined": 0.587},
    },
}
JEVBENCH_ITEMS = {"easy": 48, "standard": 72, "hard": 111}
MISC_PREFIXES = ("demo_", "latency_", "unreadable", "jevbench_hard")
GENERAL_SETS = ["general", "quiz", "multilingual", "tools", "browser", "moderation",
                "more", "semrouter", "kotoba", "domains", "games", "routing"]


def environment() -> dict[str, str]:
    env = dict(os.environ)
    keys = Path.home() / ".config" / "gavel" / "keys.env"
    if keys.exists():
        for line in keys.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if "=" in line and not line.startswith("#"):
                name, value = line.split("=", 1)
                env[name.strip()] = value.strip().strip('"').strip("'")
    env.setdefault("MEMGUARD_ALLOW_MB", "16000")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return env


# ------------------------------------------------------------------ models
_kind_cache: dict[str, tuple[float, str]] = {}


def checkpoint_kind(path: Path) -> str:
    """`train.py` saves `args`, `train_span.py` saves `flip_rate`."""
    stamp = path.stat().st_mtime
    cached = _kind_cache.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        kind = "span" if "flip_rate" in state or "mean" in state else "pair"
        if "model" in state and any(k.startswith("head.0") for k in state["model"]):
            kind = "span"
    except Exception:
        kind = "unknown"
    _kind_cache[str(path)] = (stamp, kind)
    return kind


def discover_models() -> list[dict[str, Any]]:
    models = [{"id": f"hub:{HUB_MODEL}", "kind": "pair", "run": "hub-main",
               "label": f"Hub {HUB_MODEL} (pair)", "path": HUB_MODEL}]
    if EXPORT.exists():
        for folder in sorted(EXPORT.iterdir()):
            if (folder / "config.json").exists():
                models.append({"id": f"export:{folder.name}", "kind": "pair", "run": folder.name,
                               "label": f"export/{folder.name} (pair)", "path": str(folder)})
            head = folder / "span-head.pt"
            if head.exists():
                models.append({"id": f"export:{folder.name}/span-head.pt", "kind": "span",
                               "run": f"{folder.name}-span", "label": f"export/{folder.name} span head",
                               "path": str(head)})
    if RUNS.exists():
        for folder in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            for name in ("best-calibrated.pt", "best.pt", "pair.pt"):
                file = folder / name
                if not file.exists():
                    continue
                kind = "pair" if name == "pair.pt" else checkpoint_kind(file)
                if kind == "unknown":
                    continue
                models.append({"id": f"runs:{folder.name}/{name}", "kind": kind,
                               "run": folder.name if name != "pair.pt" else f"{folder.name}-teacher",
                               "label": f"runs/{folder.name}/{name} ({kind})", "path": str(file),
                               "mtime": file.stat().st_mtime})
                if name == "best-calibrated.pt":
                    break
    return models


class Judges:
    """Loaded models, one per id, loaded on first use."""

    def __init__(self):
        self.loaded: dict[str, Any] = {}
        self.lock = threading.Lock()

    def get(self, model_id: str):
        with self.lock:
            if model_id in self.loaded:
                return self.loaded[model_id]
            spec = next((m for m in discover_models() if m["id"] == model_id), None)
            if spec is None:
                raise KeyError(model_id)
            if spec["kind"] == "span":
                from .spanapi import SpanGavel
                judge = SpanGavel.from_checkpoint(spec["path"], max_length=8192)
            elif spec["path"].endswith(".pt"):
                from .api import Gavel
                judge = Gavel.from_checkpoint(spec["path"], max_length=4096)
            else:
                from .api import Gavel
                judge = Gavel.from_pretrained(spec["path"], None, 4096)
            self.loaded[model_id] = judge
            return judge

    def drop(self, model_id: str | None = None) -> None:
        with self.lock:
            if model_id:
                self.loaded.pop(model_id, None)
            else:
                self.loaded.clear()
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ------------------------------------------------------------------ results
def load_results() -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    misc: dict[str, Any] = {}
    if RESULTS.exists():
        for file in sorted(RESULTS.glob("*.json")):
            try:
                data = json.loads(file.read_text())
            except Exception:
                continue
            name = file.stem
            if name.startswith(MISC_PREFIXES):
                misc[name] = data
                continue
            run, _, suite = name.partition("_")
            runs.setdefault(run, {})[suite] = {"data": data, "mtime": file.stat().st_mtime}
    return {"runs": runs, "misc": misc, "published": PUBLISHED,
            "jevbench_items": JEVBENCH_ITEMS}


# ------------------------------------------------------------------ jobs
class Jobs:
    def __init__(self):
        self.jobs: dict[str, dict[str, Any]] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        if JOBS_FILE.exists():
            try:
                for job in json.loads(JOBS_FILE.read_text()):
                    if job["status"] == "running":
                        job["status"] = "lost"
                    self.jobs[job["id"]] = job
            except Exception:
                pass

    def save(self) -> None:
        JOBS_FILE.parent.mkdir(exist_ok=True)
        JOBS_FILE.write_text(json.dumps(list(self.jobs.values())[-200:], indent=1))

    def start(self, kind: str, command: list[str], label: str, out: str | None) -> dict:
        job_id = time.strftime("%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        log = LOGS / f"webui_{job_id}.log"
        LOGS.mkdir(exist_ok=True)
        with open(log, "w") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    env=environment(), start_new_session=True)
        job = {"id": job_id, "kind": kind, "label": label, "command": " ".join(command),
               "log": str(log.relative_to(ROOT)), "out": out, "status": "running",
               "started": time.time(), "ended": None, "code": None, "pid": proc.pid}
        with self.lock:
            self.jobs[job_id] = job
            self.procs[job_id] = proc
            self.save()
        return job

    def poll(self) -> list[dict]:
        with self.lock:
            for job_id, proc in list(self.procs.items()):
                code = proc.poll()
                if code is None:
                    continue
                job = self.jobs[job_id]
                job["status"] = "done" if code == 0 else "failed"
                job["code"], job["ended"] = code, time.time()
                del self.procs[job_id]
                self.save()
            return sorted(self.jobs.values(), key=lambda j: j["started"], reverse=True)

    def stop(self, job_id: str) -> bool:
        with self.lock:
            proc = self.procs.get(job_id)
            if proc is None:
                return False
            try:
                os.killpg(os.getpgid(proc.pid), 15)
            except ProcessLookupError:
                pass
            return True


def model_flag(spec: dict[str, Any], script: str) -> list[str]:
    """How each evaluation script wants to be told which model to load."""
    if spec["kind"] == "span":
        if script in ("eval_jevbench",):
            return ["--span-checkpoint", spec["path"]]
        return ["--span", spec["path"]]
    if spec["path"].endswith(".pt"):
        return ["--checkpoint", spec["path"]]
    return ["--model", spec["path"]]


def build_command(kind: str, spec: dict[str, Any] | None, extra: dict[str, Any]) -> tuple[list[str], str, str | None]:
    run = spec["run"] if spec else "none"
    if kind == "jevbench":
        out = f"results/{run}_jevbench.json"
        return ([PY, "scripts/eval_jevbench.py", *model_flag(spec, "eval_jevbench"),
                 "--suite", "_scratch/jevbench", "--max-length", "4096", "--out", out],
                f"JevBench · {run}", out)
    if kind == "cbench":
        if spec["kind"] == "span":
            raise ValueError("classifier-benchmark runs on pair models only for now")
        out = f"results/{run}_cbench.json"
        return ([PY, "scripts/eval_cbench.py", *model_flag(spec, "eval_cbench"),
                 "--suites", "all", "--max-length", "512", "--out", out],
                f"classifier-benchmark v1+v2 · {run}", out)
    if kind == "general":
        test = extra.get("set", "general")
        if test not in GENERAL_SETS:
            raise ValueError(f"unknown test set {test}")
        if spec["kind"] != "pair" or not spec["path"].endswith(".pt"):
            raise ValueError("held-out sets need a pair checkpoint (.pt)")
        out = f"results/{run}_{test}.json"
        return ([PY, "scripts/eval_general.py", "--checkpoint", spec["path"],
                 "--test", f"data/test_{test}.jsonl", "--batch-size", "4",
                 "--max-length", "1024", "--out", out], f"held-out {test} · {run}", out)
    if kind == "latency":
        pair = extra.get("pair", HUB_MODEL)
        span = extra.get("span", "export/restore-base/span-head.pt")
        return ([PY, "scripts/bench_latency.py", pair, span], "latency · pair vs span",
                "results/latency_cuda.json")
    if kind == "snake":
        out = f"results/demo_snake_{run}.json"
        args = model_flag(spec, "demo") if spec["kind"] == "span" or spec["path"].endswith(".pt") else ["--model", spec["path"]]
        if spec["kind"] == "pair" and spec["path"].endswith(".pt"):
            raise ValueError("the snake demo loads pair models from the Hub or export/ only")
        return ([PY, "scripts/demo_snake.py", *args, "--games", str(extra.get("games", 20)),
                 *(["--bundled"] if extra.get("bundled") else []), "--out", out],
                f"snake · {run}", out)
    if kind == "webnav":
        out = f"results/demo_webnav_{run}.json"
        if spec["kind"] == "pair" and spec["path"].endswith(".pt"):
            raise ValueError("the web demo loads pair models from the Hub or export/ only")
        args = ["--span", spec["path"]] if spec["kind"] == "span" else ["--model", spec["path"]]
        return ([PY, "scripts/demo_webnav.py", *args, "--out", out], f"web navigation · {run}", out)
    if kind == "distil":
        steps = int(extra.get("steps", 40000))
        name = extra.get("name", f"base-span-{steps // 1000}k")
        return (["bash", "./_scratch/local_span.sh", HUB_MODEL, extra.get("revision", "main"),
                 name, str(steps), str(extra.get("batch", 8))],
                f"distil span head · {name} · {steps} steps", f"results/{name}_jevbench.json")
    raise ValueError(f"unknown job kind {kind}")


# ------------------------------------------------------------------ status
def gpu_status() -> dict[str, Any]:
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False).stdout
        name, used, total, util, temp = [x.strip() for x in text.strip().split(",")]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False,
                              timeout=5).stdout.strip().splitlines()
        return {"name": name, "used_mb": int(used), "total_mb": int(total), "util": int(util),
                "temperature": int(temp),
                "apps": [dict(zip(("pid", "name", "mb"), [x.strip() for x in a.split(",")]))
                         for a in apps if a.strip()]}
    except Exception as error:
        return {"error": str(error)[:120]}


def tail(path: Path, lines: int = 60) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 64000))
            text = handle.read().decode("utf-8", "replace")
        return "\n".join(text.splitlines()[-lines:])
    except Exception:
        return ""


def training_status() -> dict[str, Any]:
    procs = subprocess.run(["pgrep", "-af", "scripts/train"], capture_output=True, text=True, check=False).stdout
    active = [line[:160] for line in procs.splitlines() if "pgrep" not in line]
    histories = {}
    if RUNS.exists():
        for folder in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:6]:
            history = folder / "history.json"
            if history.exists():
                try:
                    histories[folder.name] = json.loads(history.read_text())
                except Exception:
                    pass
    logs = {}
    for name in ("base-span-40k", "night"):
        candidates = [LOGS / f"{name}.log", ROOT / "_scratch" / f"{name}.log"]
        for file in candidates:
            if file.exists():
                logs[name] = tail(file, 25)
    return {"active": active, "histories": histories, "logs": logs}


# ------------------------------------------------------------------ app
def build_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field

    judges, jobs = Judges(), Jobs()
    app = FastAPI(title="gavel webui")

    class Question(BaseModel):
        question: str
        options: list[str] = Field(min_length=2)

    class DecideRequest(BaseModel):
        model: str
        state: str = ""
        questions: list[Question] = Field(min_length=1)
        bundled: bool = False

    class JobRequest(BaseModel):
        kind: str
        model: str | None = None
        extra: dict[str, Any] = Field(default_factory=dict)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "web" / "index.html").read_text()

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        loaded = set(judges.loaded)
        return [{**m, "loaded": m["id"] in loaded} for m in discover_models()]

    @app.post("/api/models/unload")
    def unload(model: str | None = None) -> dict[str, Any]:
        judges.drop(model)
        return {"ok": True}

    @app.post("/api/decide")
    def decide(request: DecideRequest) -> dict[str, Any]:
        try:
            judge = judges.get(request.model)
        except KeyError:
            raise HTTPException(404, f"no model {request.model}") from None
        from .spanapi import SpanGavel
        started = time.perf_counter()
        if isinstance(judge, SpanGavel) and request.bundled:
            verdicts = judge.decide_many(request.state, [(q.question, q.options) for q in request.questions])
        else:
            verdicts = [judge.decide(request.state, q.question, q.options) for q in request.questions]
        ms = (time.perf_counter() - started) * 1000
        return {"milliseconds": round(ms, 1), "device": str(judge.device),
                "verdicts": [{"option": v.option, "confidence": v.confidence,
                              "probabilities": v.probabilities} for v in verdicts]}

    @app.get("/api/results")
    def results() -> dict[str, Any]:
        return load_results()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return jobs.poll()

    @app.post("/api/jobs")
    def start_job(request: JobRequest) -> dict[str, Any]:
        spec = None
        if request.model:
            spec = next((m for m in discover_models() if m["id"] == request.model), None)
            if spec is None:
                raise HTTPException(404, f"no model {request.model}")
        try:
            command, label, out = build_command(request.kind, spec, request.extra)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return jobs.start(request.kind, command, label, out)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        return {"stopped": jobs.stop(job_id)}

    @app.get("/api/log")
    def read_log(path: str, lines: int = 200) -> dict[str, Any]:
        file = (ROOT / path).resolve()
        if not str(file).startswith(str(ROOT)) or not re.search(r"\.(log|txt|json)$", path):
            raise HTTPException(403, "not a log")
        return {"path": path, "text": tail(file, lines)}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"gpu": gpu_status(), "training": training_status(),
                "running": [j for j in jobs.poll() if j["status"] == "running"],
                "loaded": list(judges.loaded), "time": time.time()}

    @app.exception_handler(Exception)
    async def on_error(request, error):
        return JSONResponse(status_code=500, content={"detail": str(error)[:400]})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
