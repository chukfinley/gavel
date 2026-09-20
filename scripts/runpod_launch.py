#!/usr/bin/env python3
"""Start, watch and stop a rented pod that trains this project.

The pod pulls the code from GitHub, builds every dataset from public sources
and publishes its logs, results and finished models to Hugging Face. Nothing
large travels over the home connection, and the run can be watched without a
shell on the rented machine.

    export RUNPOD_API_KEY=...            # from runpod.io/console/user/settings
    export HF_TOKEN=...                  # write token
    python scripts/runpod_launch.py start --gpu "NVIDIA GeForce RTX 4090"
    python scripts/runpod_launch.py status
    python scripts/runpod_launch.py stop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://rest.runpod.io/v1"
STATE = os.path.expanduser("~/.cache/gavel-pod.json")
# A consumer card on purpose: the point is that this model does not need a
# data-centre GPU.
DEFAULT_GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090",
                "NVIDIA GeForce RTX 3090"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def call(method: str, path: str, body: dict | None = None,
         fatal: bool = True) -> dict | None:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    request = urllib.request.Request(
        f"{BASE}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 # The endpoint sits behind a filter that rejects the default
                 # Python user agent with a 403.
                 "User-Agent": "curl/8.5.0", "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as error:
        detail = f"{error.code} {error.reason}: {error.read().decode()[:400]}"
        if fatal:
            sys.exit(detail)
        print(detail, flush=True)
        return None


def remember(pod: dict) -> None:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as handle:
        json.dump(pod, handle)


def recall() -> dict:
    try:
        with open(STATE) as handle:
            return json.load(handle)
    except Exception:
        sys.exit("no pod is remembered; start one first")


def start(args, fatal: bool = True) -> dict | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set; the pod needs it to publish results")
    command = ("apt-get update -qq && apt-get install -y -qq curl git >/dev/null 2>&1; "
               f"curl -fsSL {args.bootstrap} -o /workspace/bootstrap.sh && "
               "bash /workspace/bootstrap.sh")
    body = {
        "name": args.name,
        "computeType": "GPU",
        "gpuTypeIds": [args.gpu] if args.gpu else DEFAULT_GPUS,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "imageName": args.image,
        "containerDiskInGb": args.disk,
        "cloudType": args.cloud,
        "interruptible": args.spot,
        "ports": ["22/tcp"],
        "env": {"HF_TOKEN": token,
                "RESULTS_REPO": args.results,
                "GAVEL_REPO": args.repo,
                "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        "dockerStartCmd": ["bash", "-lc", command],
    }
    pod = call("POST", "/pods", body, fatal=fatal)
    if not pod:
        return None
    remember(pod)
    print(f"started {pod.get('id')}  {args.gpu or 'any consumer card'}  "
          f"{'spot' if args.spot else 'on demand'}")
    print(f"watch: https://huggingface.co/datasets/{args.results}")
    return pod


def status(args) -> None:
    pod = call("GET", f"/pods/{recall()['id']}")
    cost = pod.get("costPerHr")
    print(f"{pod.get('id')}  {pod.get('desiredStatus')}  "
          f"{pod.get('machine', {}).get('gpuTypeId', '')}  {cost} $/h")
    started = pod.get("lastStartedAt")
    if started:
        print(f"running since {started}")


def stop(args) -> None:
    pod = recall()
    call("DELETE", f"/pods/{pod['id']}")
    print(f"terminated {pod['id']}")


def ensure(args) -> None:
    """Rent machines until one can actually run CUDA.

    Community hosts regularly come up with a working `nvidia-smi` and a torch
    that sees no device; three in a row did on 2026-09-20. The bootstrap
    detects that, publishes its log and idles, so this loop reads the log,
    terminates the broken machine and rents the next one.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    wanted = [g.strip() for g in args.gpus.split(",") if g.strip()]
    for attempt in range(1, args.tries + 1):
        gpu = wanted[(attempt - 1) % len(wanted)]
        body_args = argparse.Namespace(**vars(args))
        body_args.gpu = gpu
        started = start(body_args, fatal=False)
        if not started:
            # Usually "no instances currently available" for that card. Try
            # the next one in the list rather than giving up on the run.
            print(f"[{attempt}/{args.tries}] {gpu} unavailable, next card",
                  flush=True)
            time.sleep(args.interval)
            continue
        pod = started["id"]
        print(f"[{attempt}/{args.tries}] {pod} on {gpu}, waiting for its log",
              flush=True)
        deadline = time.time() + args.wait
        verdict = "no log"
        while time.time() < deadline:
            time.sleep(args.interval)
            try:
                path = api.hf_hub_download(args.results, "bootstrap.log",
                                           repo_type="dataset", force_download=True)
                text = open(path).read()
            except Exception as error:
                print(f"  (waiting: {str(error)[:50]})", flush=True)
                continue
            # Match on this pod's own id, never on a phrase that an older
            # pod could have left in the same file. Getting this wrong
            # terminated a machine that had just reported a working GPU.
            if f"FATAL pod {pod} cannot run CUDA" in text:
                verdict = "no CUDA"
                break
            if f"CUDA OK on pod {pod}" in text:
                print(f"  {pod} has a working GPU and is building. "
                      f"Watch: https://huggingface.co/datasets/{args.results}")
                return
            print("  (still starting)", flush=True)
        print(f"  {pod}: {verdict}, terminating", flush=True)
        call("DELETE", f"/pods/{pod}")
    sys.exit(f"no machine out of {args.tries} could run CUDA")


def watch(args) -> None:
    """Poll the results dataset until the pod reports that it is finished."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    seen = ""
    while True:
        try:
            path = api.hf_hub_download(args.results, "bootstrap.log",
                                       repo_type="dataset", force_download=True)
            text = open(path).read()
            fresh = text[len(seen):]
            if fresh.strip():
                print(fresh, end="", flush=True)
            seen = text
            if "the pod can be stopped" in text:
                print("\nrun finished")
                return
        except Exception as error:
            print(f"(waiting: {str(error)[:60]})", flush=True)
        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    begin = sub.add_parser("start")
    begin.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    begin.add_argument("--name", default="gavel-training")
    begin.add_argument("--image", default=IMAGE)
    begin.add_argument("--disk", type=int, default=120)
    begin.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    begin.add_argument("--spot", action="store_true",
                       help="cheaper, but the pod can be taken away")
    begin.add_argument("--results", default="chukfinley/gavel-runs")
    begin.add_argument("--repo", default="https://github.com/chukfinley/gavel.git")
    begin.add_argument("--bootstrap", default="https://raw.githubusercontent.com/"
                                              "chukfinley/gavel/master/scripts/pod_bootstrap.sh")
    begin.set_defaults(function=start)

    for name, function in [("status", status), ("stop", stop)]:
        parser_ = sub.add_parser(name)
        parser_.set_defaults(function=function)

    keep = sub.add_parser("ensure", help="rent until one machine can run CUDA")
    keep.add_argument("--gpus", default="NVIDIA GeForce RTX 4090,"
                                        "NVIDIA GeForce RTX 3090,"
                                        "NVIDIA GeForce RTX 3090 Ti")
    keep.add_argument("--tries", type=int, default=6)
    keep.add_argument("--wait", type=int, default=900,
                      help="seconds to give one machine before giving up on it")
    keep.add_argument("--interval", type=int, default=40)
    keep.add_argument("--name", default="gavel-training")
    keep.add_argument("--image", default=IMAGE)
    keep.add_argument("--disk", type=int, default=120)
    keep.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    keep.add_argument("--spot", action="store_true")
    keep.add_argument("--results", default="chukfinley/gavel-runs")
    keep.add_argument("--repo", default="https://github.com/chukfinley/gavel.git")
    keep.add_argument("--bootstrap", default="https://raw.githubusercontent.com/"
                                             "chukfinley/gavel/master/scripts/pod_bootstrap.sh")
    keep.set_defaults(function=ensure)

    follow = sub.add_parser("watch")
    follow.add_argument("--results", default="chukfinley/gavel-runs")
    follow.add_argument("--interval", type=int, default=300)
    follow.set_defaults(function=watch)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
