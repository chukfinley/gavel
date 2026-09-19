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


def call(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    request = urllib.request.Request(
        f"{BASE}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as error:
        sys.exit(f"{error.code} {error.reason}: {error.read().decode()[:400]}")


def remember(pod: dict) -> None:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as handle:
        json.dump(pod, handle)


def recall() -> dict:
    try:
        with open(STATE) as handle:
            return json.load(handle)
    except Exception:                                            # noqa: BLE001
        sys.exit("no pod is remembered; start one first")


def start(args) -> None:
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
    pod = call("POST", "/pods", body)
    remember(pod)
    print(f"started {pod.get('id')}  {args.gpu or 'any consumer card'}  "
          f"{'spot' if args.spot else 'on demand'}")
    print(f"watch: https://huggingface.co/datasets/{args.results}")


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
        except Exception as error:                               # noqa: BLE001
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

    follow = sub.add_parser("watch")
    follow.add_argument("--results", default="chukfinley/gavel-runs")
    follow.add_argument("--interval", type=int, default=300)
    follow.set_defaults(function=watch)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
