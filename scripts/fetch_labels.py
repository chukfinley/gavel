#!/usr/bin/env python3
"""Pull Jev's labels and the harvested traces from the results dataset.

The workstation writes them under `data/`, which is not in git; a pod
gets them from `chukfinley/gavel-runs` before it assembles the mix.
"""
from __future__ import annotations

import os

from huggingface_hub import hf_hub_download

FILES = ("data/jev/train_v6.jsonl", "data/jev/dev_strat_v2.jsonl",
         "data/traces/webnav.jsonl", "data/traces/tasks.jsonl")


def main() -> None:
    repo = os.environ.get("RESULTS_REPO", "chukfinley/gavel-runs")
    for name in FILES:
        try:
            hf_hub_download(repo, name, repo_type="dataset", local_dir=".",
                            token=os.environ.get("HF_TOKEN"))
            print("fetched", name)
        except Exception as error:
            print("missing", name, str(error)[:80])


if __name__ == "__main__":
    main()
