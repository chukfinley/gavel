#!/usr/bin/env python3
"""Assemble every results file into one comparison document.

Published reference numbers come from the sources named below; everything else
in the table was measured on this machine, on identical rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REFERENCE = {
    "Jev (TypeSafe, closed)": {
        "typesafe102_agreement": 0.8831, "typesafe102_variation": 0.1268,
        "note": "published by TypeSafe; 102 rows aligned by OpenJev"},
    "OpenJev, frozen Qwen3.5-4B": {
        "authored144": 0.8132, "wanli256": 0.6365,
        "typesafe102_agreement": 0.8453, "typesafe102_variation": 0.1770,
        "note": "published by the OpenJev project"},
    "openjev NLI-4B (AlexWortega)": {
        "mmlu_rerank": 0.474, "note": "published in its model card"},
    "openjev NLI-0.8B (AlexWortega)": {
        "mmlu_rerank": 0.350, "note": "published in its model card"},
}


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                            # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="COMPARISON.md")
    args = parser.parse_args()

    folder = Path(args.results)
    variants: dict[str, dict] = {}
    for path in sorted(folder.glob("*.json")):
        name, _, kind = path.stem.partition("_")
        variants.setdefault(name, {})[kind] = load(path)

    lines = ["# Comparison", "",
             "Measured on one RTX 3060. Rows are identical across systems where a",
             "number was measured here; published numbers are marked as such.", ""]

    lines += ["## OpenJev fixtures", "",
              "| system | authored144 | WANLI256 | TypeSafe102 agreement | TypeSafe102 variation |",
              "|---|---:|---:|---:|---:|"]
    for name, values in REFERENCE.items():
        if "authored144" in values or "typesafe102_agreement" in values:
            lines.append(f"| {name} | {values.get('authored144', '—')} | "
                         f"{values.get('wanli256', '—')} | "
                         f"{values.get('typesafe102_agreement', '—')} | "
                         f"{values.get('typesafe102_variation', '—')} |")
    for name, blocks in sorted(variants.items()):
        fixtures = blocks.get("fixtures") or {}
        typesafe = blocks.get("typesafe") or {}
        authored = fixtures.get("authored144", {}).get("balanced_accuracy")
        wanli = fixtures.get("wanli256", {}).get("balanced_accuracy")
        lines.append(
            f"| typedec {name} | {authored:.4f} " if authored else f"| typedec {name} | — ")
        lines[-1] += (f"| {wanli:.4f} " if wanli else "| — ")
        lines[-1] += (f"| {typesafe.get('modal_agreement', float('nan')):.4f} "
                      if typesafe else "| — ")
        lines[-1] += (f"| {typesafe.get('total_variation', float('nan')):.4f} |"
                      if typesafe else "| — |")

    for section, key in [("General test set", "general"), ("Question banks", "quiz"),
                         ("Ten languages", "multilingual"), ("Tool selection", "tools"),
                         ("Browser actions", "browser")]:
        rows = {name: blocks[key] for name, blocks in variants.items()
                if blocks.get(key) and "strata" in blocks[key]}
        if not rows:
            continue
        strata = sorted({s for block in rows.values() for s in block["strata"]})
        lines += ["", f"## {section}", "",
                  "| stratum | " + " | ".join(sorted(rows)) + " |",
                  "|---" * (len(rows) + 1) + "|"]
        for stratum in strata:
            cells = []
            for name in sorted(rows):
                value = rows[name]["strata"].get(stratum, {}).get("accuracy")
                cells.append(f"{value:.4f}" if value is not None else "—")
            lines.append(f"| {stratum} | " + " | ".join(cells) + " |")
        # The cells show accuracy, thus the mean must be over the same number.
        # The stored mean is over balanced accuracy and would not match.
        means = []
        for name in sorted(rows):
            values = [block.get("accuracy") for block in rows[name]["strata"].values()
                      if block.get("accuracy") is not None]
            means.append(f"{sum(values) / len(values):.4f}" if values else "—")
        lines.append("| **mean** | " + " | ".join(means) + " |")

    routers = {name: blocks["router"] for name, blocks in variants.items()
               if blocks.get("router")}
    if routers:
        lines += ["", "## Routing economics", "",
                  "Quality is the share of routed items the chosen tier answers.",
                  "Spend is relative, one unit is the small tier.", "",
                  "| strategy | tier accuracy | quality | relative spend |", "|---|---:|---:|---:|"]
        first = routers[sorted(routers)[0]]
        for strategy, values in first.items():
            lines.append(f"| {strategy} | {values['tier_accuracy']:.4f} | "
                         f"{values['quality']:.4f} | {values['relative_spend']:.2f} |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} with {len(variants)} variants")


if __name__ == "__main__":
    main()
