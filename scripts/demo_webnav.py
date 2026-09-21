#!/usr/bin/env python3
"""Navigate a website from its HTML, one decision per page.

The demo in the field: hand the model the page as structured text (title,
headings, the links with their text) and a goal, and let it pick the link.
No screenshot, no token generation. Five hops at most.

    .venv/bin/python scripts/demo_webnav.py [--span runs/x/best.pt]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

sys.path.insert(0, "src")
from curl_cffi import requests

TASKS = [
    ("https://github.com", "the pricing page", r"pricing"),
    ("https://www.python.org", "the download page for Python", r"download"),
    ("https://huggingface.co", "the documentation", r"docs"),
    ("https://www.rust-lang.org", "how to install Rust", r"install|learn/get-started"),
    ("https://fastapi.tiangolo.com", "the tutorial", r"tutorial"),
    ("https://www.postgresql.org", "the documentation", r"docs"),
    ("https://stripe.com", "the pricing page", r"pricing"),
    ("https://nodejs.org", "the download page", r"download"),
]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title, self.headings, self.links = "", [], []
        self._tag, self._href, self._text = None, None, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self._tag, self._href, self._text = "a", attrs["href"], []
        elif tag in ("title", "h1", "h2"):
            self._tag, self._text = tag, []

    def handle_data(self, data):
        if self._tag:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag != self._tag:
            return
        text = re.sub(r"\s+", " ", " ".join(self._text)).strip()
        if tag == "title":
            self.title = text
        elif tag in ("h1", "h2") and text:
            self.headings.append(text)
        elif tag == "a" and text and 2 <= len(text) <= 60:
            self.links.append((text, self._href))
        self._tag = None


def fetch(url: str) -> Page:
    response = requests.get(url, impersonate="chrome", timeout=20)
    page = Page()
    page.feed(response.text)
    page.url = str(response.url)
    return page


def state_of(page: Page, links: list[tuple[str, str]]) -> str:
    lines = [f"Page title: {page.title}", f"URL: {page.url}"]
    if page.headings:
        lines.append("Headings: " + " | ".join(page.headings[:12]))
    lines.append("Links on the page:")
    lines += [f"- {text} -> {href}" for text, href in links]
    return "\n".join(lines)


def navigate(judge, start: str, goal: str, pattern: str, hops: int, clock: list[float]):
    url, seen, path = start, set(), []
    for _ in range(hops):
        page = fetch(url)
        seen.add(url)
        host = urlparse(page.url).netloc.split(".")[-2]
        links, keys = [], set()
        for text, href in page.links:
            full = urljoin(page.url, href)
            if host not in urlparse(full).netloc or full in seen or full.startswith("mailto"):
                continue
            key = text.lower()
            if key in keys:
                continue
            keys.add(key)
            links.append((text, full))
        links = links[:40]
        if not links:
            break
        options = [text for text, _ in links]
        start_time = time.perf_counter()
        verdict = judge.decide(state_of(page, links), f"Which link leads to {goal}?", options)
        clock.append((time.perf_counter() - start_time) * 1000)
        chosen = dict(links)[verdict.option]
        path.append((verdict.option, round(verdict.confidence, 3)))
        url = chosen
        if re.search(pattern, url, re.IGNORECASE):
            return True, path
    return False, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--span", default=None)
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--out", default="results/demo_webnav.json")
    args = parser.parse_args()
    if args.span:
        from gavel.spanapi import SpanGavel
        judge = SpanGavel.from_checkpoint(args.span, max_length=4096)
    else:
        from gavel import Gavel
        judge = Gavel.from_pretrained(args.model, None, 4096)

    clock: list[float] = []
    results = []
    for start, goal, pattern in TASKS:
        try:
            found, path = navigate(judge, start, goal, pattern, args.hops, clock)
        except Exception as error:
            found, path = False, [("error", str(error)[:60])]
        results.append({"start": start, "goal": goal, "found": found, "path": path})
        print(f"{'ok ' if found else 'no '} {start} -> {goal}: {path}", flush=True)
    solved = sum(r["found"] for r in results)
    ms = round(sorted(clock)[len(clock) // 2], 1) if clock else None
    print(f"solved {solved}/{len(results)}, median {ms} ms per page decision", flush=True)
    json.dump({"model": args.span or args.model, "solved": solved, "of": len(results),
               "median_ms": ms, "tasks": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
