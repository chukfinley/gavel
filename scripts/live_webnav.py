#!/usr/bin/env python3
"""Navigate a website in a real browser and show every decision as it happens.

Each hop writes one JSON line to the trace file and two screenshots: the
page as the model saw it, and the same page with the chosen link marked.
The WebUI reads the trace while it grows. The model gets what the demo in
the field gets: the page as text (title, headings, the visible links),
never the picture.

    .venv/bin/python scripts/live_webnav.py --span runs/x/best.pt \\
        --url https://stripe.com --goal "the pricing page" --trace-dir _scratch/traces/x
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, "src")

LINKS_JS = """
() => {
  const out = [];
  const seen = new Set();
  for (const a of document.querySelectorAll('a[href]')) {
    const r = a.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    const style = getComputedStyle(a);
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    const text = (a.innerText || a.getAttribute('aria-label') || a.title || '').replace(/\\s+/g, ' ').trim();
    if (text.length < 2 || text.length > 80) continue;
    const href = a.href || '';
    if (!href.startsWith('http')) continue;
    if (href.split('#')[0] === location.href.split('#')[0]) continue;   // same page, other anchor
    const key = text.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const main = !!a.closest('main, [role=main], #main, #mainContent, #content, article, .main');
    const chrome = !!a.closest('header, nav, footer, [role=navigation], [role=banner], [role=contentinfo], #navbar, #nav-main, #navFooter');
    out.push({text, href, main, chrome, y: r.y + window.scrollY, index: out.length});
    a.dataset.gavelIndex = String(out.length - 1);
  }
  // Content first, then navigation, then the rest; the model reads the top of the list first.
  out.sort((p, q) => (q.main - p.main) || (p.chrome - q.chrome) || (p.y - q.y));
  return {title: document.title,
          headings: [...document.querySelectorAll('h1,h2')].map(h => h.innerText.replace(/\\s+/g, ' ').trim()).filter(Boolean).slice(0, 12),
          links: out};
}
"""
MARK_JS = """
(index) => {
  const a = document.querySelector('a[data-gavel-index="' + index + '"]');
  if (!a) return false;
  a.scrollIntoView({block: 'center', inline: 'nearest'});
  const r = a.getBoundingClientRect();
  const box = document.createElement('div');
  box.style.cssText = 'position:fixed;z-index:2147483647;pointer-events:none;border:3px solid #f5b642;border-radius:6px;box-shadow:0 0 0 4px rgba(245,182,66,.35), 0 0 0 9999px rgba(0,0,0,.35);'
    + 'left:' + (r.x - 6) + 'px;top:' + (r.y - 6) + 'px;width:' + (r.width + 12) + 'px;height:' + (r.height + 12) + 'px;';
  const label = document.createElement('div');
  label.textContent = 'gavel picks this';
  label.style.cssText = 'position:absolute;left:0;top:-26px;background:#f5b642;color:#1a1200;font:600 12px system-ui;padding:3px 8px;border-radius:5px;white-space:nowrap';
  box.appendChild(label);
  document.body.appendChild(box);
  return true;
}
"""


def state_of(page: dict, links: list[dict], url: str) -> str:
    lines = [f"Page title: {page['title']}", f"URL: {url}"]
    if page["headings"]:
        lines.append("Headings: " + " | ".join(page["headings"]))
    lines.append("Links on the page:")
    lines += [f"- {link['text']} -> {link['href']}" for link in links]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--span", default=None)
    parser.add_argument("--url", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--pattern", default=None, help="regex on the URL that means the goal is reached")
    parser.add_argument("--hops", type=int, default=5)
    parser.add_argument("--max-links", type=int, default=120)
    parser.add_argument("--jev", action="store_true", help="the closed model over OpenRouter")
    parser.add_argument("--record", default="",
                        help="append every decision as a training row (label = the choice made)")
    parser.add_argument("--trace-dir", required=True)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = open(trace_dir / "trace.jsonl", "a")

    def emit(**record) -> None:
        trace.write(json.dumps(record) + "\n")
        trace.flush()

    pattern = args.pattern or "|".join(re.escape(w) for w in re.findall(r"[a-z]{4,}", args.goal.lower())
                                       if w not in {"page", "with", "that", "this", "from", "into", "what"})
    emit(kind="start", url=args.url, goal=args.goal, pattern=pattern,
         model="typesafe/jev-1.13" if args.jev else (args.span or args.model))

    if args.jev:
        from gavel.jevapi import JevGavel
        judge = JevGavel()
    elif args.span:
        from gavel.spanapi import SpanGavel
        judge = SpanGavel.from_checkpoint(args.span, max_length=4096)
    else:
        from gavel import Gavel
        judge = Gavel.from_pretrained(args.model, None, 4096)
    emit(kind="loaded", device=str(judge.device))
    recorder = open(args.record, "a") if args.record else None

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800},
                                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36")
        url, seen, found = args.url, set(), False
        # A link text chosen once is not offered again: Amazon's category
        # link carries new tracking parameters on every page, so the URL
        # filter alone let the first harvest pick "Computer & Tablets" six
        # times in a row (2026-09-21).
        chosen_texts: set[str] = set()
        for step in range(1, args.hops + 1):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(1200)
            except Exception as error:
                emit(kind="error", step=step, url=url, message=str(error)[:200])
                break
            seen.add(page.url)
            info = page.evaluate(LINKS_JS)
            host = urlparse(page.url).netloc.split(".")[-2:]
            links = [link for link in info["links"]
                     if urlparse(link["href"]).netloc.split(".")[-2:] == host
                     and link["href"] not in seen and not link["href"].startswith("mailto")
                     and link["text"].strip().lower() not in chosen_texts][: args.max_links]
            shot_page = f"step{step}_page.png"
            page.screenshot(path=str(trace_dir / shot_page))
            if not links:
                emit(kind="dead_end", step=step, url=page.url, title=info["title"], shot=shot_page)
                break
            state = state_of(info, links, page.url)
            options = [link["text"] for link in links]
            started = time.perf_counter()
            verdict = judge.decide(state, f"Which link leads to {args.goal}?", options)
            ms = round((time.perf_counter() - started) * 1000, 1)
            chosen = next(link for link in links if link["text"] == verdict.option)
            chosen_texts.add(chosen["text"].strip().lower())
            seen.add(chosen["href"])
            if recorder is not None:
                import hashlib
                row_id = hashlib.sha1(f"{page.url}|{args.goal}|{step}".encode()).hexdigest()[:20]
                recorder.write(json.dumps({
                    "id": row_id, "state": state, "question": f"Which link leads to {args.goal}?",
                    "options": [{"id": f"o{k}", "description": link["text"]} for k, link in enumerate(links)],
                    "label": links.index(chosen), "source": "webnav-jev" if args.jev else "webnav-ours",
                    "task": "choice",
                    "meta": {"url": page.url, "goal": args.goal, "href": chosen["href"],
                             "teacher": [verdict.probabilities.get(link["text"], 0.0) for link in links]
                             if args.jev else None}}) + "\n")
                recorder.flush()
            try:
                page.evaluate(MARK_JS, chosen["index"])
                page.wait_for_timeout(150)
            except Exception:
                pass
            shot_pick = f"step{step}_pick.png"
            page.screenshot(path=str(trace_dir / shot_pick))
            found = bool(pattern) and re.search(pattern, chosen["href"], re.IGNORECASE) is not None
            emit(kind="step", step=step, url=page.url, title=info["title"], ms=ms,
                 usd=round(getattr(judge, "spent_usd", 0.0), 5),
                 chosen=chosen["text"], href=chosen["href"], confidence=verdict.confidence,
                 options=[{"text": link["text"], "href": link["href"],
                           "p": verdict.probabilities.get(link["text"], 0.0)} for link in links],
                 state_chars=len(state), shot=shot_pick, shot_page=shot_page, found=found)
            if found:
                try:
                    page.goto(chosen["href"], wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(800)
                    page.screenshot(path=str(trace_dir / "final.png"))
                    emit(kind="arrived", url=page.url, title=page.title(), shot="final.png")
                except Exception:
                    pass
                break
            url = chosen["href"]
        browser.close()
    emit(kind="done", found=found)


if __name__ == "__main__":
    main()
