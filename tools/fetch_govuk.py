#!/usr/bin/env python3
"""Fetch a gov.uk corpus into JSONL for benchgate.

Content is Crown copyright, reusable under the Open Government Licence v3.0:
"Contains public sector information licensed under the Open Government Licence v3.0."

Uses the public search + content APIs (no auth). Polite: serial requests with a
delay, resumable via the output file so a re-run does not refetch.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

UA = "benchgate-corpus/0.1 (research; contact via repo)"
SEARCH = "https://www.gov.uk/api/search.json"
CONTENT = "https://www.gov.uk/api/content"


def get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Text(HTMLParser):
    """Minimal HTML→text. Keeps block boundaries, drops tags and script/style."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "li", "h1", "h2", "h3", "h4", "div", "tr", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        out = "".join(self.parts)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n\s*\n\s*", "\n\n", out)
        return out.strip()


def to_text(html: str) -> str:
    p = Text()
    p.feed(html)
    return p.text()


def body_of(details: dict) -> str:
    """gov.uk stores the body in three different shapes depending on document type.

    `answer`/`detailed_guide` put HTML in `body`; `guide` splits it across `parts`;
    `transaction` has no body at all, only short framing fields around a link out
    to the actual service. We assemble all three rather than assuming one, because
    a silent empty body would look identical to a genuinely short page.
    """
    if isinstance(details.get("body"), str):
        return to_text(details["body"])
    if details.get("parts"):
        chunks = [
            f"## {part.get('title', '')}\n{to_text(part.get('body', ''))}"
            for part in details["parts"]
        ]
        return "\n\n".join(chunks).strip()
    framing = [
        to_text(details[k])
        for k in ("introductory_paragraph", "more_information", "other_ways_to_apply")
        if isinstance(details.get(k), str)
    ]
    return "\n\n".join(x for x in framing if x).strip()


def enumerate_links(doc_types: list[str], want: int) -> list[str]:
    """Enumerate by document_type, not purpose supergroup.

    The `services` supergroup is dominated by `transaction` pages, which are thin
    start-a-service stubs with no real body. `answer` and `guide` carry the
    support-shaped prose we actually want to retrieve over.
    """
    links: list[str] = []
    start = 0
    while len(links) < want:
        q = urllib.parse.urlencode(
            {
                "filter_content_store_document_type": doc_types,
                "count": min(100, want - len(links)),
                "start": start,
                "fields": "link",
                "order": "-popularity",
            },
            doseq=True,
        )
        results = get(f"{SEARCH}?{q}").get("results", [])
        if not results:
            break
        links.extend(r["link"] for r in results if r.get("link", "").startswith("/"))
        start += len(results)
        time.sleep(0.2)
    return links[:want]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("corpus.jsonl"))
    ap.add_argument("--doc-types", default="answer,guide,detailed_guide")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--delay", type=float, default=0.25)
    args = ap.parse_args()

    seen: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["doc_id"])
        print(f"resuming: {len(seen)} already fetched", file=sys.stderr)

    links = enumerate_links([t.strip() for t in args.doc_types.split(",")], args.limit)
    print(f"enumerated {len(links)} links", file=sys.stderr)

    kept = skipped = 0
    with args.out.open("a") as fh:
        for i, link in enumerate(links, 1):
            doc_id = link.strip("/").replace("/", "_")
            if doc_id in seen:
                continue
            try:
                d = get(f"{CONTENT}{link}")
                text = body_of(d.get("details", {}))
                if len(text) < args.min_chars:
                    skipped += 1
                else:
                    fh.write(
                        json.dumps(
                            {
                                "doc_id": doc_id,
                                "title": d.get("title", ""),
                                "url": f"https://www.gov.uk{link}",
                                "updated_at": d.get("public_updated_at"),
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fh.flush()
                    kept += 1
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"  !! {link}: {e}", file=sys.stderr)
                skipped += 1
            if i % 25 == 0:
                print(f"  {i}/{len(links)} (kept {kept}, skipped {skipped})", file=sys.stderr)
            time.sleep(args.delay)

    print(f"done: kept {kept}, skipped {skipped} → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
