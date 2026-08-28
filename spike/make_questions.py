"""Build spike probe questions with REAL character offsets into the corpus.

These are NOT curated golden-set questions. They are mechanical probes whose
only purpose is to exercise the machinery at real scale. Difficulty is defined
by *lexical distance* from the source passage, which is a real and controllable
axis and gives a spread of recall values to measure against:

  easy   - reuses several content words from the passage
  medium - uses the document title plus one content word
  hard   - uses only the title's topic, no passage vocabulary

Recall numbers from these are optimistic and must not be read as quality.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fireassay.system.corpus import load_corpus
from fireassay.text import tokenize

STOP = {"the","a","an","and","or","of","to","in","for","on","you","your","is","are",
        "be","if","it","this","that","with","as","at","by","from","can","will","must",
        "not","do","does","have","has","they","their","there","which","when","what"}


def pick_sentence(text: str) -> tuple[str, int, int] | None:
    """First sentence of 90-260 chars containing a digit or a capitalised term."""
    for m in re.finditer(r"[^\n.]{90,260}\.", text):
        s = m.group(0).strip()
        if re.search(r"\d", s) or re.search(r"\b[A-Z]{2,}\b", s):
            start = text.index(s)
            return s, start, start + len(s)
    return None


def content_words(s: str, n: int) -> list[str]:
    seen: list[str] = []
    for w in tokenize(s):
        if len(w) > 4 and w not in STOP and w not in seen and not w.isdigit():
            seen.append(w)
        if len(seen) >= n:
            break
    return seen


def main() -> None:
    docs = load_corpus(Path("data/corpus.jsonl"))
    out: list[dict[str, object]] = []
    # spread across the corpus rather than taking the first N
    step = max(1, len(docs) // 60)
    for doc in docs[::step]:
        got = pick_sentence(doc.text)
        if not got:
            continue
        sent, cs, ce = got
        title = doc.title.rstrip(".")
        words = content_words(sent, 3)
        if len(words) < 2:
            continue
        tier = len(out) % 3
        if tier == 0:
            q, diff = f"What does the guidance say about {words[0]} and {words[1]}?", "easy"
        elif tier == 1:
            q, diff = f"{title}: what does it say about {words[0]}?", "medium"
        else:
            q, diff = f"{title} - what are the rules?", "hard"
        out.append({
            "text": q, "qtype": "factual", "difficulty": diff,
            "reference_answer": sent,
            "evidence_spans": [{"doc_id": doc.doc_id, "chunk_id": f"{doc.doc_id}#0000",
                                "char_start": cs, "char_end": ce, "quote": sent}],
            "provenance": "synthetic", "generator": "spike-probe@1.0.0",
            "source_doc_id": doc.doc_id,
        })
        if len(out) >= 51:
            break

    path = Path("spike/questions.jsonl")
    path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n")
    by = {}
    for o in out:
        by[o["difficulty"]] = by.get(o["difficulty"], 0) + 1
    print(f"wrote {len(out)} probes -> {path}   by difficulty: {by}")
    for o in out[:3]:
        print(f"  [{o['difficulty']:6}] {o['text'][:78]}")


if __name__ == "__main__":
    main()
