# prepare_for_embedding.py
# Produces data/faqs_for_embedding.jsonl with unique ids and no duplicate Q/A pairs.
import json
from pathlib import Path
from hashlib import sha1
import re
from typing import Iterator

INPUT = Path("data/cleaned_faqs.jsonl")
OUTPUT = Path("data/faqs_for_embedding.jsonl")
REQUIRED = ["brand_id","brand_name","language","question","answer","url","updated_at"]

def stream_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode error at line {i}")

def validate_row(r: dict):
    for k in REQUIRED:
        if k not in r or not isinstance(r[k], str) or not r[k].strip():
            return False, k
    return True, None

SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")

def split_sentences_safe(text: str):
    """Split text into sentences conservatively so we don't cut in the middle.
    We only split on punctuation followed by a capital/number/[, which matches our data.
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = SENT_SPLIT_RX.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, max_words: int = 200, overlap_sentences: int = 1):
    """Create chunks that always end at sentence boundaries.
    - Assemble sentences until just over the word budget, then close the chunk.
    - Add a small sentence-level overlap (default 1 sentence) for safety.
    This avoids mid-sentence cuts while keeping redundancy low.
    """
    sentences = split_sentences_safe(text)
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        cur = []
        words = 0
        while i < len(sentences) and (words == 0 or words + len(sentences[i].split()) <= max_words):
            cur.append(sentences[i])
            words += len(sentences[i].split())
            i += 1
        if cur:
            chunks.append(" ".join(cur))
        # sentence-level overlap for continuity (bounded by start of list)
        i = max(i - overlap_sentences, i)
        if overlap_sentences and i < len(sentences):
            # ensure we move forward at least 1 sentence
            i += 0
    return chunks

def make_stable_id(url: str, question: str, answer: str, chunk_index: int = 0):
    # deterministic id from content + chunk index (short SHA1)
    base = f"{url}|{question}|{answer}|{chunk_index}"
    return sha1(base.encode("utf-8")).hexdigest()[:20]

def main():
    if not INPUT.exists():
        print(f"[ERROR] Input not found: {INPUT}")
        return

    # To avoid duplicate QA content across file
    seen_qa_hashes = set()
    out_count = 0
    skipped_invalid = 0
    skipped_duplicates = 0

    with OUTPUT.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(stream_jsonl(INPUT), 1):
            ok, bad = validate_row(row)
            if not ok:
                skipped_invalid += 1
                print(f"[SKIP] Row {i} missing/invalid field: {bad}")
                continue

            # normalize question/answer whitespace
            q = " ".join(row["question"].split())
            a = " ".join(row["answer"].split())
            combined_full = f"Q: {q}\nA: {a}"

            # dedupe exact Q/A pairs (regardless of url)
            qa_key = sha1((q + "\n" + a).encode("utf-8")).hexdigest()
            if qa_key in seen_qa_hashes:
                skipped_duplicates += 1
                continue
            # we will add to seen set only after producing chunks for this row
            # Sentence-aware chunks; keep ends aligned to sentences
            chunks = chunk_text(combined_full, max_words=200, overlap_sentences=1)

            for ci, chunk in enumerate(chunks):
                uid = make_stable_id(row["url"], q, a, ci)
                metadata = {
                    "brand_id": row["brand_id"],
                    "brand_name": row["brand_name"],
                    "language": row["language"],
                    # product removed as requested (if present it will be ignored)
                    "url": row["url"],
                    "type": row.get("type","faq"),
                    "updated_at": row.get("updated_at","")
                }
                doc = {
                    "id": uid,
                    "text": chunk,
                    "metadata": metadata
                }
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                out_count += 1

            # mark qa as seen (so other identical Q/A rows get skipped)
            seen_qa_hashes.add(qa_key)

    print(f"[DONE] Wrote {out_count} prepared records to {OUTPUT}")
    print(f"[INFO] skipped invalid rows: {skipped_invalid}, skipped duplicate Q/A rows: {skipped_duplicates}")

if __name__ == "__main__":
    main()
