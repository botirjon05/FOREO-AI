#!/usr/bin/env python3
# evaluate_rag.py — batch questions → RAG → score (0–10) and save CSV

import argparse, csv, time, re, statistics, sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

# Import your existing RAG pieces
from rag_gemma import (
    pick_device, connect_chroma, SentenceTransformer as STEmbed,  # alias to avoid name clash
    retrieve_top_k, best_question_similarity, extractive_answer, maybe_paraphrase,
    CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME,
    GEMMA_SMALL_ID, load_gemma_small, OFFTOPIC_SIM_THRESHOLD
)

NOT_ENOUGH = "Not enough information."

def token_f1(a: str, b: str) -> float:
    """Simple token F1 on lowercased whitespace tokens."""
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens); b_set = set(b_tokens)
    overlap = len(a_set & b_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_set)
    recall    = overlap / len(b_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def score_0_to_10(gold: str, pred: str, st_embedder) -> float:
    """Combined semantic + lexical score, scaled to 0–10."""
    if pred.strip() == "" or pred.strip() == NOT_ENOUGH:
        return 0.0
    # semantic cosine
    embs = st_embedder.encode([gold, pred], convert_to_tensor=True)
    cos = float(util.cos_sim(embs[0], embs[1]))
    cos = max(0.0, min(1.0, cos))
    # lexical overlap
    f1 = token_f1(gold, pred)
    # blend & scale
    blended = 0.7 * cos + 0.3 * f1
    return round(blended * 10.0, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="CSV with columns: question,gold_answer")
    ap.add_argument("--out", required=True, help="Where to write results CSV")
    ap.add_argument("--no-paraphrase", action="store_true", help="Disable Gemma paraphrase polish")
    args = ap.parse_args()

    tests_path = Path(args.tests)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load infra
    device   = pick_device()
    coll     = connect_chroma(CHROMA_DIR, COLLECTION_NAME)
    embedder = STEmbed(EMBED_MODEL_NAME)             # for retrieval & sim
    scorer_embedder = SentenceTransformer(EMBED_MODEL_NAME)  # for cosine scoring

    tokenizer = model = None
    if not args.no_paraphrase:
        try:
            tokenizer, model = load_gemma_small(GEMMA_SMALL_ID, device)
        except Exception:
            tokenizer = model = None

    # Read tests
    rows = []
    with tests_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("question") or "").strip()
            g = (r.get("gold_answer") or "").strip()
            if q and g:
                rows.append((q, g))

    if not rows:
        print("No tests found. Ensure CSV has question,gold_answer.")
        sys.exit(1)

    # Run
    results = []
    for i, (q, gold) in enumerate(rows, 1):
        t0 = time.time()
        try:
            docs, _ = retrieve_top_k(coll, embedder, q, 3)
            sim = best_question_similarity(embedder, q, docs)

            if sim < OFFTOPIC_SIM_THRESHOLD:
                pred = NOT_ENOUGH
                off_topic = True
            else:
                pred = extractive_answer(q, docs)
                off_topic = (pred == NOT_ENOUGH)
                if (not args.no_paraphrase) and model is not None and not off_topic:
                    try:
                        pred = maybe_paraphrase(tokenizer, model, device, pred)
                    except Exception:
                        pass

            pred = re.sub(r"https?://\\S+", "", pred).strip()
        except Exception as e:
            pred = f"RUNTIME_ERROR: {e}"
            off_topic = False

        t1 = time.time()
        latency = round(t1 - t0, 3)

        # Score
        sc = 0.0 if (off_topic or pred.startswith("RUNTIME_ERROR")) else score_0_to_10(gold, pred, scorer_embedder)

        results.append({
            "idx": i,
            "question": q,
            "gold_answer": gold,
            "rag_answer": pred,
            "off_topic": off_topic,
            "score_0_10": sc,
            "latency_s": latency,
        })
        print(f"[{i:02d}] score={sc:>4}  off_topic={off_topic}  {latency}s")

    # Save CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Summary
    valid_scores = [r["score_0_10"] for r in results if isinstance(r["score_0_10"], (int, float))]
    avg = round(statistics.mean(valid_scores), 2) if valid_scores else 0.0
    p_ge8 = round(100.0 * sum(s >= 8.0 for s in valid_scores) / len(valid_scores), 1) if valid_scores else 0.0
    p_ge5 = round(100.0 * sum(s >= 5.0 for s in valid_scores) / len(valid_scores), 1) if valid_scores else 0.0
    avg_lat = round(statistics.mean([r["latency_s"] for r in results]), 2)

    print("\n=== SUMMARY ===")
    print(f"Tests           : {len(results)}")
    print(f"Average score   : {avg}/10")
    print(f">=8/10          : {p_ge8}%")
    print(f">=5/10          : {p_ge5}%")
    print(f"Avg latency     : {avg_lat}s")
    print(f"Saved to        : {out_path}")

if __name__ == "__main__":
    main()