#!/usr/bin/env python3
# rag_gemma.py — RAG using local Gemma-3-270M (no hosted API)
# Focus: sentence-aware snippets, robust off-topic detection, grounded extractive answers.

import os, re, time, sys
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.config import Settings

# ---------------- Config ----------------
CHROMA_DIR        = "./chroma_db"
COLLECTION_NAME   = "foreo_kb"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"   # same embedder used to index
TOP_K             = 3

GEMMA_SMALL_ID    = "google/gemma-3-270m"    # tiny local model
MAX_NEW_TOKENS    = 120
TEMPERATURE       = 0.2

# Off-topic + similarity settings
OFFTOPIC_SIM_THRESHOLD = 0.32   # cosine similarity to best doc question; < threshold => off-topic
QUERY_AUGMENT_IF_GENERIC = True

OFF_TOPIC_MESSAGE = "I am a FOREO chatbot. Please ask only FOREO-related questions."

# Terms to detect support intent even if brand not named
DOMAIN_KEYWORDS = [
    "warranty","guarantee","return","refund","exchange",
    "clean","cleaning","wash","charge","charging","battery",
    "manual","guide","troubleshoot","reset","app","login","account",
    "order","shipping","delivery","payment","checkout","discount","promo","coupon","code"
]
PRODUCT_HINTS = ["foreo","luna","issa","ufo","bear","espada","iris","peach","fofo"]

# ---------------- Utilities ----------------
SENT_SPLIT_RX = re.compile(r'(?<=[.?!])\s+')
QA_RX = re.compile(r'Q:\s*(.*?)\s*A:\s*(.*)', re.IGNORECASE | re.DOTALL)

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def split_sentences(s: str) -> List[str]:
    """
    More reliable sentence splitter that preserves full sentences
    and doesn't cut off trailing clauses like 'for optimal results'.
    """
    s = normalize_spaces(s)
    if not s:
        return []
    # Split only on punctuation followed by a capital letter (likely start of next sentence)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', s)
    return [p.strip() for p in parts if p.strip()]

def to_sentences(text: str, max_sentences: int = 4) -> str:
    """
    Return up to max_sentences complete sentences.
    If last sentence seems incomplete, append one more to finish the thought.
    """
    sents = split_sentences(text)
    if not sents:
        return ""
    # Take up to max_sentences, but if last one ends abruptly (no final punctuation),
    # append one more sentence if available.
    subset = sents[:max_sentences]
    if subset and not subset[-1].endswith(('.', '?', '!')) and len(sents) > max_sentences:
        subset.append(sents[max_sentences])
    return " ".join(subset).strip()

def parse_qa(doc_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract Q and A from a 'Q: ... A: ...' block."""
    if not doc_text:
        return None, None
    m = QA_RX.search(doc_text)
    if not m:
        return None, None
    q_text = normalize_spaces(m.group(1))
    a_text = normalize_spaces(m.group(2))
    return q_text, a_text

def normalize_warranty_phrasing(a_text: str) -> str:
    """Normalize 'TWO (2) YEARS' -> '2 years', keep country exceptions if present."""
    t = a_text
    t = re.sub(r'\bTWO\s*\(2\)\s*YEARS\b', '2 years', t, flags=re.IGNORECASE)
    t = re.sub(r'\bTHREE\s*\(3\)\s*YEARS\b', '3 years', t, flags=re.IGNORECASE)
    t = re.sub(r'\bONE\s*\(1\)\s*YEAR\b', '1 year', t, flags=re.IGNORECASE)
    # compress spaces
    t = normalize_spaces(t)
    return t

def looks_like_procedure(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["how do i", "how can i", "steps", "clean", "charge", "use", "reset", "connect", "pair"])

def format_procedure_steps(a_text: str) -> str:
    """
    Extract 3–5 short steps from answer text using punctuation/commas/semicolons.
    Keeps it concise and literal from the source.
    """
    a = normalize_spaces(a_text)
    # split by sentences first; if single long sentence, split at ';' or ' and '
    sentences = split_sentences(a)
    parts: List[str] = []
    for s in sentences:
        # break long sentences by ; or ' and '
        subs = re.split(r';|\band\b', s)
        for sub in subs:
            sub = normalize_spaces(sub)
            if len(sub) >= 4:
                parts.append(sub)
        if len(parts) >= 5:
            break
    if not parts:
        parts = sentences
    steps = [p.rstrip('.').strip() for p in parts[:5]]
    # Return as a single line with bullets replaced by separators (console-friendly)
    return "; ".join(f"{i+1}) {st}" for i, st in enumerate(steps[:5]))

# ---------------- Chroma connection ----------------
def connect_chroma(path: str, collection_name: str):
    PersistentClient = getattr(chromadb, "PersistentClient", None)
    if PersistentClient:
        client = PersistentClient(path=path)
    else:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=path))
    try:
        coll = client.get_collection(collection_name)
    except Exception:
        coll = client.create_collection(collection_name)
    return coll

# ---------------- Retrieval ----------------
def maybe_augment_query(q: str) -> str:
    ql = q.lower()
    if QUERY_AUGMENT_IF_GENERIC:
        has_brand = any(p in ql for p in PRODUCT_HINTS)
        has_domain = any(k in ql for k in DOMAIN_KEYWORDS)
        # Treat "my device", "my product" as brand-related intent
        generic_device = bool(re.search(r'\bmy (device|product)\b', ql))
        if (not has_brand and (has_domain or generic_device)):
            return q + " foreo"
    return q

def retrieve_top_k(coll, embedder: SentenceTransformer, question: str, k: int):
    q_text = maybe_augment_query(question)
    q_emb = embedder.encode([q_text])[0].tolist()
    res = coll.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs[:k], metas[:k]

def best_question_similarity(embedder: SentenceTransformer, user_q: str, docs: List[str]) -> float:
    """
    Compute cosine similarity between user question and each doc's Q: text.
    Use the best as our on-topic score.
    """
    if not docs:
        return 0.0
    uq = embedder.encode(user_q, convert_to_tensor=True)
    # extract Qs; fall back to whole doc if missing
    qs = []
    for d in docs:
        q_part, _ = parse_qa(d)
        qs.append(q_part if q_part else d)
    q_embs = embedder.encode(qs, convert_to_tensor=True)
    sims = util.cos_sim(uq, q_embs)[0]
    return float(torch.max(sims))

# ---------------- Answering (extractive + tiny rephrase by Gemma if needed) ----------------
def extractive_answer(user_q: str, docs: List[str]) -> str:
    """
    Extract answer strictly from A: of the best doc (or consolidate top-2 if same info).
    For procedures: format into 3–5 short steps.
    """
    # Gather A parts from top docs
    answers = []
    for d in docs:
        _, a = parse_qa(d)
        if a:
            answers.append(a)

    if not answers:
        return "Not enough information."

    # Normalize warranty phrasing if present
    answers_norm = [normalize_warranty_phrasing(a) for a in answers]

    # For procedures, present steps
    if looks_like_procedure(user_q):
        return format_procedure_steps(answers_norm[0])

    # Otherwise, produce 1–2 clean sentences from top answer (maybe enrich with exceptions from next)
    primary = to_sentences(answers_norm[0], 2)

    # Try to append jurisdictional exception once if present in another doc and not already in primary
    if len(answers_norm) > 1:
        for a2 in answers_norm[1:]:
            if "3 years" in a2.lower() and "3 years" not in primary.lower():
                extra = to_sentences(a2, 1)
                # Append only the fragment mentioning exceptions
                primary = normalize_spaces(f"{primary} {extra}")

    # Final guard
    return primary if primary else "Not enough information."

# (Optional) very light paraphrase using Gemma, but ONLY on the extracted text
def maybe_paraphrase(tokenizer, model, device, text: str) -> str:
    # Keep it deterministic; tiny models can drift, so keep paraphrase minimal
    prompt = (
        "Rewrite the following sentence to be concise and clear without adding new facts:\n\n"
        f"{text}\n\nRewritten:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_len = inputs.input_ids.shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (exclude the prompt)
    gen_ids = out[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Remove any accidental echo of the instruction/marker the model might emit
    gen_text = re.split(r"Rewrite the following sentence.*?:", gen_text, flags=re.IGNORECASE)[0].strip()
    gen_text = gen_text.split("Rewritten:")[0].strip()
    gen_text = re.sub(r"(Rewritten:?)+$", "", gen_text, flags=re.IGNORECASE).strip()

    # Fallback to original if the paraphrase is empty
    if not gen_text:
        return to_sentences(text, 2)

    # Keep only 1–2 sentences
    return to_sentences(gen_text, 2)

# ---------------- Model (kept for optional paraphrase; core is extractive) ----------------
def load_gemma_small(model_id: str, device: torch.device):
    print(f"[INFO] Loading Gemma model: {model_id} on {device}")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model

# ---------------- Main (interactive) ----------------
def main():
    device = pick_device()
    print(f"[INFO] Device: {device}")

    # Connect to Chroma + embedder
    coll = connect_chroma(CHROMA_DIR, COLLECTION_NAME)
    print(f"[INFO] Connected to Chroma collection: {COLLECTION_NAME} (path={CHROMA_DIR})")
    print(f"[INFO] Loading embedder: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # Load Gemma (used only for tiny paraphrase polishing; answer stays grounded)
    try:
        tokenizer, model = load_gemma_small(GEMMA_SMALL_ID, device)
    except Exception as e:
        print("[WARN] Gemma-3-270M not available; proceeding without paraphrase. Reason:", e)
        tokenizer = model = None

    print("\n[READY] FOREO RAG (extractive + sentence-aware snippets). Type a question (Ctrl+C to exit).\n")
    while True:
        try:
            q = input("Q> ").strip()
            if not q:
                continue

            # Retrieve docs
            t0 = time.time()
            docs, metas = retrieve_top_k(coll, embedder, q, TOP_K)
            t1 = time.time()

            # Off-topic: use semantic similarity between user Q and the retrieved Q parts
            best_sim = best_question_similarity(embedder, q, docs)
            if best_sim < OFFTOPIC_SIM_THRESHOLD:
                print(f"\n-- ANSWER --\n{OFF_TOPIC_MESSAGE}\n")
                continue

            # Show sentence-aware snippets: Q + first 1–2 sentences of A
            print("\n-- Retrieved snippets --")
            for i, d in enumerate(docs, 1):
                dq, da = parse_qa(d)
                if dq or da:
                    dq = dq or ""
                    da_seg = to_sentences(da or "", 2)
                    snippet = normalize_spaces(f"Q: {dq} A: {da_seg}")
                else:
                    snippet = to_sentences(d, 2)
                print(f"[{i}] {snippet}")
            print(f"[INFO] Retrieval in {t1 - t0:.2f}s")

            # Extractive grounded answer
            ans = extractive_answer(q, docs)

            # Optional: light paraphrase for fluency (keeps facts intact)
            if model is not None and ans not in {"Not enough information.", ""}:
                try:
                    ans = maybe_paraphrase(tokenizer, model, device, ans)
                except Exception:
                    pass

            # Clean
            ans = re.sub(r"https?://\S+", "", ans).strip()

            print(f"\n-- ANSWER --\n{ans}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("[ERROR] Runtime error:", e)

if __name__ == "__main__":
    main()