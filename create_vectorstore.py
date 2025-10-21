# create_vectorstore.py
# Robust Chroma vectorstore creator: tries PersistentClient -> Settings -> default Client
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import sys
from typing import Iterable

# config
INPUT = Path("data/faqs_for_embedding.jsonl")
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "foreo_kb"
EMB_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

def stream_prepared(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def batches(iterable, n):
    batch=[]
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch=[]
    if batch:
        yield batch

def create_client_with_fallback(persist_dir: str):
    """
    Try persistent client constructors in order:
     1) chromadb.PersistentClient(path=...)
     2) chromadb.Client(Settings(...))
     3) chromadb.Client()
    Return (client, mode_string)
    """
    # 1) try PersistentClient (modern API)
    try:
        PersistentClient = getattr(chromadb, "PersistentClient", None)
        if PersistentClient is not None:
            print(f"[INFO] Attempting chromadb.PersistentClient(path='{persist_dir}')")
            client = PersistentClient(path=persist_dir)
            return client, "PersistentClient"
    except Exception as e:
        print("[WARN] PersistentClient constructor failed:", e)

    # 2) try legacy Settings(...) constructor
    try:
        from chromadb.config import Settings
        print("[INFO] Attempting chromadb.Client(Settings(...)) (legacy pattern)")
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
        return client, "Client+Settings"
    except Exception as e:
        print("[WARN] chromadb.Client(Settings(...)) failed:", e)

    # 3) fallback to default client
    try:
        print("[INFO] Falling back to chromadb.Client() (no explicit persistence)")
        client = chromadb.Client()
        return client, "ClientDefault"
    except Exception as e:
        print("[ERROR] chromadb.Client() also failed:", e)
        raise

def main():
    if not INPUT.exists():
        print(f"[ERROR] Input file not found: {INPUT}. Run prepare_for_embedding.py first.")
        sys.exit(1)

    print("[INFO] Loading embedding model:", EMB_MODEL)
    model = SentenceTransformer(EMB_MODEL)

    # create client
    client, mode = create_client_with_fallback(PERSIST_DIR)
    print("[INFO] Chroma client created using mode:", mode)

    # create / get collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print("[INFO] Using existing collection:", COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)
        print("[INFO] Created collection:", COLLECTION_NAME)

    items = list(stream_prepared(INPUT))
    print(f"[INFO] {len(items)} items to index")

    total = 0
    for batch in batches(items, BATCH_SIZE):
        texts = [b["text"] for b in batch]
        ids = [b["id"] for b in batch]
        metadatas = [b["metadata"] for b in batch]

        embs = model.encode(texts, show_progress_bar=False)
        embs = [e.tolist() for e in embs]

        collection.add(documents=texts, embeddings=embs, metadatas=metadatas, ids=ids)
        total += len(ids)
        print(f"[INFO] Indexed {total} / {len(items)}")

    # try to persist if available
    try:
        client.persist()
        print("[INFO] client.persist() invoked.")
    except Exception as e:
        print("[WARN] client.persist() not available on this chromadb build:", e)

    print(f"[DONE] Indexed {total} docs into Chroma collection '{COLLECTION_NAME}'.")
    print("[INFO] Persistence mode:", mode)
    print("[INFO] If mode != 'PersistentClient' or 'Client+Settings', check where files are saved (see docs).")

if __name__ == "__main__":
    main()
