# clean_faqs.py
import json, re, hashlib, sys, datetime as dt

def clean_text(t): return re.sub(r"\s+", " ", (t or "").strip())

def row_key(r):
    base = f"{r.get('brand_id')}|{r.get('brand_name')}|{clean_text(r.get('product',''))}|{clean_text(r['question'])}|{clean_text(r['answer'])}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def main(inp, outp):
    seen, out = set(), []
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            r["question"] = clean_text(r.get("question",""))
            r["answer"] = clean_text(r.get("answer",""))
            r["url"] = clean_text(r.get("url",""))
            r["updated_at"] = r.get("updated_at") or now
            k = row_key(r)
            if k in seen: continue
            seen.add(k); out.append(r)
    with open(outp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(out)} rows")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
