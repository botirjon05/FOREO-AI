# Brand-Agnostic FAQ Ingestion (with fixed export schema)
# Exports rows: brand_id, brand_name, language, product, question, answer, url, updated_at

import re
import json
import datetime as dt
import urllib.parse as urlparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Set

import requests
from bs4 import BeautifulSoup
import streamlit as st

USER_AGENT = "Mozilla/5.0 (compatible; FAQBot/1.0)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
FAQ_HINT_PATTERNS = [
    r"/faq", r"/faqs", r"/help", r"/support", r"/knowledge", r"/kb/",
    r"/guide", r"/guides", r"/documentation", r"/docs", r"/questions", r"/common-questions",
]
TIMEOUT = 15

@dataclass
class QA:
    question: str
    answer: str
    source_url: str

def normalize_url(base: str, link: str) -> Optional[str]:
    if not link: return None
    link = link.strip()
    parsed = urlparse.urlparse(link)
    if parsed.scheme in ("http", "https"):
        return link.split("#")[0]
    if link.startswith("/") or not parsed.scheme:
        return urlparse.urljoin(base, link).split("#")[0]
    return None

def same_domain(a: str, b: str) -> bool:
    return urlparse.urlparse(a).netloc.lower() == urlparse.urlparse(b).netloc.lower()

def fetch(url: str) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            return r
    except requests.RequestException:
        pass
    return None

def find_sitemap_urls(base_url: str) -> List[str]:
    cands = [
        urlparse.urljoin(base_url, "/sitemap.xml"),
        urlparse.urljoin(base_url, "/sitemap_index.xml"),
        urlparse.urljoin(base_url, "/sitemap/sitemap.xml"),
    ]
    out = []
    for c in cands:
        try:
            r = requests.get(c, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200 and "xml" in r.headers.get("Content-Type", ""):
                out.append(c)
        except requests.RequestException:
            pass
    return out

def parse_sitemap_for_faq_links(smap: str, base_url: str) -> List[str]:
    try:
        r = requests.get(smap, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "xml")
        links = []
        for loc in soup.find_all("loc"):
            u = loc.get_text().strip()
            if u.endswith(".xml") and same_domain(u, base_url):
                links.extend(parse_sitemap_for_faq_links(u, base_url))
            else:
                links.append(u)
        faq_links = [u for u in links if same_domain(u, base_url) and any(k in u.lower() for k in ["faq","help","support","knowledge","docs","guide","question"])]
        seen, out = set(), []
        for u in faq_links:
            if u not in seen:
                seen.add(u); out.append(u)
        return out
    except requests.RequestException:
        return []

def discover_candidate_pages(base_url: str, homepage_html: str) -> List[str]:
    soup = BeautifulSoup(homepage_html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        u = normalize_url(base_url, a["href"])
        if u and same_domain(u, base_url):
            links.add(u)
    return sorted([u for u in links if any(re.search(p, u, re.I) for p in FAQ_HINT_PATTERNS)])

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def extract_jsonld_faq(html: str) -> List[Tuple[str,str]]:
    soup = BeautifulSoup(html, "html.parser")
    qas = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try: data = json.loads(tag.string or "{}")
        except json.JSONDecodeError: continue
        blocks = data if isinstance(data, list) else [data]
        for b in blocks:
            if not isinstance(b, dict): continue
            if b.get("@type") == "FAQPage" and isinstance(b.get("mainEntity"), list):
                for item in b["mainEntity"]:
                    if isinstance(item, dict) and item.get("@type") in ["Question", "FAQPage"]:
                        q = item.get("name") or item.get("question") or ""
                        ans = ""
                        acc = item.get("acceptedAnswer")
                        if isinstance(acc, dict): ans = acc.get("text") or ""
                        elif isinstance(acc, list) and acc: ans = acc[0].get("text") or ""
                        if q and ans: qas.append((clean_text(q), clean_text(ans)))
    return qas

def extract_heuristic_qas(html: str) -> List[Tuple[str,str]]:
    soup = BeautifulSoup(html, "html.parser")
    qas = []
    # Headings with '?' + following blocks
    for h in soup.find_all(re.compile("^h[1-6]$")):
        q = clean_text(h.get_text(" "))
        if "?" not in q: continue
        parts = []
        for sib in h.find_all_next():
            if sib.name and re.match(r"^h[1-6]$", sib.name, re.I): break
            if sib.name in ["p","li","div"]:
                t = clean_text(sib.get_text(" "))
                if t: parts.append(t)
            if len(" ".join(parts)) > 1200: break
        if parts: qas.append((q, " ".join(parts)))
    # dt/dd pairs
    for dl in soup.find_all("dl"):
        dts, dds = dl.find_all("dt"), dl.find_all("dd")
        if len(dts)==len(dds)>0:
            for dt_, dd_ in zip(dts, dds):
                q, a = clean_text(dt_.get_text(" ")), clean_text(dd_.get_text(" "))
                if q and a: qas.append((q,a))
    # class hints
    for qnode in soup.select("[class*='question'], [class*='faq']"):
        q = clean_text(qnode.get_text(" "))
        sib = qnode.find_next_sibling()
        a = clean_text(sib.get_text(" ")) if sib else ""
        if q and a: qas.append((q,a))
    # dedupe
    seen, out = set(), []
    for q,a in qas:
        key = (q.lower(), a.lower())
        if key not in seen:
            seen.add(key); out.append((q,a))
    return out

def crawl_for_faqs(base_url: str, max_pages: int = 40):
    visited: Set[str] = set()
    to_visit: List[str] = []
    qas: List[QA] = []
    notes: List[str] = []

    home = fetch(base_url)
    if not home:
        notes.append("Failed to fetch homepage or not HTML.")
        return [], notes
    notes.append(f"Fetched homepage: {base_url}")

    candidates = discover_candidate_pages(base_url, home.text)
    if candidates:
        notes.append(f"Discovered {len(candidates)} candidate FAQ/help pages on homepage.")
        to_visit.extend(candidates)
    else:
        notes.append("No FAQ-like links on homepage.")

    for sm in find_sitemap_urls(base_url):
        notes.append(f"Found sitemap: {sm}")
        faq_links = parse_sitemap_for_faq_links(sm, base_url)
        if faq_links:
            to_visit.extend(faq_links[:50])
            notes.append(f"Added {len(faq_links)} links from sitemap likely to be FAQs/help.")

    # domain-guard + dedupe
    to_visit = [u for u in dict.fromkeys(to_visit) if same_domain(u, base_url)]
    if not to_visit:
        soup = BeautifulSoup(home.text, "html.parser")
        for a in soup.find_all("a", href=True):
            u = normalize_url(base_url, a["href"])
            if u and same_domain(u, base_url): to_visit.append(u)
        to_visit = [u for u in dict.fromkeys(to_visit) if any(k in u.lower() for k in ["faq","help","support","knowledge","docs","guide"])]

    for url in to_visit[:max_pages]:
        if url in visited: continue
        visited.add(url)
        r = fetch(url)
        if not r: continue
        html = r.text
        for q,a in extract_jsonld_faq(html): qas.append(QA(q,a,url))
        for q,a in extract_heuristic_qas(html): qas.append(QA(q,a,url))

    # final dedupe
    uniq, seen = [], set()
    for qa in qas:
        key = (qa.question.lower(), qa.answer.lower())
        if key not in seen and qa.question and qa.answer:
            seen.add(key); uniq.append(qa)

    notes.append(f"Extracted {len(uniq)} unique Q&A pairs from {len(visited)} pages.")
    return uniq, notes

# ---------------- UI ----------------

st.set_page_config(page_title="FAQ Ingestion", page_icon="📥", layout="wide")
st.title("FAQ Ingestion → Fixed Schema Export")

with st.sidebar:
    st.header("Brand metadata (applied to every row)")
    brand_id = st.text_input("brand_id", value="foreo")
    brand_name = st.text_input("brand_name", value="Foreo")
    language = st.text_input("language", value="eng")  # e.g., 'eng' or 'en'
    product = st.text_input("product", value="luna 4")  # free text

    st.divider()
    st.header("Crawl")
    base_url = st.text_input("Main website URL", value=st.session_state.get("default_url",""), placeholder="https://www.example.com")
    max_pages = st.slider("Max pages to visit", 5, 200, 60)
    st.session_state["default_url"] = base_url

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("1) Discover & Extract")
    start = st.button("Run", type="primary", disabled=not base_url)
with col2:
    st.subheader("2) Export")
    export_fmt = st.radio("Format", ["JSONL","CSV"], horizontal=True)
    fname = st.text_input("File name", value="faqs_export")
    do_export = st.button("Download Export")

if start and base_url:
    with st.spinner("Crawling and extracting FAQs..."):
        qas, notes = crawl_for_faqs(base_url, max_pages=max_pages)
    st.success(f"Done. Extracted {len(qas)} Q&A pairs.")
    st.session_state["raw_qas"] = [asdict(x) for x in qas]
    st.session_state["notes"] = notes

if "notes" in st.session_state:
    with st.expander("Crawl log / notes"):
        for n in st.session_state["notes"]:
            st.markdown(f"- {n}")

# Editable detail view
if "raw_qas" in st.session_state and st.session_state["raw_qas"]:
    st.subheader("Review & Edit")
    edited = []
    for i, r in enumerate(st.session_state["raw_qas"]):
        with st.expander(f"Q{i+1}: {r['question'][:80]}{'...' if len(r['question'])>80 else ''}"):
            q = st.text_area("Question", r["question"], key=f"q_{i}")
            a = st.text_area("Answer", r["answer"], key=f"a_{i}")
            src = st.text_input("URL", r["source_url"], key=f"s_{i}")
            edited.append({"question": q.strip(), "answer": a.strip(), "source_url": src.strip()})
    st.session_state["raw_qas"] = edited

# Build final rows with your schema
def build_rows() -> List[dict]:
    rows = []
    now_iso = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for r in st.session_state.get("raw_qas", []):
        if not r.get("question") or not r.get("answer"):
            continue
        rows.append({
            "brand_id": brand_id,
            "brand_name": brand_name,
            "language": language,
            "product": product,
            "question": r["question"],
            "answer": r["answer"],
            "url": r["source_url"],
            "updated_at": now_iso,
        })
    return rows

if do_export:
    rows = build_rows()
    if not rows:
        st.warning("Nothing to export.")
    else:
        if export_fmt == "JSONL":
            payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            st.download_button("Download JSONL", payload, file_name=f"{fname}.jsonl", mime="application/json")
        else:
            import io, csv
            buf = io.StringIO()
            fieldnames = ["brand_id","brand_name","language","product","question","answer","url","updated_at"]
            w = csv.DictWriter(buf, fieldnames=fieldnames)
            w.writeheader()
            for row in rows: w.writerow(row)
            st.download_button("Download CSV", buf.getvalue(), file_name=f"{fname}.csv", mime="text/csv")

st.markdown("---")
st.caption("Exports rows with fixed brand metadata and UTC ISO8601 updated_at. Add more products by re-running or extend UI for per-row product if needed.")
