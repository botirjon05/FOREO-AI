#!/usr/bin/env python3
# app.py — Streamlit Chat UI for RAG Gemma chatbot (polished FOREO styling)

from dotenv import load_dotenv
from streamlit import query_params

load_dotenv()

import os
import re
import time
import base64
import json
import uuid
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

from db import init_db, get_session
import db as db_module  # shared SQLAlchemy helpers

from rag_gemma import (
    pick_device,
    connect_chroma,
    SentenceTransformer,
    retrieve_top_k,
    best_question_similarity,
    extractive_answer,
    maybe_paraphrase,
    GEMMA_SMALL_ID,
    OFFTOPIC_SIM_THRESHOLD,
    OFF_TOPIC_MESSAGE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    load_gemma_small,
)

from intent_detection import (
    classify_intent,
    extract_device_type,
    simple_extract_slots,
    needs_clarification,
    extract_country,
    extract_region,
    extract_email,
    extract_name,
    needs_ticket_info,
)
from troubleshooting import get_troubleshooting_steps
from ticket_management import create_ticket

# Upload + indexing helpers
from document_loader import load_document
from document_chunker import chunk_documents
from lc_index_uploaded import index_uploaded_chunks

# Uploaded-doc answering (HF API generation inside this module)
from uploaded_docs_answering import answer_from_uploaded_docs

from dataset_schema import infer_schema, normalize_dataframe
from generic_structured_engine import answer_structured  # Import the structured data answer engine

# -----------------------------
# Streamlit Setup & Styling
# -----------------------------
st.set_page_config(
    page_title="FOREO Chatbot (Gemma-3-270M RAG)",
    page_icon="💗",
    layout="centered"
)

# Global CSS for background and chat bubbles
st.markdown('''<style>
body, .stApp {
  background: radial-gradient(1200px 600px at 15% 0%, #ffeaf3 0%, rgba(255,234,243,0) 60%),
              radial-gradient(1000px 500px at 100% 20%, #efe7ff 0%, rgba(239,231,255,0) 55%),
              linear-gradient(180deg, #faf8fc 0%, #f7f5fb 100%);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
}

.chat-wrap {
  margin: 26px auto 90px auto;
  background: rgba(255,255,255,0.72);
  backdrop-filter: blur(8px);
}

.header {
  display:flex; align-items:center; gap:14px;
  padding: 12px 14px;
  border-radius: 14px;
  background: linear-gradient(135deg, #ff8cc6 0%, #a78bfa 100%);
  color:#fff;
  box-shadow: 0 6px 18px rgba(167,139,250,.24);
  margin-bottom: 10px;
  margin-top: -30px;
}
.header .title { font-weight: 800; letter-spacing:.3px; font-size: 1.4rem; text-shadow: 0 1px 3px rgba(0,0,0,.25); }
.header .sub { font-size: .9rem; opacity: .9; margin-top: 2px; }

.logo-wrap { width: 48px; height: 48px; border-radius: 50%; background: rgba(255,255,255,.22);
  display:flex; align-items:center; justify-content:center; border: 1px solid rgba(255,255,255,.35); overflow:hidden; }

.chat-bubble {
  border-radius: 18px; padding: 10px 14px; margin: 8px 0; line-height: 1.5; max-width: 88%;
  word-wrap: break-word; box-shadow: 0 6px 16px rgba(15, 23, 42, .06); animation: fadeIn .12s ease-in;
}
@keyframes fadeIn { from {opacity:0; transform: translateY(4px)} to {opacity:1; transform:none} }
.user-bubble { margin-left: auto; background: linear-gradient(135deg, #87b7ff 0%, #6ae0ea 100%); color: #fff; }
.bot-bubble { margin-right: auto; color: #0f172a; background: #f5f6fb; border: 1px solid #eef1f6; }

.think-bubble { display:inline-flex; align-items:center; gap:8px; margin-right: auto; color:#6b7280;
  background:#f5f6fb; border:1px solid #eef1f6; padding: 10px 14px; border-radius: 18px; }
.dot { width:6px; height:6px; border-radius:50%; background:#a3a8b6; display:inline-block; animation: blink 1.2s infinite;}
.dot:nth-child(2){ animation-delay:.2s;} .dot:nth-child(3){ animation-delay:.4s;}
@keyframes blink { 0%, 80%, 100% { opacity:.25 } 40% { opacity:1 } }

.footer { text-align:center; color:#8a90a6; font-size:.88rem; margin-top: 12px; }
</style>''', unsafe_allow_html=True)

# -----------------------------
# Secrets and Helper Functions
# -----------------------------
def _get_secret(name: str, default: str = "") -> str:
    """Read from st.secrets first, fallback to env var."""
    try:
        if name in st.secrets:
            v = st.secrets.get(name, default)
            return v if v is not None else default
    except Exception:
        pass
    return os.getenv(name, default)

def explain_with_ollama(text: str) -> str:
    """Use local Ollama API (Mistral model) to generate a short explanation."""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"Explain the following result in one short sentence:\n{text}",
                "stream": False
            },
            timeout=30
        )
        r.raise_for_status()
        resp_json = r.json()
        return resp_json.get("response", "").strip() if isinstance(resp_json, dict) else "".strip()
    except Exception:
        return text

def paraphrase_with_gemma_api(text: str) -> str:
    """Optional: Paraphrase text via HuggingFace API (Gemma model) for conciseness."""
    token = _get_secret("HF_TOKEN", "")
    if not token:
        return text  # Skip if no API token available
    model = _get_secret("GEMMA_API_MODEL", "google/gemma-2-2b-it")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    prompt = (
        "Paraphrase the following answer to be concise, friendly, and avoid duplication. "
        "Do not add new facts. Keep a brand-safe tone.\n\nAnswer:\n"
        f"{text}\n\nParaphrase:"
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 128, "temperature": 0.3}}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        # The HF API might return a list or dict depending on the model
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
            out = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            out = data["generated_text"]
        else:
            out = str(data)
        if "Paraphrase:" in out:
            out = out.split("Paraphrase:", 1)[-1].strip()
        return out.strip() if out.strip() else text
    except Exception:
        return text

def _looks_like_structured_question(q: str) -> bool:
    q = (q or "").lower()

    # Questions about columns/schema
    if any(k in q for k in [
        "column", "columns", "schema", "fields", "header", "headers",
        "what columns", "show columns", "list columns"
    ]):
        return True

    # Analytics/aggregation keywords
    if any(k in q for k in [
        "highest", "lowest", "max", "min", "average", "mean", "sum", "total",
        "count", "rows", "top", "bottom", "compare", "correlation", "trend"
    ]):
        return True

    # Common “data” words
    if any(k in q for k in [
        "dataset", "csv", "excel", "table", "dataframe", "report"
    ]):
        return True

    return False

def _infer_device_from_history(messages) -> str:
    """Infer device type from the user's last few messages (if mentioned)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        d = extract_device_type(msg.get("content", ""))
        if d:
            return d
    return ""


def try_answer_csv(query: str, file_path: str):
    """Attempt to answer the query using an uploaded structured CSV or Excel file."""
    # Use cached DataFrame if available
    cached_path = st.session_state.get("uploaded_df_path")
    df = st.session_state.get("uploaded_df")

    if df is None or cached_path != file_path:
        try:
            df_raw = load_structured_file_safely(file_path)
            df = normalize_dataframe(df_raw)
            st.session_state["uploaded_df"] = df
            st.session_state["uploaded_df_path"] = file_path

            schema = infer_schema(df)
            st.session_state["uploaded_schema"] = schema
        except Exception:
            return None, None


    # Use existing inferred schema if available; otherwise infer new schema
    schema = st.session_state.get("uploaded_schema")
    if schema is None:
        try:
            schema = infer_schema(df)
            st.session_state["uploaded_schema"] = schema
        except Exception:
            schema = None

    # Normalize DataFrame for consistent querying
    try:
        df_normalized = normalize_dataframe(df)
    except Exception:
        df_normalized = df

    if not _looks_like_structured_question(query):
        return None, None

    # Use the structured data engine to get an answer
    try:
        if schema is not None:
            answer = answer_structured(query, df_normalized, schema)
        else:
            answer = answer_structured(query, df_normalized, {})
        # `answer_structured` returns either a string (answer) or a tuple/list/dict including evidence
    except Exception:
        return None, None

    if answer is None or (isinstance(answer, str) and answer.strip() == ""):
        return None, None

    # Parse the answer output (which might include an evidence or source)
    result = None
    evidence = None
    if isinstance(answer, (tuple, list)):
        if len(answer) >= 2:
            result, evidence = answer[0], answer[1]
        elif len(answer) == 1:
            result = answer[0]
    elif isinstance(answer, dict):
        result = answer.get("answer") or answer.get("result") or answer.get("output")
        evidence = answer.get("evidence") or answer.get("source")
    else:
        result = answer

    if result is not None:
        result = str(result)
    if evidence is not None:
        evidence = str(evidence)

    return result, evidence


def load_structured_file_safely(file_path: str) -> pd.DataFrame:
    """
    Loads CSV/XLSX robustly.
    Tries multiple header rows and selects the one with the fewest Unnamed columns.
    Also drops completely empty columns.
    """
    ext = file_path.lower().split(".")[-1]

    def score_columns(cols) -> int:
        # lower score = better
        unnamed = sum(str(c).lower().startswith("unnamed") for c in cols)
        empty = sum(str(c).strip() == "" for c in cols)
        return unnamed * 10 + empty

    def post_clean(df: pd.DataFrame) -> pd.DataFrame:
        # drop fully empty columns
        df = df.dropna(axis=1, how="all")
        # normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        return df

    candidates = []

    if ext == "csv":
        for h in [0, 1, 2, 3]:
            try:
                df = pd.read_csv(file_path, header=h)
                df = post_clean(df)
                candidates.append((score_columns(df.columns), df))
            except Exception:
                pass

    elif ext in ("xls", "xlsx"):
        for h in [0, 1, 2, 3]:
            try:
                df = pd.read_excel(file_path, header=h)
                df = post_clean(df)
                candidates.append((score_columns(df.columns), df))
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported structured file type: {ext}")

    if not candidates:
        raise ValueError("Could not load structured file with any header row.")

    # pick best candidate (lowest score)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]



def _make_dataset_id(company_id: str, filename: str) -> str:
    """Stable-ish id per company + filename (simple and predictable)"""
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", filename.strip())
    return f"{company_id}::{safe}"

def set_active_dataset(dataset_id: str):
    """
    Sync the chosen dataset into your Existing single-dataset keys:
        -latest_structured_file
        -uploaded_df
        -uploaded_schema
    So your current answering path keeps working without refactoring
    """

    ds = st.session_state["datasets"].get(dataset_id)
    if not ds:
        return

    st.session_state["active_dataset_id"] = dataset_id

    if ds.get("type") == "structured":
        st.session_state["latest_structured_file"] = ds.get("path")
        st.session_state["uploaded_df"] = ds.get("df")
        st.session_state["uploaded_schema"] = ds.get("schema")
        st.session_state["uploaded_df_path"] = ds.get("path")

# -----------------------------
# Initialize Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your FOREO assistant. Ask me about warranty, cleaning, charging, orders, or account help."}
    ]
if "failed_attempts" not in st.session_state:
    st.session_state.failed_attempts = 0
if "ticket_state" not in st.session_state:
    st.session_state.ticket_state = None
if "ticket_slots" not in st.session_state:
    st.session_state["ticket_slots"] = {}
if "use_uploaded_kb" not in st.session_state:
    st.session_state.use_uploaded_kb = True
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "ticket_just_created" not in st.session_state:
    st.session_state.ticket_just_created = False

device = pick_device()


# Dataset registry (multi-file support)

if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}

if "active_dataset_id" not in st.session_state:
    st.session_state["active_dataset_id"] = None

@st.cache_resource
def load_all_components():
    """Load embeddings & DB; load local Gemma model if enabled."""
    coll = connect_chroma(CHROMA_DIR, COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    tokenizer = model = None
    if _get_secret("USE_LOCAL_GEMMA", os.getenv("USE_LOCAL_GEMMA", "0")) == "1":
        try:
            tokenizer, model = load_gemma_small(GEMMA_SMALL_ID, device)
        except Exception:
            tokenizer = model = None
    return coll, embedder, tokenizer, model

coll, embedder, tokenizer, model = load_all_components()
init_db()

# -----------------------------
# Chat Interface and Logo
# -----------------------------
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

# Top logo and header
st.markdown('''<style>
.logo-top { text-align: center; margin-top: -100px; margin-bottom: -100px; }
.logo-top img { max-width: 220px; width: 50%; height: auto; opacity: 0.96; transition: transform 0.3s ease, opacity 0.3s ease; }
.logo-top img:hover { transform: scale(1.05); opacity: 1; }
@media (max-width: 768px) { .logo-top img { max-width: 160px; width: 35%; } }
</style>''', unsafe_allow_html=True)

LOGO_LOCAL_PATH = "assets/foreo_logo.png"
LOGO_URL_ENV = os.environ.get("FOREO_LOGO_URL", "").strip()

b64_logo = None
header_logo_html = "💗"  # default emoji if no image
if os.path.exists(LOGO_LOCAL_PATH):
    with open(LOGO_LOCAL_PATH, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode()
    header_logo_html = f'<img src="data:image/png;base64,{b64_logo}" width="30" height="30" alt="FOREO" />'
elif LOGO_URL_ENV:
    header_logo_html = f'<img src="{LOGO_URL_ENV}" width="30" height="30" alt="FOREO" />'

if b64_logo:
    st.markdown(f'<div class="logo-top"><img src="data:image/png;base64,{b64_logo}" alt="FOREO Logo"></div>', unsafe_allow_html=True)

st.markdown(
    f"""
<div class="header">
  <div class="logo-wrap">{header_logo_html}</div>
  <div>
    <div class="title">FOREO AI Assistant</div>
    <div class="sub">RAG + Reasoning Loop + Gemma-3-270M</div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    css_class = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {css_class}">{content}</div>', unsafe_allow_html=True)

# -----------------------------
# File Upload & Indexing Section
# -----------------------------
with st.container():
    uploaded_file = st.file_uploader(
        "📎 Upload a document",
        type=["pdf", "csv", "xlsx"],
        label_visibility="collapsed"
    )

company_id = st.session_state.get("company_id", "demo_company")
file_path = None

if uploaded_file is not None:
    upload_dir = f"data/uploads/{company_id}"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    # Save the uploaded file to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"📄 {uploaded_file.name} uploaded successfully")

    dataset_id = _make_dataset_id(company_id, uploaded_file.name)
    ext = uploaded_file.name.lower().split(".")[-1]

    ds_record = {
        "id": dataset_id,
        "company_id": company_id,
        "name": uploaded_file.name,
        "path": file_path,
        "ext": ext,
        "uploaded_at": datetime.utcnow().isoformat(),
        "type": "unstructured",
        "df": None,
        "schema": None,
    }

    # If structured => load df + schema
    if uploaded_file.name.lower().endswith((".csv", ".xlsx")):
        try:
            df_raw = load_structured_file_safely(file_path)

            # ✅ normalize BEFORE schema
            df_norm = normalize_dataframe(df_raw)

            for col in df_norm.columns:
                if col.lower().replace(" ", "_") in ("quality_score", "score", "rating"):
                    df_norm[col] = pd.to_numeric(df_norm[col], errors = "coerce")
            schema = infer_schema(df_norm)

            ds_record["type"] = "structured"
            ds_record["df"] = df_norm
            ds_record["schema"] = schema

        except Exception as e:
            st.warning(f"Could not read structured file: {e}")

    # Store into registry
    st.session_state["datasets"][dataset_id] = ds_record
    set_active_dataset(dataset_id)

    if ds_record["type"] == "structured":
        st.session_state["latest_structured_file"] = file_path
    else:
        st.session_state["latest_unstructured_file"] = None


# If a CSV/Excel is uploaded, show indexing and toggle options
if uploaded_file is not None and uploaded_file.name.lower().endswith((".csv", ".xlsx")):
    # Save latest structured file path
    st.session_state["latest_structured_file"] = file_path

    colA, colB = st.columns([1, 2])
    with colA:
        index_clicked = st.button("Index document")
    with colB:
        st.session_state.use_uploaded_kb = st.toggle(
            "Use uploaded docs in answers",
            value=st.session_state.use_uploaded_kb
        )

    if index_clicked:
        if uploaded_file.name in st.session_state.indexed_files:
            st.info("Already indexed ✅")
        else:
            with st.spinner("Indexing document into vector DB..."):
                docs = load_document(file_path)
                chunks = chunk_documents(docs)
                count = index_uploaded_chunks(chunks, company_id, uploaded_file.name)
                st.session_state.indexed_files.add(uploaded_file.name)
            st.success(f"Indexed {count} chunks/rows ✅")

# -----------------------------
# Dataset + Analysis panel (sidebar)
# -----------------------------
datasets = st.session_state.get("datasets", {})
active_id = st.session_state.get("active_dataset_id")

st.sidebar.markdown("## 📁 Uploaded files")

if not datasets:
    st.sidebar.info("Upload a file to begin.")
else:
    # ---- Active dataset picker ----
    dataset_ids = list(datasets.keys())
    label_map = {did: datasets[did].get("name", did) for did in dataset_ids}

    default_index = dataset_ids.index(active_id) if active_id in dataset_ids else 0
    chosen = st.sidebar.radio(
        "Active dataset",
        options=dataset_ids,
        format_func=lambda did: label_map.get(did, did),
        index=default_index
    )
    if chosen and chosen != active_id:
        set_active_dataset(chosen)
        active_id = chosen

    ds = datasets.get(active_id)

    # ---- Quick info ----
    if ds:
        st.sidebar.caption(f"Type: **{ds.get('type', 'unknown')}**")
        if ds.get("type") == "structured" and ds.get("df") is not None:
            df = ds["df"]
            st.sidebar.caption(f"Rows: **{len(df)}** | Cols: **{len(df.columns)}**")

    st.sidebar.markdown("---")

    # =============================
    # 📊 Analysis panel (ONLY structured datasets)
    # =============================
    if ds and ds.get("type") == "structured" and ds.get("df") is not None and ds.get("schema") is not None:
        df = ds["df"]
        schema = ds["schema"]

        st.sidebar.markdown("## 📊 Analysis")

        # ---- Variables (columns) in checkbox format ----
        # schema expected: {col: {"role": "numeric"/"categorical"/"datetime"/"text", ...}, ...}
        all_cols = list(df.columns)

        # Split columns by role (safer UI)
        numeric_cols = [c for c in all_cols if schema.get(c, {}).get("role") == "numeric"]
        cat_cols = [c for c in all_cols if schema.get(c, {}).get("role") in ("categorical", "text")]
        dt_cols = [c for c in all_cols if schema.get(c, {}).get("role") == "datetime"]

        # Store selections in session
        var_key = f"vars_{active_id}"
        if var_key not in st.session_state:
            st.session_state[var_key] = []

        st.sidebar.write("### ✅ Variables")
        # A clean multi-select is more scalable than 30 checkboxes, but still “checkbox format”
        st.session_state[var_key] = st.sidebar.multiselect(
            "Select variables (columns)",
            options=all_cols,
            default=st.session_state[var_key],
        )

        # ---- Typical analysis types (checkbox format) ----
        st.sidebar.write("### ✅ Typical analysis")

        # Keep these stable (no surprises)
        analysis_options = [
            "Highest (max)",
            "Lowest (min)",
            "Average (mean)",
            "Sum (total)",
            "Count rows",
            "Show first row",
            "Show rows for a date",
            "Compare two dates",
            "Correlation (numeric vs numeric)",
            "Top category by metric",
            "Bottom category by metric",
        ]

        ana_key = f"analysis_{active_id}"
        if ana_key not in st.session_state:
            st.session_state[ana_key] = []

        st.session_state[ana_key] = st.sidebar.multiselect(
            "Select analysis",
            options=analysis_options,
            default=st.session_state[ana_key],
        )

        # ---- Extra parameters (only when needed) ----
        # Target metric (numeric)
        target_metric = None
        if any(x in st.session_state[ana_key] for x in ["Highest (max)", "Lowest (min)", "Average (mean)", "Sum (total)"]):
            if numeric_cols:
                target_metric = st.sidebar.selectbox("Target metric", numeric_cols, index=0)
            else:
                st.sidebar.warning("No numeric columns detected for max/min/avg/sum.")

        # Date inputs (for date-based queries)
        date_value = ""
        if "Show rows for a date" in st.session_state[ana_key]:
            date_value = st.sidebar.text_input("Date (YYYY-MM-DD)", value="")

        # Compare dates inputs
        d1, d2 = "", ""
        if "Compare two dates" in st.session_state[ana_key]:
            d1 = st.sidebar.text_input("Date 1 (YYYY-MM-DD)", value="", key=f"d1_{active_id}")
            d2 = st.sidebar.text_input("Date 2 (YYYY-MM-DD)", value="", key=f"d2_{active_id}")

        if "Compare two dates" in st.session_state[ana_key] and numeric_cols and target_metric is None:
            target_metric = st.sidebar.selectbox("Metric for compare", numeric_cols, index=0, key = f"metric_compare_{active_id}")

        # Correlation params (numeric vs numeric)
        corr_x, corr_y = None, None
        if "Correlation (numeric vs numeric)" in st.session_state[ana_key]:
            if len(numeric_cols) >= 2:
                corr_x = st.sidebar.selectbox("X (numeric)", numeric_cols, index=0, key=f"corrx_{active_id}")
                corr_y = st.sidebar.selectbox("Y (numeric)", numeric_cols, index=1, key=f"corry_{active_id}")
            else:
                st.sidebar.warning("Need at least 2 numeric columns for correlation.")

        # Category breakdown params
        group_col = None
        if any(x in st.session_state[ana_key] for x in ["Top category by metric", "Bottom category by metric"]):
            if cat_cols:
                group_col = st.sidebar.selectbox("Group by (category)", cat_cols, index=0, key=f"group_{active_id}")
            else:
                st.sidebar.warning("No categorical columns detected for category breakdown.")
            if target_metric is None and numeric_cols:
                target_metric = st.sidebar.selectbox("Metric for grouping", numeric_cols, index=0, key=f"metric_group_{active_id}")

        st.sidebar.markdown("---")

        # ---- Run analysis button ----
        run = st.sidebar.button("▶ Run analysis", use_container_width=True)

        if run:
            selected_vars = st.session_state[var_key]
            selected_analysis = st.session_state[ana_key]

            query_parts = []

            for a in selected_analysis:
                if a == "Count rows":
                    query_parts.append("count rows")

                elif a == "Show first row":
                    query_parts.append("show first row")
                elif a == "Highest (max)" and target_metric:
                    query_parts.append(f"highest {target_metric}")
                elif a == "Lowest (min)" and target_metric:
                    query_parts.append(f"lowest {target_metric}")
                elif a == "Average (mean)" and target_metric:
                    query_parts.append(f"average {target_metric}")
                elif a == "Sum (total)" and target_metric:
                    query_parts.append(f"sum {target_metric}")
                elif a == "Show rows for a date" and date_value:
                    query_parts.append(f"rows for {date_value}")

                elif a == "Compare two dates" and d1 and d2:
                    metric = target_metric
                    if not metric and selected_vars:
                        metric = selected_vars[0]

                    if metric:
                        query_parts.append(f"compare {metric} between {d1} and {d2}")

                elif a == "Correlation (numeric vs numeric)" and corr_x and corr_y:
                    query_parts.append(f"correlation between {corr_x} and {corr_y}")
                elif a == "Top category by metric" and group_col and target_metric:
                    query_parts.append(f"top {group_col} by {target_metric}")
                elif a == "Bottom category by metric" and group_col and target_metric:
                    query_parts.append(f"bottom {group_col} by {target_metric}")

            if selected_vars:
                query_parts.append("variables: " + ", ".join(selected_vars))

            built_query = " | ".join([p for p in query_parts if p.strip()])

            if not built_query.strip():
                st.sidebar.warning("Pick at least one analysis and required fields (metric/date)")
            else:
                st.session_state["pending_structured_query"] = built_query
                st.sidebar.markdown(f"Queued ✅: {built_query}")


# -----------------------------
# Chat Input and Response Logic
# -----------------------------
user_query = st.chat_input("Ask your FOREO question here...")
if not user_query and st.session_state.get("pending_structured_query"):
    user_query = st.session_state.pop("pending_structured_query")
if user_query:
    q = user_query  # alias for convenience
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": q})
    st.markdown(f'<div class="chat-bubble user-bubble">{q}</div>', unsafe_allow_html=True)

    # Show typing indicator while processing
    thinking = st.empty()
    with thinking.container():
        st.markdown(
            '<div class="think-bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>',
            unsafe_allow_html=True
        )

    ticket_collection_complete = False

    # ----- Support Ticket Collection Flow -----
    if st.session_state.ticket_state == "collecting":
        q_lower = q.lower().strip()
        cancel_phrases = ["never mind", "nevermind", "don't need", "dont need", "no thanks", "no thank you", "not needed"]
        is_cancel = any(phrase in q_lower for phrase in cancel_phrases)
        if not is_cancel:
            cancel_keywords = ["no", "don't", "dont", "cancel", "skip"]
            for kw in cancel_keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, q_lower):
                    is_cancel = True
                    break
        if is_cancel:
            # User canceled the ticket creation
            st.session_state.ticket_state = None
            st.session_state["ticket_slots"] = {}
            bot_reply = "No problem! How else can I help you today?"
            thinking.empty()
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        elif not extract_email(q) and not extract_name(q) and len(q.split()) > 3:
            # User response doesn't contain needed info (likely off track), exit ticket flow
            st.session_state.ticket_state = None
            st.session_state["ticket_slots"] = {}
        else:
            # Collect ticket info
            ticket_slots = st.session_state.get("ticket_slots", {})
            name = extract_name(q) or ticket_slots.get("name")
            email = extract_email(q) or ticket_slots.get("email")
            device_for_slots = extract_device_type(q) or ticket_slots.get("device")
            issue = ticket_slots.get("issue", "")
            if name:
                ticket_slots["name"] = name
            if email:
                ticket_slots["email"] = email
            if device_for_slots:
                ticket_slots["device"] = device_for_slots
            if not issue and st.session_state.get("failed_attempts", 0) > 0:
                # If issue not set yet, use recent user queries as issue summary
                recent_queries = [m["content"] for m in st.session_state.messages[-5:] if m["role"] == "user"]
                issue = " | ".join(recent_queries[:3])
                ticket_slots["issue"] = issue
            st.session_state["ticket_slots"] = ticket_slots
            needs_info, missing_q = needs_ticket_info(ticket_slots)
            if needs_info:
                bot_reply = f"To create a support ticket, {missing_q}"
            else:
                # All info collected, create the support ticket
                device_for_ticket = ticket_slots.get("device") or _infer_device_from_history(st.session_state.messages)
                ticket_slots["device"] = device_for_ticket
                ticket = create_ticket(
                    name=ticket_slots["name"],
                    email=ticket_slots["email"],
                    device=device_for_ticket,
                    issue=ticket_slots.get("issue"),
                    chat_history=st.session_state.messages.copy(),
                    metadata={"failed_attempts": st.session_state.get("failed_attempts", 0)}
                )
                session = get_session()
                try:
                    db_module.create_ticket(
                        session=session,
                        ticket_id=ticket["ticket_id"],
                        chat_history_json=json.dumps(st.session_state.messages),
                        user_name=ticket_slots["name"],
                        user_email=ticket_slots["email"],
                        device_type=device_for_ticket,
                        issue_summary=ticket_slots.get("issue"),
                        intent=ticket_slots.get("intent"),
                        escalated_by_bot=True,
                    )
                except Exception as e:
                    # If database write fails, still show success to user but log the error
                    import traceback
                    st.error(f"⚠️ Ticket created in memory, but database save failed: {str(e)}")
                    st.code(traceback.format_exc())
                    # Continue anyway - the ticket was created in the ticket_management module
                finally:
                    session.close()
                bot_reply = (f"✅ Support ticket created! Your ticket ID is **{ticket['ticket_id']}**. "
                             f"Our support team will contact you at {ticket_slots['email']} within 24 hours.")
                st.session_state.ticket_state = None
                st.session_state.failed_attempts = 0
                st.session_state["ticket_slots"] = {}
                st.session_state["ticket_just_created"] = True
                ticket_collection_complete = True
            # Respond to user
            thinking.empty()
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # ----- Normal Query Handling (not in ticket flow) -----
    if not ticket_collection_complete and st.session_state.ticket_state != "collecting":
        # Determine user intent and any slots from query
        if st.session_state.get("reasoning_state"):
            # Use previous clarification context if available
            old_intent = st.session_state["reasoning_state"].get("intent")
            old_slots = st.session_state["reasoning_state"].get("slots", {})
            country = extract_country(q)
            region = extract_region(q)
            device_type = extract_device_type(q)
            # Heuristic: detect issue type by keywords for troubleshooting
            q_low = q.lower()
            if any(kw in q_low for kw in ["charge", "charging", "battery", "power"]):
                issue = "charging"
            elif any(kw in q_low for kw in ["turn on", "won't turn", "wont turn", "start", "power on"]):
                issue = "not_turning_on"
            elif any(kw in q_low for kw in ["clean", "cleaning", "wash"]):
                issue = "cleaning"
            elif any(kw in q_low for kw in ["button", "buttons"]):
                issue = "buttons"
            elif any(kw in q_low for kw in ["weak", "slow", "performance"]):
                issue = "performance"
            else:
                issue = None
            # Update slots with new info
            slots = old_slots.copy()
            if country:
                slots["country"] = country
            if region:
                slots["region"] = region
            if device_type:
                slots["device_type"] = device_type
            if issue:
                slots["issue"] = issue
            intent = old_intent
        else:
            intent, _ = classify_intent(q)
            if st.session_state.get("latest_structured_file"):
                analytics_words = ["highest", "lowest", "average", "sum", "top", "bottom", "compare", "rows", "count"]
                if any(w in q.lower() for w in analytics_words):
                    intent = "analytics"
            slots = simple_extract_slots(q)

        # Check if user just responded "yes" to create a ticket
        if st.session_state.get("ticket_just_created", False):
            # Reset the flag, and do not escalate again immediately
            st.session_state["ticket_just_created"] = False
            should_escalate = False
        else:
            q_low = q.lower()
            def has_word(w: str) -> bool:
                return re.search(rf"\b{re.escape(w)}\b", q_low) is not None

            user_accepts_ticket = (
                has_word("yes") or has_word("yeah") or has_word("sure") or has_word("okay") or has_word("ok")
                or ("create ticket" in q_low)
                or("support ticket" in q_low)
            )
            should_escalate = (intent =="escalation") or user_accepts_ticket

        if should_escalate:
            # Begin support ticket collection process
            st.session_state.ticket_state = "collecting"
            st.session_state["ticket_slots"] = {
                "device": slots.get("device_type"),
                "issue": slots.get("issue", ""),
                "intent": intent,
            }
            bot_reply = ("I understand you'd like to speak with our support team. "
                         "To create a support ticket, I'll need a few details. What's your name?")
            thinking.empty()
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        else:
            needs_clar, clarification_q = needs_clarification(intent, slots)
            if needs_clar:
                # Ask a clarification question
                st.session_state["reasoning_state"] = {"intent": intent, "slots": slots}
                bot_reply = f"To help you better, {clarification_q}"
            else:
                # Build augmented query if needed for certain intents
                was_in_clarification = st.session_state.get("reasoning_state") is not None
                is_short_query = len(q.split()) <= 3
                is_country_response = was_in_clarification and (slots.get("country") or slots.get("region")) and intent in ["warranty", "orders"]
                if was_in_clarification and (is_short_query or is_country_response):
                    # User answered clarification (like providing country)
                    if intent == "warranty":
                        augmented_query = "warranty information"
                        if slots.get("country"):
                            augmented_query += f" in {slots['country']}"
                    elif intent == "orders":
                        augmented_query = "order and shipping information"
                        loc = slots.get("country") or slots.get("region")
                        if loc:
                            augmented_query += f" in {loc}"
                    else:
                        augmented_query = q
                else:
                    augmented_query = q
                    if intent in ["warranty", "orders"]:
                        loc = slots.get("country") or slots.get("region")
                        if loc:
                            augmented_query += f" in {loc}"

                # Intent-specific shortcuts
                if intent == "troubleshooting" and slots.get("issue"):
                    bot_reply = get_troubleshooting_steps(slots)
                elif intent == "cleaning" and slots.get("issue"):
                    bot_reply = get_troubleshooting_steps(slots)
                elif intent == "charging":
                    if not slots.get("issue"):
                        slots["issue"] = "charging"
                    bot_reply = get_troubleshooting_steps(slots)
                else:
                    # Routing: Uploaded docs vs. FAQ knowledge base
                    use_uploaded = st.session_state.get("use_uploaded_kb", False)
                    use_hf_api = _get_secret("USE_HF_API", os.getenv("USE_HF_API", "0")) == "1"
                    structured_path = st.session_state.get("latest_structured_file")

                    if use_uploaded and structured_path and _looks_like_structured_question(augmented_query):
                        # First try answering from structured file if available
                        result, evidence = try_answer_csv(augmented_query, structured_path)
                        if result is not None and str(result).strip() != "":
                            result_str = str(result)
                            # If the result looks like a table or code block, present directly
                            looks_structured = ("```" in result_str) or ("\n" in result_str and len(result_str) > 60)
                            wants_explanation = any(
                                w in augmented_query.lower() for w in ["explain", "summary", "summarize", "insights"])

                            if looks_structured and not wants_explanation:
                                bot_reply = result_str
                            else:
                                try:
                                    # If it's a big table, don't send the whole thing—send a short safe summary instead
                                    if looks_structured:
                                        safe_prompt = (
                                            "Give 1 short sentence explaining what this table represents. "
                                            "Do NOT invent numbers. Do NOT change values.\n\n"
                                            f"{result_str[:1500]}"
                                        )
                                        explanation = explain_with_ollama(safe_prompt)
                                        bot_reply = f"{explanation}\n\n{result_str}"
                                    else:
                                        explanation = explain_with_ollama(result_str)
                                        bot_reply = f"{explanation}\n\n{result_str}"
                                except Exception:
                                    bot_reply = result_str
                            if evidence is not None and str(evidence).strip() != "":
                                bot_reply += f"\n\n```text\n{evidence}\n```"
                        else:
                            # If structured file did not answer, try unstructured docs (PDF, etc.)
                            bot_reply = answer_from_uploaded_docs(augmented_query, company_id)
                    else:
                        # Fall back to FAQ knowledge base via vector search
                        docs, _ = retrieve_top_k(coll, embedder, augmented_query, k=3)
                        best_sim = best_question_similarity(embedder, augmented_query, docs)
                        if best_sim < OFFTOPIC_SIM_THRESHOLD:
                            bot_reply = OFF_TOPIC_MESSAGE
                            st.session_state.failed_attempts = st.session_state.get("failed_attempts", 0) + 1
                        else:
                            ans = extractive_answer(augmented_query, docs)
                            # Optionally paraphrase the answer for fluency
                            use_local = _get_secret("USE_LOCAL_GEMMA", os.getenv("USE_LOCAL_GEMMA", "0")) == "1"
                            use_hf_api2 = _get_secret("USE_HF_API", os.getenv("USE_HF_API", "0")) == "1"
                            if use_local and model is not None and ans not in {"Not enough information.", ""}:
                                try:
                                    ans = maybe_paraphrase(tokenizer, model, device, ans)
                                except Exception:
                                    pass
                            elif use_hf_api2 and ans not in {"Not enough information.", ""}:
                                ans = paraphrase_with_gemma_api(ans)
                            bot_reply = re.sub(r"https?://\S+", "", ans).strip()
                            if bot_reply and bot_reply.strip() and bot_reply != OFF_TOPIC_MESSAGE:
                                st.session_state.failed_attempts = 0

                # Clear any stored clarification context if we provided a direct answer
                if bot_reply != OFF_TOPIC_MESSAGE:
                    st.session_state["reasoning_state"] = None

            # Handle cases where answer is empty or off-topic
            if not bot_reply or not bot_reply.strip() or bot_reply == OFF_TOPIC_MESSAGE:
                failed_count = st.session_state.get("failed_attempts", 0)
                st.session_state.failed_attempts = failed_count + 1
                if not bot_reply or not bot_reply.strip():
                    bot_reply = "I didn't catch that. Could you rephrase or provide a bit more detail?"
                elif bot_reply == OFF_TOPIC_MESSAGE:
                    if failed_count == 0:
                        bot_reply = "I am a FOREO chatbot. Please ask only FOREO-related questions."
                    elif failed_count == 1:
                        bot_reply = "I can only help with FOREO product questions. Could you ask something about FOREO devices, warranty, orders, or support?"
                    else:
                        bot_reply = ("I'm designed to help with FOREO-related questions only. If you'd like, I can connect you with our support team "
                                     "for further assistance. Would you like me to create a support ticket?")

            # Show bot response
            thinking.empty()
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# Footer
st.markdown("""
<div class="footer">
  <small>Powered by Gemma-3-270M · Embeddings: all-MiniLM-L6-v2 · Vector DB: Chroma</small>
</div>
</div>
""", unsafe_allow_html=True)