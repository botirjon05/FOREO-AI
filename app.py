#!/usr/bin/env python3
# app.py — Streamlit Chat UI for RAG Gemma chatbot (polished FOREO styling)
# Requires: rag_gemma.py (same directory)

import os
import streamlit as st
import time
import re

from rag_gemma import (
    pick_device,
    connect_chroma,
    SentenceTransformer,
    retrieve_top_k,
    best_question_similarity,
    extractive_answer,
    maybe_paraphrase,
    to_sentences,
    parse_qa,
    GEMMA_SMALL_ID,
    OFFTOPIC_SIM_THRESHOLD,
    OFF_TOPIC_MESSAGE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    load_gemma_small,
)

# ----------------------------------------
# ----------- Streamlit Setup ------------
# ----------------------------------------

st.set_page_config(
    page_title="FOREO Chatbot (Gemma-3-270M RAG)",
    page_icon="💗",
    layout="centered"
)

# ---- Brand assets (optional) ----
LOGO_LOCAL_PATH = "assets/foreo_logo.png"  # put your logo here if you have it
LOGO_URL_ENV = os.environ.get("FOREO_LOGO_URL", "").strip()

# ----------------------------------------
# -------------- Global CSS --------------
# ----------------------------------------
st.markdown("""
<style>
/* Soft brand background */
body, .stApp {
  background: radial-gradient(1200px 600px at 15% 0%, #ffeaf3 0%, rgba(255,234,243,0) 60%),
              radial-gradient(1000px 500px at 100% 20%, #efe7ff 0%, rgba(239,231,255,0) 55%),
              linear-gradient(180deg, #faf8fc 0%, #f7f5fb 100%);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
}





/* Chat container (glass card) */
.chat-wrap {
  
  margin: 26px auto 90px auto;
  background: rgba(255,255,255,0.72);
  backdrop-filter: blur(8px);
 
  
  
  
}

/* Header bar */
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
.header .title {
  font-weight: 800; letter-spacing:.3px;
  font-size: 1.4rem; text-shadow: 0 1px 3px rgba(0,0,0,.25);
}
.header .sub {
  font-size: .9rem; opacity: .9; margin-top: 2px;
}

/* Logo circle */
.logo-wrap {
  width: 48px; height: 48px; border-radius: 50%;
  background: rgba(255,255,255,.22);
  display:flex; align-items:center; justify-content:center;
  border: 1px solid rgba(255,255,255,.35);
  overflow:hidden;
}

/* Chat bubbles */
.chat-bubble {
    border-radius: 18px;
    padding: 10px 14px;
    margin: 8px 0;
    line-height: 1.5;
    max-width: 88%;
    word-wrap: break-word;
    box-shadow: 0 6px 16px rgba(15, 23, 42, .06);
    animation: fadeIn .12s ease-in;
}
@keyframes fadeIn { from {opacity:0; transform: translateY(4px)} to {opacity:1; transform:none} }

.user-bubble {
    margin-left: auto;
    color: #0b1220;
    background: linear-gradient(135deg, #87b7ff 0%, #6ae0ea 100%);
    color: #fff;
}
.bot-bubble {
    margin-right: auto;
    color: #0f172a;
    background: #f5f6fb;
    border: 1px solid #eef1f6;
}

/* Thinking bubble with animated 3 dots */
.think-bubble {
  display:inline-flex; align-items:center; gap:8px;
  margin-right: auto;
  color:#6b7280; background:#f5f6fb; border:1px solid #eef1f6;
  padding: 10px 14px; border-radius: 18px;
}
.dot { width:6px; height:6px; border-radius:50%; background:#a3a8b6; display:inline-block; animation: blink 1.2s infinite;}
.dot:nth-child(2){ animation-delay:.2s;}
.dot:nth-child(3){ animation-delay:.4s;}
@keyframes blink { 0%, 80%, 100% { opacity:.25 } 40% { opacity:1 } }

/* Quick chips (optional if you add later) */
.pill {
  border: 2px solid #8b5cf6;
  color: #373c4a;
  background: #ffffff;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: .92rem;
  cursor: pointer;
}
.pill:hover { background:#f3e8ff; }

/* Footer */
.footer {
  text-align:center; color:#8a90a6; font-size:.88rem; margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)





# ----------------------------------------
# ---------- Initialize state ------------
# ----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "👋Hi! I’m your FOREO assistant. Ask me about warranty, cleaning, charging, orders, or account help."}
    ]

device = pick_device()

@st.cache_resource
def load_all_components():
    """Load embeddings, DB, and Gemma model once."""
    coll = connect_chroma(CHROMA_DIR, COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    tokenizer, model = load_gemma_small(GEMMA_SMALL_ID, device)
    return coll, embedder, tokenizer, model

coll, embedder, tokenizer, model = load_all_components()

# ----------------------------------------
# ----------- Chat interface -------------
# ----------------------------------------

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
# Responsive FOREO logo above everything
st.markdown("""
<style>
.logo-top {
  text-align: center;
  margin-top: -100px;        /* lift logo up */
  margin-bottom: -100px;       /* tighten spacing below */
}

.logo-top img {
  max-width: 220px;         /* slightly larger on desktop */
  width: 50%;
  height: auto;
  opacity: 0.96;
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.logo-top img:hover {
  transform: scale(1.05);
  opacity: 1;
}

@media (max-width: 768px) {
  .logo-top img {
    max-width: 160px;       /* smaller for phones */
    width: 35%;
  }
}
</style>
""", unsafe_allow_html=True)

import base64
with open("assets/foreo_logo.png", "rb") as f:
    b64_logo = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
<div class="logo-top">
  <img src="data:image/png;base64,{b64_logo}" alt="FOREO Logo">
</div>
""",
    unsafe_allow_html=True
)


# Header with logo (local path or URL), fallback to emoji
logo_rendered = False
logo_html = ""
if os.path.exists(LOGO_LOCAL_PATH):
    # Use base64 to embed image inline
    import base64
    with open(LOGO_LOCAL_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{b64}" width="30" height="30" alt="FOREO" />'
    logo_rendered = True
elif LOGO_URL_ENV:
    logo_html = f'<img src="{LOGO_URL_ENV}" width="30" height="30" alt="FOREO" />'
    logo_rendered = True

st.markdown(
    f"""
<div class="header">
  <div class="logo-wrap">{logo_html if logo_rendered else "💗"}</div>
  <div>
    <div class="title">FOREO AI Assistant</div>
    <div class="sub">RAG + Gemma-3-270M • Chroma vector store</div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# Render chat history
for msg in st.session_state.messages:
    role, content = msg["role"], msg["content"]
    css_class = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {css_class}">{content}</div>', unsafe_allow_html=True)

# User input
q = st.chat_input("Ask your FOREO question here...")

if q:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": q})
    st.markdown(f'<div class="chat-bubble user-bubble">{q}</div>', unsafe_allow_html=True)

    # “Thinking…” three-dot bubble
    thinking = st.empty()
    with thinking.container():
        st.markdown(
            '<div class="think-bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>',
            unsafe_allow_html=True
        )

    # RAG pipeline
    t0 = time.time()
    docs, metas = retrieve_top_k(coll, embedder, q, 3)
    t1 = time.time()

    best_sim = best_question_similarity(embedder, q, docs)
    if best_sim < OFFTOPIC_SIM_THRESHOLD:
        bot_reply = OFF_TOPIC_MESSAGE
    else:
        ans = extractive_answer(q, docs)
        if model is not None and ans not in {"Not enough information.", ""}:
            try:
                ans = maybe_paraphrase(tokenizer, model, device, ans)
            except Exception:
                pass
        bot_reply = re.sub(r"https?://\\S+", "", ans).strip()

    # Replace thinking bubble with bot reply
    thinking.empty()
    st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)

    # Append to session
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# Footer
st.markdown("""
<div class="footer">
  <small>Powered by Gemma-3-270M · Embeddings: all-MiniLM-L6-v2 · Vector DB: Chroma</small>
</div>
</div>
""", unsafe_allow_html=True)