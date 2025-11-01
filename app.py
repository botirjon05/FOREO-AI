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
    GEMMA_SMALL_ID,
    OFFTOPIC_SIM_THRESHOLD,
    OFF_TOPIC_MESSAGE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    load_gemma_small,
)

from intent_detection import classify_intent, extract_device_type, simple_extract_slots, needs_clarification, extract_country, extract_region
from troubleshooting import get_troubleshooting_steps

# ----------------------------------------
# ----------- Streamlit Setup ------------
# ----------------------------------------

st.set_page_config(
    page_title="FOREO Chatbot (Gemma-3-270M RAG)",
    page_icon="💗",
    layout="centered"
)

# ---- Brand assets (optional) ----
LOGO_LOCAL_PATH = "assets/foreo_logo.png"
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
         "content": "👋Hi! I'm your FOREO assistant. Ask me about warranty, cleaning, charging, orders, or account help."}
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

# Load logo once and reuse for both top logo and header
import base64
logo_rendered = False
b64_logo = None
header_logo_html = "💗"

if os.path.exists(LOGO_LOCAL_PATH):
    with open(LOGO_LOCAL_PATH, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode()
    logo_rendered = True
    header_logo_html = f'<img src="data:image/png;base64,{b64_logo}" width="30" height="30" alt="FOREO" />'
elif LOGO_URL_ENV:
    logo_rendered = True
    header_logo_html = f'<img src="{LOGO_URL_ENV}" width="30" height="30" alt="FOREO" />'

# Display top logo (large)
if b64_logo:
    st.markdown(
        f"""
    <div class="logo-top">
      <img src="data:image/png;base64,{b64_logo}" alt="FOREO Logo">
    </div>
    """,
        unsafe_allow_html=True
    )

# Header with logo
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

    # REASONING LOOP with clarification support
    # Step 1: Check if we're in a clarification flow FIRST
    if st.session_state.get("reasoning_state"):
        # We're continuing a clarification flow - merge with existing context
        old_intent = st.session_state.get("reasoning_state", {}).get("intent")
        old_slots = st.session_state.get("reasoning_state", {}).get("slots", {})
        
        # Extract any additional info from the current query (country, device, issue, etc.)
        country = extract_country(q)
        region = extract_region(q)
        device = extract_device_type(q)
        
        # Extract issue type
        q_lower = q.lower()
        if any(kw in q_lower for kw in ["charge", "charging", "battery", "power"]):
            issue = "charging"
        elif any(kw in q_lower for kw in ["turn on", "won't turn", "wont turn", "start", "power on"]):
            issue = "not_turning_on"
        elif any(kw in q_lower for kw in ["clean", "cleaning", "wash"]):
            issue = "cleaning"
        elif any(kw in q_lower for kw in ["button", "buttons"]):
            issue = "buttons"
        elif any(kw in q_lower for kw in ["weak", "slow", "performance"]):
            issue = "performance"
        else:
            issue = None
        
        # Start with old slots and update with new info
        slots = old_slots.copy()
        
        # Add to slots if found
        if country:
            slots["country"] = country
        if region:
            slots["region"] = region
        if device:
            slots["device_type"] = device
        if issue:
            slots["issue"] = issue
        
        # Use the original intent
        intent = old_intent
    else:
        # Not in clarification - classify intent and extract slots normally
        intent, confidence = classify_intent(q)
        slots = simple_extract_slots(q)
    
    # Step 3: Check if clarification is needed
    needs_clar, clarification_q = needs_clarification(intent, slots)
    
    if needs_clar:
        # Store current state for next turn
        st.session_state["reasoning_state"] = {
            "intent": intent,
            "slots": slots
        }
        bot_reply = f"To help you better, {clarification_q}"
    else:
        # All info gathered - provide answer
        # Check if we just got a clarification response
        was_in_clarification = st.session_state.get("reasoning_state") is not None
        is_short_query = len(q.split()) <= 3
        is_country_response = was_in_clarification and (slots.get("country") or slots.get("region")) and intent in ["warranty", "orders"]
        
        if was_in_clarification and (is_short_query or is_country_response):
            # This is a clarification response - use the intent to construct a proper query
            if intent == "warranty":
                augmented_query = "warranty information"
                if slots.get("country"):
                    augmented_query = f"{augmented_query} in {slots.get('country')}"
            elif intent == "orders":
                augmented_query = "order and shipping information"
                if slots.get("country"):
                    augmented_query = f"{augmented_query} in {slots.get('country')}"
                elif slots.get("region"):
                    augmented_query = f"{augmented_query} in {slots.get('region')}"
            else:
                # For device-related clarifications, just use the original query
                augmented_query = q
        else:
            # Normal query - use as is
            augmented_query = q
            
            # Augment query with country if available and relevant
            if intent in ["warranty", "orders"]:
                if slots.get("country"):
                    augmented_query = f"{augmented_query} in {slots.get('country')}"
                elif slots.get("region"):
                    augmented_query = f"{augmented_query} in {slots.get('region')}"
        
        # For troubleshooting intent, use troubleshooting flow
        if intent == "troubleshooting" and slots.get("issue"):
            bot_reply = get_troubleshooting_steps(slots)
        # For cleaning intent, use troubleshooting flow
        elif intent == "cleaning" and slots.get("issue"):
            bot_reply = get_troubleshooting_steps(slots)
        # For other intents, use RAG pipeline
        else:
            # Charging intent can also use troubleshooting guidance
            if intent == "charging":
                if not slots.get("issue"):
                    slots["issue"] = "charging"
                bot_reply = get_troubleshooting_steps(slots)
            else:
                t0 = time.time()
                docs, metas = retrieve_top_k(coll, embedder, augmented_query, 3)
                t1 = time.time()

                best_sim = best_question_similarity(embedder, augmented_query, docs)
                if best_sim < OFFTOPIC_SIM_THRESHOLD:
                    bot_reply = OFF_TOPIC_MESSAGE
                else:
                    ans = extractive_answer(augmented_query, docs)
                    if model is not None and ans not in {"Not enough information.", ""}:
                        try:
                            ans = maybe_paraphrase(tokenizer, model, device, ans)
                        except Exception:
                            pass
                    bot_reply = re.sub(r"https?://\\S+", "", ans).strip()

        # Clear the clarification state only if we didn't go off-topic
        if bot_reply != OFF_TOPIC_MESSAGE:
            st.session_state["reasoning_state"] = None

    # Safety fallback: if bot_reply is missing/empty, guide the user
    if not bot_reply or not bot_reply.strip():
        bot_reply = "I didn't catch that. Could you rephrase or provide a bit more detail?"
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