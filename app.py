#!/usr/bin/env python3
# app.py — Streamlit Chat UI for RAG Gemma chatbot (polished FOREO styling)

import os
import re
import time
import base64
import json
import uuid
import requests
import streamlit as st
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

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(
    page_title="FOREO Chatbot (Gemma-3-270M RAG)",
    page_icon="💗",
    layout="centered"
)

LOGO_LOCAL_PATH = "assets/foreo_logo.png"
LOGO_URL_ENV = os.environ.get("FOREO_LOGO_URL", "").strip()

# -----------------------------
# Global CSS
# -----------------------------
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paraphrase via HF Inference API (hosted Gemma)
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

def paraphrase_with_gemma_api(text: str) -> str:
    """Use HF Inference API to lightly paraphrase/clean the answer."""
    token = _get_secret("HF_TOKEN", "")
    if not token:
        return text  # no token -> skip

    model = _get_secret("GEMMA_API_MODEL", "google/gemma-2-2b-it")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    prompt = (
        "Paraphrase the following answer to be concise, friendly, and avoid duplication. "
        "Do not add new facts. Keep brand-safe tone.\n\nAnswer:\n"
        f"{text}\n\nParaphrase:"
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 128, "temperature": 0.3}}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        # HF responses vary: handle both text-generation and chat templates
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
            out = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            out = data["generated_text"]
        else:
            # Some models return a list of dicts with 'generated_text'
            out = str(data)
        # Heuristic: return only the part after "Paraphrase:" if present
        if "Paraphrase:" in out:
            out = out.split("Paraphrase:", 1)[-1].strip()
        return out.strip() if out.strip() else text
    except Exception:
        return text  # fail-safe


def _infer_device_from_history(messages) -> str:
    """
    Best-effort device extraction from the full chat.

    This is used as a fallback for ticket metadata so the Support Portal
    still sees a device even if the final escalation messages don't mention it.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        d = extract_device_type(msg.get("content", ""))
        if d:
            return d
    return ""

# -----------------------------
# Initialize state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "👋Hi! I'm your FOREO assistant. Ask me about warranty, cleaning, charging, orders, or account help."}
    ]

# Initialize escalation tracking
if "failed_attempts" not in st.session_state:
    st.session_state.failed_attempts = 0
if "ticket_state" not in st.session_state:
    st.session_state.ticket_state = None  # None, "collecting", or ticket dict

device = pick_device()

@st.cache_resource
def load_all_components():
    """Load embeddings & DB; load local Gemma ONLY when enabled."""
    coll = connect_chroma(CHROMA_DIR, COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    tokenizer = model = None

    # Local-only Gemma load (use on your laptop)
    if _get_secret("USE_LOCAL_GEMMA", os.getenv("USE_LOCAL_GEMMA", "0")) == "1":
        try:
            tokenizer, model = load_gemma_small(GEMMA_SMALL_ID, device)
        except Exception:
            tokenizer = model = None  # never crash

    return coll, embedder, tokenizer, model

coll, embedder, tokenizer, model = load_all_components()

init_db()

# -----------------------------
# Chat interface
# -----------------------------
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

# Top logo styling & render
st.markdown("""
<style>
.logo-top { text-align: center; margin-top: -100px; margin-bottom: -100px; }
.logo-top img { max-width: 220px; width: 50%; height: auto; opacity: 0.96; transition: transform 0.3s ease, opacity 0.3s ease; }
.logo-top img:hover { transform: scale(1.05); opacity: 1; }
@media (max-width: 768px) { .logo-top img { max-width: 160px; width: 35%; } }
</style>
""", unsafe_allow_html=True)

b64_logo = None
header_logo_html = "💗"
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

# Render history
for msg in st.session_state.messages:
    role, content = msg["role"], msg["content"]
    css_class = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {css_class}">{content}</div>', unsafe_allow_html=True)

# Input
q = st.chat_input("Ask your FOREO question here...")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    st.markdown(f'<div class="chat-bubble user-bubble">{q}</div>', unsafe_allow_html=True)

    thinking = st.empty()
    with thinking.container():
        st.markdown(
            '<div class="think-bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>',
            unsafe_allow_html=True
        )

    # Check if we're collecting ticket information
    ticket_collection_complete = False
    if st.session_state.ticket_state == "collecting":
        # Check if user wants to cancel/exit ticket collection
        # Use word boundaries to avoid false matches (e.g., "no" in email addresses)
        q_lower = q.lower().strip()
        # Check for cancel phrases first (multi-word)
        cancel_phrases = ["never mind", "nevermind", "don't need", "dont need", "no thanks", "no thank you", "not needed"]
        is_cancel = any(phrase in q_lower for phrase in cancel_phrases)
        # Then check for single-word cancel keywords with word boundaries
        if not is_cancel:
            cancel_keywords = ["no", "don't", "dont", "cancel", "skip"]
            # Use word boundaries to match whole words only
            for kw in cancel_keywords:
                # Match as whole word (not part of another word)
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, q_lower):
                    is_cancel = True
                    break
        
        if is_cancel:
            # User wants to cancel - exit ticket collection mode
            st.session_state.ticket_state = None
            st.session_state["ticket_slots"] = {}
            bot_reply = "No problem! How else can I help you today?"
            thinking.empty()
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        # Check if user is asking a normal question (not providing ticket info)
        elif not extract_email(q) and not extract_name(q) and len(q.split()) > 3:
            # Looks like a normal question - exit ticket collection and process normally
            st.session_state.ticket_state = None
            st.session_state["ticket_slots"] = {}
            # Continue with normal flow (will be handled below)
        else:
            # Extract ticket info from current query
            ticket_slots = st.session_state.get("ticket_slots", {})
            name = extract_name(q) or ticket_slots.get("name")
            email = extract_email(q) or ticket_slots.get("email")
            device = extract_device_type(q) or ticket_slots.get("device")
            issue = ticket_slots.get("issue", "")
            
            # Update ticket slots
            if name:
                ticket_slots["name"] = name
            if email:
                ticket_slots["email"] = email
            if device:
                ticket_slots["device"] = device
            if not issue and st.session_state.get("failed_attempts", 0) > 0:
                # Use the last few queries as issue description
                recent_queries = [msg["content"] for msg in st.session_state.messages[-5:] if msg["role"] == "user"]
                issue = " | ".join(recent_queries[:3])
                ticket_slots["issue"] = issue
            
            st.session_state["ticket_slots"] = ticket_slots
            
            # Check if we have all required info
            needs_info, missing_q = needs_ticket_info(ticket_slots)
            if needs_info:
                bot_reply = f"To create a support ticket, {missing_q}"
            else:
                # All info collected - create ticket
                # Ensure we have a best-effort device for metadata
                device_for_ticket = ticket_slots.get("device") or _infer_device_from_history(
                    st.session_state.messages
                )
                ticket_slots["device"] = device_for_ticket

                ticket = create_ticket(
                    name=ticket_slots["name"],
                    email=ticket_slots["email"],
                    device=device_for_ticket,
                    issue=ticket_slots.get("issue"),
                    chat_history=st.session_state.messages.copy(),
                    metadata={"failed_attempts": st.session_state.get("failed_attempts", 0)}
                )

                # Mirror ticket into SQLite for the new Support Portal
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
                finally:
                    session.close()
                
                bot_reply = f"✅ Support ticket created! Your ticket ID is **{ticket['ticket_id']}**. Our support team will contact you at {ticket_slots['email']} within 24 hours."
                st.session_state.ticket_state = None
                st.session_state.failed_attempts = 0
                st.session_state["ticket_slots"] = {}
                # Mark that we just created a ticket to prevent immediate re-escalation
                st.session_state["ticket_just_created"] = True
                ticket_collection_complete = True
            
            thinking.empty()
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    
    # Normal query flow - continue with reasoning loop (if not in ticket collection or just completed it)
    # Skip normal flow if we just completed ticket collection (to prevent re-processing)
    if not ticket_collection_complete and st.session_state.ticket_state != "collecting":
        # ----- Reasoning loop with clarification -----
        if st.session_state.get("reasoning_state"):
            old_intent = st.session_state["reasoning_state"].get("intent")
            old_slots = st.session_state["reasoning_state"].get("slots", {})
            country = extract_country(q)
            region = extract_region(q)
            device_type = extract_device_type(q)

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

            slots = old_slots.copy()
            if country: slots["country"] = country
            if region: slots["region"] = region
            if device_type: slots["device_type"] = device_type
            if issue: slots["issue"] = issue
            intent = old_intent
        else:
            intent, _ = classify_intent(q)
            slots = simple_extract_slots(q)

        # Check for escalation intent or user accepting ticket creation
        # Only trigger escalation if:
        # 1. User explicitly requests escalation, OR
        # 2. User accepts ticket creation, OR
        # 3. Failed attempts >= 2 AND current query is also likely to fail (we'll check after RAG)
        # BUT: Don't trigger if we just created a ticket (prevent immediate re-escalation)
        if st.session_state.get("ticket_just_created", False):
            # Clear the flag - only skip escalation for this one turn
            st.session_state["ticket_just_created"] = False
            should_escalate = False
        else:
            user_accepts_ticket = any(kw in q.lower() for kw in ["yes", "yeah", "sure", "okay", "ok", "create ticket", "support ticket"])
            should_escalate = intent == "escalation" or user_accepts_ticket
        
        # Don't trigger escalation here if it's just failed attempts - we'll check after RAG
        if should_escalate:
            # Trigger escalation flow
            st.session_state.ticket_state = "collecting"
            st.session_state["ticket_slots"] = {
                "device": slots.get("device_type"),
                "issue": slots.get("issue", ""),
                "intent": intent,
            }
            bot_reply = "I understand you'd like to speak with our support team. To create a support ticket, I'll need a few details. What's your name?"
            thinking.empty()
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        else:
            needs_clar, clarification_q = needs_clarification(intent, slots)

            if needs_clar:
                st.session_state["reasoning_state"] = {"intent": intent, "slots": slots}
                bot_reply = f"To help you better, {clarification_q}"
            else:
                was_in_clarification = st.session_state.get("reasoning_state") is not None
                is_short_query = len(q.split()) <= 3
                is_country_response = was_in_clarification and (slots.get("country") or slots.get("region")) and intent in ["warranty", "orders"]

                if was_in_clarification and (is_short_query or is_country_response):
                    if intent == "warranty":
                        augmented_query = "warranty information"
                        if slots.get("country"):
                            augmented_query += f" in {slots['country']}"
                    elif intent == "orders":
                        augmented_query = "order and shipping information"
                        loc = slots.get("country") or slots.get("region")
                        if loc: augmented_query += f" in {loc}"
                    else:
                        augmented_query = q
                else:
                    augmented_query = q
                    if intent in ["warranty", "orders"]:
                        loc = slots.get("country") or slots.get("region")
                        if loc: augmented_query += f" in {loc}"

                # Intent-specific flows
                if intent == "troubleshooting" and slots.get("issue"):
                    bot_reply = get_troubleshooting_steps(slots)
                elif intent == "cleaning" and slots.get("issue"):
                    bot_reply = get_troubleshooting_steps(slots)
                elif intent == "charging":
                    if not slots.get("issue"):
                        slots["issue"] = "charging"
                    bot_reply = get_troubleshooting_steps(slots)
                else:
                    # ----- RAG retrieval -----
                    docs, _ = retrieve_top_k(coll, embedder, augmented_query, 3)
                    best_sim = best_question_similarity(embedder, augmented_query, docs)
                    if best_sim < OFFTOPIC_SIM_THRESHOLD:
                        bot_reply = OFF_TOPIC_MESSAGE
                        # Track failed attempt
                        st.session_state.failed_attempts = st.session_state.get("failed_attempts", 0) + 1
                    else:
                        ans = extractive_answer(augmented_query, docs)

                        # Paraphrase: local Gemma first (if enabled), else HF API (if enabled)
                        use_local = _get_secret("USE_LOCAL_GEMMA", os.getenv("USE_LOCAL_GEMMA", "0")) == "1"
                        use_hf_api = _get_secret("USE_HF_API", os.getenv("USE_HF_API", "0")) == "1"

                        if use_local and model is not None and ans not in {"Not enough information.", ""}:
                            try:
                                ans = maybe_paraphrase(tokenizer, model, device, ans)
                            except Exception:
                                pass
                        elif use_hf_api and ans not in {"Not enough information.", ""}:
                            ans = paraphrase_with_gemma_api(ans)

                        bot_reply = re.sub(r"https?://\\S+", "", ans).strip()
                        
                        # Reset failed attempts on successful answer
                        if bot_reply and bot_reply.strip() and bot_reply != OFF_TOPIC_MESSAGE:
                            st.session_state.failed_attempts = 0

                if bot_reply != OFF_TOPIC_MESSAGE:
                    st.session_state["reasoning_state"] = None

            # Track failed attempts for empty or off-topic answers
            if not bot_reply or not bot_reply.strip() or bot_reply == OFF_TOPIC_MESSAGE:
                # Get failed count BEFORE incrementing
                failed_count = st.session_state.get("failed_attempts", 0)
                st.session_state.failed_attempts = failed_count + 1
                
                if not bot_reply or not bot_reply.strip():
                    bot_reply = "I didn't catch that. Could you rephrase or provide a bit more detail?"
                elif bot_reply == OFF_TOPIC_MESSAGE:
                    # Vary the off-topic message to avoid repetition
                    if failed_count == 0:
                        bot_reply = "I am a FOREO chatbot. Please ask only FOREO-related questions."
                    elif failed_count == 1:
                        bot_reply = "I can only help with FOREO product questions. Could you ask something about FOREO devices, warranty, orders, or support?"
                    else:
                        bot_reply = "I'm designed to help with FOREO-related questions only. If you'd like, I can connect you with our support team who can assist you further. Would you like me to create a support ticket?"

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