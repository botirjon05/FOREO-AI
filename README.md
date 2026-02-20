# FOREO AI Support Suite 💗

A simple AI support project for FOREO that combines:

- a customer-facing chatbot (RAG + troubleshooting),
- an FAQ ingestion UI to build/update the knowledge base,
- and an internal support portal for ticket handling.

This README is intentionally short and practical.

## What This Project Does ✨

The system helps answer FOREO support questions using a local knowledge base and creates support tickets when needed.

Main capabilities:

- 🔎 Retrieves answers from FAQ data using semantic search (ChromaDB + embeddings)
- 🧠 Detects intent (troubleshooting, warranty, account, order, etc.)
- ❓ Asks clarification questions when details are missing (device, country, issue type)
- 🎫 Supports internal ticket workflows for human agents

## The 3 UIs (Explained) 🖥️

### 1) Customer Chatbot UI (`app.py`) 💬

**Who it is for:** Customers/end users  
**Purpose:** Ask support questions and get answers in chat format

What users can do:

- Ask product/support questions in natural language
- Get grounded answers from indexed FAQ data
- Receive follow-up questions when context is missing
- Trigger ticket creation/escalation flows when required

Run it:

```bash
streamlit run app.py
```

Quick example:

```text
User: "My FOREO device is not charging."
Bot: "Which FOREO device are you using?"
User: "LUNA 4"
Bot: "Try these steps: ... (device-specific troubleshooting)"
```

---

### 2) FAQ Ingestion UI (`faq_ingest.py`) 📥

**Who it is for:** Content/admin team  
**Purpose:** Crawl help/FAQ pages and export structured Q&A data

What users can do:

- Enter a website URL and crawl likely FAQ/help pages
- Extract Q&A pairs (JSON-LD + heuristic extraction)
- Review/edit extracted entries in the UI
- Export in `JSONL` or `CSV` with fixed schema fields

Run it:

```bash
streamlit run faq_ingest.py
```

Quick example:

```text
Input URL: https://www.foreo.com
Action: Click "Run" to crawl + extract Q&A
Result: Review extracted entries, edit if needed, then export as JSONL
```

---

### 3) Support Portal UI (`support_portal.py`) 🛠️

**Who it is for:** Internal support agents  
**Purpose:** Manage escalated tickets and collaborate on resolution

What users can do:

- View ticket list with filters/search
- Open ticket details and chat transcript
- Update ticket status (`open`, `in_progress`, `resolved`)
- Add internal notes for team collaboration

Run it:

```bash
streamlit run support_portal.py
```

Quick example:

```text
1) Open ticket FOREO-000021
2) Change status from "open" to "in_progress"
3) Add note: "Customer asked for replacement policy details"
4) Resolve ticket when issue is completed
```

## Simple End-to-End Example 🔄

```text
Step 1: Use FAQ Ingestion UI to extract fresh FAQs from support pages.
Step 2: Rebuild vector store so chatbot uses the latest knowledge.
Step 3: Customer asks a question in Chatbot UI.
Step 4: If unresolved, chatbot creates/escalates a ticket.
Step 5: Agent handles the ticket in Support Portal UI.
```

Useful rebuild commands after new FAQ export:

```bash
python3 prepare_for_embedding.py
python3 create_vectorstore.py
```

## Quick Start 🚀

### 1) Install dependencies 📦

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare knowledge base data (first time) 🧰

```bash
python3 clean_faqs.py
python3 prepare_for_embedding.py
python3 create_vectorstore.py
```

### 3) Launch the UI you need ▶️

```bash
# Customer chatbot
streamlit run app.py

# FAQ ingestion tool
streamlit run faq_ingest.py

# Agent support portal
streamlit run support_portal.py
```

## Project Structure 🗂️

```text
Foreo/
├── app.py
├── faq_ingest.py
├── support_portal.py
├── rag_gemma.py
├── intent_detection.py
├── troubleshooting.py
├── ticket_management.py
├── db.py
├── clean_faqs.py
├── prepare_for_embedding.py
├── create_vectorstore.py
├── requirements.txt
├── data/
└── chroma_db/
```

## Notes 📝

- ⚠️ `chroma_db/` should exist before running the chatbot, otherwise retrieval will fail.
- 🗄️ `foreo_support.db` is created/used for ticket storage.
- ✅ Cleanest flow: run ingestion/export first, then rebuild embeddings/vector store, then use chatbot.

## Troubleshooting 🧯

- **No answers from chatbot:** run `python3 create_vectorstore.py` again and verify `chroma_db/` is populated.
- **Module/import errors:** confirm virtual environment is active and dependencies are installed.
- **Portal has no tickets:** create tickets via chatbot escalation flow first.

