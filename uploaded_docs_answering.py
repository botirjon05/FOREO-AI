# uploaded_docs_answering.py
import os
import requests

from lc_retrieve_uploaded import retrieve_company_docs
from ollama_client import generate_with_ollama

FAIL_MSG = "I don't have enough information in the uploaded documents to answer this question."

def dedupe_docs(docs):
    seen = set()
    out = []
    for d in docs:
        t = (d.page_content or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(d)
    return out

def answer_from_uploaded_docs(question: str, company_id: str) -> str:

    #Retrieve relevant chunks from chroma
    docs = retrieve_company_docs(question, company_id, k=8)
    docs = dedupe_docs(docs)

    if not docs:
        return FAIL_MSG

    #Build context
    context = "\n".join(d.page_content for d in docs)

    #prompt for ollama

    prompt = f"""
You are a business data assistant.

Rules:
- Use ONLY the context below.
- Do NOT guess or invent numbers.
- If the answer is not in the context, reply exactly: 
"{FAIL_MSG}"

Context:
{context}

Question:
{question}

Answer in 1–4 sentences. If helpful, include the exact values you used.
""".strip()

    return generate_with_ollama(prompt)