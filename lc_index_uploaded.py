from lc_vectorstore import get_uploads_vs

def index_uploaded_chunks(chunks, company_id: str, filename: str) -> int:
    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata.update({
            "company_id": company_id,
            "kb": "uploaded",
            "filename": filename,
        })

    vs = get_uploads_vs()
    vs.add_documents(chunks)
    return len(chunks)