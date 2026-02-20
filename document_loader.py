import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def load_csv(path):
    df = pd.read_csv(path)
    documents = []

    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content = text))

    return documents


def load_document(path):
    if path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".csv"):
        return load_csv(path)
    else:
        raise ValueError(f"Unsupported file type")