from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"
UPLOADS_COLLECTION = "foreo_uploads"
FAQ_COLLECTION = "foreo_kb"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

def get_uploads_vs():
    return Chroma(
        collection_name = UPLOADS_COLLECTION,
        persist_directory = PERSIST_DIR,
        embedding_function = get_embeddings()
    )

def get_faq_vs():
    return Chroma(
        collection_name = FAQ_COLLECTION,
        persist_directory = PERSIST_DIR,
        embedding_function = get_embeddings()
    )