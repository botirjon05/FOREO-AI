from lc_vectorstore import get_uploads_vs

def retrieve_company_docs (query: str, company_id: str, k: int = 5 ):
    vs = get_uploads_vs()
    return vs.similarity_search(query, k=k, filter = {'company_id': company_id})