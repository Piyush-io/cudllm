from typing import List, Dict, Any, Optional
import os
import chromadb


def fetch_doc(
    *,
    query: str,
    collection: str,
    api_key: Optional[str] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
    n_results: int = 8,
) -> str:
    mode = (os.environ.get("CHROMA_MODE") or "").lower()
    persist_path = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_db")

    if mode == "persistent" or persist_path:
        client = chromadb.PersistentClient(path=persist_path)
    else:
        client = chromadb.CloudClient(
            api_key=api_key or os.environ.get("CHROMA_API_KEY"),
            tenant=tenant or os.environ.get("CHROMA_TENANT"),
            database=database or os.environ.get("CHROMA_DATABASE"),
        )
    col = client.get_collection(name=collection)
    res = col.query(query_texts=[query], n_results=n_results)
    docs_matrix = res.get("documents") or []
    docs = docs_matrix[0] if docs_matrix else []
    out: List[str] = []
    total = 0
    for d in docs:
        if not isinstance(d, str):
            continue
        sep = "\n---\n"
        add_len = len(d) + len(sep)
        if total + add_len > 6000:
            break
        out.append(d)
        total += add_len
    return "\n---\n".join(out)
