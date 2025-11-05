"""Document retrieval from ChromaDB."""

from typing import List
from .chroma_client import get_chroma_client


def fetch_doc(
    *,
    query: str,
    collection: str,
    n_results: int = 8,
) -> str:
    """
    Fetch documents from ChromaDB collection.

    Args:
        query: Search query text.
        collection: Name of the ChromaDB collection to query.
        n_results: Maximum number of results to retrieve.

    Returns:
        Concatenated document snippets separated by '\\n---\\n'.
    """
    client = get_chroma_client()
    col = client.get_collection(name=collection)
    res = col.query(query_texts=[query], n_results=n_results)

    docs_matrix = res.get("documents") or []
    docs = docs_matrix[0] if docs_matrix else []

    out: List[str] = []
    total = 0
    max_chars = 6000

    for d in docs:
        if not isinstance(d, str):
            continue
        sep = "\n---\n"
        add_len = len(d) + len(sep)
        if total + add_len > max_chars:
            break
        out.append(d)
        total += add_len

    return "\n---\n".join(out)


__all__ = ["fetch_doc"]
