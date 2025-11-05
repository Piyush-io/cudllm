"""Vector store search utilities."""

from typing import List
from .docs_retriever import fetch_doc
from .chroma_config import STAGE_COLLECTIONS


def search_stage(*, stage: str, query: str, k: int = 3) -> List[str]:
    """
    Search a knowledge stage collection.

    Args:
        stage: Knowledge stage name (concepts, patterns, hardware, api, examples).
        query: Search query text.
        k: Maximum number of snippets to return.

    Returns:
        List of document snippets (up to k items).
    """
    collection = STAGE_COLLECTIONS.get(stage)
    if not collection:
        return []

    raw = fetch_doc(query=query, collection=collection, n_results=k)
    snippets = [s.strip() for s in raw.split("\n---\n") if s.strip()]

    return snippets[:k]


__all__ = ["search_stage"]
