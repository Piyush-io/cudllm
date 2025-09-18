from typing import List, Optional
from .docs_retriever import fetch_doc
from ..chroma import STAGE_COLLECTIONS


def search_collection(
    *,
    query: str,
    collection: str,
    api_key: Optional[str] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
    k: int = 3,
) -> List[str]:
    raw = fetch_doc(
        query=query,
        collection=collection,
        api_key=api_key,
        tenant=tenant,
        database=database,
        n_results=k,
    )
    return [s for s in raw.split("\n---\n") if s.strip()]


def search_stage(*, stage: str, query: str, k: int = 3) -> List[str]:
    collection = STAGE_COLLECTIONS.get(stage)
    if not collection:
        return []
    raw = fetch_doc(query=query, collection=collection)
    return [s for s in raw.split("\n---\n") if s.strip()][:k]
