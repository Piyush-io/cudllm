"""ChromaDB client factory - cloud-first."""

import os
import logging
from typing import Optional
import chromadb
from chromadb import CloudClient, PersistentClient


_logger = logging.getLogger(__name__)


def get_chroma_client(
    mode: Optional[str] = None,
    api_key: Optional[str] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
    persist_path: Optional[str] = None,
):
    """
    Get ChromaDB client based on mode (cloud/persistent).

    Args:
        mode: 'cloud' or 'persistent'. Defaults to CHROMA_MODE env var or 'cloud'.
        api_key: Cloud API key. Defaults to CHROMA_API_KEY env var.
        tenant: Cloud tenant ID. Defaults to CHROMA_TENANT env var.
        database: Cloud database name. Defaults to CHROMA_DATABASE env var.
        persist_path: Persistent storage path. Defaults to CHROMA_PERSIST_PATH env var or './chroma_db'.

    Returns:
        ChromaDB client instance (CloudClient or PersistentClient).

    Raises:
        ValueError: If mode is invalid or required credentials are missing.
    """
    mode = mode or os.environ.get("CHROMA_MODE", "cloud").lower()

    if mode == "cloud":
        api_key = api_key or os.environ.get("CHROMA_API_KEY")
        tenant = tenant or os.environ.get("CHROMA_TENANT")
        database = database or os.environ.get("CHROMA_DATABASE")

        if not all([api_key, tenant, database]):
            raise ValueError(
                "Cloud mode requires CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE. "
                "Please set these environment variables or pass them as arguments."
            )

        _logger.info("Using ChromaDB Cloud: tenant=%s database=%s", tenant, database)
        return CloudClient(api_key=api_key, tenant=tenant, database=database)

    elif mode == "persistent":
        persist_path = persist_path or os.environ.get(
            "CHROMA_PERSIST_PATH", "./chroma_db"
        )
        _logger.info("Using ChromaDB Persistent: path=%s", persist_path)
        return PersistentClient(path=persist_path)

    else:
        raise ValueError(
            f"Unknown CHROMA_MODE='{mode}'. Must be 'cloud' or 'persistent'. "
            f"Set CHROMA_MODE environment variable or pass mode argument."
        )


__all__ = ["get_chroma_client"]
