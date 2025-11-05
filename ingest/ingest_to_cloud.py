"""Ingest documents to ChromaDB Cloud."""

import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.chroma_client import get_chroma_client
from src.core.chroma_config import STAGE_COLLECTIONS

# Load environment variables from .env file
load_dotenv()


def extract_pdf_text(path: Path) -> str:
    """Extract text from PDF."""
    text_parts = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        except Exception as e:
            print(f"WARNING: Failed to extract page {i} from {path}: {e}")
    return "\n".join(text_parts)


def extract_html_text(path: Path) -> str:
    """Extract text from HTML."""
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    """Chunk text with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_file(path: Path, stage: str, client, batch_size: int = 100):
    """Ingest a file into ChromaDB."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        raw_text = extract_pdf_text(path)
    elif ext == ".html":
        raw_text = extract_html_text(path)
    else:
        print(f"Skipping unsupported file: {path}")
        return

    if not raw_text or not raw_text.strip():
        print(f"WARNING: No text extracted from {path}")
        return

    chunks = chunk_text(raw_text)
    collection_name = STAGE_COLLECTIONS[stage]
    collection = client.get_or_create_collection(name=collection_name)

    # Upload in batches
    for i in range(0, len(chunks), batch_size):
        batch_docs = chunks[i : i + batch_size]
        batch_ids = [f"{path.name}-{i + j}" for j in range(len(batch_docs))]
        batch_meta = [{"source": path.name, "stage": stage}] * len(batch_docs)

        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
        except Exception as e:
            print(f"WARNING: Failed to add batch {i} for {path.name}: {e}")
            continue

    print(f"Ingested {path.name} -> {stage} ({len(chunks)} chunks)")


def main():
    """Main ingestion."""
    print("\nChromaDB Cloud Ingestion\n")

    # Ensure cloud mode
    os.environ["CHROMA_MODE"] = "cloud"

    try:
        client = get_chroma_client()
    except ValueError as e:
        print(f"ERROR: Failed to connect to ChromaDB: {e}")
        print("\nPlease ensure you have set the following environment variables:")
        print("  - CHROMA_API_KEY")
        print("  - CHROMA_TENANT")
        print("  - CHROMA_DATABASE")
        print("\nOr create a .env file with these values.\n")
        return

    # Define files to ingest (adjust paths as needed)
    files_to_ingest = {
        "concepts": [
            Path("ingest/concepts/cuda_guide.html"),
            Path("ingest/concepts/cuda_guide.pdf"),
            Path("ingest/concepts/warp_primitives.html"),
        ],
        "patterns": [
            Path("ingest/patterns/reduction.pdf"),
        ],
        "hardware": [
            Path("ingest/hardware/ampere_tuning.html"),
            Path("ingest/hardware/ampere_tuning.pdf"),
            Path("ingest/hardware/ampere_compat.html"),
        ],
        "api": [
            Path("ingest/api/cuda_guide.html"),
            Path("ingest/api/best_practices.html"),
        ],
        "examples": [
            Path("ingest/examples/reduction.pdf"),
        ],
    }

    total_files = sum(len(paths) for paths in files_to_ingest.values())
    processed = 0

    for stage, paths in files_to_ingest.items():
        print(f"\nProcessing stage: {stage}")
        for path in paths:
            if path.exists():
                ingest_file(path, stage, client)
                processed += 1
            else:
                print(f"WARNING: File not found: {path}")

    print("\n" + "=" * 60)
    print(f"Ingestion complete. Processed {processed}/{total_files} files\n")


if __name__ == "__main__":
    main()
