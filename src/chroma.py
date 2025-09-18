import os
import re
from pypdf import PdfReader
import chromadb
from bs4 import BeautifulSoup

# -------------------------
# 1. Initialize Chroma Cloud client
# -------------------------
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# -------------------------
# 2. Stage -> Collection mapping
# -------------------------
STAGE_COLLECTIONS = {
    "concepts": "cuda-concepts",
    "patterns": "cuda-patterns",
    "hardware": "cuda-hardware",
    "api": "cuda-api",
    "examples": "cuda-examples",
}

collections = {
    stage: client.get_or_create_collection(name=col_name)
    for stage, col_name in STAGE_COLLECTIONS.items()
}

# -------------------------
# 3. Helper: clean + extract PDF text
# -------------------------
def extract_pdf_text(path: str) -> str:
    text_parts = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        except Exception as e:
            print(f"⚠️ Failed to extract page {i} from {path}: {e}")
    return "\n".join(text_parts)


# -------------------------
# 4. Helper: clean + extract HTML text
# -------------------------
def extract_html_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style/nav/footer tags
    for tag in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Compact whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


# -------------------------
# 5. Helper: chunk text
# -------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# -------------------------
# 6. Ingest function
# -------------------------
def ingest_file(path: str, stage: str, batch_size: int = 500):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_pdf_text(path)
    elif ext == ".html":
        raw_text = extract_html_text(path)
    else:
        print(f"Skipping unsupported file: {path}")
        return

    chunks = chunk_text(raw_text)
    collection = collections[stage]

    # Upload in smaller batches
    for i in range(0, len(chunks), batch_size):
        batch_docs = chunks[i : i + batch_size]
        batch_ids = [f"{os.path.basename(path)}-{i+j}" for j in range(len(batch_docs))]
        batch_meta = [{"source": os.path.basename(path), "stage": stage}] * len(batch_docs)

        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
        )

    print(f"✅ Ingested {path} into {stage} ({len(chunks)} chunks in batches of {batch_size})")


# -------------------------
# 7. File ingestion paths
# -------------------------
FILES_TO_INGEST = {
    "concepts": [
        "ingest/concepts/cuda_guide.html",
        "ingest/concepts/cuda_guide.pdf",
        "ingest/concepts/warp_primitives.html",
    ],
    "patterns": [
        "ingest/patterns/reduction.pdf",
    ],
    "hardware": [
        "ingest/hardware/ampere_tuning.html",
        "ingest/hardware/ampere_tuning.pdf",
        "ingest/hardware/ampere_compat.html",
    ],
    "api": [
        "ingest/api/cuda_guide.html",
        "ingest/api/best_practices.html",
    ],
    "examples": [
        "ingest/examples/reduction.pdf",
    ],
}

# -------------------------
# 8. Run ingestion
# -------------------------
def main():
    for stage, files in FILES_TO_INGEST.items():
        for path in files:
            if os.path.exists(path):
                ingest_file(path, stage)
            else:
                print(f"⚠️ File not found: {path}")


if __name__ == "__main__":
    main()
