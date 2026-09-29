import hashlib
import logging
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .vectorstore import get_index

logger = logging.getLogger(__name__)

_embedder = GoogleGenerativeAIEmbeddings(
    model=settings.embedding_model,
    google_api_key=settings.gemini_api_key,
)


def load_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) tuples (1-indexed page numbers)."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    source_name: str,
) -> list[dict]:
    """Split each page's text and attach page metadata to every chunk.

    source_name is included in the chunk ID so IDs are scoped to a document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_num, text in pages:
        for idx, chunk_text in enumerate(splitter.split_text(text)):
            chunk_id = hashlib.sha256(
                f"{source_name}:{page_num}:{idx}".encode()
            ).hexdigest()[:16]
            chunks.append(
                {"id": chunk_id, "text": chunk_text, "page": page_num}
            )
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the Gemini embedding model."""
    return _embedder.embed_documents(texts)


def ingest(pdf_path: Path) -> int:
    """Full ingestion pipeline. Returns the number of vectors upserted."""
    logger.info("Loading %s", pdf_path)
    pages = load_pages(pdf_path)
    chunks = chunk_pages(pages, source_name=pdf_path.name)

    index = get_index()
    batch_size = 100
    upserted = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed([c["text"] for c in batch])
        records = [
            {
                "id": c["id"],
                "values": v,
                "metadata": {"text": c["text"], "page": c["page"]},
            }
            for c, v in zip(batch, vectors)
        ]
        index.upsert(vectors=records)
        upserted += len(records)
        logger.info("Upserted %d / %d chunks", upserted, len(chunks))
        # gemini-embedding-2 free tier: 100 requests/min. Sleep between batches.
        if start + batch_size < len(chunks):
            logger.info("Rate-limit pause: sleeping 65 s")
            time.sleep(65)

    return upserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/Ebook-Agentic-AI.pdf")
    total = ingest(path)
    print(f"Done. {total} vectors upserted.")
