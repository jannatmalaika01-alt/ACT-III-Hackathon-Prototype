"""Builds the index from every PDF in data/raw and exposes a ready-to-use singleton."""
import os
from typing import Optional

from ..chunking.chunker import DocumentChunker
from ..ingestion.pdf_parser import PDFIngestor
from .vector_store import KnowledgeVectorStore

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw")


def build_store(raw_dir: str = RAW_DIR) -> KnowledgeVectorStore:
    store, parser, chunker = KnowledgeVectorStore(), PDFIngestor(), DocumentChunker()
    chunks = []
    if os.path.isdir(raw_dir):
        for name in sorted(os.listdir(raw_dir)):
            if name.lower().endswith(".pdf"):
                chunks.extend(chunker.chunk_pages(parser.parse_pdf(os.path.join(raw_dir, name))))
    store.add_chunks(chunks)
    return store


_STORE: Optional[KnowledgeVectorStore] = None


def get_store() -> KnowledgeVectorStore:
    global _STORE
    if _STORE is None:
        _STORE = build_store()
    return _STORE
