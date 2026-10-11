"""Split pages into retrieval chunks that never cross a page (so every chunk has an exact page citation)."""
import re
from typing import Any, Dict, List


class DocumentChunker:
    def __init__(self, chunk_size: int = 600, overlap_sentences: int = 1):
        self.chunk_size = chunk_size
        self.overlap = overlap_sentences

    @staticmethod
    def _sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue
            source, num = page["source"], page["page"]
            if len(text) <= self.chunk_size:
                groups = [text]
            else:
                groups, cur = [], []
                for s in self._sentences(text):
                    if cur and len(" ".join(cur + [s])) > self.chunk_size:
                        groups.append(" ".join(cur))
                        cur = cur[-self.overlap:] if self.overlap else []
                    cur.append(s)
                if cur:
                    groups.append(" ".join(cur))
            for i, g in enumerate(groups):
                chunks.append({"chunk_id": f"{source}_p{num}_c{i}", "source": source, "page": num, "text": g})
        return chunks
