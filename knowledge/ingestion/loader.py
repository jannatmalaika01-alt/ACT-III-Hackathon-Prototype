"""Turn any uploaded file into plain text (for document-type inputs)."""
import os

from .pdf_parser import PDFIngestor, ocr_available, ocr_image


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "\n".join(p["text"] for p in PDFIngestor().parse_pdf(path))
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):  # photographed / scanned document
        if not ocr_available():
            return ""
        from PIL import Image
        with Image.open(path) as img:
            return ocr_image(img)
    return open(path, encoding="utf-8", errors="replace").read()
