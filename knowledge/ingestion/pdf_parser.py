"""PDF -> list of pages {"source", "page", "text", "is_ocr"}.

Text PDFs use pypdf. A page with (almost) no text layer is treated as a scan: it is rasterised with
pypdfium2 and read with Tesseract if both are installed. If OCR is not available the page is returned with
empty text and `ocr_unavailable=True`; text is NEVER invented, because everything we return is later cited
as evidence.
"""
import os
import shutil
from typing import Any, Dict, List

from pypdf import PdfReader

MIN_TEXT_CHARS = 10


def ocr_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pypdfium2  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_image(img) -> str:
    import pytesseract
    return (pytesseract.image_to_string(img.convert("L")) or "").strip()


def _ocr_page(pdf_path: str, idx: int) -> str:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        img = pdf[idx].render(scale=2.5).to_pil()
        return ocr_image(img)
    finally:
        pdf.close()


class PDFIngestor:
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found: {pdf_path}")
        filename = os.path.basename(pdf_path)
        can_ocr = ocr_available()
        pages = []
        for idx, page in enumerate(PdfReader(pdf_path).pages):
            text = (page.extract_text() or "").strip()
            is_ocr, unavailable = False, False
            if len(text) < MIN_TEXT_CHARS:
                if can_ocr:
                    try:
                        text, is_ocr = _ocr_page(pdf_path, idx), True
                    except Exception:  # a bad page must not stop the whole document
                        text, unavailable = "", True
                else:
                    text, unavailable = "", True
            pages.append({"source": filename, "page": idx + 1, "text": text, "is_ocr": is_ocr,
                          "ocr_unavailable": unavailable})
        return pages
