import fitz  # PyMuPDF
from pypdf import PdfReader
import pytesseract
from PIL import Image
import io

def extract_text_fast(path: str) -> str:
    """Extract text from simple PDFs quickly."""
    
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text() or ""
        text += content + "\n"
    return text.strip()


def extract_text_layout(path: str) -> str:
    """Extract text from complex PDFs (columns, tables) using PyMuPDF."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()


def extract_text_ocr(path: str) -> str:
    """Extract text from scanned / image-only PDFs using OCR."""
    doc = fitz.open(path)
    final_text = ""
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        final_text += text + "\n"

    return final_text.strip()


def extract_text_from_pdf(path: str) -> str:
    """
    Auto-detect PDF type:
    1. Try fast extractor (pypdf)
    2. If text too small → use layout extractor
    3. If still empty → use OCR
    """

    # Step 1: Fast text extraction
    fast_text = extract_text_fast(path)
    if len(fast_text.split()) > 50:
        return fast_text

    # Step 2: Layout extraction
    layout_text = extract_text_layout(path)
    if len(layout_text.split()) > 50:
        return layout_text

    # Step 3: OCR for scanned PDFs
    ocr_text = extract_text_ocr(path)
    return ocr_text
