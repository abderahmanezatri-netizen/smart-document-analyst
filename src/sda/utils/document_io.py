from __future__ import annotations

import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

SUPPORTED = {
    ".txt",
    ".md",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEXT_EXTS = {".txt", ".md"}


def validate_document_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(
            f"Unsupported file type {path.suffix}. "
            f"Supported: {sorted(SUPPORTED)}"
        )

    return path


def extract_text(path: Path) -> str:
    """
    Extract text from a supported document.

    For the final real-dataset version:
    - text files are read directly
    - PDFs are parsed with pypdf
    - images are processed with OCR using pytesseract
    """
    validate_document_path(path)
    suffix = path.suffix.lower()

    if suffix in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace").strip()

    if suffix == ".pdf":
        return extract_text_from_pdf(path)

    if suffix in IMAGE_EXTS:
        return extract_text_from_image(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts).strip()

        if not text:
            return "[PDF warning] No readable text was extracted from this PDF."

        return text

    except Exception as exc:
        return f"[PDF extraction error] {exc}"


def extract_text_from_image(path: Path) -> str:
    """
    OCR extraction for real document images.
    Requires:
        pip install pytesseract
    and Tesseract OCR installed on Windows.
    """
    try:
        import pytesseract

        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        image = Image.open(path).convert("RGB")
        text = pytesseract.image_to_string(image)

        text = text.strip()

        if not text:
            return "[OCR warning] No readable text was extracted from this image."

        return text

    except Exception as exc:
        return (
            "[OCR error] Text extraction failed for this image. "
            f"Reason: {exc}"
        )


def render_text_to_image(text: str, out_path: Path, size: int = 128) -> Path:
    """
    Kept for compatibility with text/PDF inputs.
    The final classifier uses real image documents, but this helper allows
    text documents to be converted into simple image representations if needed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("L", (512, 720), color=255)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    y = 20
    lines = text.splitlines() or [text]

    for i, line in enumerate(lines[:38]):
        wrapped = textwrap.wrap(line, width=62) or [""]

        for part in wrapped[:3]:
            if y > 690:
                break

            draw.text(
                (25, y),
                part,
                fill=0,
                font=title_font if i == 0 else font,
            )
            y += 18

    img = img.resize((size, size))
    img.save(out_path)

    return out_path


def document_to_image(
    path: Path,
    cache_dir: Path = Path("outputs/cache"),
    size: int = 224,
) -> Path:
    """
    Convert supported documents to an image path.

    For real image documents, this keeps the image format compatible
    with the classifier pipeline.
    """
    validate_document_path(path)
    suffix = path.suffix.lower()

    out = cache_dir / f"{path.stem}_{size}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    if suffix in IMAGE_EXTS:
        img = Image.open(path).convert("RGB").resize((size, size))
        img.save(out)
        return out

    text = extract_text(path)
    return render_text_to_image(text, out, size=size)