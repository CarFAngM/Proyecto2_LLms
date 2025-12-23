"""
Módulo para procesamiento de archivos PDF en modo funcional.
Extrae texto por página para permitir chunking y limpieza consistente.
"""

from pathlib import Path
import hashlib
import re
from typing import Dict, List, Optional
import unicodedata

import PyPDF2


def clean_text(text: str) -> str:
    """Limpia texto extraído del PDF manteniendo acentos y eliminando ruido."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = "".join(char for char in text if unicodedata.category(char)[0] != "C" or char in "\n\t ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n\n+", "\n\n", text)

    replacements = {
        "•": "",
        "◦": "",
        "▪": "",
        "▫": "",
        "●": "",
        "○": "",
        "►": "",
        "▸": "",
        "➢": "",
        "➤": "",
        "■": "",
        "□": "",
        "▶": "",
        "◆": "",
        "◇": "",
        "★": "",
        "☆": "",
        "✓": "",
        "✔": "",
        "✗": "",
        "✘": "",
        "⚫": "",
        "⚪": "",
        "🔹": "",
        "🔸": "",
        "▪️": "",
        "•️": "",
        "\uf0b7": "",
        "\uf0a7": "",
        "\uf0d8": "",
        "\uf076": "",
        "\uf0fc": "",
        "“": "\"",
        "”": "\"",
        "„": "\"",
        "‚": "'",
        "«": "\"",
        "»": "\"",
        "‹": "'",
        "›": "'",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "…": "...",
        "\xa0": " ",
        "\u2003": " ",
        "\u2002": " ",
        "\u2009": " ",
        "\u200b": "",
        "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    lines = [line.lstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    allowed_chars: List[str] = []
    for char in text:
        if 32 <= ord(char) <= 126:
            allowed_chars.append(char)
        elif char in "\n\t":
            allowed_chars.append(char)
        elif unicodedata.category(char) in ("Ll", "Lu", "Lt", "Lo", "Nd", "Nl", "No", "Pd", "Ps", "Pe", "Po"):
            allowed_chars.append(char)
        else:
            allowed_chars.append(" ")
    text = "".join(allowed_chars)

    text = re.sub(r"[ ]+", " ", text)
    lines = [line for line in text.split("\n") if not re.match(r"^[\s\-_.]+$", line)]
    text = "\n".join(lines)

    return text.strip()


def extract_page_texts(pdf_path: str) -> List[str]:
    """Extrae texto limpio por página."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            pages: List[str] = []
            for page in reader.pages:
                raw_text = page.extract_text() or ""
                pages.append(clean_text(raw_text))
            return pages
    except Exception as exc:  # pragma: no cover - depende del archivo
        raise Exception(f"Error al extraer texto del PDF {pdf_path}: {exc}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae y limpia texto completo del PDF."""
    return "\n".join(extract_page_texts(pdf_path))


def generate_document_hash(content: str) -> str:
    """Genera hash MD5 del contenido."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def extract_metadata(text: str, filename: str, total_pages: int) -> Dict[str, str]:
    """Extrae metadata básica y hash."""
    metadata: Dict[str, str] = {
        "filename": filename,
        "document_hash": generate_document_hash(text),
        "word_count": len(text.split()),
        "char_count": len(text),
        "total_pages": total_pages,
    }

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails = re.findall(email_pattern, text)
    if emails:
        metadata["email"] = emails[0]

    phone_pattern = r"(?:\+?502[\s.-]?)?(?:\(?502\)?[\s.-]?)?\b\d{4}[-\s.]?\d{4}\b"
    phones = re.findall(phone_pattern, text)
    if phones:
        metadata["phone"] = phones[0]

    return metadata


def validate_pdf(pdf_path: str) -> bool:
    """Valida que el archivo sea un PDF legible."""
    try:
        path = Path(pdf_path)
        if path.suffix.lower() != ".pdf" or not path.exists():
            return False
        with open(pdf_path, "rb") as file:
            PyPDF2.PdfReader(file)
        return True
    except Exception:
        return False


def process_cv(pdf_path: str) -> Optional[Dict]:
    """Procesa un PDF: texto completo, páginas limpias y metadata."""
    try:
        if not validate_pdf(pdf_path):
            raise ValueError(f"Archivo PDF inválido: {pdf_path}")

        page_texts = extract_page_texts(pdf_path)
        if not page_texts:
            raise ValueError(f"No se pudo extraer texto del PDF: {pdf_path}")

        full_text = "\n".join(page_texts)
        metadata = extract_metadata(full_text, Path(pdf_path).name, len(page_texts))

        return {
            "text": full_text,
            "page_texts": page_texts,
            "metadata": metadata,
        }
    except Exception as exc:
        print(f"Error procesando CV {pdf_path}: {exc}")
        return None
