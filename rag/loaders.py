from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from pypdf import PdfReader


@dataclass
class DocumentChunk:
    """Represents a page-level text unit with metadata (pre-chunking)."""
    text: str
    metadata: Dict[str, Any]


def _clean_text(text: str) -> str:
    """Light cleanup for embedding & LLM prompting."""
    if not text:
        return ""
    # Normalize whitespace
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    return text.strip()


def load_pdfs_from_folder(
    folder_path: str | Path,
    doc_type: str = "radiology_reference",
    min_chars: int = 200,
) -> List[DocumentChunk]:
    """
    Load PDFs from a folder and return page-level DocumentChunks.
    Each chunk contains text + metadata: source, page, doc_type.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    docs: List[DocumentChunk] = []

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)

        for i in range(num_pages):
            page = reader.pages[i]
            raw_text = page.extract_text() or ""
            text = _clean_text(raw_text)

            # Skip empty/too-short pages (often headers, references, etc.)
            if len(text) < min_chars:
                continue

            docs.append(
                DocumentChunk(
                    text=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": i + 1,  # human-readable page index
                        "doc_type": doc_type,
                        "path": str(pdf_path),
                    },
                )
            )

    return docs


def load_txts_from_folder(
    folder_path: str | Path,
    doc_type: str = "radiology_reference",
    min_chars: int = 200,
    encoding: str = "utf-8",
) -> List[DocumentChunk]:
    """
    Load .txt files from a folder. Useful if you add text docs later.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {folder}")

    txt_files = sorted(folder.glob("*.txt"))
    docs: List[DocumentChunk] = []

    for txt_path in txt_files:
        raw = txt_path.read_text(encoding=encoding, errors="ignore")
        text = _clean_text(raw)
        if len(text) < min_chars:
            continue

        docs.append(
            DocumentChunk(
                text=text,
                metadata={
                    "source": txt_path.name,
                    "page": 1,
                    "doc_type": doc_type,
                    "path": str(txt_path),
                },
            )
        )

    return docs


def load_knowledge_base(
    kb_folder: str | Path = "data/knowledge_base",
    doc_type: str = "radiology_reference",
) -> List[DocumentChunk]:
    """
    Load all KB documents (PDF + TXT) into page-level chunks.
    """
    pdf_docs = load_pdfs_from_folder(kb_folder, doc_type=doc_type)
    txt_docs = load_txts_from_folder(kb_folder, doc_type=doc_type)
    return pdf_docs + txt_docs
