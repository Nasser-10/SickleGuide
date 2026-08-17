from pathlib import Path
from typing import List, Dict

import pymupdf


def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF page by page.

    Each page is returned with:
    - source: PDF file name
    - page_number: page number
    - text: extracted text
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                {
                    "source": path.name,
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def load_all_pdfs(data_dir: str = "data/raw") -> List[Dict]:
    """
    Load all PDF files from the data/raw directory.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    all_pages = []

    pdf_files = sorted(data_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {data_path}"
        )

    for pdf_file in pdf_files:
        pages = load_pdf(str(pdf_file))
        all_pages.extend(pages)

    return all_pages