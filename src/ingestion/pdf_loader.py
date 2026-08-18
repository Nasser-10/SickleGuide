from pathlib import Path
from typing import List

import pymupdf4llm
from langchain_core.documents import Document


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Extract one PDF into page-level LangChain Documents.

    PyMuPDF4LLM handles:
    - Markdown extraction
    - tables
    - images / pictures detection
    - automatic OCR when needed
    - header/footer filtering
    - page-level metadata
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF file, got: {path.suffix}")

    pages = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True,
        header=False,
        footer=False,
        use_ocr=True,
        force_ocr=False,
        force_text=True,
        table_strategy="lines_strict",
        ignore_images=False,
        embed_images=False,
        write_images=False,
        ignore_graphics=False,
        show_progress=True,
    )

    documents: List[Document] = []

    total_pages = len(pages)

    for index, page in enumerate(pages, start=1):

        text = page.get("text", "").strip()

        if not text:
            continue

        raw_metadata = page.get("metadata", {})

        tables = page.get("tables", [])
        images = page.get("images", [])
        page_boxes = page.get("page_boxes", [])

        # Keep only useful layout information.
        layout_classes = [
            box.get("class")
            for box in page_boxes
            if isinstance(box, dict) and box.get("class")
        ]

        metadata = {
            "source": path.name,
            "file_path": str(path),
            "file_type": "pdf",

            "page_number": raw_metadata.get(
                "page_number",
                index,
            ),

            "total_pages": raw_metadata.get(
                "page_count",
                total_pages,
            ),

            "title": raw_metadata.get(
                "title",
                "",
            ),

            "author": raw_metadata.get(
                "author",
                "",
            ),

            "has_table": bool(tables),
            "table_count": len(tables),

            "has_image": bool(images),
            "image_count": len(images),

            "layout_classes": layout_classes,

            # Useful for citation / document classification.
            "document_type": "medical_guideline",
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


def load_all_pdfs(
    data_dir: str = "data/raw",
) -> List[Document]:
    """
    Load all PDFs from a directory.

    This function is kept for testing / utility use.
    The main ingestion runner processes PDFs one at a time.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}"
        )

    pdf_files = sorted(
        data_path.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {data_path}"
        )

    documents: List[Document] = []

    for pdf_file in pdf_files:
        documents.extend(
            load_pdf(str(pdf_file))
        )

    return documents