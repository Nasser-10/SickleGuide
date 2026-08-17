import re
from collections import Counter
from typing import List, Dict


def remove_repeated_headers_footers(
    pages: List[Dict],
    min_occurrences: int = 3,
) -> List[Dict]:
    """
    Remove repeated headers and footers from PDF pages.

    Headers and footers are detected separately for each PDF.
    A line is considered repeated if it appears in the top or
    bottom part of at least `min_occurrences` pages.
    """

    # Group pages by PDF source
    pages_by_source = {}

    for page in pages:
        pages_by_source.setdefault(page["source"], []).append(page)

    cleaned_pages = []

    # Process each PDF separately
    for source, source_pages in pages_by_source.items():

        top_lines = []
        bottom_lines = []

        # Collect possible header/footer lines
        for page in source_pages:

            lines = [
                line.strip()
                for line in page["text"].splitlines()
                if line.strip()
            ]

            if not lines:
                continue

            # First 3 lines → possible headers
            top_lines.extend(lines[:3])

            # Last 3 lines → possible footers
            bottom_lines.extend(lines[-3:])

        # Count repeated lines
        top_counts = Counter(top_lines)
        bottom_counts = Counter(bottom_lines)

        # Detect repeated headers
        repeated_headers = {
            line
            for line, count in top_counts.items()
            if count >= min_occurrences
        }

        # Detect repeated footers
        repeated_footers = {
            line
            for line, count in bottom_counts.items()
            if count >= min_occurrences
        }

        # Remove detected headers/footers
        for page in source_pages:

            lines = [
                line.strip()
                for line in page["text"].splitlines()
                if line.strip()
            ]

            filtered_lines = [
                line
                for line in lines
                if line not in repeated_headers
                and line not in repeated_footers
            ]

            cleaned_pages.append(
                {
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "text": "\n".join(filtered_lines),
                }
            )

    return cleaned_pages


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving
    the clinical meaning and paragraph structure.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Fix words broken across lines by PDF extraction
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert single line breaks inside paragraphs to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Keep paragraph breaks, but remove excessive ones
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Clean all extracted PDF pages while preserving metadata.
    """

    cleaned_pages = []

    for page in pages:

        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        cleaned_pages.append(
            {
                "source": page["source"],
                "page_number": page["page_number"],
                "text": cleaned_text,
            }
        )

    return cleaned_pages