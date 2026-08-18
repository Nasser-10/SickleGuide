import re
from typing import List

from langchain_core.documents import Document


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace without destroying Markdown."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+$",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"[ \t]+([,.;:!?])",
        r"\1",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_bullets(text: str) -> str:
    """Normalize common PDF bullet characters."""

    return re.sub(
        r"^[ \t]*[•●▪◦]\s+",
        "- ",
        text,
        flags=re.MULTILINE,
    )


def remove_consecutive_duplicates(text: str) -> str:
    """
    Remove only immediately duplicated lines.

    Intentionally conservative for medical content.
    """

    lines = text.splitlines()

    result = []
    previous = None

    for line in lines:

        normalized = " ".join(
            line.strip().split()
        )

        if (
            normalized
            and normalized == previous
            and len(normalized) >= 10
        ):
            continue

        result.append(line)

        if normalized:
            previous = normalized

    return "\n".join(result)


def clean_markdown(text: str) -> str:
    """Clean extracted Markdown safely."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return ""

    text = normalize_whitespace(text)
    text = normalize_bullets(text)
    text = remove_consecutive_duplicates(text)

    return normalize_whitespace(text)


def clean_documents(
    documents: List[Document],
) -> List[Document]:
    """Clean LangChain Documents while preserving metadata."""

    cleaned = []

    for document in documents:

        text = clean_markdown(
            document.page_content
        )

        if not text:
            continue

        cleaned.append(
            Document(
                page_content=text,
                metadata=document.metadata.copy(),
            )
        )

    return cleaned