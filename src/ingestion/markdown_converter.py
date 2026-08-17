from typing import List, Dict


def escape_markdown(text: str) -> str:
    """
    Escape characters that may interfere with Markdown formatting.
    """
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
    )


def page_to_markdown(page: Dict) -> Dict:
    """
    Convert a single processed page into Markdown format.

    Preserves:
    - source
    - page number
    - text
    - tables
    - figures/images metadata when available
    """

    source = page["source"]
    page_number = page["page_number"]
    text = page.get("text", "").strip()

    markdown_parts = []

    # Document metadata
    markdown_parts.append(f"# Source: {source}")
    markdown_parts.append(f"**Page:** {page_number}")

    # Main text
    if text:
        markdown_parts.append(text)

    # Tables
    tables = page.get("tables", [])

    for table_index, table in enumerate(tables, start=1):
        markdown_parts.append(
            f"## Table {table_index}"
        )

        if isinstance(table, str):
            markdown_parts.append(table)

        elif isinstance(table, list) and table:
            # Convert table rows to Markdown
            rows = []

            for row in table:
                cleaned_row = [
                    escape_markdown(str(cell))
                    for cell in row
                ]
                rows.append(cleaned_row)

            if rows:
                # Header
                header = rows[0]
                markdown_parts.append(
                    "| " + " | ".join(header) + " |"
                )

                markdown_parts.append(
                    "| "
                    + " | ".join(["---"] * len(header))
                    + " |"
                )

                # Remaining rows
                for row in rows[1:]:
                    markdown_parts.append(
                        "| " + " | ".join(row) + " |"
                    )

    # Figures / Images
    figures = page.get("figures", [])

    for figure_index, figure in enumerate(figures, start=1):
        markdown_parts.append(
            f"## Figure {figure_index}"
        )

        if isinstance(figure, dict):
            image_path = figure.get("image_path")
            caption = figure.get("caption")

            if image_path:
                markdown_parts.append(
                    f"![Figure {figure_index}]({image_path})"
                )

            if caption:
                markdown_parts.append(
                    f"**Caption:** {caption}"
                )

        elif isinstance(figure, str):
            markdown_parts.append(
                f"![Figure {figure_index}]({figure})"
            )

    markdown_text = "\n\n".join(markdown_parts)

    return {
        "source": source,
        "page_number": page_number,
        "text": markdown_text,
    }


def pages_to_markdown(pages: List[Dict]) -> List[Dict]:
    """
    Convert all processed pages into Markdown documents.
    """

    markdown_pages = []

    for page in pages:
        markdown_page = page_to_markdown(page)

        if not markdown_page["text"].strip():
            continue

        markdown_pages.append(markdown_page)

    return markdown_pages
