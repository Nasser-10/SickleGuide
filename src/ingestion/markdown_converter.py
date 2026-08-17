from typing import List, Dict


def page_to_markdown(page: Dict) -> Dict:
    """
    Convert one cleaned PDF page into a Markdown document
    while preserving source and page metadata.
    """

    source = page["source"]
    page_number = page["page_number"]
    text = page["text"].strip()

    markdown = f"# Source: {source}\n\n"
    markdown += f"**Page:** {page_number}\n\n"
    markdown += text

    return {
        "source": source,
        "page_number": page_number,
        "text": markdown,
    }


def pages_to_markdown(pages: List[Dict]) -> List[Dict]:
    """
    Convert all cleaned pages to Markdown.
    """

    return [page_to_markdown(page) for page in pages]