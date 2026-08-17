from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.cleaner import (
    clean_pages,
    remove_repeated_headers_footers,
)
from src.ingestion.markdown_converter import pages_to_markdown


def main():
    print("=" * 60)
    print("SickleGuide - Document Ingestion")
    print("=" * 60)

    # Step 1: Load all PDF documents
    print("\n[1/4] Loading PDF documents...")
    pages = load_all_pdfs("data/raw")
    print(f"Loaded {len(pages)} pages.")

    # Step 2: Remove repeated headers and footers
    print("\n[2/4] Removing repeated headers and footers...")
    pages = remove_repeated_headers_footers(pages)
    print(f"Processed {len(pages)} pages.")

    # Step 3: Clean extracted text
    print("\n[3/4] Cleaning extracted text...")
    cleaned_pages = clean_pages(pages)
    print(f"Cleaned {len(cleaned_pages)} pages.")

    # Step 4: Convert pages to Markdown
    print("\n[4/4] Converting pages to Markdown...")
    markdown_pages = pages_to_markdown(cleaned_pages)
    print(f"Converted {len(markdown_pages)} pages.")

    # Display a small sample
    if markdown_pages:
        print("\n" + "=" * 60)
        print("SAMPLE DOCUMENT")
        print("=" * 60)
        print(markdown_pages[0]["text"][:1000])

    print("\n" + "=" * 60)
    print("Ingestion completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()