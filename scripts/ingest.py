from pathlib import Path
import json
import gc
import sys
import traceback
import importlib


# ============================================================
# Project root
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "chunks.json"


# ============================================================
# Safe import helper
# ============================================================

def import_pipeline_modules():
    """
    Import project modules one by one.

    We catch BaseException intentionally here because errors such as
    SystemExit do not inherit from Exception and can otherwise terminate
    the process without giving us useful diagnostics.
    """

    modules = {}

    imports = [
        (
            "pdf_loader",
            "src.ingestion.pdf_loader",
            "load_pdf",
        ),
        (
            "cleaner",
            "src.ingestion.cleaner",
            "clean_documents",
        ),
        (
            "markdown_converter",
            "src.ingestion.markdown_converter",
            "convert_documents_to_markdown",
        ),
        (
            "chunker",
            "src.chunking.chunker",
            "create_chunks",
        ),
        (
            "metadata",
            "src.chunking.metadata",
            "enrich_chunk_metadata",
        ),
    ]

    print(
        "\n[1/5] Loading pipeline modules...",
        flush=True,
    )

    for name, module_name, function_name in imports:

        print(
            f"      Importing {name}...",
            flush=True,
        )

        try:
            module = importlib.import_module(
                module_name
            )

            function = getattr(
                module,
                function_name,
            )

            modules[name] = function

            print(
                f"      {name} OK",
                flush=True,
            )

        except BaseException as exc:

            print(
                "\n" + "=" * 70,
                flush=True,
            )

            print(
                f"FAILED IMPORT: {name}",
                flush=True,
            )

            print(
                f"Module: {module_name}",
                flush=True,
            )

            print(
                f"Function: {function_name}",
                flush=True,
            )

            print(
                f"Exception type: {type(exc).__name__}",
                flush=True,
            )

            print(
                f"Exception: {exc}",
                flush=True,
            )

            print(
                f"Python executable: {sys.executable}",
                flush=True,
            )

            print(
                f"Project root: {ROOT_DIR}",
                flush=True,
            )

            print(
                "=" * 70,
                flush=True,
            )

            traceback.print_exc()

            raise

    print(
        "      All pipeline modules loaded successfully.",
        flush=True,
    )

    return modules


# ============================================================
# Main pipeline
# ============================================================

def main():

    print("=" * 70, flush=True)
    print(
        "SickleGuide - Ingestion Pipeline",
        flush=True,
    )
    print("=" * 70, flush=True)

    print(
        f"\nPython: {sys.executable}",
        flush=True,
    )

    print(
        f"Project root: {ROOT_DIR}",
        flush=True,
    )

    try:

        # ------------------------------------------------------
        # 1. Load modules
        # ------------------------------------------------------

        pipeline = import_pipeline_modules()

        load_pdf = pipeline["pdf_loader"]
        clean_documents = pipeline["cleaner"]
        convert_documents_to_markdown = (
            pipeline["markdown_converter"]
        )
        create_chunks = pipeline["chunker"]
        enrich_chunk_metadata = pipeline["metadata"]

        # ------------------------------------------------------
        # 2. Validate directories
        # ------------------------------------------------------

        print(
            "\n[2/5] Validating data directories...",
            flush=True,
        )

        if not RAW_DIR.exists():
            raise FileNotFoundError(
                f"Raw directory not found: {RAW_DIR}"
            )

        pdf_files = sorted(
            RAW_DIR.glob("*.pdf")
        )

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in: {RAW_DIR}"
            )

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"      PDFs found: {len(pdf_files)}",
            flush=True,
        )

        for pdf in pdf_files:
            print(
                f"      - {pdf.name}",
                flush=True,
            )

        # ------------------------------------------------------
        # 3. Process PDFs one by one
        # ------------------------------------------------------

        print(
            "\n[3/5] Processing PDFs...",
            flush=True,
        )

        all_chunks = []

        total_pdfs = len(pdf_files)

        for pdf_index, pdf_file in enumerate(
            pdf_files,
            start=1,
        ):

            print(
                "\n" + "-" * 70,
                flush=True,
            )

            print(
                f"PDF {pdf_index}/{total_pdfs}: "
                f"{pdf_file.name}",
                flush=True,
            )

            # --------------------------------------------------
            # Load
            # --------------------------------------------------

            print(
                "  -> Loading with PyMuPDF4LLM...",
                flush=True,
            )

            documents = load_pdf(
                str(pdf_file)
            )

            print(
                f"  -> Pages loaded: "
                f"{len(documents)}",
                flush=True,
            )

            if not documents:
                print(
                    "  -> WARNING: no documents extracted.",
                    flush=True,
                )
                continue

            # --------------------------------------------------
            # Clean
            # --------------------------------------------------

            print(
                "  -> Cleaning...",
                flush=True,
            )

            documents = clean_documents(
                documents
            )

            print(
                f"  -> After cleaning: "
                f"{len(documents)}",
                flush=True,
            )

            # --------------------------------------------------
            # Markdown normalization
            # --------------------------------------------------

            print(
                "  -> Normalizing Markdown...",
                flush=True,
            )

            documents = (
                convert_documents_to_markdown(
                    documents
                )
            )

            print(
                f"  -> Markdown documents: "
                f"{len(documents)}",
                flush=True,
            )

            # --------------------------------------------------
            # Chunking
            # --------------------------------------------------

            print(
                "  -> Creating LangChain chunks...",
                flush=True,
            )

            pdf_chunks = create_chunks(
                documents
            )

            print(
                f"  -> Chunks created: "
                f"{len(pdf_chunks)}",
                flush=True,
            )

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------

            print(
                "  -> Enriching metadata...",
                flush=True,
            )

            pdf_chunks = enrich_chunk_metadata(
                pdf_chunks
            )

            all_chunks.extend(
                pdf_chunks
            )

            print(
                f"  -> Accumulated chunks: "
                f"{len(all_chunks)}",
                flush=True,
            )

            # Release memory before next PDF.
            del documents
            del pdf_chunks

            gc.collect()

        # ------------------------------------------------------
        # 4. Validate final output
        # ------------------------------------------------------

        print(
            "\n[4/5] Validating final chunks...",
            flush=True,
        )

        if not all_chunks:
            raise RuntimeError(
                "Pipeline completed but produced zero chunks."
            )

        print(
            f"      Final chunks: "
            f"{len(all_chunks)}",
            flush=True,
        )

        # ------------------------------------------------------
        # 5. Save output
        # ------------------------------------------------------

        print(
            "\n[5/5] Saving processed data...",
            flush=True,
        )

        serialized_chunks = []

        for chunk in all_chunks:

            serialized_chunks.append(
                {
                    "page_content": (
                        chunk.page_content
                    ),
                    "metadata": (
                        chunk.metadata
                    ),
                }
            )

        with OUTPUT_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                serialized_chunks,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        print(
            f"      Saved to: {OUTPUT_FILE}",
            flush=True,
        )

        # ------------------------------------------------------
        # Summary
        # ------------------------------------------------------

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "INGESTION COMPLETED SUCCESSFULLY",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        print(
            f"PDF files : {len(pdf_files)}",
            flush=True,
        )

        print(
            f"Final chunks : {len(all_chunks)}",
            flush=True,
        )

        print(
            f"Output : {OUTPUT_FILE}",
            flush=True,
        )

        print(
            "\nFirst chunk preview:",
            flush=True,
        )

        print(
            all_chunks[0].page_content[:1000],
            flush=True,
        )

        print(
            "\nFirst chunk metadata:",
            flush=True,
        )

        print(
            all_chunks[0].metadata,
            flush=True,
        )

        print(
            "\n" + "=" * 70,
            flush=True,
        )

    except BaseException as exc:

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "INGESTION FAILED",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        print(
            f"Exception type: {type(exc).__name__}",
            flush=True,
        )

        print(
            f"Exception: {exc}",
            flush=True,
        )

        print(
            f"Python: {sys.executable}",
            flush=True,
        )

        print(
            f"Project root: {ROOT_DIR}",
            flush=True,
        )

        traceback.print_exc()

        print(
            "=" * 70,
            flush=True,
        )

        raise


if __name__ == "__main__":
    main()