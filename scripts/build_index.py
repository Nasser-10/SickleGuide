from pathlib import Path
import json
import sys
import traceback


# ============================================================
# Project root
# ============================================================

ROOT_DIR = Path(
    __file__
).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# Paths
# ============================================================

CHUNKS_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "chunks.json"
)

CHROMA_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "chroma"
)


# ============================================================
# Build stable document identity
# ============================================================

def build_document_key(
    source: str,
    chunk_id: str,
) -> str:
    """
    Build a globally unique chunk identity.

    chunk_id is local to a PDF, so source must be included.
    """

    source = str(
        source
        or "unknown_source"
    ).strip()

    chunk_id = str(
        chunk_id
        or "unknown_chunk"
    ).strip()

    return (
        f"{source}::{chunk_id}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "SickleGuide - Vector Index Builder",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    try:

        # ----------------------------------------------------
        # 1. Validate chunks
        # ----------------------------------------------------

        print(
            "\n[1/4] Validating processed chunks...",
            flush=True,
        )

        if not CHUNKS_FILE.exists():

            raise FileNotFoundError(
                f"Chunks file not found: "
                f"{CHUNKS_FILE}"
            )

        with CHUNKS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            list,
        ):

            raise ValueError(
                "chunks.json must contain a JSON list."
            )

        if not data:

            raise ValueError(
                "chunks.json is empty."
            )

        print(
            f"      Raw chunk records: "
            f"{len(data)}",
            flush=True,
        )

        # ----------------------------------------------------
        # 2. Prepare LangChain Documents
        # ----------------------------------------------------

        print(
            "\n[2/4] Preparing LangChain documents...",
            flush=True,
        )

        from langchain_core.documents import Document

        documents = []

        seen_document_keys = set()

        duplicate_count = 0

        invalid_count = 0

        for index, item in enumerate(
            data
        ):

            if not isinstance(
                item,
                dict,
            ):
                invalid_count += 1
                continue

            text = (
                item.get(
                    "page_content",
                    "",
                )
                or ""
            ).strip()

            metadata = item.get(
                "metadata",
                {},
            )

            if not text:
                invalid_count += 1
                continue

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            metadata = metadata.copy()

            source = metadata.get(
                "source",
                metadata.get(
                    "file_path",
                    "unknown_source",
                ),
            )

            chunk_id = metadata.get(
                "chunk_id",
                index,
            )

            source = str(
                source
            )

            chunk_id = str(
                chunk_id
            )

            # IMPORTANT:
            # chunk_id restarts inside every PDF.
            # Therefore source + chunk_id is the true global ID.
            document_key = (
                build_document_key(
                    source=source,
                    chunk_id=chunk_id,
                )
            )

            if (
                document_key
                in seen_document_keys
            ):

                duplicate_count += 1
                continue

            seen_document_keys.add(
                document_key
            )

            metadata[
                "chunk_id"
            ] = chunk_id

            metadata[
                "document_key"
            ] = document_key

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        print(
            f"      Valid documents: "
            f"{len(documents)}",
            flush=True,
        )

        print(
            f"      Invalid records: "
            f"{invalid_count}",
            flush=True,
        )

        print(
            f"      Duplicate records: "
            f"{duplicate_count}",
            flush=True,
        )

        if not documents:

            raise RuntimeError(
                "No valid documents were prepared."
            )

        # ----------------------------------------------------
        # IMPORTANT VALIDATION
        # ----------------------------------------------------

        expected_count = len(
            documents
        )

        print(
            "\n      Source distribution:",
            flush=True,
        )

        source_counts = {}

        for document in documents:

            source = document.metadata.get(
                "source",
                "unknown",
            )

            source_counts[source] = (
                source_counts.get(
                    source,
                    0,
                )
                + 1
            )

        for source, count in (
            source_counts.items()
        ):

            print(
                f"        {count:4d} | {source}",
                flush=True,
            )

        # ----------------------------------------------------
        # 3. Load vector store
        # ----------------------------------------------------

        print(
            "\n[3/4] Loading vector store...",
            flush=True,
        )

        from src.retrieval.vector_store import (
            create_vector_store,
        )

        CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        vector_store = (
            create_vector_store()
        )

        existing_count = (
            vector_store.count()
        )

        print(
            f"      Existing vectors: "
            f"{existing_count}",
            flush=True,
        )

        print(
            f"      Expected vectors: "
            f"{expected_count}",
            flush=True,
        )

        # ----------------------------------------------------
        # Existing index is already correct
        # ----------------------------------------------------

        if (
            existing_count
            == expected_count
        ):

            print(
                "\n      Vector index already matches "
                "chunks.json.",
                flush=True,
            )

            print(
                "      No re-indexing required.",
                flush=True,
            )

        # ----------------------------------------------------
        # Empty index
        # ----------------------------------------------------

        elif existing_count == 0:

            print(
                "\n      Vector store is empty.",
                flush=True,
            )

            print(
                "      Indexing documents...",
                flush=True,
            )

            added = (
                vector_store.add_documents(
                    documents
                )
            )

            print(
                f"      Added: {added}",
                flush=True,
            )

            final_count = (
                vector_store.count()
            )

            if (
                final_count
                != expected_count
            ):

                raise RuntimeError(
                    "Vector count mismatch after indexing. "
                    f"Expected {expected_count}, "
                    f"got {final_count}."
                )

        # ----------------------------------------------------
        # Mismatch
        # ----------------------------------------------------

        else:

            raise RuntimeError(
                "\n"
                "Chroma index count does not match "
                "chunks.json.\n"
                f"Existing vectors : {existing_count}\n"
                f"Expected vectors : {expected_count}\n\n"
                "The script intentionally refuses to append "
                "blindly because that could create duplicate "
                "vectors.\n"
                "Since the existing Chroma index was already "
                "validated earlier in this project, do NOT "
                "delete it automatically."
            )

        # ----------------------------------------------------
        # 4. Final validation
        # ----------------------------------------------------

        print(
            "\n[4/4] Final index validation...",
            flush=True,
        )

        final_count = (
            vector_store.count()
        )

        print(
            f"      Final vectors: "
            f"{final_count}",
            flush=True,
        )

        if (
            final_count
            != expected_count
        ):

            raise RuntimeError(
                "Final vector count mismatch. "
                f"Expected {expected_count}, "
                f"got {final_count}."
            )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "VECTOR INDEX BUILD COMPLETED SUCCESSFULLY",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        print(
            f"Chunk records     : {len(data)}",
            flush=True,
        )

        print(
            f"Valid documents   : {expected_count}",
            flush=True,
        )

        print(
            f"Vector count      : {final_count}",
            flush=True,
        )

        print(
            f"Vector DB         : {CHROMA_DIR}",
            flush=True,
        )

        print(
            "\nFirst document:",
            flush=True,
        )

        print(
            documents[0].page_content[:500],
            flush=True,
        )

        print(
            "\nFirst metadata:",
            flush=True,
        )

        print(
            documents[0].metadata,
            flush=True,
        )

        print(
            "\n" + "=" * 70,
            flush=True,
        )

    except Exception as exc:

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "INDEX BUILD FAILED",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        print(
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        traceback.print_exc()

        raise


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()