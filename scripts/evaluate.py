from pathlib import Path
import json
import sys
import traceback
import argparse


# ============================================================
# Project root
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run SickleGuide retrieval "
            "and optional end-to-end evaluation."
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run full end-to-end evaluation "
            "including Qwen generation."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "evaluation_report.json"
        ),
        help=(
            "Output JSON report path."
        ),
    )

    args = parser.parse_args()

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "SickleGuide - Evaluation Runner",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    try:

        from src.evaluation.evaluate import (
            run_evaluation,
        )

        report = run_evaluation(
            run_end_to_end=args.full
        )

        output_path = (
            ROOT_DIR
            / args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "EVALUATION COMPLETED",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        retrieval_summary = report[
            "retrieval"
        ][
            "summary"
        ]

        print(
            "\nRetrieval:",
            flush=True,
        )

        print(
            f"Recall@5          : "
            f"{retrieval_summary['candidate_recall@5']:.3f}",
            flush=True,
        )

        print(
            f"Recall@10         : "
            f"{retrieval_summary['candidate_recall@10']:.3f}",
            flush=True,
        )

        print(
            f"Recall@20         : "
            f"{retrieval_summary['candidate_recall@20']:.3f}",
            flush=True,
        )

        print(
            f"Reranked Recall@5 : "
            f"{retrieval_summary['reranked_recall@5']:.3f}",
            flush=True,
        )

        print(
            f"MRR               : "
            f"{retrieval_summary['mrr']:.3f}",
            flush=True,
        )

        if report.get(
            "end_to_end"
        ):

            e2e_summary = report[
                "end_to_end"
            ][
                "summary"
            ]

            print(
                "\nEnd-to-End:",
                flush=True,
            )

            print(
                f"Grounded rate     : "
                f"{e2e_summary['grounded_rate']:.3f}",
                flush=True,
            )

            print(
                f"Citation validity : "
                f"{e2e_summary['citation_valid_rate']:.3f}",
                flush=True,
            )

            print(
                f"Answer coverage   : "
                f"{e2e_summary['answer_term_coverage']:.3f}",
                flush=True,
            )

        print(
            "\nReport:",
            flush=True,
        )

        print(
            output_path,
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
            "EVALUATION FAILED",
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


if __name__ == "__main__":
    main()