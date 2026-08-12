import json
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORT RETRIEVERS
# ============================================================

from rag.retriever import FinancialRetriever
from rag.bm25_retriever import search as bm25_search
from rag.hybrid_retriever import HybridRetriever

# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

TOP_K_VALUES = [1, 3, 5]


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():
    """Load evaluation questions from JSON."""

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# EXTRACT COMPANY AND YEAR
# ============================================================

def get_company_and_year(
    result,
    retriever_name,
):
    """Extract company and fiscal year from retrieval result."""

    # Vector results store metadata inside "metadata".
    metadata = result.get("metadata")

    if metadata:
        return (
            metadata.get("company"),
            metadata.get("year"),
        )

    # BM25 results store metadata at the top level.
    return (
        result.get("company"),
        result.get("year"),
    )


# ============================================================
# CHECK EXPECTED DOCUMENT
# ============================================================

def matches_expected(
    result,
    expected_company,
    expected_year,
    retriever_name,
):
    """Check whether a result matches expected company/year."""

    company, year = get_company_and_year(
        result,
        retriever_name,
    )

    if company != expected_company:
        return False

    try:
        return int(year) == int(expected_year)

    except (TypeError, ValueError):
        return False


# ============================================================
# EVALUATE VECTOR SEARCH
# ============================================================

def evaluate_vector_search(
    retriever,
    questions,
    top_k=5,
):
    """Evaluate semantic/vector retrieval."""

    print("\n" + "=" * 70)
    print("VECTOR SEARCH EVALUATION")
    print("=" * 70)

    accuracy_by_k = {}

    for k in TOP_K_VALUES:

        correct = 0

        print(
            f"\n--- Vector Search Top-{k} ---"
        )

        for i, item in enumerate(
            questions,
            start=1,
        ):

            question = item["question"]

            expected_company = item["company"]

            expected_year = item["year"]

            results = retriever.search(
                question,
                top_k=top_k,
            )

            found = False
            found_rank = None

            for rank, result in enumerate(
                results[:k],
                start=1,
            ):

                if matches_expected(
                    result,
                    expected_company,
                    expected_year,
                    "Vector Search",
                ):

                    found = True
                    found_rank = rank
                    break

            if found:

                correct += 1

                print(
                    f"Question {i}: PASS "
                    f"(rank {found_rank})"
                )

            else:

                print(
                    f"Question {i}: FAIL"
                )

        accuracy = (
            correct / len(questions)
            if questions
            else 0
        )

        accuracy_by_k[k] = accuracy

        print(
            f"\nVector Search Top-{k} Accuracy: "
            f"{accuracy:.2%}"
        )

        print(
            f"Correct: "
            f"{correct}/{len(questions)}"
        )

    return accuracy_by_k


# ============================================================
# EVALUATE BM25
# ============================================================

def evaluate_bm25(
    questions,
    top_k=5,
):
    """Evaluate BM25 retrieval."""

    print("\n" + "=" * 70)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 70)

    accuracy_by_k = {}

    for k in TOP_K_VALUES:

        correct = 0

        print(
            f"\n--- BM25 Top-{k} ---"
        )

        for i, item in enumerate(
            questions,
            start=1,
        ):

            question = item["question"]

            expected_company = item["company"]

            expected_year = item["year"]

            results = bm25_search(
                question,
                top_k=top_k,
            )

            found = False
            found_rank = None

            for rank, result in enumerate(
                results[:k],
                start=1,
            ):

                if matches_expected(
                    result,
                    expected_company,
                    expected_year,
                    "BM25",
                ):

                    found = True
                    found_rank = rank
                    break

            if found:

                correct += 1

                print(
                    f"Question {i}: PASS "
                    f"(rank {found_rank})"
                )

            else:

                print(
                    f"Question {i}: FAIL"
                )

        accuracy = (
            correct / len(questions)
            if questions
            else 0
        )

        accuracy_by_k[k] = accuracy

        print(
            f"\nBM25 Top-{k} Accuracy: "
            f"{accuracy:.2%}"
        )

        print(
            f"Correct: "
            f"{correct}/{len(questions)}"
        )

    return accuracy_by_k

# ============================================================
# EVALUATE HYBRID RRF
# ============================================================

def evaluate_hybrid(
    retriever,
    questions,
    top_k=5,
):
    """Evaluate hybrid Vector + BM25 retrieval using RRF."""

    print("\n" + "=" * 70)
    print("HYBRID RRF RETRIEVAL EVALUATION")
    print("=" * 70)

    accuracy_by_k = {}

    for k in TOP_K_VALUES:

        correct = 0

        print(
            f"\n--- Hybrid RRF Top-{k} ---"
        )

        for i, item in enumerate(
            questions,
            start=1,
        ):

            question = item["question"]

            expected_company = item["company"]

            expected_year = item["year"]

            results = retriever.search(
                question,
                top_k=top_k,
            )

            found = False
            found_rank = None

            for rank, result in enumerate(
                results[:k],
                start=1,
            ):

                if matches_expected(
                    result,
                    expected_company,
                    expected_year,
                    "Hybrid RRF",
                ):

                    found = True
                    found_rank = rank
                    break

            if found:

                correct += 1

                print(
                    f"Question {i}: PASS "
                    f"(rank {found_rank})"
                )

            else:

                print(
                    f"Question {i}: FAIL"
                )

        accuracy = (
            correct / len(questions)
            if questions
            else 0
        )

        accuracy_by_k[k] = accuracy

        print(
            f"\nHybrid RRF Top-{k} Accuracy: "
            f"{accuracy:.2%}"
        )

        print(
            f"Correct: "
            f"{correct}/{len(questions)}"
        )

    return accuracy_by_k

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinRAG AI - RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # INITIALIZE VECTOR RETRIEVER
    # --------------------------------------------------------

    print(
        "\nInitializing vector retriever..."
    )

    vector_retriever = FinancialRetriever()

    print(
        "\nInitializing hybrid retriever..."
    )

    hybrid_retriever = HybridRetriever()

    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    vector_results = evaluate_vector_search(
        vector_retriever,
        questions,
        top_k=max(TOP_K_VALUES),
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = evaluate_bm25(
        questions,
        top_k=max(TOP_K_VALUES),
    )

    # --------------------------------------------------------
    # HYBRID RRF
    # --------------------------------------------------------

    hybrid_results = evaluate_hybrid(
        hybrid_retriever,
        questions,
        top_k=max(TOP_K_VALUES),
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 70)

    print(
        "\nRetrieval Accuracy:"
    )

    print(
        "\n                    Top-1       Top-3       Top-5"
    )

    print(
        "-" * 70
    )

    print(
        "Vector Search       "
        f"{vector_results[1]:.2%}      "
        f"{vector_results[3]:.2%}      "
        f"{vector_results[5]:.2%}"
    )

    print(
        "BM25                "
        f"{bm25_results[1]:.2%}      "
        f"{bm25_results[3]:.2%}      "
        f"{bm25_results[5]:.2%}"
    )

    print(
        "Hybrid RRF          "
        f"{hybrid_results[1]:.2%}      "
        f"{hybrid_results[3]:.2%}      "
        f"{hybrid_results[5]:.2%}"
    )

    print(
        "\n" + "-" * 70
    )

    # --------------------------------------------------------
    # DETERMINE BEST RETRIEVER
    # --------------------------------------------------------

    vector_top5 = vector_results[5]

    bm25_top5 = bm25_results[5]

    hybrid_top5 = hybrid_results[5]

    retriever_scores = {
        "Vector Search": vector_top5,
        "BM25": bm25_top5,
        "Hybrid RRF": hybrid_top5,
    }

    best_retriever = max(
        retriever_scores,
        key=retriever_scores.get,
    )

    best_score = retriever_scores[
        best_retriever
    ]

    print(
    f"BEST RETRIEVER: {best_retriever}"
)

    print(
       f"Best Top-5 Accuracy: {best_score:.2%}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\nEvaluation completed successfully."
    )
    # ============================================================
    # ENTRY POINT
    # ============================================================

if __name__ == "__main__":
    main()