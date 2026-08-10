import json
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import FinancialRetriever
from rag.bm25_retriever import search as bm25_search


QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"


def load_questions():
    """Load evaluation questions."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_company_and_year(result, retriever_name):
    """Extract company and year from either retriever format."""

    if retriever_name == "Vector Search":
        metadata = result.get("metadata", {})
        return (
            metadata.get("company"),
            metadata.get("year"),
        )

    return (
        result.get("company"),
        result.get("year"),
    )


def evaluate_vector_search(retriever, questions, top_k=5):
    """Evaluate semantic/vector retrieval."""

    print("\n" + "=" * 70)
    print("VECTOR SEARCH EVALUATION")
    print("=" * 70)

    correct = 0

    for i, item in enumerate(questions, start=1):

        question = item["question"]
        expected_company = item["company"]
        expected_year = item["year"]

        print(f"\nQuestion {i}: {question}")
        print(
            f"Expected: {expected_company} "
            f"{expected_year}"
        )

        results = retriever.search(
            question,
            top_k=top_k,
        )

        found = False

        for rank, result in enumerate(
            results,
            start=1,
        ):

            company, year = get_company_and_year(
                result,
                "Vector Search",
            )

            if (
                company == expected_company
                and int(year) == int(expected_year)
            ):
                found = True
                print(
                    f"PASS - Found at rank {rank}"
                )
                break

        if not found:
            print("FAIL - Expected document not found")

        if found:
            correct += 1

    accuracy = correct / len(questions)

    print("\n" + "-" * 70)
    print(
        f"Vector Search Top-{top_k} Accuracy: "
        f"{accuracy:.2%}"
    )
    print(
        f"Correct: {correct}/{len(questions)}"
    )
    print("-" * 70)

    return accuracy


def evaluate_bm25(questions, top_k=5):
    """Evaluate BM25 retrieval."""

    print("\n" + "=" * 70)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 70)

    correct = 0

    for i, item in enumerate(questions, start=1):

        question = item["question"]
        expected_company = item["company"]
        expected_year = item["year"]

        print(f"\nQuestion {i}: {question}")
        print(
            f"Expected: {expected_company} "
            f"{expected_year}"
        )

        results = bm25_search(
            question,
            top_k=top_k,
        )

        found = False

        for rank, result in enumerate(
            results,
            start=1,
        ):

            company, year = get_company_and_year(
                result,
                "BM25",
            )

            if (
                company == expected_company
                and int(year) == int(expected_year)
            ):
                found = True
                print(
                    f"PASS - Found at rank {rank}"
                )
                break

        if not found:
            print("FAIL - Expected document not found")

        if found:
            correct += 1

    accuracy = correct / len(questions)

    print("\n" + "-" * 70)
    print(
        f"BM25 Top-{top_k} Accuracy: "
        f"{accuracy:.2%}"
    )
    print(
        f"Correct: {correct}/{len(questions)}"
    )
    print("-" * 70)

    return accuracy


def main():

    print("=" * 70)
    print("FinRAG AI - RETRIEVAL EVALUATION")
    print("=" * 70)

    questions = load_questions()

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    # Load vector retriever once
    print("\nInitializing vector retriever...")
    vector_retriever = FinancialRetriever()

    # Evaluate both retrieval approaches
    vector_accuracy = evaluate_vector_search(
        vector_retriever,
        questions,
        top_k=5,
    )

    bm25_accuracy = evaluate_bm25(
        questions,
        top_k=5,
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 70)

    print(
        f"Vector Search : {vector_accuracy:.2%}"
    )

    print(
        f"BM25          : {bm25_accuracy:.2%}"
    )

    if vector_accuracy > bm25_accuracy:

        print(
            "\nBEST RETRIEVER: Vector Search"
        )

    elif bm25_accuracy > vector_accuracy:

        print(
            "\nBEST RETRIEVER: BM25"
        )

    else:

        print(
            "\nBEST RETRIEVER: Tie"
        )


if __name__ == "__main__":
    main()