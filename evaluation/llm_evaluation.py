import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import FinRAGPipeline


QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_pipeline(pipeline, questions):
    results = []

    for i, item in enumerate(questions, start=1):

        question = item["question"]
        expected_company = item["company"]
        expected_year = item["year"]

        print(f"\nQuestion {i}: {question}")

        try:
            result = pipeline.answer(
                question=question,
                top_k=5,
            )

            answer = result["answer"]
            sources = result["sources"]

            company_found = any(
                source.get("company") == expected_company
                for source in sources
            )

            year_found = any(
                str(source.get("year")) == str(expected_year)
                for source in sources
            )

            grounded = company_found and year_found

            if grounded:
                print("PASS - Expected company/year found in sources")
            else:
                print("FAIL - Expected company/year not found in sources")

            print(f"Answer: {answer}")

            results.append(
                {
                    "question": question,
                    "company": expected_company,
                    "year": expected_year,
                    "grounded": grounded,
                    "answer": answer,
                    "sources": sources,
                }
            )

        except Exception as e:
            print(f"ERROR: {e}")

            results.append(
                {
                    "question": question,
                    "company": expected_company,
                    "year": expected_year,
                    "grounded": False,
                    "answer": None,
                    "sources": [],
                    "error": str(e),
                }
            )

    return results


def main():

    print("=" * 60)
    print("FinRAG AI - LLM EVALUATION")
    print("=" * 60)

    questions = load_questions()

    print(f"\nEvaluation questions: {len(questions)}")

    pipeline = FinRAGPipeline()

    results = evaluate_pipeline(
        pipeline,
        questions,
    )

    passed = sum(
        result["grounded"]
        for result in results
    )

    total = len(results)

    score = passed / total if total else 0

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nGrounded: {passed}/{total}")
    print(f"Score: {score:.2%}")

    print("\nFinRAG LLM evaluation completed.")


if __name__ == "__main__":
    main()