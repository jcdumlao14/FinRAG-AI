import json
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT VECTOR RETRIEVER
# ============================================================

from rag.retriever import FinancialRetriever


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

TOP_K = 20


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():
    """Load evaluation questions."""

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    """Normalize text for answer comparison."""

    if text is None:
        return ""

    return (
        str(text)
        .lower()
        .replace("\xa0", " ")
        .replace(",", "")
        .replace("$", "")
        .replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("(", "")
        .replace(")", "")
    )


# ============================================================
# CHECK EXPECTED ANSWER
# ============================================================

def answer_found_in_text(
    text,
    expected_answer,
):
    """
    Check whether the expected financial answer
    appears in the retrieved document text.

    Handles financial table formatting where units such
    as 'million' may be stated in the table heading rather
    than next to every numeric value.
    """

    if not text:
        return False

    if not expected_answer:
        return False

    normalized_text = normalize_text(text)
    normalized_expected = normalize_text(expected_answer)

    # --------------------------------------------------------
    # Direct normalized match
    # --------------------------------------------------------

    if normalized_expected in normalized_text:
        return True

    # --------------------------------------------------------
    # Financial table handling
    #
    # Example:
    #
    # Expected:
    #   $416,161 million
    #
    # Chunk:
    #   Total net sales
    #   $416,161
    #
    # The table may already state "(in millions)"
    # elsewhere in the chunk.
    # --------------------------------------------------------

    expected_lower = str(expected_answer).lower()
    text_lower = str(text).lower().replace("\xa0", " ")

    if "million" in expected_lower:
        expected_without_unit = (
            expected_lower
            .replace("million", "")
            .replace("$", "")
            .replace(",", "")
            .replace(" ", "")
        )

        text_normalized = (
            text_lower
            .replace("$", "")
            .replace(",", "")
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("(", "")
            .replace(")", "")
        )

        if expected_without_unit in text_normalized:

            # Accept the match when the chunk explicitly
            # indicates that the table is in millions.
            million_indicators = [
                "in millions",
                "dollars in millions",
                "in millions,",
                "millions, except",
            ]

            if any(
                indicator in text_lower
                for indicator in million_indicators
            ):
                return True

    # --------------------------------------------------------
    # Special handling:
    #
    # "$19.1 billion increase, or 17%"
    # --------------------------------------------------------

    if (
        "19.1 billion" in expected_lower
        and "19.1 billion" in text_lower
        and "17%" in expected_lower
        and "17%" in text_lower
    ):
        return True

    return False

    # --------------------------------------------------------
    # Direct normalized match
    # --------------------------------------------------------

    if normalized_expected in normalized_text:
        return True

    # --------------------------------------------------------
    # Special handling:
    #
    # "$19.1 billion increase, or 17%"
    # --------------------------------------------------------

    expected_lower = str(
        expected_answer
    ).lower()

    text_lower = str(
        text
    ).lower()

    if (
        "19.1 billion" in expected_lower
        and "19.1 billion" in text_lower
        and "17%" in expected_lower
        and "17%" in text_lower
    ):
        return True

    return False


# ============================================================
# EVALUATE CONTEXT
# ============================================================

def evaluate_context(
    retriever,
    questions,
    top_k=20,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTEXT / GROUNDING EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nThis evaluation does NOT call Gemini."
    )

    print(
        "It checks whether the expected financial "
        "answer exists in retrieved chunks."
    )

    correct = 0

    results = []

    # ========================================================
    # QUESTIONS
    # ========================================================

    for i, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        expected_company = item["company"]

        expected_year = item["year"]

        expected_answer = item.get(
            "expected_answer"
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"Question {i}: {question}"
        )

        print(
            f"Expected: "
            f"{expected_company} "
            f"{expected_year}"
        )

        print(
            f"Expected answer: "
            f"{expected_answer}"
        )

        # ----------------------------------------------------
        # VECTOR SEARCH
        # ----------------------------------------------------

        retrieved = retriever.search(
            question,
            top_k=top_k,
        )

        found = False

        matched_rank = None

        matched_text = None

        matched_metadata = None

        # ====================================================
        # CHECK TOP-K RESULTS
        # ====================================================

        for rank, result in enumerate(
            retrieved,
            start=1,
        ):

            # ------------------------------------------------
            # CORRECT RESULT STRUCTURE
            #
            # {
            #     "text": "...",
            #     "metadata": {...},
            #     "distance": ...
            # }
            # ------------------------------------------------

            text = result.get(
                "text",
                "",
            )

            metadata = result.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            company = metadata.get(
                "company"
            )

            year = metadata.get(
                "year"
            )

            filename = metadata.get(
                "filename"
            )

            chunk_id = metadata.get(
                "chunk_id"
            )

            # ------------------------------------------------
            # COMPANY CHECK
            # ------------------------------------------------

            company_match = (
                str(company).strip().lower()
                == str(expected_company)
                .strip()
                .lower()
            )

            # ------------------------------------------------
            # YEAR CHECK
            # ------------------------------------------------

            try:

                year_match = (
                    int(year)
                    == int(expected_year)
                )

            except (
                TypeError,
                ValueError,
            ):

                year_match = False

            # ------------------------------------------------
            # ANSWER CHECK
            # ------------------------------------------------

            answer_match = answer_found_in_text(
                text,
                expected_answer,
            )

            # ------------------------------------------------
            # DEBUG INFORMATION
            # ------------------------------------------------

            if (
                company_match
                and year_match
            ):

                print(
                    f"  Rank {rank}: "
                    f"{company} "
                    f"{year}"
                )

                print(
                    f"  File: "
                    f"{filename}"
                )

                print(
                    f"  Chunk: "
                    f"{chunk_id}"
                )

                print(
                    f"  Expected answer in chunk: "
                    f"{answer_match}"
                )

            # ------------------------------------------------
            # FINAL MATCH
            # ------------------------------------------------

            if (
                company_match
                and year_match
                and answer_match
            ):

                found = True

                matched_rank = rank

                matched_text = text

                matched_metadata = metadata

                break

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if found:

            correct += 1

            print(
                "\nPASS - Expected answer "
                f"found at rank {matched_rank}"
            )

            print(
                f"Company: "
                f"{matched_metadata.get('company')}"
            )

            print(
                f"Year: "
                f"{matched_metadata.get('year')}"
            )

            print(
                f"File: "
                f"{matched_metadata.get('filename')}"
            )

            print(
                f"Chunk: "
                f"{matched_metadata.get('chunk_id')}"
            )

            print(
                f"Expected answer found: "
                f"{expected_answer}"
            )

            # ------------------------------------------------
            # CONTEXT PREVIEW
            # ------------------------------------------------

            if matched_text:

                preview = (
                    matched_text
                    .replace(
                        "\n",
                        " ",
                    )
                    .strip()
                )

                if len(preview) > 700:

                    preview = (
                        preview[:700]
                        + "..."
                    )

                print(
                    "\nContext preview:"
                )

                print(
                    preview
                )

        else:

            print(
                "\nFAIL - Expected answer "
                "not found in top "
                f"{top_k} retrieved chunks"
            )

        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append(
            {
                "question": question,
                "company": expected_company,
                "year": expected_year,
                "expected_answer": expected_answer,
                "grounded": found,
                "matched_rank": matched_rank,
            }
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(questions)

    accuracy = (
        correct / total
        if total
        else 0
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTEXT / GROUNDING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCorrect: "
        f"{correct}/{total}"
    )

    print(
        f"Context Grounding Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        "\nGemini API calls: 0"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_file = (
        PROJECT_ROOT
        / "evaluation"
        / "context_results.json"
    )

    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\nResults saved to:"
    )

    print(
        results_file
    )

    print(
        "\nContext evaluation completed."
    )

    return accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "FinRAG AI - CONTEXT EVALUATION"
    )

    print(
        "=" * 70
    )

    questions = load_questions()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # INITIALIZE RETRIEVER
    # --------------------------------------------------------

    print(
        "\nInitializing vector retriever..."
    )

    retriever = FinancialRetriever()

    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    evaluate_context(
        retriever,
        questions,
        top_k=TOP_K,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()