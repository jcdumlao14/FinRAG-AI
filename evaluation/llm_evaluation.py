import sys
import json
import time
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT FINRAG PIPELINE
# ============================================================

from rag.pipeline import FinRAGPipeline


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

TOP_K = 5

# Maximum number of retries for temporary
# per-minute rate-limit errors.
MAX_RETRIES = 3

# Delay between temporary rate-limit retries.
RETRY_DELAY_SECONDS = 45


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# CHECK ANSWER
# ============================================================

def answer_matches_expected(
    answer,
    expected_answer,
):

    if not answer:
        return False

    answer_text = str(answer).lower()
    expected_text = str(
        expected_answer
    ).lower()

    # --------------------------------------------------------
    # Direct match
    # --------------------------------------------------------

    if expected_text in answer_text:
        return True

    # --------------------------------------------------------
    # Remove common formatting differences
    # --------------------------------------------------------

    normalized_answer = (
        answer_text
        .replace(",", "")
        .replace("$", "")
        .replace(" ", "")
    )

    normalized_expected = (
        expected_text
        .replace(",", "")
        .replace("$", "")
        .replace(" ", "")
    )

    if normalized_expected in normalized_answer:
        return True

    # --------------------------------------------------------
    # Special handling for billion/million wording
    # --------------------------------------------------------

    if (
        "$19.1 billion" in expected_text
        and "19.1 billion" in answer_text
    ):
        return True

    # --------------------------------------------------------
    # Special handling for percentage
    # --------------------------------------------------------

    if (
        "17%" in expected_text
        and "17%" in answer_text
        and "increase" in answer_text
    ):
        return True

    return False


# ============================================================
# CHECK IF ERROR IS GEMINI DAILY QUOTA
# ============================================================

def is_daily_quota_error(error):

    error_text = str(error).lower()

    return (
        "generaterequestsperday" in error_text
        or "generate_requests_per_day" in error_text
        or "perday" in error_text
        or (
            "quota exceeded" in error_text
            and "per-day" in error_text
        )
    )


# ============================================================
# CHECK IF ERROR IS TEMPORARY RATE LIMIT
# ============================================================

def is_rate_limit_error(error):

    error_text = str(error).lower()

    return (
        "429" in error_text
        or "resource_exhausted" in error_text
        or "quota exceeded" in error_text
        or "rate limit" in error_text
    )


# ============================================================
# RUN ONE QUESTION WITH RETRIES
# ============================================================

def run_question(
    pipeline,
    question,
):

    total_attempts = MAX_RETRIES + 1

    for attempt in range(
        1,
        total_attempts + 1,
    ):

        try:

            result = pipeline.answer(
                question=question,
                top_k=TOP_K,
            )

            return (
                result,
                None,
                attempt,
            )

        except Exception as error:

            # ------------------------------------------------
            # DAILY QUOTA ERROR
            #
            # Do NOT retry.
            # The daily quota cannot be fixed by waiting
            # 45 seconds.
            # ------------------------------------------------

            if is_daily_quota_error(error):

                print(
                    "\nGemini DAILY quota exhausted."
                )

                print(
                    "Stopping LLM evaluation "
                    "to avoid unnecessary retries."
                )

                return (
                    None,
                    str(error),
                    attempt,
                )

            # ------------------------------------------------
            # TEMPORARY RATE LIMIT ERROR
            #
            # Retry because this can be a per-minute limit.
            # ------------------------------------------------

            if is_rate_limit_error(error):

                if attempt >= total_attempts:

                    print(
                        "\nGemini rate limit reached "
                        "after maximum retries."
                    )

                    return (
                        None,
                        str(error),
                        attempt,
                    )

                print(
                    f"ERROR on attempt "
                    f"{attempt}/{total_attempts}: "
                    f"Gemini rate limit reached."
                )

                print(
                    f"Waiting "
                    f"{RETRY_DELAY_SECONDS} seconds "
                    "before retry..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            else:

                print(
                    f"ERROR: {error}"
                )

                return (
                    None,
                    str(error),
                    attempt,
                )

    return (
        None,
        "Unknown error",
        total_attempts,
    )


# ============================================================
# EVALUATE PIPELINE
# ============================================================

def evaluate_pipeline(
    pipeline,
    questions,
):

    results = []

    grounded_count = 0

    answer_match_count = 0

    rate_limit_errors = 0

    other_errors = 0

    total_retries = 0

    stopped_by_daily_quota = False

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
            f"\nQuestion {i}: {question}"
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
        # RUN QUESTION
        # ----------------------------------------------------

        result, error, attempts = run_question(
            pipeline,
            question,
        )

        total_retries += max(
            0,
            attempts - 1,
        )

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        if error is not None:

            if is_daily_quota_error(error):

                rate_limit_errors += 1

            elif is_rate_limit_error(error):

                rate_limit_errors += 1

            else:

                other_errors += 1

            print(
                f"FAILED: {error}"
            )

            results.append(
                {
                    "question": question,
                    "company": expected_company,
                    "year": expected_year,
                    "expected_answer": expected_answer,
                    "grounded": False,
                    "answer_match": False,
                    "answer": None,
                    "sources": [],
                    "error": error,
                }
            )

            # ------------------------------------------------
            # STOP ENTIRE EVALUATION IF DAILY QUOTA IS HIT
            # ------------------------------------------------

            if is_daily_quota_error(error):

                stopped_by_daily_quota = True

                print(
                    "\nDaily Gemini quota reached."
                )

                print(
                    "Stopping the remaining "
                    "LLM evaluation questions."
                )

                break

            continue

        # ----------------------------------------------------
        # ANSWER + SOURCES
        # ----------------------------------------------------

        answer = result.get(
            "answer",
            "",
        )

        sources = result.get(
            "sources",
            [],
        )

        # ----------------------------------------------------
        # GROUNDING CHECK
        # ----------------------------------------------------

        company_found = any(
            source.get("company")
            == expected_company
            for source in sources
        )

        year_found = any(
            str(source.get("year"))
            == str(expected_year)
            for source in sources
        )

        grounded = (
            company_found
            and year_found
        )

        # ----------------------------------------------------
        # EXPECTED ANSWER CHECK
        # ----------------------------------------------------

        answer_match = answer_matches_expected(
            answer,
            expected_answer,
        )

        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        if grounded:

            grounded_count += 1

        if answer_match:

            answer_match_count += 1

        # ----------------------------------------------------
        # DISPLAY GROUNDING RESULT
        # ----------------------------------------------------

        if grounded:

            print(
                "PASS - Expected "
                "company/year found in sources"
            )

        else:

            print(
                "FAIL - Expected "
                "company/year not found in sources"
            )

        # ----------------------------------------------------
        # DISPLAY ANSWER RESULT
        # ----------------------------------------------------

        if answer_match:

            print(
                "PASS - Expected answer "
                "found in generated response"
            )

        else:

            print(
                "FAIL - Expected answer "
                "not found in generated response"
            )

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        print(
            f"Answer: {answer}"
        )

        # ----------------------------------------------------
        # DISPLAY SOURCES
        # ----------------------------------------------------

        print(
            "Sources:"
        )

        for source_number, source in enumerate(
            sources,
            start=1,
        ):

            print(
                f"  {source_number}. "
                f"{source.get('company')} "
                f"({source.get('year')}) - "
                f"{source.get('filename')} "
                f"[Chunk {source.get('chunk_id')}]"
            )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "question": question,
                "company": expected_company,
                "year": expected_year,
                "expected_answer": expected_answer,
                "grounded": grounded,
                "answer_match": answer_match,
                "answer": answer,
                "sources": sources,
                "error": None,
            }
        )

    return (
        results,
        grounded_count,
        answer_match_count,
        rate_limit_errors,
        other_errors,
        total_retries,
        stopped_by_daily_quota,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "FinRAG AI - LLM EVALUATION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # INITIALIZE PIPELINE
    # --------------------------------------------------------

    pipeline = FinRAGPipeline()

    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    (
        results,
        grounded_count,
        answer_match_count,
        rate_limit_errors,
        other_errors,
        total_retries,
        stopped_by_daily_quota,
    ) = evaluate_pipeline(
        pipeline,
        questions,
    )

    # --------------------------------------------------------
    # CALCULATE SCORES
    # --------------------------------------------------------

    total_questions = len(
        questions
    )

    evaluated_questions = len(
        results
    )

    grounded_score = (
        grounded_count / evaluated_questions
        if evaluated_questions
        else 0
    )

    answer_score = (
        answer_match_count / evaluated_questions
        if evaluated_questions
        else 0
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = (
        PROJECT_ROOT
        / "evaluation"
        / "llm_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"\nQuestions in evaluation set: "
        f"{total_questions}"
    )

    print(
        f"Questions evaluated: "
        f"{evaluated_questions}"
    )

    print(
        f"\nGrounded: "
        f"{grounded_count}/"
        f"{evaluated_questions}"
    )

    print(
        f"Grounding Score: "
        f"{grounded_score:.2%}"
    )

    print(
        f"\nCorrect Answers: "
        f"{answer_match_count}/"
        f"{evaluated_questions}"
    )

    print(
        f"Answer Score: "
        f"{answer_score:.2%}"
    )

    print(
        f"\nRate-limit errors: "
        f"{rate_limit_errors}"
    )

    print(
        f"Other errors: "
        f"{other_errors}"
    )

    print(
        f"Total retries: "
        f"{total_retries}"
    )

    if stopped_by_daily_quota:

        print(
            "\nSTATUS: "
            "Stopped because Gemini daily "
            "quota was exhausted."
        )

    else:

        print(
            "\nSTATUS: "
            "Evaluation completed."
        )

    # --------------------------------------------------------
    # RESULTS FILE
    # --------------------------------------------------------

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )

    print(
        "\nFinRAG LLM evaluation completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

