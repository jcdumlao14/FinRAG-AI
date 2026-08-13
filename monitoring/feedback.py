import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEEDBACK_FILE = (
    PROJECT_ROOT
    / "monitoring"
    / "feedback.json"
)


def save_feedback(
    question,
    answer,
    feedback,
    sources=None,
):
    """
    Save user feedback for a FinRAG AI response.
    """

    if sources is None:
        sources = []

    record = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "sources": sources,
    }

    records = []

    if FEEDBACK_FILE.exists():

        try:

            with open(
                FEEDBACK_FILE,
                "r",
                encoding="utf-8",
            ) as f:

                records = json.load(f)

        except (
            json.JSONDecodeError,
            OSError,
        ):

            records = []

    records.append(record)

    with open(
        FEEDBACK_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            records,
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_feedback():
    """
    Load all stored feedback records.
    """

    if not FEEDBACK_FILE.exists():

        return []

    try:

        with open(
            FEEDBACK_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except (
        json.JSONDecodeError,
        OSError,
    ):

        return []
