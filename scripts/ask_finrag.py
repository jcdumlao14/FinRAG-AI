import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import FinRAGPipeline


def main():

    print("=" * 70)
    print("FinRAG AI - FINANCIAL RESEARCH ASSISTANT")
    print("=" * 70)

    pipeline = FinRAGPipeline()

    question = input(
        "\nAsk a financial question: "
    )

    result = pipeline.answer(
        question,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for i, source in enumerate(
        result["sources"],
        start=1,
    ):

        print(
            f"{i}. "
            f"{source['company']} "
            f"({source['year']}) - "
            f"{source['filename']} "
            f"[Chunk {source['chunk_id']}]"
        )


if __name__ == "__main__":
    main()