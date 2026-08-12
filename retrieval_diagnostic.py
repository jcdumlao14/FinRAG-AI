import json
from pathlib import Path

from rag.retriever import FinancialRetriever

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

print("=" * 80)
print("FinRAG AI - VECTOR RETRIEVAL DIAGNOSTIC")
print("=" * 80)

retriever = FinancialRetriever()

for i, item in enumerate(questions, start=1):

    question = item["question"]
    expected_answer = item["expected_answer"]

    print("\n" + "=" * 80)
    print(f"QUESTION {i}")
    print("=" * 80)

    print(f"Question: {question}")
    print(f"Expected: {expected_answer}")

    results = retriever.search(
        question,
        top_k=5,
    )

    for rank, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})
        text = result.get("text", "")
        distance = result.get("distance")

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"Company: {metadata.get('company')}")
        print(f"Year: {metadata.get('year')}")
        print(f"File: {metadata.get('filename')}")
        print(f"Chunk: {metadata.get('chunk_id')}")
        print(f"Local chunk: {metadata.get('local_chunk_id')}")
        print(f"Distance: {distance}")
        print("-" * 80)
        print(text[:2000])

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETED")
print("=" * 80)
