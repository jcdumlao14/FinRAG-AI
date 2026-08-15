import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"


def tokenize(text):
    """Convert text into simple lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    """Load document chunks from chunks.json."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25():
    """Build the BM25 index."""
    chunks = load_chunks()

    documents = [chunk["text"] for chunk in chunks]
    tokenized_documents = [tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25, chunks


# Build the BM25 index once when the module is imported.
# Subsequent searches reuse the in-memory index instead of
# rebuilding it for every query.
BM25_INDEX, BM25_CHUNKS = build_bm25()


def search(query, top_k=5):
    """Search the pre-built BM25 index."""
    query_tokens = tokenize(query)

    scores = BM25_INDEX.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        chunk = BM25_CHUNKS[index].copy()
        chunk["bm25_score"] = float(scores[index])
        results.append(chunk)

    return results


if __name__ == "__main__":
    query = "What was Apple's total net sales in fiscal year 2025?"

    print("=" * 60)
    print("FinRAG AI - BM25 RETRIEVAL")
    print("=" * 60)

    results = search(query, top_k=5)

    for i, result in enumerate(results, start=1):
        print(f"\nResult #{i}")
        print(f"Company: {result.get('company')}")
        print(f"Year: {result.get('year')}")
        print(f"Document: {result.get('filename')}")
        print(f"Chunk: {result.get('chunk_id')}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print("\nText:")
        print(result["text"][:1000])
