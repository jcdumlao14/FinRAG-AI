from rag.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

query = "What was Microsoft net income in fiscal year 2025?"

results = retriever.search(
    query,
    top_k=20,
    candidate_k=20,
)

print("\n" + "=" * 70)
print("HYBRID MICROSOFT NET INCOME TEST")
print("=" * 70)

for i, result in enumerate(results, start=1):

    metadata = result.get("metadata", {})

    print(
        f"Rank {i}: "
        f"Chunk {metadata.get('chunk_id')} | "
        f"{metadata.get('filename')} | "
        f"RRF Score {result.get('rrf_score', 0):.6f} | "
        f"Contains 101,832: "
        f"{'101,832' in result.get('text', '')}"
    )
