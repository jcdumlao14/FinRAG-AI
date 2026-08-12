from rag.bm25_retriever import search

results = search(
    "What was Microsoft net income in fiscal year 2025?",
    top_k=3,
)

print("\n" + "=" * 70)
print("BM25 RESULT STRUCTURE")
print("=" * 70)

for i, result in enumerate(results, start=1):

    print(f"\nRESULT {i}")
    print("Keys:", list(result.keys()))
    print("Filename:", repr(result.get("filename")))
    print("Chunk ID:", repr(result.get("chunk_id")))
    print("Company:", repr(result.get("company")))
    print("Year:", repr(result.get("year")))
