from rag.retriever import FinancialRetriever
from rag.bm25_retriever import search as bm25_search

query = "What was Microsoft net income in fiscal year 2025?"

print("\n" + "=" * 70)
print("RETRIEVER IDENTITY COMPARISON")
print("=" * 70)

# --------------------------------------------------
# VECTOR
# --------------------------------------------------

vector = FinancialRetriever()

vector_results = vector.search(
    query,
    top_k=20,
)

print("\nVECTOR RESULTS")
print("-" * 70)

for rank, result in enumerate(vector_results, start=1):

    metadata = result.get("metadata", {})

    print(
        f"Rank {rank}: "
        f"filename={metadata.get('filename')!r}, "
        f"chunk_id={metadata.get('chunk_id')!r}, "
        f"company={metadata.get('company')!r}, "
        f"year={metadata.get('year')!r}"
    )

# --------------------------------------------------
# BM25
# --------------------------------------------------

bm25_results = bm25_search(
    query,
    top_k=20,
)

print("\nBM25 RESULTS")
print("-" * 70)

for rank, result in enumerate(bm25_results, start=1):

    print(
        f"Rank {rank}: "
        f"filename={result.get('filename')!r}, "
        f"chunk_id={result.get('chunk_id')!r}, "
        f"company={result.get('company')!r}, "
        f"year={result.get('year')!r}"
    )
