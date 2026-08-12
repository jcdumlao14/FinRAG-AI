from rag.retriever import FinancialRetriever
from rag.bm25_retriever import search as bm25_search


class HybridRetriever:

    def __init__(self):
        print("Initializing Hybrid Retriever...")

        self.vector_retriever = FinancialRetriever()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        """
        Hybrid retrieval using:
        - Vector semantic search
        - BM25 lexical search
        - Reciprocal Rank Fusion (RRF)
        """

        vector_results = self.vector_retriever.search(
            query,
            top_k=candidate_k,
        )

        bm25_results = bm25_search(
            query,
            top_k=candidate_k,
        )

        fused = {}

        # --------------------------------------------------
        # VECTOR RESULTS
        # --------------------------------------------------

        for rank, result in enumerate(
            vector_results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            key = (
                str(metadata.get("filename")),
                str(metadata.get("chunk_id")),
            )

            if key not in fused:
                fused[key] = {
                    "result": result,
                    "rrf_score": 0.0,
                }

            fused[key]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )

        # --------------------------------------------------
        # BM25 RESULTS
        # --------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            key = (
                str(result.get("filename")),
                str(result.get("chunk_id")),
            )

            if key not in fused:
                fused[key] = {
                    "result": result,
                    "rrf_score": 0.0,
                }

            fused[key]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )

        # --------------------------------------------------
        # SORT FUSED RESULTS
        # --------------------------------------------------

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        results = []

        for item in ranked[:top_k]:

            result = item["result"].copy()

            # Normalize BM25 results to the same
            # metadata structure used by Vector Search.
            if "metadata" not in result:

                result["metadata"] = {
                    "company": result.get("company"),
                    "year": result.get("year"),
                    "filename": result.get("filename"),
                    "chunk_id": str(
                        result.get("chunk_id")
                    ),
                }

            result["rrf_score"] = item[
                "rrf_score"
            ]

            results.append(result)

        return results
