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
        - Financial evidence boosting
        - Company diversity for broad financial questions
        """

        # --------------------------------------------------
        # QUERY EXPANSION
        # --------------------------------------------------

        retrieval_query = query

        query_lower = query.lower()

        broad_financial_question = any(
            phrase in query_lower
            for phrase in [
                "perform financially",
                "financial performance",
                "how did the companies perform",
                "how did the company perform",
                "financially during",
            ]
        )

        if broad_financial_question:
            retrieval_query = (
                f"{query} "
                "revenue net sales net income "
                "operating income gross profit "
                "gross margin earnings EPS "
                "financial results total revenue"
            )

        # --------------------------------------------------
        # VECTOR SEARCH
        # --------------------------------------------------

        vector_results = self.vector_retriever.search(
            retrieval_query,
            top_k=candidate_k,
        )

        # --------------------------------------------------
        # BM25 SEARCH
        # --------------------------------------------------

        bm25_results = bm25_search(
            retrieval_query,
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

            bm25_weight = 1.5 if broad_financial_question else 1.0

            fused[key]["rrf_score"] += (
                bm25_weight / (rrf_k + rank)
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

            bm25_weight = 1.5 if broad_financial_question else 1.0

            fused[key]["rrf_score"] += (
                bm25_weight / (rrf_k + rank)
            )

        # --------------------------------------------------
        # FINANCIAL EVIDENCE BOOST
        # --------------------------------------------------

        financial_terms = [
            "revenue",
            "net sales",
            "net income",
            "operating income",
            "operating profit",
            "gross profit",
            "gross margin",
            "earnings",
            "earnings per share",
            "eps",
            "financial results",
            "total revenue",
            "total net sales",
        ]

        # Core company-level financial metrics.
        # A chunk containing several of these metrics is
        # stronger evidence for broad financial-performance
        # questions than a chunk containing only generic
        # financial terminology.
        core_financial_metrics = [
            "revenue",
            "net sales",
            "gross margin",
            "operating income",
            "net income",
            "earnings per share",
        ]

        is_financial_question = (
            broad_financial_question
            or any(
                term in query_lower
                for term in financial_terms
            )
        )

        if is_financial_question:

            for item in fused.values():

                text = item["result"].get(
                    "text",
                    "",
                ).lower()

                evidence_matches = sum(
                    1
                    for term in financial_terms
                    if term in text
                )

                core_matches = sum(
                    1
                    for term in core_financial_metrics
                    if term in text
                )

                # Keep the original small terminology boost.
                item["rrf_score"] += (
                    min(evidence_matches, 5)
                    * 0.001
                )

                # Stronger bonus for chunks containing
                # multiple core financial-performance metrics.
                if broad_financial_question:
                    item["rrf_score"] += (
                        min(core_matches, 5)
                        * 0.003
                    )

        # --------------------------------------------------
        # SORT FUSED RESULTS
        # --------------------------------------------------

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # --------------------------------------------------
        # COMPANY DIVERSITY
        # --------------------------------------------------

        if broad_financial_question:

            diverse_results = []
            remaining_results = []

            companies_seen = set()

            # First pass: prefer the strongest result
            # from each company.
            for item in ranked:

                result = item["result"]
                metadata = result.get(
                    "metadata",
                    {},
                )

                company = metadata.get(
                    "company"
                )

                if company and company not in companies_seen:
                    diverse_results.append(item)
                    companies_seen.add(company)

                else:
                    remaining_results.append(item)

            # Fill remaining positions using normal
            # RRF ranking.
            ranked = (
                diverse_results
                + remaining_results
            )

        # --------------------------------------------------
        # BUILD FINAL RESULTS
        # --------------------------------------------------

        results = []

        for item in ranked[:top_k]:

            result = item["result"].copy()

            # Normalize BM25 results to the same
            # metadata structure used by Vector Search.
            if "metadata" not in result:

                result["metadata"] = {
                    "company": result.get(
                        "company"
                    ),
                    "year": result.get(
                        "year"
                    ),
                    "filename": result.get(
                        "filename"
                    ),
                    "chunk_id": str(
                        result.get(
                            "chunk_id"
                        )
                    ),
                }

            result["rrf_score"] = item[
                "rrf_score"
            ]

            results.append(result)

        return results



