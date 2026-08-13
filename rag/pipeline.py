from rag.hybrid_retriever import HybridRetriever
from llm.generator import FinancialLLM



class FinRAGPipeline:

    def __init__(self):

        print("Initializing FinRAG AI...")

        self.retriever = HybridRetriever()
        self.llm = FinancialLLM()

    def build_context(self, results):

        context_parts = []

        for i, result in enumerate(
            results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            company = metadata.get(
                "company",
                "Unknown",
            )

            year = metadata.get(
                "year",
                "Unknown",
            )

            filename = metadata.get(
                "filename",
                "Unknown",
            )

            chunk_id = metadata.get(
                "chunk_id",
                "Unknown",
            )

            text = result.get(
                "text",
                "",
            )

            context_parts.append(
                f"""
SOURCE {i}
Company: {company}
Fiscal Year: {year}
Document: {filename}
Chunk: {chunk_id}

{text}
"""
            )

        return "\n".join(context_parts)

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        results = self.retriever.search(
            question,
            top_k=top_k,
        )

        context = self.build_context(
            results
        )

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        sources = []

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            sources.append(
                {
                    "company": metadata.get(
                        "company"
                    ),
                    "year": metadata.get(
                        "year"
                    ),
                    "filename": metadata.get(
                        "filename"
                    ),
                    "chunk_id": metadata.get(
                        "chunk_id"
                    ),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }
