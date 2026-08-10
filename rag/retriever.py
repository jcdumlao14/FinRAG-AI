from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "finrag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FinancialRetriever:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Connecting to ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Collection loaded: {self.collection.name}"
        )

        print(
            f"Documents available: {self.collection.count():,}"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):

        query_embedding = self.model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_documents = []

        for i in range(len(results["documents"][0])):

            retrieved_documents.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved_documents


if __name__ == "__main__":

    retriever = FinancialRetriever()

    query = "What was Apple's net sales in 2025?"
    query = "What were Microsoft's major sources of revenue?"
    query = (
        "How did the companies perform financially "
        "during the reported fiscal year?"
    )
    results = retriever.search(
        query,
        top_k=5,
    )

    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 60)

    print(f"\nQuery: {query}")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        print("\n" + "-" * 60)

        print(f"Result #{rank}")

        print(
            f"Company: {metadata.get('company')}"
        )

        print(
            f"Year: {metadata.get('year')}"
        )

        print(
            f"Document: {metadata.get('filename')}"
        )

        print(
            f"Chunk: {metadata.get('chunk_id')}"
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print("\nText:")

        print(
            result["text"][:1000]
        )