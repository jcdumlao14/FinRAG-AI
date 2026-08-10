from pathlib import Path
import json

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "embeddings.json"
)

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "finrag_documents"


def main():

    print("=" * 60)
    print("FinRAG AI - CHROMADB VECTOR DATABASE")
    print("=" * 60)

    print(f"\nLoading embeddings from:")
    print(EMBEDDINGS_FILE)

    with EMBEDDINGS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    print(f"Total records: {len(records):,}")

    print("\nInitializing ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Financial 10-K document collection"
        },
    )

    print(f"Collection: {COLLECTION_NAME}")

    # Clear existing records so the script is reproducible.
    existing = collection.count()

    if existing > 0:
        print(
            f"Existing records found: {existing:,}"
        )

        client.delete_collection(
            name=COLLECTION_NAME
        )

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Financial 10-K document collection"
            },
        )

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, record in enumerate(records):

        ids.append(
            f"{record['company']}_{record['chunk_id']}"
        )

        documents.append(
            record["text"]
        )

        embeddings.append(
            record["embedding"]
        )

        metadatas.append(
            {
                "filename": record["filename"],
                "company": record["company"],
                "document_type": record["document_type"],
                "year": str(record["year"]),
                "source": record["source"],
                "chunk_id": str(record["chunk_id"]),
            }
        )

    print("\nAdding documents to ChromaDB...")

    batch_size = 500

    for start in range(
        0,
        len(ids),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(ids)
        )

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

        print(
            f"Added {end:,}/{len(ids):,}"
        )

    print("\nVector database created successfully.")

    print(
        f"Stored documents: {collection.count():,}"
    )

    print(
        f"Database location: {CHROMA_PATH}"
    )

    print("\n" + "=" * 60)
    print("CHROMADB BUILD COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()