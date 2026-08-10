from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "chunks" / "embeddings.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    print("=" * 60)
    print("FinRAG AI - EMBEDDING GENERATION")
    print("=" * 60)

    print(f"\nLoading chunks from:")
    print(CHUNKS_FILE)

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print(f"Total chunks: {len(chunks):,}")

    print(f"\nLoading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(f"\nEmbedding shape: {embeddings.shape}")

    output = []

    for chunk, embedding in zip(chunks, embeddings):

        record = {
            **chunk,
            "embedding": embedding.tolist(),
        }

        output.append(record)

    print("\nSaving embeddings...")

    with EMBEDDINGS_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file
        )

    print(f"Output: {EMBEDDINGS_FILE}")

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()