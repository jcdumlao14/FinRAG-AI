from pathlib import Path
import csv
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "documents.csv"


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    return text.strip()


def create_chunks(text: str, chunk_size: int, overlap: int):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def load_metadata():

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return {
            row["filename"]: row
            for row in csv.DictReader(file)
        }


def main():

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    all_chunks = []

    chunk_id = 0

    print("=" * 60)
    print("FinRAG AI - DOCUMENT CHUNKING")
    print("=" * 60)

    for text_file in sorted(PROCESSED_DIR.glob("*.txt")):

        filename = text_file.name.replace(".txt", ".pdf")

        if filename not in metadata:
            print(f"WARNING: No metadata found for {filename}")
            continue

        print(f"\nProcessing: {filename}")

        text = text_file.read_text(
            encoding="utf-8"
        )

        cleaned_text = clean_text(text)

        chunks = create_chunks(
            cleaned_text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        company_metadata = metadata[filename]

        print(f"Original characters: {len(text):,}")
        print(f"Cleaned characters:  {len(cleaned_text):,}")
        print(f"Chunks created:     {len(chunks)}")

        for local_id, chunk_text in enumerate(chunks):

            record = {
                "chunk_id": chunk_id,
                "company": company_metadata["company"],
                "document_type": company_metadata["document_type"],
                "year": company_metadata["year"],
                "source": company_metadata["source"],
                "filename": filename,
                "local_chunk_id": local_id,
                "text": chunk_text,
            }

            all_chunks.append(record)

            chunk_id += 1

    output_file = CHUNKS_DIR / "chunks.json"

    output_file.write_text(
        json.dumps(
            all_chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)

    print(f"Total chunks: {len(all_chunks):,}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()