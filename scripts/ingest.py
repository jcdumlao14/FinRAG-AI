from pathlib import Path
import csv
import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "documents.csv"


def extract_text(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF."""

    text_parts = []

    with fitz.open(pdf_path) as document:
        for page in document:
            text = page.get_text()

            if text.strip():
                text_parts.append(text)

    return "\n".join(text_parts)


def load_metadata():
    """Load document metadata from documents.csv."""

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    print("=" * 60)
    print("FinRAG AI - PDF INGESTION")
    print("=" * 60)

    for document in metadata:

        filename = document["filename"]
        pdf_path = RAW_DIR / filename

        if not pdf_path.exists():
            print(f"WARNING: Missing file: {filename}")
            continue

        print(f"\nProcessing: {filename}")

        text = extract_text(pdf_path)

        output_filename = pdf_path.stem + ".txt"
        output_path = PROCESSED_DIR / output_filename

        output_path.write_text(
            text,
            encoding="utf-8"
        )

        with fitz.open(pdf_path) as document:
            page_count = len(document)

        print(f"Pages: {page_count}")
        print(f"Characters: {len(text):,}")
        print(f"Output: {output_path}")

    print("\n" + "=" * 60)
    print("PDF INGESTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()