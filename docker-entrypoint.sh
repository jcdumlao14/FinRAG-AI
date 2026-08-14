#!/bin/sh
set -e

echo "=============================================="
echo "FinRAG AI - Container Startup"
echo "=============================================="

if [ ! -f "/app/data/chunks/embeddings.json" ]; then
    echo "ERROR: embeddings.json not found."
    exit 1
fi

echo "Building ChromaDB vector database..."

python scripts/build_vector_db.py

echo "Starting Streamlit..."

exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
