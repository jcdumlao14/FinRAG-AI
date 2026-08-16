# 📊 FinRAG AI

## Financial Research Assistant with Hybrid Retrieval and Grounded Generation

FinRAG AI is an end-to-end **Retrieval-Augmented Generation (RAG)** application designed to answer questions about company financial reports.

The system combines **semantic vector search**, **BM25 lexical retrieval**, and **Hybrid Reciprocal Rank Fusion (RRF)** to retrieve relevant financial evidence before generating an answer with **Google Gemini 2.5 Flash**.

The project was built as a complete RAG application covering the full lifecycle:

**Data → Ingestion → Chunking → Knowledge Base → Retrieval → Hybrid Search → LLM Generation → Evaluation → User Interface → Feedback → Monitoring → Docker**

---

## 🎯 Problem Description

Financial reports such as annual reports and 10-K filings contain large amounts of detailed financial information. Finding a specific figure or explanation manually can be time-consuming, especially when users need to compare financial information across companies and fiscal years.

Traditional keyword search can miss semantically related information, while pure vector search may fail when exact financial terms, company names, years, or numerical concepts are important.

FinRAG AI addresses this problem by combining:

- **Vector retrieval** for semantic similarity
- **BM25 retrieval** for exact lexical matching
- **Reciprocal Rank Fusion (RRF)** for combining retrieval signals
- **LLM generation** for producing natural-language answers
- **Source attribution** so users can inspect the retrieved evidence
- **Evaluation pipelines** to measure retrieval and answer quality

The goal is to provide a financial research assistant that produces **grounded answers from the indexed financial documents instead of relying only on the LLM's internal knowledge**.

---

# 🚀 Project Objectives

The project implements the major components required for an end-to-end RAG application:

- Build a searchable financial knowledge base
- Ingest financial reports
- Clean and chunk documents
- Store document representations for retrieval
- Implement semantic vector retrieval
- Implement lexical BM25 retrieval
- Combine retrieval methods using Hybrid RRF
- Generate grounded answers using Gemini 2.5 Flash
- Evaluate retrieval quality
- Evaluate final LLM answers
- Provide a user-facing Streamlit application
- Collect user feedback
- Provide monitoring functionality
- Containerize the application with Docker Compose
- Document reproducible setup and execution

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │   Financial Reports  │
                         │   PDF / Documents    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Ingestion Pipeline │
                         │   Cleaning / Chunking│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Knowledge Base     │
                         │      ChromaDB        │
                         └──────────┬───────────┘
                                    │
                         User Query │
                                    ▼
                    ┌─────────────────────────────┐
                    │       Retrieval Layer       │
                    ├─────────────────────────────┤
                    │ Vector Search                │
                    │ BM25 Lexical Search         │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Hybrid Reciprocal Rank      │
                    │ Fusion (RRF)                 │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Retrieved Financial Context │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Gemini 2.5 Flash        │
                    │      Grounded Generation     │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     Answer + Sources        │
                    └──────────────┬──────────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                 ┌──────────────┐    ┌──────────────┐
                 │  Streamlit   │    │   Feedback /  │
                 │     UI       │    │  Monitoring   │
                 └──────────────┘    └──────────────┘

---
📚 Knowledge Base

The current evaluation knowledge base contains financial reports for:

Apple — Fiscal Year 2025
Microsoft — Fiscal Year 2025
NVIDIA — Fiscal Year 2026

The indexed ChromaDB collection currently contains approximately 1,746 document chunks.

The application retrieves financial evidence from these indexed documents before generating an answer.

🔎 Retrieval Pipeline

FinRAG AI uses multiple retrieval strategies.

1. Vector Search

Vector search represents the user query and document chunks as embeddings and retrieves semantically similar financial content.

This helps when the wording of the question differs from the wording used in the financial report.

2. BM25

BM25 provides lexical retrieval based on token-level matching.

This is particularly useful for:

Company names
Fiscal years
Financial terminology
Exact phrases
Numerical concepts

The BM25 index is built once and reused during subsequent searches.

3. Hybrid Retrieval

The project combines vector and BM25 retrieval using Reciprocal Rank Fusion (RRF).

This allows the system to benefit from both:

semantic similarity
exact lexical matching

The hybrid retriever is the primary retrieval strategy used by the application.

⚡ Retrieval Performance

Warm retrieval benchmarks were performed against the 1,746-document-chunk knowledge base.

Vector Retrieval
Metric    Result
Mean    18.73 ms
Median    17.94 ms
Minimum    12.28 ms
Maximum    25.56 ms
P95    25.62 ms
Warm Hybrid Retrieval
Metric    Result
Mean    21.38 ms
Median    19.30 ms
Minimum    16.93 ms
Maximum    29.88 ms
P95    30.64 ms
BM25 Internal Profile

The optimized in-memory BM25 index was profiled internally:

Operation    Time
Tokenization    0.01 ms
BM25 scoring    2.20 ms
Top-k sorting    0.48 ms
Result construction    0.01 ms
Total measured    2.70 ms

The BM25 index is reused rather than rebuilt for every query.

🤖 LLM Generation

FinRAG AI uses:

Gemini 2.5 Flash

The generation pipeline is designed to answer using the retrieved financial context.

The system instructs the model to:

Use the retrieved financial information
Avoid inventing financial figures
Provide the requested financial answer
State when the information is unavailable from the retrieved documents
Include source information for retrieved evidence

This grounding strategy helps reduce unsupported financial claims.

📊 LLM Evaluation

The project includes an automated LLM evaluation set containing 9 financial questions.

The evaluation checks:

Whether the expected company and fiscal year appear in the retrieved sources
Whether the expected answer appears in the generated response

Latest evaluation result:

Metric    Result
Questions evaluated    9
Grounded answers    9/9
Grounding score    100%
Correct answers    9/9
Answer score    100%
Rate-limit errors    0
Other errors    0
Total retries    1

The evaluation completed successfully and saved results to:

evaluation/llm_results.json

One Gemini rate-limit event occurred during Question 7, after which the application successfully retried the request.

🧪 Evaluation Questions

The evaluation covers financial questions involving:

Apple revenue
Apple operating expenses
Apple net income
Microsoft revenue
Microsoft operating income
Microsoft net income
NVIDIA revenue
NVIDIA R&D expenses
NVIDIA operating expenses

Example:

What was Apple's total net sales in fiscal year 2025?

Expected answer:

$416,161 million

The system successfully returned the expected answer with supporting Apple 2025 source chunks.

🖥️ User Interface

FinRAG AI provides a Streamlit web interface.

The interface allows users to:

Enter financial questions
Retrieve relevant financial evidence
Generate grounded answers
View source information
Submit feedback

The application is exposed on:

http://localhost:8501
📈 Monitoring and Feedback

The project includes a monitoring module:

monitoring/
├── dashboard.py
├── feedback.py
└── __init__.py

The application also includes feedback collection through the Streamlit interface.

Feedback and monitoring are intended to provide a foundation for tracking system behavior and improving retrieval and generation quality over time.

📥 Ingestion Pipeline

The project includes scripts for processing the source documents.

The ingestion workflow is:

Source Documents
       ↓
Document Loading
       ↓
Text Extraction
       ↓
Cleaning
       ↓
Chunking
       ↓
Metadata
       ↓
Embeddings
       ↓
ChromaDB

Important project directories include:

data/
├── raw/
├── processed/
├── chunks/
└── metadata/

The ingestion and retrieval components are implemented as Python modules and scripts so the knowledge base can be rebuilt when the source documents change.

🗂️ Project Structure
FinRAG-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   └── metadata/
│
├── evaluation/
│   ├── questions.json
│   ├── retrieval_evaluation.py
│   ├── context_evaluation.py
│   ├── llm_evaluation.py
│   └── *.json
│
├── llm/
│   └── generator.py
│
├── monitoring/
│   ├── dashboard.py
│   ├── feedback.py
│   └── __init__.py
│
├── rag/
│   ├── pipeline.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── chunker.py
│
├── scripts/
│   ├── ingest.py
│   └── ...
│
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
🐳 Containerization

FinRAG AI is containerized using Docker.

The project uses:

Python 3.11
Docker
Docker Compose
Streamlit
ChromaDB
Gemini 2.5 Flash

The main application is defined in:

Dockerfile
docker-compose.yml

Docker Compose exposes Streamlit on port:

8501

ChromaDB data is persisted through the Docker volume:

chroma_data
⚙️ Installation
Prerequisites

Install:

Git
Docker Desktop
Docker Compose

A Google Gemini API key is required for LLM generation.

🔐 Environment Variables

Create a local .env file in the project root.

Example:

GOOGLE_API_KEY=your_google_api_key

Do not commit .env to GitHub.

The .env file contains secrets and should remain local.

▶️ Running the Application

From the project root:

docker compose build
docker compose up -d

Check the container:

docker compose ps

The application should be available at:

http://localhost:8501

To stop the application:

docker compose down
🔬 Running Evaluations
LLM Evaluation

Run:

docker compose exec finrag sh -c "PYTHONPATH=/app python evaluation/llm_evaluation.py"

The evaluation reports:

Grounding
Expected answer matching
Evaluation success
Retry information
Final evaluation summary
Retrieval Evaluation

The repository includes retrieval evaluation scripts for assessing the retrieval pipeline.

The project evaluates multiple retrieval approaches, including:

Vector retrieval
BM25 retrieval
Hybrid retrieval

This allows the retrieval strategies to be compared rather than relying on a single retrieval method.

🔁 Reproducibility

The application is designed to be reproducible through Docker Compose.

The main dependencies are specified in:

requirements.txt

The Docker environment uses:

python:3.11-slim

The source financial documents used to build the knowledge base are organized under the project's data directories.

To reproduce the application:

git clone https://github.com/jcdumlao14/FinRAG-AI.git
cd FinRAG-AI
docker compose build
docker compose up -d

Then open:

http://localhost:8501
🏆 Evaluation Criteria Coverage

The project was designed around the DataTalksClub LLM Zoomcamp capstone requirements.

Requirement    Implementation
Problem description    Financial research and retrieval problem clearly defined
Knowledge base    Financial 10-K reports
Retrieval flow    Vector + BM25 + Hybrid RRF
LLM generation    Gemini 2.5 Flash
Retrieval evaluation    Vector, BM25 and Hybrid evaluation
LLM evaluation    Automated 9-question evaluation
Interface    Streamlit
Ingestion    Python ingestion pipeline
Monitoring    Monitoring module + feedback collection
Containerization    Dockerfile + Docker Compose
Reproducibility    Docker-based setup and requirements
Hybrid search    Vector + BM25 + RRF
Document re-ranking    RRF-based rank fusion
Query rewriting    Not currently implemented
Additional embedding model    Not currently implemented
Cloud deployment    Not currently implemented

The project prioritizes a complete, working RAG system over adding experimental features immediately before the deadline.

🧠 Design Decisions
Why Hybrid Search?

Financial questions frequently contain exact entities, fiscal years, financial terms, and numerical concepts.

BM25 is strong at lexical matching, while vector retrieval captures semantic similarity.

Combining them improves retrieval robustness.

Why RRF?

Reciprocal Rank Fusion provides a simple and effective method for combining rankings produced by different retrieval systems without requiring their raw scores to be directly comparable.

Why Gemini 2.5 Flash?

Gemini 2.5 Flash provides a practical balance between response quality and latency for the final answer-generation stage.

Why ChromaDB?

ChromaDB provides a lightweight vector database suitable for storing and retrieving the document embeddings used by the application.

🛡️ Grounding and Reliability

FinRAG AI is designed to reduce hallucination risk by providing retrieved financial evidence to the LLM.

The application does not treat the LLM as the primary source of financial facts.

Instead:

User Question
     ↓
Retrieve Evidence
     ↓
Build Context
     ↓
LLM Generation
     ↓
Grounded Answer
     ↓
Sources

The evaluation results provide evidence that the current evaluation questions were answered correctly and with expected source grounding.

⚠️ Current Limitations

The current version intentionally focuses on a strong and reproducible core RAG architecture.

The following features are not currently implemented:

Cloud deployment
Dedicated neural document reranker
Query rewriting
Multiple embedding-model comparison
Major architectural redesign
Additional financial datasets
Alternative LLM evaluation methodology
Major UI redesign

These are potential future improvements rather than requirements for the current working system.

🚀 Future Improvements

Possible future development includes:

Cloud deployment
Neural document re-ranking
Query rewriting
Evaluation of alternative embedding models
Larger financial datasets
Additional evaluation metrics
More advanced monitoring dashboards
Improved user feedback analysis
Authentication and multi-user support
Production deployment and scaling
📌 Key Results

The current FinRAG AI system demonstrates:

1,746 indexed document chunks
Hybrid vector + BM25 retrieval
RRF rank fusion
Warm hybrid retrieval mean: 21.38 ms
Vector retrieval mean: 18.73 ms
BM25 internal measured time: 2.70 ms
9/9 grounded evaluation questions
100% grounding score
9/9 correct answers
100% answer score
Streamlit interface
Docker Compose deployment
User feedback capability
Monitoring module
Source-aware financial answers
👩‍💻 Author

Jocelyn C. Dumlao

Data Scientist | Kaggle GrandMaster

GitHub:

https://github.com/jcdumlao14

Project:

https://github.com/jcdumlao14/FinRAG-AI

📄 License

This project is intended for educational and portfolio purposes as part of the LLM Zoomcamp capstone project.

🙏 Acknowledgments

This project was developed as part of the DataTalksClub LLM Zoomcamp and applies concepts from the course including:

Retrieval-Augmented Generation
Vector Search
BM25
Hybrid Search
Reciprocal Rank Fusion
LLM evaluation
Docker
Application monitoring

⭐ FinRAG AI

Search financial evidence. Retrieve the right context. Generate grounded answers.
