<div align="Left">

# 📊 FinRAG AI

### Financial Research Assistant with Hybrid Retrieval and Grounded Generation

<p>
  <strong>Retrieval-Augmented Generation for Financial Research</strong>
</p>

<p>
  Vector Search • BM25 • Hybrid RRF • Gemini 2.5 Flash • ChromaDB • Streamlit • Docker
</p>

<p>
  <a href="https://github.com/jcdumlao14/FinRAG-AI">
    <strong>GitHub Repository</strong>
  </a>
</p>

</div>

---

## 📌 Overview

**FinRAG AI** is an end-to-end **Retrieval-Augmented Generation (RAG)** application designed to answer questions about company financial reports.

Instead of relying entirely on the language model's internal knowledge, FinRAG AI first retrieves relevant financial evidence from an indexed document collection and then provides that context to the LLM for grounded answer generation.

The system combines:

- 🔎 Semantic Vector Search
- 📚 BM25 Lexical Retrieval
- 🔀 Hybrid Retrieval
- 🧮 Reciprocal Rank Fusion (RRF)
- 🤖 Gemini 2.5 Flash
- 🗄️ ChromaDB
- 🖥️ Streamlit
- 📊 Evaluation pipelines
- 👍 User feedback
- 📈 Monitoring dashboard
- 🐳 Docker and Docker Compose

The project was developed as part of the **DataTalksClub LLM Zoomcamp 2026 capstone project**.

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

## 🎯 Project Goal

Financial reports contain large amounts of detailed information covering revenue, expenses, profitability, earnings, and other financial metrics.

Finding a specific financial figure manually can be time-consuming.

Traditional keyword search can fail when the wording of a question differs from the wording in the source document. Pure vector search can also struggle when exact company names, fiscal years, financial terminology, or numerical concepts are important.

FinRAG AI addresses this by combining **semantic retrieval and lexical retrieval** before generating the final answer.

The overall workflow is:

<pre>
Financial Documents
        │
        ▼
Document Ingestion
        │
        ▼
Text Extraction
        │
        ▼
Cleaning & Chunking
        │
        ▼
Metadata
        │
        ▼
Embeddings
        │
        ▼
ChromaDB Knowledge Base
        │
        ▼
      User Query
        │
        ├───────────────┐
        ▼               ▼
 Vector Search       BM25 Search
        │               │
        └───────┬───────┘
                ▼
        Reciprocal Rank
          Fusion (RRF)
                │
                ▼
      Retrieved Financial
            Context
                │
                ▼
       Gemini 2.5 Flash
                │
                ▼
       Grounded Answer
                │
                ▼
        Sources + Feedback
                │
                ▼
          Monitoring
</pre>

---

# 🧠 Core RAG Architecture

FinRAG AI follows a modular RAG architecture:

<pre>
                    ┌─────────────────────────┐
                    │   Financial Reports     │
                    │      PDF / Documents    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Ingestion Pipeline    │
                    │ Cleaning + Chunking      │
                    │ Metadata Extraction      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Knowledge Base      │
                    │        ChromaDB          │
                    └────────────┬────────────┘
                                 │
                                 │
                         ┌───────▼───────┐
                         │   User Query  │
                         └───────┬───────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
        ┌──────────────────┐          ┌──────────────────┐
        │  Vector Search   │          │   BM25 Search    │
        │ Semantic Search  │          │ Lexical Search   │
        └────────┬─────────┘          └────────┬─────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ Reciprocal Rank Fusion  │
                    │          RRF             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Financial Evidence      │
                    │ + Evidence Boosting      │
                    │ + Company Diversity     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Retrieved Context    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Gemini 2.5 Flash     │
                    │   Grounded Generation   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Answer + Source Context │
                    └────────────┬────────────┘
                                 │
                         ┌───────┴────────┐
                         ▼                ▼
                    Streamlit        Feedback &
                        UI            Monitoring
</pre>

---

# 🔎 Retrieval System

FinRAG AI uses multiple retrieval strategies rather than depending on a single search method.

## 1. Vector Retrieval

The vector retriever performs semantic search over the indexed financial document chunks.

This allows the system to identify relevant information even when the wording of the user's question differs from the wording in the source document.

The vector retrieval component is implemented in:

<pre>
rag/retriever.py
</pre>

---

## 2. BM25 Retrieval

BM25 provides lexical retrieval based on token-level matching.

This is particularly useful for financial questions involving:

- Company names
- Fiscal years
- Financial terminology
- Exact phrases
- Financial metrics
- Numerical concepts

The BM25 implementation builds the index once when the module is imported:

<pre>
BM25_INDEX, BM25_CHUNKS = build_bm25()
</pre>

The index is then reused during subsequent searches instead of being rebuilt for every query.

Implementation:

<pre>
rag/bm25_retriever.py
</pre>

---

## 3. Hybrid Retrieval

The hybrid retriever combines:

<pre>
Vector Search
      +
BM25 Search
      ↓
Reciprocal Rank Fusion
      ↓
Financial Evidence Boosting
      ↓
Company Diversity
      ↓
Final Ranked Results
</pre>

Implementation:

<pre>
rag/hybrid_retriever.py
</pre>

The hybrid retriever retrieves a larger candidate set and combines the rankings using **Reciprocal Rank Fusion (RRF)**.

---

# 🔀 Reciprocal Rank Fusion

RRF allows results from different retrieval systems to be combined without requiring their raw scores to be directly comparable.

Conceptually:

<pre>
RRF Score = Σ weight / (k + rank)
</pre>

FinRAG AI uses separate ranking signals from:

- Vector retrieval
- BM25 retrieval

The resulting candidates are then ranked according to their fused score.

This provides a practical way to combine semantic and lexical retrieval.

---

# 💰 Financial Evidence Boosting

The hybrid retriever contains additional financial evidence scoring.

Financial terms considered include:

- Revenue
- Net sales
- Net income
- Operating income
- Operating profit
- Gross profit
- Gross margin
- Earnings
- Earnings per share
- EPS
- Financial results
- Total revenue
- Total net sales

For broad financial-performance questions, the retriever also gives additional importance to core financial metrics.

Core metrics include:

- Revenue
- Net sales
- Gross margin
- Operating income
- Net income
- Earnings per share

This helps prioritize chunks containing multiple relevant financial indicators.

---

# 🏢 Company Diversity

For broad questions involving financial performance, FinRAG AI applies a company-diversity step.

The retriever first prefers strong results from different companies before filling the remaining result positions according to the normal ranking.

This helps prevent the retrieved context from being dominated by multiple chunks from only one company when a broader comparison is requested.

---

# 📚 Knowledge Base

The current evaluation knowledge base contains financial reports for:

| Company | Fiscal Year |
|---|---:|
| Apple | 2025 |
| Microsoft | 2025 |
| NVIDIA | 2026 |

The indexed ChromaDB collection currently contains approximately:

<div align="center">

### 📦 1,746 indexed document chunks

</div>

These chunks provide the evidence used by the retrieval and generation pipeline.

---

# ⚡ Retrieval Performance

Warm retrieval benchmarks were performed against the current knowledge base.

## Vector Retrieval

| Metric | Result |
|---|---:|
| Mean | 18.73 ms |
| Median | 17.94 ms |
| Minimum | 12.28 ms |
| Maximum | 25.56 ms |
| P95 | 25.62 ms |

## Warm Hybrid Retrieval

| Metric | Result |
|---|---:|
| Mean | 21.38 ms |
| Median | 19.30 ms |
| Minimum | 16.93 ms |
| Maximum | 29.88 ms |
| P95 | 30.64 ms |

## BM25 Internal Profile

| Operation | Time |
|---|---:|
| Tokenization | 0.01 ms |
| BM25 scoring | 2.20 ms |
| Top-k sorting | 0.48 ms |
| Result construction | 0.01 ms |
| **Total measured** | **2.70 ms** |

The BM25 index is built once and reused during subsequent searches.

---

# 🤖 LLM Generation

FinRAG AI uses:

<div align="center">

### Gemini 2.5 Flash

</div>

The generation pipeline is implemented in:

<pre>
llm/generator.py
</pre>

The LLM is instructed to:

- Use the retrieved financial context
- Avoid inventing financial figures
- Answer using the retrieved evidence
- State when information is unavailable from the retrieved documents
- Provide source information for the retrieved evidence

The overall generation flow is:

<pre>
User Question
      │
      ▼
Hybrid Retrieval
      │
      ▼
Retrieved Financial Evidence
      │
      ▼
Context Construction
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Grounded Financial Answer
</pre>

This architecture is designed to reduce unsupported financial claims.

---

# 🛡️ Grounding Strategy

FinRAG AI does not treat the LLM as the primary source of financial facts.

Instead, the system follows:

<pre>
Question
   ↓
Retrieve Evidence
   ↓
Build Context
   ↓
Generate Answer
   ↓
Return Sources
</pre>

This makes the retrieved financial documents the evidence layer for the generated response.

---

# 🔬 Evaluation

FinRAG AI includes a dedicated evaluation framework.

The evaluation directory contains:

<pre>
evaluation/
├── context_evaluation.py
├── context_results.json
├── llm_evaluation.py
├── llm_results.json
├── questions.json
├── retrieval_evaluation.py
└── retrieval_results.json
</pre>

The project evaluates both retrieval and final answer quality.

---

# 🧪 LLM Evaluation

The automated LLM evaluation currently contains:

<div align="center">

### 9 Financial Questions

</div>

The evaluation checks:

- Whether the expected company appears in the retrieved evidence
- Whether the expected fiscal year appears in the retrieved evidence
- Whether the expected answer appears in the generated response
- Evaluation success
- Retry information
- Final evaluation results

## Latest Evaluation Results

| Metric | Result |
|---|---:|
| Questions evaluated | 9 |
| Grounded answers | 9 / 9 |
| Grounding score | **100%** |
| Correct answers | 9 / 9 |
| Answer score | **100%** |
| Rate-limit errors | 0 |
| Other errors | 0 |
| Total retries | 1 |

Results are saved to:

<pre>
evaluation/llm_results.json
</pre>

One Gemini rate-limit event occurred during Question 7, after which the evaluation successfully retried the request.

---

# ❓ Evaluation Questions

The current evaluation set covers financial questions involving:

- Apple revenue
- Apple operating expenses
- Apple net income
- Microsoft revenue
- Microsoft operating income
- Microsoft net income
- NVIDIA revenue
- NVIDIA R&D expenses
- NVIDIA operating expenses

### Example

<pre>
What was Apple's total net sales in fiscal year 2025?
</pre>

Expected answer:

<pre>
$416,161 million
</pre>

The system successfully returned the expected answer using supporting Apple 2025 source chunks.

---

# 🔍 Retrieval Evaluation

The repository includes:

<pre>
evaluation/retrieval_evaluation.py
</pre>

The retrieval evaluation framework is designed to compare:

| Retrieval Strategy |
|---|
| Vector Retrieval |
| BM25 Retrieval |
| Hybrid Retrieval |

This makes it possible to evaluate the retrieval layer rather than assuming that one retrieval strategy is always optimal.

Results are stored in:

<pre>
evaluation/retrieval_results.json
</pre>

---

# 🖥️ Streamlit Application

FinRAG AI provides a Streamlit web interface.

Application:

<pre>
app/streamlit_app.py
</pre>

The interface allows users to:

- Enter financial questions
- Retrieve financial evidence
- Generate grounded answers
- View source information
- Submit feedback

The application runs on:

<pre>
http://localhost:8501
</pre>

---

# 👍 Feedback System

FinRAG AI includes user feedback collection.

Implementation:

<pre>
monitoring/feedback.py
</pre>

Feedback records contain:

- Timestamp
- User question
- Generated answer
- Feedback type
- Retrieved sources

The feedback file is stored at:

<pre>
monitoring/feedback.json
</pre>

Feedback can be used to understand how users interact with the system and identify areas for future improvement.

---

# 📊 Monitoring Dashboard

FinRAG AI includes a Streamlit monitoring dashboard.

Implementation:

<pre>
monitoring/dashboard.py
</pre>

The dashboard provides information about:

- Total queries
- Positive feedback
- Negative feedback
- Positive feedback rate
- Feedback distribution
- Queries over time
- Feedback over time
- Questions by company
- Retrieved sources by document
- Recent questions

The dashboard provides a foundation for monitoring application usage and feedback.

---

# 📥 Ingestion Pipeline

The project includes a complete document-processing pipeline.

The workflow is:

<pre>
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
</pre>

Important scripts include:

<pre>
scripts/
├── ingest.py
├── chunk_documents.py
├── build_embeddings.py
├── build_vector_db.py
└── ask_finrag.py
</pre>

The data directories are organized as:

<pre>
data/
├── raw/
├── processed/
├── chunks/
└── metadata/
</pre>

This modular design allows the knowledge base to be rebuilt when source documents change.

---

# 🗂️ Project Structure

The current repository structure is:

<pre>
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
│   ├── context_evaluation.py
│   ├── context_results.json
│   ├── llm_evaluation.py
│   ├── llm_results.json
│   ├── questions.json
│   ├── retrieval_evaluation.py
│   └── retrieval_results.json
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
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── pipeline.py
│   ├── retriever.py
│   └── __init__.py
│
├── scripts/
│   ├── ask_finrag.py
│   ├── build_embeddings.py
│   ├── build_vector_db.py
│   ├── chunk_documents.py
│   └── ingest.py
│
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
</pre>

---

# 🧩 Main Components

| Component | Purpose |
|---|---|
| `app/` | Streamlit user interface |
| `rag/` | Retrieval and RAG pipeline |
| `llm/` | LLM answer generation |
| `evaluation/` | Retrieval and LLM evaluation |
| `monitoring/` | Feedback and monitoring |
| `scripts/` | Data ingestion and knowledge-base construction |
| `data/` | Source documents and processed data |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Application orchestration |

---

# 🐳 Containerization

FinRAG AI is containerized using Docker.

The environment uses:

- Python 3.11
- Docker
- Docker Compose
- Streamlit
- ChromaDB
- Gemini 2.5 Flash

The primary container configuration is defined in:

<pre>
Dockerfile
docker-compose.yml
docker-entrypoint.sh
</pre>

The Streamlit application is exposed on:

<pre>
8501
</pre>

ChromaDB data is persisted through the Docker volume configured by the Docker Compose setup.

---

# ⚙️ Installation

## Prerequisites

Install:

- Git
- Docker Desktop
- Docker Compose

A Google Gemini API key is required for LLM generation.

---

# 🔐 Environment Variables

Create a local `.env` file in the project root.

Example:

<pre>
GOOGLE_API_KEY=your_google_api_key
</pre>

Do **not** commit `.env` to GitHub.

The `.env` file contains credentials and should remain local.

---

# ▶️ Running the Application

Clone the repository:

<pre>
git clone https://github.com/jcdumlao14/FinRAG-AI.git
cd FinRAG-AI
</pre>

Build the Docker environment:

<pre>
docker compose build
</pre>

Start the application:

<pre>
docker compose up -d
</pre>

Check the running containers:

<pre>
docker compose ps
</pre>

Open the application:

<pre>
http://localhost:8501
</pre>

To stop the application:

<pre>
docker compose down
</pre>

---

# 🔬 Running Evaluations

## LLM Evaluation

Run:

<pre>
docker compose exec finrag sh -c "PYTHONPATH=/app python evaluation/llm_evaluation.py"
</pre>

The evaluation produces results covering:

- Grounding
- Expected answer matching
- Evaluation success
- Retry information
- Final evaluation summary

Results are written to:

<pre>
evaluation/llm_results.json
</pre>

---

## Retrieval Evaluation

Run the retrieval evaluation script from the project environment:

<pre>
python evaluation/retrieval_evaluation.py
</pre>

The evaluation compares:

<pre>
Vector Retrieval
       │
       ├─────────────┐
       │             │
       ▼             ▼
     BM25        Hybrid RRF
       │             │
       └──────┬──────┘
              ▼
       Retrieval Results
</pre>

Results are stored in:

<pre>
evaluation/retrieval_results.json
</pre>

---

# 🔁 Reproducibility

The application is designed to be reproducible using Docker Compose.

Dependencies are specified in:

<pre>
requirements.txt
</pre>

The Docker environment uses:

<pre>
python:3.11-slim
</pre>

To reproduce the application:

<pre>
git clone https://github.com/jcdumlao14/FinRAG-AI.git

cd FinRAG-AI

docker compose build

docker compose up -d
</pre>

Then open:

<pre>
http://localhost:8501
</pre>

---

# 🏆 Evaluation Criteria Coverage

The project was designed around the major DataTalksClub LLM Zoomcamp capstone requirements.

| Requirement | Implementation |
|---|---|
| Problem description | Financial research and retrieval problem |
| Knowledge base | Financial reports |
| Retrieval | Vector + BM25 + Hybrid RRF |
| LLM generation | Gemini 2.5 Flash |
| Retrieval evaluation | Vector, BM25 and Hybrid evaluation |
| LLM evaluation | Automated 9-question evaluation |
| Interface | Streamlit |
| Ingestion | Python ingestion pipeline |
| Monitoring | Monitoring dashboard |
| Feedback | User feedback collection |
| Containerization | Dockerfile + Docker Compose |
| Reproducibility | Docker-based environment |
| Hybrid search | Vector + BM25 + RRF |
| Rank fusion | Reciprocal Rank Fusion |
| Query rewriting | Not currently implemented |
| Additional embedding model | Not currently implemented |
| Cloud deployment | Not currently implemented |

The project prioritizes a complete, working and reproducible RAG system rather than adding experimental features without sufficient evaluation.

---

# 🧠 Design Decisions

## Why Hybrid Search?

Financial questions frequently contain:

- Company names
- Fiscal years
- Exact financial terminology
- Financial metrics
- Numerical concepts

BM25 is strong at lexical matching, while vector retrieval captures semantic similarity.

Combining both provides a more robust retrieval strategy.

---

## Why BM25?

BM25 is particularly useful when exact terminology matters.

For example:

<pre>
Apple
2025
net sales
operating income
earnings per share
</pre>

These terms can provide strong lexical signals when retrieving financial evidence.

---

## Why Vector Search?

Vector search captures semantic relationships between the question and document content.

This is useful when the user's wording differs from the wording used in the financial report.

---

## Why RRF?

Reciprocal Rank Fusion provides a practical mechanism for combining independent rankings.

It avoids requiring vector similarity scores and BM25 scores to have the same scale.

---

## Why Gemini 2.5 Flash?

Gemini 2.5 Flash provides a practical balance between:

- Answer generation quality
- Response latency
- RAG application usability

The model is used only after relevant financial context has been retrieved.

---

## Why ChromaDB?

ChromaDB provides a lightweight vector database suitable for storing and retrieving document embeddings.

It fits well with the project's local and Docker-based architecture.

---

# 🛡️ Reliability Principles

FinRAG AI follows several principles designed to improve answer reliability.

### 1. Retrieve Before Generate

The system retrieves financial evidence before asking the LLM to generate an answer.

### 2. Ground the Answer

The retrieved context is passed directly to the generation stage.

### 3. Avoid Invented Financial Figures

The LLM is instructed not to invent financial figures.

### 4. Acknowledge Missing Information

When the retrieved documents do not contain the requested information, the model is instructed to indicate that the information is unavailable.

### 5. Preserve Source Information

The pipeline maintains source metadata including:

- Company
- Fiscal year
- Filename
- Chunk ID

---

# 📈 Key Results

<div align="center">

| Metric | Result |
|---|---:|
| 📦 Indexed document chunks | **1,746** |
| 🔎 Vector retrieval mean | **18.73 ms** |
| 🔀 Hybrid retrieval mean | **21.38 ms** |
| 📚 BM25 internal profile | **2.70 ms** |
| 🧪 Evaluation questions | **9** |
| 🛡️ Grounding score | **100%** |
| ✅ Correct answers | **9 / 9** |
| 🎯 Answer score | **100%** |
| 🔄 Evaluation retries | **1** |
| ❌ Rate-limit errors | **0** |
| ❌ Other errors | **0** |

</div>

---

# ⚠️ Current Limitations

The current version focuses on a strong and reproducible core RAG architecture.

The following features are not currently implemented:

- Cloud deployment
- Dedicated neural document reranking
- Query rewriting
- Multiple embedding-model comparison
- Additional financial datasets
- Authentication
- Multi-user support
- Production-scale deployment
- Advanced LLM evaluation methodologies

These are potential future improvements rather than missing components of the current working RAG pipeline.

---

# 🚀 Future Improvements

Potential future development includes:

- ☁️ Cloud deployment
- 🧠 Neural document reranking
- 🔄 Query rewriting
- 🧪 Additional retrieval experiments
- 🤖 Alternative embedding models
- 📚 Larger financial datasets
- 📊 Additional evaluation metrics
- 📈 More advanced monitoring
- 💬 Improved feedback analysis
- 🔐 Authentication
- 👥 Multi-user support
- ⚙️ Production deployment
- 📦 Scalable infrastructure

---

# 🧪 Technical Stack

<div align="center">

| Technology | Role |
|---|---|
| 🐍 Python 3.11 | Core programming language |
| 🤖 Gemini 2.5 Flash | LLM generation |
| 🗄️ ChromaDB | Vector database |
| 🔎 Vector Search | Semantic retrieval |
| 📚 BM25 | Lexical retrieval |
| 🔀 RRF | Hybrid rank fusion |
| 🖥️ Streamlit | User interface |
| 📊 Pandas | Monitoring and analysis |
| 🐳 Docker | Containerization |
| 🐙 Git | Version control |

</div>

---

# 📁 Important Files

### RAG

<pre>
rag/retriever.py
rag/bm25_retriever.py
rag/hybrid_retriever.py
rag/pipeline.py
</pre>

### LLM

<pre>
llm/generator.py
</pre>

### Application

<pre>
app/streamlit_app.py
</pre>

### Evaluation

<pre>
evaluation/context_evaluation.py
evaluation/llm_evaluation.py
evaluation/retrieval_evaluation.py
</pre>

### Monitoring

<pre>
monitoring/feedback.py
monitoring/dashboard.py
</pre>

### Data Pipeline

<pre>
scripts/ingest.py
scripts/chunk_documents.py
scripts/build_embeddings.py
scripts/build_vector_db.py
scripts/ask_finrag.py
</pre>

---

# 👩‍💻 Author

<div align="center">

### Jocelyn C. Dumlao

<strong>Independent Data Scientist | Machine Learning Engineer </strong>

<br><br>

<a href="https://github.com/jcdumlao14">
GitHub Profile
</a>

&nbsp;&nbsp;•&nbsp;&nbsp;

<a href="https://github.com/jcdumlao14/FinRAG-AI">
FinRAG AI Repository
</a>

</div>

---

# 🎓 Project Context

FinRAG AI was developed as part of the:

<strong>DataTalksClub LLM Zoomcamp 2026</strong>

The project applies concepts including:

- Retrieval-Augmented Generation
- Vector Search
- BM25
- Hybrid Search
- Reciprocal Rank Fusion
- LLM Evaluation
- Docker
- Application Monitoring
- User Feedback

---

# 📄 License

This project is intended for **educational and portfolio purposes** as part of the LLM Zoomcamp capstone project.

---

# 🙏 Acknowledgments

Special thanks to the **Alexey Grigorev - Founder of DataTalks.Club** **DataTalksClub LLM Zoomcamp** for providing the learning framework and practical foundation for building an end-to-end LLM application.

The project applies concepts learned throughout the course to a financial research use case.

---

<div align="center">

# ⭐ FinRAG AI

### Search financial evidence.
### Retrieve the right context.
### Generate grounded answers.

<br>

<strong>Financial Research • Hybrid Retrieval • Grounded Generation</strong>

</div>
