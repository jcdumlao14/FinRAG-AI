# FinRAG-AI

**Financial Research Assistant using Retrieval-Augmented Generation (RAG)**

FinRAG-AI is a financial question-answering system built as part of the **LLM Zoomcamp 2026** project. It combines semantic vector search, BM25 lexical retrieval, Reciprocal Rank Fusion (RRF), and Gemini 2.5 Flash to answer questions about company financial reports.

The system is designed to provide **grounded financial answers from retrieved documents** rather than relying solely on the language model's internal knowledge.

---

## Project Overview

Financial reports contain large amounts of structured and unstructured information, making it difficult to efficiently locate specific financial figures.

FinRAG-AI addresses this problem by building a Retrieval-Augmented Generation pipeline that:

1. Ingests financial 10-K reports.
2. Extracts and processes document text.
3. Splits documents into searchable chunks.
4. Generates vector embeddings.
5. Stores embeddings in ChromaDB.
6. Performs semantic vector retrieval.
7. Performs BM25 lexical retrieval.
8. Combines retrieval results using Reciprocal Rank Fusion (RRF).
9. Sends the retrieved financial context to Gemini 2.5 Flash.
10. Generates an answer grounded in the retrieved documents.
11. Provides source information for the retrieved evidence.
12. Evaluates retrieval and LLM answer quality.

---

## Architecture

```text
                    Financial 10-K Reports
                             |
                             v
                    +----------------+
                    | Document       |
                    | Ingestion      |
                    +----------------+
                             |
                             v
                    +----------------+
                    | Text           |
                    | Processing     |
                    +----------------+
                             |
                             v
                    +----------------+
                    | Chunking       |
                    +----------------+
                             |
                  +----------+----------+
                  |                     |
                  v                     v
          +---------------+     +---------------+
          | Embeddings    |     | BM25 Index    |
          +---------------+     +---------------+
                  |                     |
                  v                     v
          +---------------+     +---------------+
          | ChromaDB      |     | BM25 Search   |
          +---------------+     +---------------+
                  |                     |
                  +----------+----------+
                             |
                             v
                    +----------------+
                    | Hybrid RRF     |
                    | Retrieval      |
                    +----------------+
                             |
                             v
                    Retrieved Context
                             |
                             v
                    +----------------+
                    | Gemini         |
                    | 2.5 Flash      |
                    +----------------+
                             |
                             v
                    Grounded Answer
                             |
                             v
                    Sources / Evidence
Data

The current evaluation corpus contains financial reports for:
| Company   | Fiscal Year | Document                 |
| --------- | ----------: | ------------------------ |
| Apple     |        2025 | `apple_10k_2025.pdf`     |
| Microsoft |        2025 | `microsoft_10k_2025.pdf` |
| NVIDIA    |        2026 | `nvidia_10k_2026.pdf`    |

The indexed ChromaDB collection currently contains 1,746 documents/chunks.

The source PDFs and generated ChromaDB files are intentionally excluded from Git tracking through .gitignore.
Retrieval System

FinRAG-AI implements three retrieval approaches.

1. Vector Search

Semantic retrieval is performed using embeddings and ChromaDB.

This allows the system to retrieve financially relevant passages even when the wording of the question differs from the wording in the source document.

2. BM25

BM25 provides lexical retrieval based on keyword matching.

This is useful when queries contain specific financial terminology, company names, fiscal years, or financial metrics.

3. Hybrid RRF

The hybrid retriever combines vector and BM25 rankings using Reciprocal Rank Fusion (RRF).

The goal is to combine:

semantic similarity from vector search
lexical relevance from BM25

This provides a second retrieval signal and improves robustness across different types of financial questions.

Retrieval Evaluation

The retrieval system was evaluated using 9 financial questions.

The evaluation checks whether the retrieved result belongs to the expected company and fiscal year.
Results
| Retriever     |       Top-1 |       Top-3 |       Top-5 |
| ------------- | ----------: | ----------: | ----------: |
| Vector Search | **100.00%** | **100.00%** | **100.00%** |
| BM25          |      77.78% |      88.89% |      88.89% |
| Hybrid RRF    | **100.00%** | **100.00%** | **100.00%** |
Interpretation

Vector Search achieved perfect retrieval accuracy across all evaluated cutoff levels.

Hybrid RRF also achieved:

100% Top-1
100% Top-3
100% Top-5

BM25 performed well but was less reliable on several questions.

For the current 9-question evaluation set, Vector Search and Hybrid RRF tie for the best Top-5 retrieval accuracy at 100%.

The complete results are stored in:

evaluation/retrieval_results.json

LLM Evaluation

FinRAG-AI also evaluates the generated answers using a dedicated LLM evaluation pipeline.

The evaluation considers whether the generated answer:

contains the expected financial answer
is grounded in retrieved context
avoids unsupported financial claims
provides relevant source information

Evaluation artifacts are stored in:
evaluation/context_results.json
evaluation/llm_results.json

The evaluation questions are stored in:

evaluation/questions.json

Grounded Answering

The generator is designed to answer using retrieved financial context.

When the required information is not available in the retrieved documents, the system is instructed to state that the information is unavailable rather than inventing a financial figure.

For example, when asked:

What was Tesla's total revenue in fiscal year 2025?

the system correctly responded that the information was not available in the retrieved documents because the current corpus contains financial information for Apple, Microsoft, and NVIDIA.

This behavior is important for financial applications because unsupported numerical claims can be misleading.

Example Questions
Apple
What was Apple's total net sales in fiscal year 2025?

Example answer:

Apple's total net sales in fiscal year 2025 was
$416,161 million.
Microsoft
What was Microsoft's total revenue in fiscal year 2025?

Example answer:

Microsoft's total revenue in fiscal year 2025 was
$281,724 million.
NVIDIA
What was NVIDIA's revenue in fiscal year 2026?

Example answer:

NVIDIA's revenue in fiscal year 2026 was
$215,938 million.

The CLI also displays the retrieved source documents and chunks used for the answer.

Project Structure
FinRAG-AI/
│
├── README.md
├── .gitignore
│
├── data/
│   └── metadata/
│       └── documents.csv
│
├── evaluation/
│   ├── questions.json
│   ├── retrieval_evaluation.py
│   ├── retrieval_results.json
│   ├── context_evaluation.py
│   ├── context_results.json
│   ├── llm_evaluation.py
│   └── llm_results.json
│
├── llm/
│   ├── __init__.py
│   └── generator.py
│
├── rag/
│   ├── __init__.py
│   ├── retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── pipeline.py
│
└── scripts/
    ├── ask_finrag.py
    ├── ingest.py
    ├── chunk_documents.py
    ├── build_embeddings.py
    └── build_vector_db.py

Technologies
Python
Google Gemini 2.5 Flash
ChromaDB
Sentence embeddings
BM25
Reciprocal Rank Fusion (RRF)
JSON
Git / GitHub
Setup
1. Clone the repository
git clone https://github.com/jcdumlao14/FinRAG-AI.git
cd FinRAG-AI

2. Create a virtual environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
3. Install dependencies

Install the project's required Python packages in the virtual environment.

4. Configure the Gemini API key

Create a .env file:

GEMINI_API_KEY=your_api_key_here

The .env file is excluded from Git tracking.

Building the Data Pipeline

The project provides scripts for the document processing pipeline.

Ingest documents
python ".\scripts\ingest.py"
Chunk documents
python ".\scripts\chunk_documents.py"
Build embeddings
python ".\scripts\build_embeddings.py"
Build the vector database
python ".\scripts\build_vector_db.py"

These steps produce the processed document chunks, embeddings, and ChromaDB collection used by the retrieval system.

Ask FinRAG-AI a Question

Run:

python ".\scripts\ask_finrag.py"

Then enter a financial question, for example:

What was Apple's total net sales in fiscal year 2025?

The application returns:

The generated answer.
Company and fiscal-year information when available.
Retrieved source documents.
Relevant chunk identifiers.
Running Evaluation
Retrieval evaluation

Run:

python ".\evaluation\retrieval_evaluation.py"

This evaluates:

Vector Search
BM25
Hybrid RRF

at:

Top-1
Top-3
Top-5

The results are saved to:

evaluation/retrieval_results.json
Context evaluation

Run:

python ".\evaluation\context_evaluation.py"

Results are stored in:

evaluation/context_results.json
LLM evaluation

Run:

python ".\evaluation\llm_evaluation.py"

Results are stored in:

evaluation/llm_results.json
Evaluation Philosophy

The project evaluates the RAG pipeline at multiple stages rather than evaluating only the final generated answer.
Document
   |
   v
Retrieval
   |
   +--> Vector Search evaluation
   |
   +--> BM25 evaluation
   |
   +--> Hybrid RRF evaluation
   |
   v
Retrieved Context
   |
   v
LLM Generation
   |
   +--> Context grounding evaluation
   |
   +--> Answer matching evaluation
   |
   v
Final Answer

his makes it possible to distinguish between:

retrieval failures
context/grounding problems
generation problems

rather than treating every incorrect answer as an LLM problem.

Limitations

The current implementation has several limitations.

Limited corpus

The current dataset contains financial reports for only three companies:

Apple
Microsoft
NVIDIA

Therefore, questions about companies outside the corpus may not be answerable.

Small evaluation set

Retrieval performance is currently measured using 9 evaluation questions.

Although the current results are strong, a larger evaluation dataset would provide a more reliable estimate of general retrieval performance.

Financial-document scope

The system is designed around the currently indexed financial reports and should not be treated as a general financial advisor.

External knowledge

The system is intentionally grounded in retrieved documents and does not attempt to supplement missing financial information from external sources.

Future Improvements

Potential improvements include:

Expand the financial-document corpus.
Increase the evaluation dataset.
Add reranking models after initial retrieval.
Tune RRF parameters.
Add retrieval latency measurements.
Add answer-generation latency measurements.
Add automated evaluation dashboards.
Add more detailed citation tracking.
Add an interactive web interface.
Add Docker deployment.
Add monitoring and observability.
Compare additional embedding models.
Evaluate retrieval performance on more complex multi-document questions.
Project Status

Status: Completed core RAG pipeline and evaluation

Current implementation includes:

Document ingestion
Text processing
Chunking
Embedding generation
ChromaDB vector storage
Vector retrieval
BM25 retrieval
Hybrid RRF retrieval
Gemini 2.5 Flash generation
Source reporting
Retrieval evaluation
Context evaluation
LLM evaluation
Grounded-answer behavior
Evaluation result persistence
Current retrieval benchmark

Best Top-5 Accuracy: 100.00%

Achieved by:

Vector Search
Hybrid RRF

on the current 9-question evaluation set.

LLM Zoomcamp 2026

This project was developed as part of the LLM Zoomcamp 2026 learning and project work, with a focus on applying Retrieval-Augmented Generation techniques to a practical financial research use case.

Author

Jocelyn C. Dumlao

Data Science / Machine Learning

GitHub:

https://github.com/jcdumlao14

