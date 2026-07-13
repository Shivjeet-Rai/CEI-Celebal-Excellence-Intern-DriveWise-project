# DriveWise: Metadata-Aware Automotive RAG Assistant

## 🚀 Live Demo
**[https://cei-celebal-excellence-intern-drivewise.onrender.com](https://cei-celebal-excellence-intern-drivewise.onrender.com)**

> Note: This app is hosted on Render's free tier, so it may take 30–60 seconds to wake up if it hasn't been visited recently. Please be patient on first load.

---

DriveWise is a modular, production-ready Retrieval-Augmented Generation (RAG) assistant designed to answer user queries using official car brochure PDFs. It integrates dense semantic search, metadata filtering, cross-encoder reranking, and the Google Gemini API to deliver highly accurate and context-aware responses about vehicle configurations, specifications, performance, and features.

---

## Overview

Automotive brochures contain dense technical specifications, feature tables, and marketing literature that standard RAG systems often fail to retrieve accurately. DriveWise solves this by implementing:
1. **Metadata-Aware Ingestion**: Chunks are enriched with vehicle-specific tags (Brand, Model, Year, Section Category) to enable precise pre-filtering or post-filtering.
2. **Dense Vector Retrieval**: Employs FAISS with lightweight ONNX-based embeddings (via fastembed) for fast, memory-efficient semantic search.
3. **Cross-Encoder Reranking**: Re-orders the top retrieved passages using an ONNX-based reranker model before context feeding to maximize answer factualness and minimize hallucinations.
4. **Streamlit UI**: An intuitive, premium chat interface allowing users to filter by car metadata, trace sources, and view retrieved brochure pages.

---

## Architecture

```mermaid
graph TD
    %% Ingestion Pipeline
    subgraph Ingestion Pipeline
        A[Raw Brochures PDFs] --> B[Parser: PyMuPDF / fitz]
        B --> C[Metadata Extractor]
        B --> D[Text & Table Chunker]
        C --> E[Metadata-Chunk Associator]
        D --> E
        E --> F[Embedding Generator: fastembed ONNX]
        F --> G[FAISS Vector Index]
    end

    %% Inference Pipeline
    subgraph Inference & Generation
        I[User Query] --> J[Dense Retriever]
        G --> J
        J --> K[Reranker: fastembed ONNX Cross-Encoder]
        K --> L[Generator: Gemini API]
        L --> M[Streamlit Chat UI]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
    style M fill:#bfb,stroke:#333,stroke-width:2px
```

---

## Folder Structure
DriveWise/
├── app/                  # User Interface components and layout views
│   ├── components.py     # Reusable Streamlit widgets (message bubbles, filters)
│   └── ui.py             # Main Streamlit view definitions
│
├── rag/                  # Core RAG pipeline module
│   ├── init.py       # Package initializer
│   ├── parser.py         # PDF text and table parser (PyMuPDF)
│   ├── chunker.py        # Text splitting & layout preservation strategies
│   ├── metadata.py       # Auto-extraction & tagging of car model metadata
│   ├── embeddings.py     # fastembed (ONNX) vector generation interface
│   ├── vectorstore.py    # FAISS read/write/update operations
│   ├── retriever.py      # Dense retriever with metadata-filtered search
│   ├── reranker.py       # fastembed (ONNX) cross-encoder reranking
│   ├── generator.py      # LLM response completion (Gemini API integration)
│   ├── logger.py         # Performance, latency, and cost tracing
│   ├── evaluator.py      # Automatic pipeline benchmarking (correctness & latency)
│   └── utils.py          # General directory and data formatting utility helpers
│
├── data/                 # Local data storage
│   ├── brochures/        # Sub-directories of car brochure PDFs grouped by manufacturer
│   ├── processed/        # Parsed text chunks, tables, and JSON metadata cache
│   ├── chunks/           # Structured section-aware chunk JSON files
│   ├── enriched/         # Chunk files enriched with metadata tags and keywords
│   └── cache/            # Model and API temporary caches
│
├── vectorstore/          # Serialized FAISS index and metadata
│   ├── index.faiss       # FAISS IndexFlatIP dense vector database
│   └── metadata.json     # JSON metadata mapping matching vector indices
├── prompts/              # System and prompt template configuration files
│   └── system_prompt.txt # Prompt instructions guiding LLM generation constraints
├── logs/                 # Rotation application and performance log files
├── tests/                # Automated pytest unit and integration tests
│
├── config.py             # Global constants, file paths, and model configuration options
├── ingest.py             # Script to run raw brochure PDF ingestion and vector indexing
├── rebuild_index.py      # One-time script to rebuild FAISS index with fastembed
├── main.py               # Principal entrypoint to launch UI or CLI runner
├── requirements.txt      # Project Python package dependencies
├── .env.example          # Sample environment configuration template
├── .gitignore            # System, IDE, cache, and key file excludes
└── LICENSE               # MIT License file

---

## Features

- **Lightweight ONNX-Based Retrieval**: Uses fastembed (ONNX Runtime) for both embeddings and reranking — no PyTorch dependency, optimized for low-memory deployment environments.
- **Structured Metadata Filtering**: Scope your search to specific parameters like `Make: Hyundai`, `Year: 2024` or categories like `Safety`.
- **Factual Reranking**: Re-evaluates retrieval candidates with a Cross-Encoder to prioritize pages holding actual facts.
- **Failsafe Gemini Prompting**: Custom system prompt constraints that force the assistant to cite brochures or state "I do not know" instead of guessing.
- **Developer-Friendly Instrumentation**: Easy execution tracking, response time reporting, and automatic retrieval quality validation.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Streamlit and standard toolchain tools

### Step-by-Step Setup

1. **Clone the Repository**:
```bash
   git clone https://github.com/Shivjeet-Rai/CEI-Celebal-Excellence-Intern-DriveWise-project.git
   cd DriveWise
```

2. **Set up a Virtual Environment**:
```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
```

3. **Install Dependencies**:
```bash
   pip install -r requirements.txt
```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your Gemini API key:
```bash
   cp .env.example .env
```
   Add your API key inside `.env`:
```env
   GEMINI_API_KEY=your_gemini_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ENABLE_RERANKER=true
```

5. **Incorporate Brochure Files**:
   Place brochure PDFs in `data/brochures/` nested within manufacturer directories:

data/brochures/
├── Hyundai/
├── Kia/
├── Mahindra/
└── Toyota/
6. **Run Ingestion Pipeline**:
```bash
   python ingest.py
```

7. **Launch the App**:
```bash
   streamlit run main.py
```
   Or, for CLI mode:
```bash
   python main.py
```

---

## Deployment Notes

This project is deployed on **Render's free tier (512 MB RAM)**. To keep the memory footprint within this limit:
- Embeddings and reranking run on **fastembed (ONNX Runtime)** instead of PyTorch/sentence-transformers, significantly reducing baseline memory usage.
- The RAG pipeline is loaded lazily via `st.cache_resource`, ensuring a single shared instance across all user sessions rather than duplicating models per session.
- Set `ENABLE_RERANKER=false` in environment variables if further memory savings are needed (this disables the reranking step while keeping all other functionality intact).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Steps to finish:

Copy this into README.md in your IDE, overwriting the old content
Go to your GitHub repo page → click the ⚙️ gear icon next to "About" (top-right) → paste your Render URL into the "Website" field → Save. This makes the link show up immediately at the top of the repo, no scrolling needed.
Commit and push: