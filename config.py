"""
DriveWise Configuration Constants.

Defines primary local paths and model identifiers for RAG execution.
"""

DATA_DIR = "data"
BROCHURE_DIR = "data/brochures"
PROCESSED_DIR = "data/processed"
CHUNKS_DIR = "data/chunks"
ENRICHED_DIR = "data/enriched"
CACHE_DIR = "data/cache"
VECTORSTORE_DIR = "vectorstore"
LOG_DIR = "logs"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "gemini-flash-latest"

# Hybrid Retrieval Settings
FAISS_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 5
RRF_K = 60

# Reranking Settings
TOP_RERANKED_CHUNKS = 8
