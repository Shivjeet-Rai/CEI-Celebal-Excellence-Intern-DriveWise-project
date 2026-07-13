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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Smaller reranker (~22M params instead of ~278M)
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "gemini-flash-latest"

# FAISS Retrieval Settings
FAISS_TOP_K = 10
FINAL_TOP_K = 5

# Reranking Settings
TOP_RERANKED_CHUNKS = 8