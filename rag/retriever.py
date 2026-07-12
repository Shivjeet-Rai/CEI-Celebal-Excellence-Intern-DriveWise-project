"""Hybrid Retriever Module.

Combines dense semantic search (FAISS) with sparse lexical search (BM25) to
retrieve the most relevant chunks from ingested car brochures.
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import faiss
import numpy as np

import config
from rag.bm25_index import tokenize_text
from rag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved text chunk along with its RRF fusion score."""

    chunk_id: str
    brand: str
    model: str
    year: int
    page_number: int
    section: str
    subsection: str
    keywords: List[str]
    source_file: str
    text: str
    score: float


class HybridRetriever:
    """Orchestrates hybrid dense-sparse search over brochure chunk indices."""

    def __init__(self, store_dir: Union[str, Path] = config.VECTORSTORE_DIR) -> None:
        """Initializes retriever by loading FAISS index, BM25, and metadata."""
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.faiss"
        self.bm25_path = self.store_dir / "bm25.pkl"
        self.metadata_path = self.store_dir / "metadata.json"

        self.faiss_index: Any = None
        self.bm25_index: Any = None
        self.metadata: List[Dict[str, Any]] = []

        self.embedding_generator = EmbeddingGenerator(model_name=config.EMBEDDING_MODEL)
        self._load_indices()

    def _load_indices(self) -> None:
        """Loads index and metadata files from the vector store directory."""
        try:
            # Shared metadata load
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(
                    "Loaded metadata mappings containing %d entries",
                    len(self.metadata)
                )
            else:
                logger.warning("Metadata file not found: %s", self.metadata_path)

            # FAISS index load
            if self.index_path.exists():
                self.faiss_index = faiss.read_index(str(self.index_path))
                logger.info(
                    "Loaded FAISS index containing %d vectors",
                    self.faiss_index.ntotal
                )
            else:
                logger.warning("FAISS index file not found: %s", self.index_path)

            # BM25 index load
            if self.bm25_path.exists():
                with open(self.bm25_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                logger.info("Loaded BM25 index file successfully")
            else:
                logger.warning("BM25 index file not found: %s", self.bm25_path)

        except Exception as e:
            logger.error("Failed to load hybrid search indices: %s", e)

    def _dense_search(self, query: str, top_k: int) -> List[int]:
        """Performs dense semantic search using FAISS and returns indices."""
        dense_results: List[int] = []
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            try:
                query_embeddings = self.embedding_generator.generate_embeddings([query])
                if query_embeddings:
                    query_vector = np.array(query_embeddings).astype("float32")
                    k_val = min(top_k, self.faiss_index.ntotal)
                    distances, indices = self.faiss_index.search(query_vector, k_val)
                    for idx in indices[0]:
                        if idx != -1:
                            dense_results.append(int(idx))
            except Exception as e:
                logger.error("Error during FAISS retrieval search: %s", e)
        return dense_results

    def _sparse_search(self, query: str, top_k: int) -> List[int]:
        """Performs sparse lexical search using BM25 and returns indices."""
        sparse_results: List[int] = []
        if self.bm25_index is not None:
            try:
                tokenized_query = tokenize_text(query)
                scores = self.bm25_index.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1]
                for idx in top_indices:
                    # Ignore zero or negative scores
                    if scores[idx] <= 0:
                        break
                    sparse_results.append(int(idx))
                    if len(sparse_results) >= top_k:
                        break
            except Exception as e:
                logger.error("Error during BM25 retrieval search: %s", e)
        return sparse_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[int],
        sparse_results: List[int],
        rrf_k: int
    ) -> List[tuple]:
        """Fuses dense and sparse results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[int, float] = {}

        # Update scores from dense rank
        for rank, doc_idx in enumerate(dense_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (
                rrf_k + (rank + 1)
            )

        # Update scores from sparse rank
        for rank, doc_idx in enumerate(sparse_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (
                rrf_k + (rank + 1)
            )

        # Sort combined results by RRF score descending
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _build_retrieved_chunks(
        self,
        rrf_results: List[tuple],
        final_top_k: int
    ) -> List[RetrievedChunk]:
        """Maps top RRF results to RetrievedChunk dataclass instances."""
        top_results = rrf_results[:final_top_k]
        retrieved_chunks: List[RetrievedChunk] = []

        for doc_idx, score in top_results:
            if doc_idx < 0 or doc_idx >= len(self.metadata):
                logger.error(
                    "Index %d out of bounds for metadata size %d",
                    doc_idx,
                    len(self.metadata)
                )
                continue
            meta = self.metadata[doc_idx]
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    brand=meta["brand"],
                    model=meta["model"],
                    year=meta["year"],
                    page_number=meta["page_number"],
                    section=meta["section"],
                    subsection=meta["subsection"],
                    keywords=meta["keywords"],
                    source_file=meta["source_file"],
                    text=meta["text"],
                    score=score
                )
            )
        return retrieved_chunks

    def retrieve(
        self,
        query: str,
        faiss_top_k: int = config.FAISS_TOP_K,
        bm25_top_k: int = config.BM25_TOP_K,
        final_top_k: int = config.FINAL_TOP_K,
        rrf_k: int = config.RRF_K
    ) -> List[RetrievedChunk]:
        """Performs hybrid dense and sparse search and merges results using RRF."""
        if not query:
            return []

        # Graceful handling for missing metadata or indices
        if not self.metadata or (self.faiss_index is None and self.bm25_index is None):
            logger.warning(
                "No index or metadata loaded. Returning empty retrieval list."
            )
            return []

        # Orchestrate the private search, fusion and build steps
        dense_results = self._dense_search(query, faiss_top_k)
        sparse_results = self._sparse_search(query, bm25_top_k)
        rrf_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, rrf_k
        )
        retrieved_chunks = self._build_retrieved_chunks(rrf_results, final_top_k)

        logger.info(
            "Hybrid retrieval successfully retrieved %d chunks for query",
            len(retrieved_chunks)
        )
        return retrieved_chunks

    def retrieve_for_vehicle(
        self,
        query: str,
        brand: str,
        model: str,
        faiss_top_k: int = config.FAISS_TOP_K,
        bm25_top_k: int = config.BM25_TOP_K,
        final_top_k: int = config.FINAL_TOP_K,
        rrf_k: int = config.RRF_K
    ) -> List[RetrievedChunk]:
        """Performs hybrid search filtering results to a specific vehicle (brand and model)."""
        if not query or not brand or not model:
            return []

        # Graceful handling for missing metadata or indices
        if not self.metadata or (self.faiss_index is None and self.bm25_index is None):
            logger.warning(
                "No index or metadata loaded. Returning empty retrieval list."
            )
            return []

        # 1. Dense search filtered by vehicle brand/model
        dense_results: List[int] = []
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            try:
                query_embeddings = self.embedding_generator.generate_embeddings([query])
                if query_embeddings:
                    query_vector = np.array(query_embeddings).astype("float32")
                    # Search all vectors in index to locate matching brand/model elements
                    distances, indices = self.faiss_index.search(query_vector, self.faiss_index.ntotal)
                    for idx in indices[0]:
                        if idx != -1:
                            meta = self.metadata[idx]
                            if (
                                meta["brand"].lower() == brand.lower()
                                and meta["model"].lower() == model.lower()
                            ):
                                dense_results.append(int(idx))
                                if len(dense_results) >= faiss_top_k:
                                    break
            except Exception as e:
                logger.error("Error during vehicle-filtered FAISS search: %s", e)

        # 2. Sparse search filtered by vehicle brand/model
        sparse_results: List[int] = []
        if self.bm25_index is not None:
            try:
                tokenized_query = tokenize_text(query)
                scores = self.bm25_index.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1]
                for idx in top_indices:
                    if scores[idx] <= 0:
                        break
                    meta = self.metadata[idx]
                    if (
                        meta["brand"].lower() == brand.lower()
                        and meta["model"].lower() == model.lower()
                    ):
                        sparse_results.append(int(idx))
                        if len(sparse_results) >= bm25_top_k:
                            break
            except Exception as e:
                logger.error("Error during vehicle-filtered BM25 search: %s", e)

        # 3. Apply RRF and construct chunks
        rrf_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, rrf_k
        )
        retrieved_chunks = self._build_retrieved_chunks(rrf_results, final_top_k)

        logger.info(
            "Vehicle-filtered retrieval successfully retrieved %d chunks for query (Vehicle: %s %s)",
            len(retrieved_chunks),
            brand,
            model
        )
        return retrieved_chunks

