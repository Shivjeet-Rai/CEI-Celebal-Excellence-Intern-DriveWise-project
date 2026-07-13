"""FAISS Retriever Module.

Performs dense semantic retrieval over brochure chunks using FAISS.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import faiss
import numpy as np

import config
from rag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved brochure chunk."""

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
    """Dense FAISS retriever."""

    def __init__(self, store_dir: Union[str, Path] = config.VECTORSTORE_DIR) -> None:
        self.store_dir = Path(store_dir)

        self.index_path = self.store_dir / "index.faiss"
        self.metadata_path = self.store_dir / "metadata.json"

        self.faiss_index: Any = None
        self.metadata: List[Dict[str, Any]] = []

        self.embedding_generator = EmbeddingGenerator(
            model_name=config.EMBEDDING_MODEL
        )

        self._load_indices()

    def _load_indices(self) -> None:
        """Loads FAISS index and metadata."""

        try:

            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                logger.info(
                    "Loaded %d metadata entries",
                    len(self.metadata),
                )

            else:
                logger.warning(
                    "Metadata file not found: %s",
                    self.metadata_path,
                )

            if self.index_path.exists():

                self.faiss_index = faiss.read_index(
                    str(self.index_path)
                )

                logger.info(
                    "Loaded FAISS index with %d vectors",
                    self.faiss_index.ntotal,
                )

            else:
                logger.warning(
                    "FAISS index not found: %s",
                    self.index_path,
                )

        except Exception as e:
            logger.error(
                "Failed loading FAISS index: %s",
                e,
            )

    def _dense_search(self, query: str, top_k: int) -> List[int]:
        """Performs dense semantic search using FAISS."""

        dense_results: List[int] = []

        if self.faiss_index is None:
            return dense_results

        if self.faiss_index.ntotal == 0:
            return dense_results

        try:
            query_embeddings = self.embedding_generator.generate_embeddings([query])

            if not query_embeddings:
                return dense_results

            query_vector = np.array(query_embeddings).astype("float32")

            k = min(top_k, self.faiss_index.ntotal)

            _, indices = self.faiss_index.search(query_vector, k)

            for idx in indices[0]:
                if idx != -1:
                    dense_results.append(int(idx))

        except Exception as e:
            logger.error("Dense retrieval failed: %s", e)

        return dense_results

    def _build_retrieved_chunks(
        self,
        indices: List[int],
        final_top_k: int
    ) -> List[RetrievedChunk]:
        """Build RetrievedChunk objects from FAISS indices."""

        retrieved_chunks: List[RetrievedChunk] = []

        for idx in indices[:final_top_k]:

            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

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
                    score=1.0,
                )
            )

        return retrieved_chunks


    def retrieve(
        self,
        query: str,
        faiss_top_k: int = config.FAISS_TOP_K,
        final_top_k: int = config.FINAL_TOP_K,
    ) -> List[RetrievedChunk]:

        if not query:
            return []

        if self.faiss_index is None or not self.metadata:
            return []

        dense_results = self._dense_search(query, faiss_top_k)

        retrieved_chunks = self._build_retrieved_chunks(
            dense_results,
            final_top_k,
        )

        logger.info(
            "Retrieved %d chunks",
            len(retrieved_chunks),
        )

        return retrieved_chunks


    def retrieve_for_vehicle(
        self,
        query: str,
        brand: str,
        model: str,
        faiss_top_k: int = config.FAISS_TOP_K,
        final_top_k: int = config.FINAL_TOP_K,
    ) -> List[RetrievedChunk]:

        if not query:
            return []

        if self.faiss_index is None or not self.metadata:
            return []

        dense_results = self._dense_search(
            query,
            self.faiss_index.ntotal,
        )

        filtered_indices = []

        for idx in dense_results:

            meta = self.metadata[idx]

            if (
                meta["brand"].lower() == brand.lower()
                and meta["model"].lower() == model.lower()
            ):
                filtered_indices.append(idx)

                if len(filtered_indices) >= faiss_top_k:
                    break

        retrieved_chunks = self._build_retrieved_chunks(
            filtered_indices,
            final_top_k,
        )

        logger.info(
            "Retrieved %d chunks for %s %s",
            len(retrieved_chunks),
            brand,
            model,
        )

        return retrieved_chunks

