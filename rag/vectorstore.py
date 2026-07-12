"""Vector Store Interface.

Provides read/write wrappers around FAISS for storing chunk embeddings,
along with support for saving indices to disk.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import faiss
import numpy as np

from rag.utils import safe_directory_create

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """A vector store implementation backed by FAISS IndexFlatIP."""

    def __init__(self, store_dir: Union[str, Path], dimension: int = 384) -> None:
        """Initializes the FAISSVectorStore with a storage directory."""
        self.store_dir = Path(store_dir)
        self.dimension = dimension
        self.index_path = self.store_dir / "index.faiss"
        self.metadata_path = self.store_dir / "metadata.json"

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_chunks(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ) -> None:
        """Appends embeddings and their corresponding metadata to the store."""
        if not embeddings:
            return

        if len(embeddings) != len(metadata_list):
            raise ValueError(
                "Size mismatch: embeddings and metadata lists must be equal."
            )

        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)
        self.metadata.extend(metadata_list)
        logger.info("Added %d embeddings to the vector store", len(embeddings))

    def save(self) -> None:
        """Persists the FAISS index and metadata mapping to disk."""
        safe_directory_create(self.store_dir)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)

        logger.info(
            "Vector store successfully saved to %s (Vectors: %d)",
            self.store_dir,
            self.index.ntotal
        )
