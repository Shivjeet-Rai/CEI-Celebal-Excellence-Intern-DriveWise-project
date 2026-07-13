"""Embeddings Generation Module.

Generates dense vector embeddings for text chunks using a quantized ONNX
model via fastembed (defaulting to sentence-transformers/all-MiniLM-L6-v2).
No torch dependency — runs on ONNX Runtime for a much smaller memory footprint.
"""

import logging
from typing import List

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Wrapper for fastembed to generate text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initializes the fastembed TextEmbedding model (ONNX-backed)."""
        logger.info("Loading embedding model: %s", model_name)
        self.model = TextEmbedding(model_name=model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized vector embeddings for a list of texts."""
        if not texts:
            return []

        logger.info("Generating embeddings for %d text segments", len(texts))
        # fastembed's MiniLM models are already L2-normalized by default
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]