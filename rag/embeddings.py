"""Embeddings Generation Module.

Generates dense vector embeddings for text chunks using local sentence-transformer
models (defaulting to BAAI/bge-small-en-v1.5).
"""

import gc
import logging
from typing import List

import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Wrapper for sentence-transformers to generate text embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Initializes the SentenceTransformer model."""
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, device="cpu")
        try:
            self.model.half()  # fp16 weights, roughly halves resident memory
        except RuntimeError as e:
            logger.warning(
                "fp16 not supported for this op on CPU, falling back to fp32: %s", e
            )
        gc.collect()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized vector embeddings for a list of texts."""
        if not texts:
            return []

        logger.info("Generating embeddings for %d text segments", len(texts))
        # normalize_embeddings=True automatically normalizes outputs
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()