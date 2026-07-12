"""Cross-Encoder Reranking Module.

Reranks retrieved document chunks using local Cross-Encoder models (defaulting to
BAAI/bge-reranker-base) to prioritize the most relevant facts.
"""

import logging
from typing import List

from sentence_transformers import CrossEncoder

import config
from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class BrochureReranker:
    """Reranks candidates retrieved by HybridRetriever using a Cross-Encoder."""

    def __init__(self, model_name: str = config.RERANKER_MODEL) -> None:
        """Initializes the CrossEncoder model."""
        logger.info("Loading Cross-Encoder reranker: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        top_n: int = config.TOP_RERANKED_CHUNKS
    ) -> List[RetrievedChunk]:
        """Reranks retrieved chunks for a query using Cross-Encoder scores."""
        if not retrieved_chunks:
            return []

        if not query:
            logger.warning(
                "Empty query provided for reranking. Returning inputs as-is."
            )
            return retrieved_chunks[:top_n]

        logger.info(
            "Reranking %d chunks using Cross-Encoder", len(retrieved_chunks)
        )

        # Prepare pairs for the Cross-Encoder: (query, text)
        pairs = [[query, chunk.text] for chunk in retrieved_chunks]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Update scores inside chunks
        for chunk, score in zip(retrieved_chunks, scores):
            chunk.score = float(score)

        # Sort chunks by score in descending order
        reranked_chunks = sorted(
            retrieved_chunks, key=lambda x: x.score, reverse=True
        )

        logger.info(
            "Reranking complete. Returning top %d chunks",
            min(top_n, len(reranked_chunks))
        )
        return reranked_chunks[:top_n]
