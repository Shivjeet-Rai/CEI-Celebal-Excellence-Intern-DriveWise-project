"""Cross-Encoder Reranking Module.

Reranks retrieved document chunks using a lightweight Cross-Encoder model
(defaulting to cross-encoder/ms-marco-MiniLM-L-6-v2) to prioritize the
most relevant brochure sections. Runs on fastembed's ONNX-backed
TextCrossEncoder — no torch dependency.
"""

import logging
from typing import List

from fastembed.rerank.cross_encoder import TextCrossEncoder

import config
from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class BrochureReranker:
    """Reranks candidates retrieved by HybridRetriever using a Cross-Encoder."""

    def __init__(self, model_name: str = config.RERANKER_MODEL) -> None:
        """Stores model name and lazy-loads the CrossEncoder only when needed."""
        self.model_name = model_name
        self.model = None

    def _load_model(self) -> None:
        """Loads the CrossEncoder model only on first use."""
        if self.model is None:
            logger.info("Loading Cross-Encoder reranker: %s", self.model_name)
            self.model = TextCrossEncoder(model_name=self.model_name)

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

        # Lazy-load the model only when actually required
        self._load_model()

        logger.info(
            "Reranking %d chunks using Cross-Encoder", len(retrieved_chunks)
        )

        # Prepare documents for reranking
        documents = [chunk.text for chunk in retrieved_chunks]

        # Predict relevance scores
        scores = list(self.model.rerank(query, documents))

        # Assign scores
        for chunk, score in zip(retrieved_chunks, scores):
            chunk.score = float(score)

        # Return top ranked chunks
        reranked_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x.score,
            reverse=True
        )

        logger.info(
            "Reranking complete. Returning top %d chunks",
            min(top_n, len(reranked_chunks))
        )

        return reranked_chunks[:top_n]