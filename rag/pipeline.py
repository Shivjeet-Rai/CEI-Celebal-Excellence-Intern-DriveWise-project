"""Orchestration Pipeline Module.

Combines retrieval, reranking, and Gemini response generation into a single
public interface.
"""

import logging
import os
import re
from typing import Any, Dict, List

import config
from rag.generator import BrochureGenerator
from rag.reranker import BrochureReranker
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Set ENABLE_RERANKER=false in Render's environment variables to skip
# reranker loading under low-memory conditions (e.g. free-tier hosting).
# All reranking code remains fully intact and functional either way —
# this only controls whether it's invoked at runtime.
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"


class BrochureRAGPipeline:
    """Orchestrates RAG components."""

    def __init__(self) -> None:
        """Lazy initialization of pipeline components."""
        logger.info("Initializing BrochureRAGPipeline (lazy mode)...")

        self.retriever = None
        self.reranker = None
        self.generator = None

        logger.info(
            "Pipeline initialized successfully. Reranker enabled: %s",
            ENABLE_RERANKER,
        )

    def _load_components(self) -> None:
        """Loads heavy components only when first needed."""

        if self.retriever is None:
            logger.info("Loading retriever...")
            self.retriever = HybridRetriever()

        if ENABLE_RERANKER and self.reranker is None:
            logger.info("Loading reranker...")
            self.reranker = BrochureReranker()

        if self.generator is None:
            logger.info("Loading generator...")
            self.generator = BrochureGenerator()

    def _rerank_or_passthrough(
        self,
        query: str,
        chunks: List[Any],
    ) -> List[Any]:
        """Applies reranking if enabled, otherwise returns top-N retrieved chunks as-is."""
        if ENABLE_RERANKER and self.reranker is not None:
            return self.reranker.rerank(query, chunks)

        logger.info(
            "Reranker disabled (ENABLE_RERANKER=false) — "
            "returning top %d retrieved chunks without reranking.",
            config.TOP_RERANKED_CHUNKS,
        )
        return chunks[: config.TOP_RERANKED_CHUNKS]

    def _extract_matched_vehicles(self, query_str: str) -> List[tuple]:
        """Extracts unique vehicles (brand, model) mentioned in the query."""

        if not self.retriever.metadata:
            return []

        available_vehicles = set()

        for chunk in self.retriever.metadata:
            available_vehicles.add(
                (
                    chunk["brand"].lower().strip(),
                    chunk["model"].lower().strip(),
                )
            )

        query_lower = query_str.lower()

        matched = []
        seen = set()

        for brand, model in available_vehicles:

            pattern = rf"\b{re.escape(model)}\b"

            if re.search(pattern, query_lower):

                key = (brand, model)

                if key not in seen:

                    seen.add(key)

                    orig_brand = None
                    orig_model = None

                    for chunk in self.retriever.metadata:

                        if (
                            chunk["brand"].lower().strip() == brand
                            and chunk["model"].lower().strip() == model
                        ):

                            orig_brand = chunk["brand"].strip()
                            orig_model = chunk["model"].strip()
                            break

                    matched.append(
                        (
                            orig_brand or brand,
                            orig_model or model,
                        )
                    )

        return matched

    def ask(self, query: str) -> Dict[str, Any]:
        """Runs the complete RAG pipeline."""

        fallback_response = {
            "answer": "I could not find this information in the provided brochures.",
            "sources": [],
        }

        if not query or not query.strip():
            logger.warning("Empty query received.")
            return fallback_response

        query_str = query.strip()

        # Load heavy models only when a user actually asks something
        self._load_components()

        logger.info("Query received: %s", query_str)

        query_lower = query_str.lower()

        comparison_triggers = [
            "compare",
            "vs",
            "versus",
            "difference between",
        ]

        is_comparison = any(
            trigger in query_lower
            for trigger in comparison_triggers
        )

        try:

            if is_comparison:

                matched_vehicles = self._extract_matched_vehicles(query_str)

                if matched_vehicles:

                    logger.info(
                        "Comparison mode: %s",
                        matched_vehicles,
                    )

                    all_chunks = []

                    for brand, model in matched_vehicles:

                        retrieved = self.retriever.retrieve_for_vehicle(
                            query_str,
                            brand,
                            model,
                        )

                        reranked = self._rerank_or_passthrough(
                            query_str,
                            retrieved,
                        )

                        all_chunks.extend(reranked)

                    if not all_chunks:
                        return fallback_response

                    return self.generator.generate_comparison(
                        query_str,
                        all_chunks,
                    )

            retrieved_chunks = self.retriever.retrieve(query_str)

            logger.info(
                "Retrieved %d chunks",
                len(retrieved_chunks),
            )

            if not retrieved_chunks:
                return fallback_response

            reranked_chunks = self._rerank_or_passthrough(
                query_str,
                retrieved_chunks,
            )

            logger.info(
                "Reranked to %d chunks",
                len(reranked_chunks),
            )

            if not reranked_chunks:
                return fallback_response

            return self.generator.generate_answer(
                query_str,
                reranked_chunks,
            )

        except Exception as e:

            logger.error(
                "Pipeline execution failed: %s",
                e,
            )

            return fallback_response