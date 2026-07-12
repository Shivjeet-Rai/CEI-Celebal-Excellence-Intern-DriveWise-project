"""Orchestration Pipeline Module.

Combines hybrid dense-sparse retrieval, Cross-Encoder reranking, and Gemini
response generation into a single public interface.
"""

import logging
import re
from typing import Any, Dict, List

from rag.generator import BrochureGenerator
from rag.reranker import BrochureReranker
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class BrochureRAGPipeline:
    """Orchestrates RAG components to answer brochure-grounded queries."""

    def __init__(self) -> None:
        """Initializes retrieval, reranking, and generation engines once."""
        logger.info("Initializing BrochureRAGPipeline components...")
        self.retriever = HybridRetriever()
        self.reranker = BrochureReranker()
        self.generator = BrochureGenerator()
        logger.info("BrochureRAGPipeline successfully initialized.")

    def _extract_matched_vehicles(self, query_str: str) -> List[tuple]:
        """Extracts unique vehicles (brand, model) mentioned in the query."""
        if not self.retriever.metadata:
            return []

        # Find all unique lowercased brands and models from metadata
        available_vehicles = set()
        for chunk in self.retriever.metadata:
            available_vehicles.add(
                (
                    chunk["brand"].lower().strip(),
                    chunk["model"].lower().strip()
                )
            )

        query_lower = query_str.lower()
        matched = []
        seen = set()

        for brand, model in available_vehicles:
            # We match model name with word boundaries
            pattern = rf"\b{re.escape(model)}\b"
            if re.search(pattern, query_lower):
                key = (brand, model)
                if key not in seen:
                    seen.add(key)
                    # Find original capitalization from metadata
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
                    matched.append((orig_brand or brand, orig_model or model))

        return matched

    def ask(self, query: str) -> Dict[str, Any]:
        """Runs the query through retrieval, reranking, and grounded LLM generation."""
        fallback_response = {
            "answer": "I could not find this information in the provided brochures.",
            "sources": []
        }

        if not query or not query.strip():
            logger.warning("Empty query received. Returning fallback response.")
            return fallback_response

        query_str = query.strip()
        logger.info("Query received: '%s'", query_str)

        # Check if the query is a comparison query
        query_lower = query_str.lower()
        comparison_triggers = ["compare", "vs", "versus", "difference between"]
        is_comparison = any(
            trigger in query_lower for trigger in comparison_triggers
        )

        try:
            if is_comparison:
                matched_vehicles = self._extract_matched_vehicles(query_str)
                if matched_vehicles:
                    logger.info(
                        "Comparison mode routed. Matched vehicles: %s",
                        matched_vehicles
                    )

                    all_comparison_chunks = []
                    for brand, model in matched_vehicles:
                        # Retrieve context independently for each vehicle
                        vehicle_retrieved = self.retriever.retrieve_for_vehicle(
                            query_str, brand, model
                        )
                        # Rerank retrieved chunks for the vehicle
                        vehicle_reranked = self.reranker.rerank(
                            query_str, vehicle_retrieved
                        )
                        all_comparison_chunks.extend(vehicle_reranked)

                    if not all_comparison_chunks:
                        return fallback_response

                    logger.info("Grounded comparison generation started.")
                    response = self.generator.generate_comparison(
                        query_str, all_comparison_chunks
                    )
                    logger.info("Grounded comparison generation finished.")
                    return response

            # Standard ask path for normal questions
            retrieved_chunks = self.retriever.retrieve(query_str)
            logger.info("Retrieved %d candidate chunks.", len(retrieved_chunks))

            if not retrieved_chunks:
                logger.info("No candidates retrieved. Returning fallback response.")
                return fallback_response

            reranked_chunks = self.reranker.rerank(query_str, retrieved_chunks)
            logger.info("Reranked to %d chunks.", len(reranked_chunks))

            if not reranked_chunks:
                logger.info("Reranking returned zero chunks. Returning fallback.")
                return fallback_response

            logger.info("Grounded answer generation started.")
            response = self.generator.generate_answer(query_str, reranked_chunks)
            logger.info("Grounded answer generation finished.")
            return response

        except Exception as e:
            logger.error("Error occurred during RAG pipeline execution: %s", e)
            return fallback_response
