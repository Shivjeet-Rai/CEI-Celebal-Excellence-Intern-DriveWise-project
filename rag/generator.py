"""Response Generator Module.

Interacts with the Google Gemini API (defaulting to gemini-2.5-flash) to generate
human-like, accurate answers based on the retrieved context passages and system prompt.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field

import config
from rag.retriever import RetrievedChunk

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

# Configure Google Generative AI API client
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GEMINI_API_KEY not found in environment or .env file.")


# Define Pydantic models for structured output schema
class SourceInfo(BaseModel):
    brand: str = Field(
        description="The manufacturer or brand of the vehicle."
    )
    model: str = Field(description="The vehicle model name.")
    page_number: int = Field(
        description="The brochure page number where information was found."
    )
    source_file: str = Field(
        description="The PDF source brochure filename."
    )


class AnswerResponse(BaseModel):
    answer: str = Field(
        description="The factual answer based strictly on the provided context."
    )
    sources: List[SourceInfo] = Field(
        description="List of brochure chunks cited to support the answer."
    )


class ComparisonResponse(BaseModel):
    answer: str = Field(
        description=(
            "The factual comparison answer. This MUST be formatted as a markdown "
            "comparison table comparing the vehicles across key specifications/features. "
            "If a specification is not available in the brochure context for a vehicle, "
            "mark it as 'Unavailable' in the table. Keep the output neat, professional, "
            "and strictly grounded in the context."
        )
    )
    sources: List[SourceInfo] = Field(
        description="List of brochure chunks cited to support the comparison."
    )



class BrochureGenerator:
    """Answers user queries grounded in brochure chunks using Gemini."""

    def __init__(self, model_name: str = config.LLM_MODEL) -> None:
        """Initializes the generator and loads the system prompt instructions."""
        self.model_name = model_name
        self.system_prompt = ""

        # Locate and load the system prompt file
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "prompts"
            / "system_prompt.txt"
        )
        if not prompt_path.exists():
            prompt_path = Path("prompts/system_prompt.txt")

        if prompt_path.exists():
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self.system_prompt = f.read().strip()
                logger.info("Loaded system prompt from: %s", prompt_path)
            except Exception as e:
                logger.error("Failed to read system prompt file: %s", e)
        else:
            logger.warning(
                "System prompt file not found at: %s. Using default fallback.",
                prompt_path
            )
            self.system_prompt = (
                "You are DriveWise, a modular metadata-aware automotive assistant. "
                "Answer ONLY from the provided brochure context. If the answer cannot "
                "be found, reply: 'I could not find this information in the provided brochures.'"
            )

        # Initialize GenerativeModel client with system instruction
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt
        )

    def generate_answer(
        self,
        query: str,
        reranked_chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """Generates a structured, grounded answer from the reranked brochure chunks."""
        # Handle empty context or query safely
        fallback_answer = {
            "answer": "I could not find this information in the provided brochures.",
            "sources": []
        }

        if not query:
            return fallback_answer

        if not reranked_chunks:
            logger.info("Empty context list provided. Returning fallback answer.")
            return fallback_answer

        # Construct context text preserving key metadata attributes
        context_blocks = []
        for i, chunk in enumerate(reranked_chunks, 1):
            block = (
                f"--- CONTEXT BLOCK {i} ---\n"
                f"Brand: {chunk.brand}\n"
                f"Model: {chunk.model}\n"
                f"Year: {chunk.year}\n"
                f"Page Number: {chunk.page_number}\n"
                f"Section: {chunk.section}\n"
                f"Source File: {chunk.source_file}\n"
                f"Text:\n{chunk.text}\n"
            )
            context_blocks.append(block)

        context_text = "\n".join(context_blocks)
        prompt_payload = f"User Query: {query}\n\nBrochure Context:\n{context_text}"

        try:
            logger.info("Requesting grounded answer from Gemini API...")
            # Request Structured JSON output using GenerationConfig
            response = self.model.generate_content(
                prompt_payload,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=AnswerResponse
                )
            )

            if not response.text:
                logger.error("Received empty response text from Gemini API.")
                return fallback_answer

            # Parse and return structured JSON dictionary
            return json.loads(response.text)

        except Exception as e:
            logger.error("Error during answer generation or JSON parsing: %s", e)
            return fallback_answer

    def generate_comparison(
        self,
        query: str,
        comparison_chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """Generates a structured comparison of vehicles using a markdown table grounded in brochure context."""
        fallback_answer = {
            "answer": "I could not find this information in the provided brochures.",
            "sources": []
        }

        if not query:
            return fallback_answer

        if not comparison_chunks:
            logger.info("Empty context list provided for comparison. Returning fallback answer.")
            return fallback_answer

        # Group chunks by vehicle to create a structured context section
        grouped_chunks: Dict[str, List[RetrievedChunk]] = {}
        for chunk in comparison_chunks:
            key = f"{chunk.brand} {chunk.model}"
            if key not in grouped_chunks:
                grouped_chunks[key] = []
            grouped_chunks[key].append(chunk)

        context_parts = []
        for vehicle_key, chunks in grouped_chunks.items():
            context_parts.append(f"=== VEHICLE: {vehicle_key} ===")
            for i, chunk in enumerate(chunks, 1):
                block = (
                    f"Block {i} (Page {chunk.page_number}, File {chunk.source_file}):\n"
                    f"{chunk.text}\n"
                )
                context_parts.append(block)

        context_text = "\n".join(context_parts)
        prompt_payload = (
            f"User Query: {query}\n\n"
            f"Instructions:\n"
            f"- Compare the vehicles mentioned in the query based ONLY on the provided brochure contexts.\n"
            f"- You MUST present the comparison using a markdown table formatting key features/specifications.\n"
            f"- If a detail is missing or not provided for any vehicle, clearly state it in the table as 'Unavailable' or 'Not in Brochure'.\n"
            f"- Cite brochure page numbers and source files directly inside your answers and inside the table.\n"
            f"- Never hallucinate any information. If no brochure details are available for any compared vehicle, state that the information is unavailable.\n\n"
            f"Brochure Context:\n{context_text}"
        )

        try:
            logger.info("Requesting grounded comparison from Gemini API...")
            response = self.model.generate_content(
                prompt_payload,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ComparisonResponse
                )
            )

            if not response.text:
                logger.error("Received empty response text from Gemini API during comparison.")
                return fallback_answer

            return json.loads(response.text)

        except Exception as e:
            logger.error("Error during comparison generation or JSON parsing: %s", e)
            return fallback_answer

