"""Streamlit Application View.

Main view orchestrator for the Streamlit UI, coordinating layout,
chat session management, and rendering pipeline responses.
"""

import logging

import streamlit as st

from app.components import render_chat_message, render_sidebar_filters
from rag.pipeline import BrochureRAGPipeline

logger = logging.getLogger(__name__)


@st.cache_resource
def get_pipeline() -> BrochureRAGPipeline:
    """Creates a single shared pipeline instance across all sessions."""
    return BrochureRAGPipeline()


def main() -> None:
    """Main Streamlit application entry point."""

    st.set_page_config(
        page_title="DriveWise – Automotive Brochure Assistant",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 DriveWise – Automotive Brochure Assistant")

    st.write(
        "Ask technical specification questions directly grounded "
        "in official manufacturer brochures."
    )

    # Shared pipeline instance across all users/sessions
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        logger.exception("Pipeline initialization failed")
        return

    # Load retriever/metadata now so we can show available vehicles.
    # Wrapped in try/except so any failure here doesn't crash the whole page.
    available_vehicles = []
    try:
        pipeline._load_components()
        if pipeline.retriever and pipeline.retriever.metadata:
            available_vehicles = sorted(set(
                (c["brand"], c["model"]) for c in pipeline.retriever.metadata
            ))
    except Exception:
        logger.exception("Could not preload metadata for vehicle list display")

    # Sidebar
    render_sidebar_filters(available_vehicles)

    # Show what's actually in the dataset so users know what to ask,
    # plus clickable example queries generated from real data.
    if available_vehicles:
        vehicle_names = [f"{b} {m}" for b, m in available_vehicles]
        with st.expander(f"📋 Available Vehicles ({len(vehicle_names)})", expanded=True):
            st.write(", ".join(vehicle_names))

        st.markdown("**Try asking:**")
        example_cols = st.columns(3)
        examples = [
            f"Compare {vehicle_names[0]} and {vehicle_names[1]}"
            if len(vehicle_names) > 1 else "Compare available cars",
            f"What safety features does {vehicle_names[0]} have?",
            f"Tell me the engine specs of {vehicle_names[-1]}",
        ]
        for col, example in zip(example_cols, examples):
            if col.button(example, use_container_width=True):
                st.session_state["pending_query"] = example

    # Chat history (kept per-session intentionally)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
        )

    # User input — always render the chat bar so it never disappears,
    # regardless of whether an example button was just clicked.
    typed_query = st.chat_input(
        "Ask about vehicle specs, safety features, colors..."
    )

    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")
    else:
        query = typed_query

    if query:

        render_chat_message("user", query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        try:

            with st.spinner("Searching brochures..."):

                response = pipeline.ask(query)

            answer = response.get(
                "answer",
                "I could not find this information in the provided brochures.",
            )

            sources = response.get("sources", [])

            render_chat_message(
                "assistant",
                answer,
                sources,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as e:

            logger.exception("Error generating answer")

            st.error(
                f"An error occurred while processing your question:\n\n{e}"
            )