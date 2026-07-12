
"""Streamlit Application View.

Main view orchestrator for the Streamlit UI, coordinating layout,
chat session management, and rendering pipeline responses.
"""

import logging

import streamlit as st

from app.components import render_chat_message, render_sidebar_filters
from rag.pipeline import BrochureRAGPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="DriveWise – Automotive Brochure Assistant",
        page_icon="🚗",
        layout="wide"
    )

    st.title("DriveWise – Automotive Brochure Assistant")
    st.write(
        "Ask technical specification questions directly grounded "
        "in official manufacturer brochures."
    )

    # 1. Render sidebar filters
    render_sidebar_filters()

    # 2. Lazy-initialize pipeline in session state
    if "pipeline" not in st.session_state:
        with st.spinner("Initializing models... Please wait."):
            try:
                st.session_state.pipeline = BrochureRAGPipeline()
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
                logger.error("Failed to initialize RAG pipeline: %s", e)
                return

    # 3. Maintain chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources")
        )

    # React to user input
    if query := st.chat_input(
        "Ask about vehicle specs, safety features, colors..."
    ):
        # Display user message in chat message container
        render_chat_message("user", query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate assistant response
        try:
            with st.spinner("Consulting brochures..."):
                response = st.session_state.pipeline.ask(query)
                answer = response.get(
                    "answer",
                    "I could not find this information in the provided brochures."
                )
                sources = response.get("sources", [])

            # Display assistant response in chat message container
            render_chat_message("assistant", answer, sources)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )
        except Exception as e:
            logger.error("Error generating answer: %s", e)
            st.error(f"An error occurred: {e}")
