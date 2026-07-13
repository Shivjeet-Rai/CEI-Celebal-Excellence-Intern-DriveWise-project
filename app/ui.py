"""Streamlit Application View.

Main view orchestrator for the Streamlit UI, coordinating layout,
chat session management, and rendering pipeline responses.
"""

import logging
import subprocess

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

    # TEMPORARY DIAGNOSTIC — remove after debugging
    with st.expander("🔧 Debug Info (temporary)"):
        try:
            result = subprocess.run(["pip", "show", "torch"], capture_output=True, text=True)
            st.code(result.stdout)
            result2 = subprocess.run(["pip", "list"], capture_output=True, text=True)
            nvidia_lines = [l for l in result2.stdout.splitlines() if "nvidia" in l.lower()]
            st.code("\n".join(nvidia_lines) if nvidia_lines else "No nvidia packages found")
        except Exception as e:
            st.error(f"Diagnostic failed: {e}")

    # Sidebar
    render_sidebar_filters()

    # Shared pipeline instance across all users/sessions
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        logger.exception("Pipeline initialization failed")
        return

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

    # User input
    query = st.chat_input(
        "Ask about vehicle specs, safety features, colors..."
    )

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