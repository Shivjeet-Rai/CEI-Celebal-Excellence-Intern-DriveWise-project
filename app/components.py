"""UI Components Module.

Provides reusable components and widgets for the Streamlit interface,
such as chat messages, source citations, and sidebars.
"""

from typing import Any, Dict, List

import streamlit as st


def render_sidebar_filters(available_vehicles: List[tuple] = None) -> Dict[str, Any]:
    """Renders the sidebar with filters for Brand, Model, and Year."""
    st.sidebar.header("Filter Brochures")

    if available_vehicles:
        brands = sorted(set(v[0] for v in available_vehicles))
        brand = st.sidebar.selectbox("Brand", ["All Brands"] + brands)

        if brand != "All Brands":
            models = sorted(set(v[1] for v in available_vehicles if v[0] == brand))
        else:
            models = sorted(set(v[1] for v in available_vehicles))
        model = st.sidebar.selectbox("Model", ["All Models"] + models)
    else:
        brand = st.sidebar.selectbox(
            "Brand",
            ["All Brands", "Hyundai", "Kia", "Mahindra", "Toyota"]
        )
        model = st.sidebar.text_input("Model", placeholder="e.g. Alcazar, Seltos...")

    year = st.sidebar.slider("Year", min_value=2020, max_value=2026, value=(2020, 2026))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About DriveWise**")
    st.sidebar.markdown(
        "DriveWise is a metadata-aware automotive assistant. "
        "It queries official brochure PDFs to answer technical specifications "
        "and details."
    )

    return {
        "brand": brand,
        "model": model,
        "year": year
    }


def render_chat_message(
    role: str,
    content: str,
    sources: List[Dict[str, Any]] = None
) -> None:
    """Renders a single chat message with optional source citations."""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            # Fix literal escape sequences and <br> tags coming from Gemini's
            # JSON output so Markdown tables and line breaks render correctly.
            display_content = content.replace("\\n", "\n").replace("<br>", "  \n")
            st.markdown(display_content, unsafe_allow_html=True)

            # Renders clean cited sources underneath assistant responses
            if sources:
                st.markdown("### Sources cited:")
                for idx, src in enumerate(sources, 1):
                    brand = src.get("brand", "Unknown")
                    model = src.get("model", "Unknown")
                    page = src.get("page_number", "?")
                    filename = src.get("source_file", "Unknown")
                    st.markdown(
                        f"**{idx}. [{brand} {model}]** Page {page} *(File: {filename})*"
                    )