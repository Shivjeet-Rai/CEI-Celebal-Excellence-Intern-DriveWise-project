"""DriveWise Application Entrypoint.

Launches the core Streamlit interface or command-line query assistant.
Coordinates UI component loads, config loading, and session states.
"""

import sys
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

from rag.pipeline import BrochureRAGPipeline


def run_cli() -> None:
    """CLI loop for the DriveWise brochure assistant."""
    print("\n" + "=" * 60)
    print("Welcome to the DriveWise RAG Assistant CLI!")
    print("Type your questions below. Type 'exit' or 'quit' to close.")
    print("=" * 60 + "\n")

    # Initialize the RAG pipeline
    try:
        pipeline = BrochureRAGPipeline()
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        sys.exit(1)

    while True:
        try:
            query = input("\nAsk DriveWise > ").strip()
            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print("\nThinking...")
            response = pipeline.ask(query)

            # Print structured answer
            print("\n" + "-" * 40)
            print("ANSWER:")
            print("-" * 40)
            print(response.get("answer", "No answer generated."))
            print("-" * 40)

            # Print sources
            sources = response.get("sources", [])
            if sources:
                print("SOURCES:")
                for idx, src in enumerate(sources, 1):
                    brand = src.get("brand", "Unknown")
                    model = src.get("model", "Unknown")
                    page = src.get("page_number", "?")
                    filename = src.get("source_file", "Unknown")
                    print(
                        f" {idx}. [{brand} {model}] Page {page} (File: {filename})"
                    )
            else:
                print("SOURCES: None cited.")
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def main() -> None:
    """Orchestrates runtime selection (Streamlit vs. CLI)."""
    # Detect if we are running within a Streamlit process
    if st.runtime.exists():
        from app.ui import main as run_ui
        run_ui()
    else:
        run_cli()


if __name__ == "__main__":
    main()
