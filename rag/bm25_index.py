"""BM25 Lexical Indexing Module.

Builds and persists a BM25 lexical index using the rank-bm25 library.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, List, Union

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text using simple lowercase whitespace tokenization."""
    return text.lower().split()


class BM25Indexer:
    """Orchestrates building and saving a BM25 index over brochure chunks."""

    def __init__(self, index_path: Union[str, Path]) -> None:
        """Initializes the BM25Indexer with destination path for the pickle file."""
        self.index_path = Path(index_path)
        self.bm25: Any = None

    def build_index(self, texts: List[str]) -> None:
        """Tokenizes text corpus and builds the BM25 Okapi index."""
        if not texts:
            logger.warning("Empty text corpus. Skipping BM25 index build.")
            return

        logger.info("Tokenizing %d documents for BM25 indexing", len(texts))
        tokenized_corpus = [tokenize_text(t) for t in texts]

        logger.info("Building BM25 Okapi index")
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self) -> None:
        """Saves the built BM25 index as a serialized pickle file."""
        if self.bm25 is None:
            logger.warning("No BM25 index built. Cannot save.")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)

        logger.info("BM25 index successfully saved to %s", self.index_path)
