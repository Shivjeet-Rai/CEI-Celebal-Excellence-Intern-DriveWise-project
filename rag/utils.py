"""General Utilities Module.

Contains helper functions for parsing configurations, manipulating directories/files,
cleaning strings, and processing lists.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Union


def safe_directory_create(directory_path: Union[str, Path]) -> Path:
    """Safely creates a directory and any parent directories if they do not exist.

    Args:
        directory_path: The path of the directory to create.

    Returns:
        Path: The resolved Path object of the created directory.
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_write(data: Any, filepath: Union[str, Path]) -> None:
    """Safely writes serializable data to a pretty-printed JSON file in UTF-8.

    Args:
        data: The JSON-serializable data to write.
        filepath: The destination file path.
    """
    path = Path(filepath)
    # Ensure parent directory exists
    safe_directory_create(path.parent)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_timestamp() -> str:
    """Generates a standard timestamp string for logging.

    Returns:
        str: The ISO formatted local timestamp.
    """
    return datetime.now().isoformat()
