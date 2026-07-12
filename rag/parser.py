"""Document Parser Module.

Responsible for reading PDF files (using PyMuPDF/fitz), extracting text,
extracting brand/model metadata, and serializing parsed documents as JSON.
"""

from dataclasses import dataclass, asdict
import fitz
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
import logging

from rag.utils import safe_json_write

logger = logging.getLogger(__name__)


@dataclass
class BrochureDocument:
    """Dataclass representing a parsed car brochure.

    Attributes:
        brand: The manufacturer brand (e.g., Hyundai).
        model: The vehicle model (e.g., Creta).
        year: The release/model year (e.g., 2025).
        document_name: The original PDF document filename.
        total_pages: Total number of pages in the original PDF.
        pages: List of dictionaries containing page_number and extracted text.
    """
    brand: str
    model: str
    year: int
    document_name: str
    total_pages: int
    pages: List[Dict[str, Any]]


class PDFParser:
    """Parser class for discovering, validating, and extracting content from car brochures."""

    def __init__(self, brochure_dir: Union[str, Path], processed_dir: Union[str, Path]) -> None:
        """Initializes the PDFParser with directories.

        Args:
            brochure_dir: Directory containing raw brochure PDFs.
            processed_dir: Directory where processed JSON files will be saved.
        """
        self.brochure_dir = Path(brochure_dir)
        self.processed_dir = Path(processed_dir)

    def discover_brochures(self) -> List[Path]:
        """Recursively searches the brochure directory for PDF files.

        Returns:
            List[Path]: A list of paths to discovered PDF brochures.
        """
        # Case-insensitive search for PDF files
        pdf_paths = [
            p for p in self.brochure_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        ]
        logger.info("Starting ingestion: found %d PDF brochures in %s", len(pdf_paths), self.brochure_dir)
        return pdf_paths

    def extract_metadata(self, pdf_path: Union[str, Path]) -> Tuple[str, str, int]:
        """Extracts brand, model, and year from a brochure PDF path.

        Args:
            pdf_path: The file path of the brochure PDF.

        Returns:
            Tuple[str, str, int]: A tuple of (brand, model, year).
        """
        path = Path(pdf_path)
        filename = path.stem
        parts = filename.split("_")
        
        # 1. Determine brand: check parent folder name first
        parent_dir = path.parent.name
        if parent_dir and parent_dir.lower() != "brochures" and parent_dir != ".":
            brand = parent_dir
        else:
            brand = parts[0] if parts else "Unknown"
            
        # 2. Extract year: check last part of filename
        year = 2025  # Default fallback
        if parts and parts[-1].isdigit():
            year = int(parts[-1])
            parts = parts[:-1]
            
        # 3. Determine model
        known_brands = {"hyundai", "kia", "mahindra", "toyota"}
        if parts:
            first_part_lower = parts[0].lower()
            # If the filename starts with the brand or another known brand, drop the prefix to clean model
            if first_part_lower == brand.lower() or first_part_lower in known_brands:
                parts = parts[1:]
                
        model = "_".join(parts)
        if not model:
            model = filename
            
        return brand, model, year

    def validate_pdf(self, pdf_path: Union[str, Path]) -> None:
        """Validates that a PDF file exists, has a .pdf extension, and can be opened.

        Args:
            pdf_path: The file path to validate.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .pdf or the PDF is corrupted.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
            
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid file extension (must be .pdf): {path.suffix}")
            
        try:
            doc = fitz.open(path)
            doc.close()
        except Exception as e:
            raise ValueError(f"Could not open PDF file with PyMuPDF: {e}") from e

    def parse_pdf(self, pdf_path: Union[str, Path]) -> BrochureDocument:
        """Parses a brochure PDF, extracting text page-by-page.

        Args:
            pdf_path: The path of the PDF file to parse.

        Returns:
            BrochureDocument: The structured document object.

        Raises:
            ValueError: If the PDF contains no pages or no extractable text.
        """
        path = Path(pdf_path)
        self.validate_pdf(path)
        
        brand, model, year = self.extract_metadata(path)
        
        doc = fitz.open(path)
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError(f"PDF document is empty (0 pages): {path.name}")
            
        pages_list = []
        for i in range(total_pages):
            page = doc[i]
            text = page.get_text().strip()
            if text:
                pages_list.append({
                    "page_number": i + 1,
                    "text": text
                })
                
        doc.close()
        
        if not pages_list:
            raise ValueError(f"PDF contains no extractable text: {path.name}")
            
        logger.info(
            "Current brochure: %s | Metadata extracted: brand=%s, model=%s, year=%d | Pages extracted: %d/%d",
            path.name, brand, model, year, len(pages_list), total_pages
        )
        
        return BrochureDocument(
            brand=brand,
            model=model,
            year=year,
            document_name=path.name,
            total_pages=total_pages,
            pages=pages_list
        )

    def save_json(self, document: BrochureDocument) -> Path:
        """Saves a BrochureDocument as a pretty-printed JSON file.

        Args:
            document: The document to serialize.

        Returns:
            Path: The resolved destination path of the written JSON file.
        """
        # Normalize name: brand_model_year.json
        clean_model = document.model.replace(" ", "_")
        json_filename = f"{document.brand}_{clean_model}_{document.year}.json"
        destination_path = self.processed_dir / json_filename
        
        # Write using utility
        safe_json_write(asdict(document), destination_path)
        logger.info("JSON saved: %s", destination_path.name)
        return destination_path
