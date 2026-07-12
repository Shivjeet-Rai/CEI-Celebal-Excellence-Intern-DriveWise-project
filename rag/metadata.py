"""Metadata Extraction Module.

Extracts structured metadata, keywords, and subsections from text chunks.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from rag.utils import safe_json_write

logger = logging.getLogger(__name__)

# Lexicon mapping search terms to display-ready keywords
KEYWORD_MAPPINGS: Dict[str, str] = {
    # Engine & Fuel Types
    "mstallion": "mStallion",
    "mhawk": "mHawk",
    "turbo petrol": "Turbo Petrol",
    "diesel": "Diesel",
    "u2 crdi": "U2 CRDi",
    "turbo gdi": "Turbo GDi",
    "hybrid": "Hybrid",
    "hev": "HEV",
    "petrol": "Petrol",
    "electric": "Electric",
    "ev": "EV",
    # ADAS
    "adas": "ADAS",
    "smartsense": "SmartSense",
    "level 2 adas": "Level 2 ADAS",
    "lane keep assist": "Lane Keep Assist",
    "cruise control": "Cruise Control",
    "collision avoidance": "Collision Avoidance",
    "automatic emergency braking": "Automatic Emergency Braking",
    # Airbags
    "6 airbags": "6 Airbags",
    "airbag": "Airbags",
    "dual airbags": "Dual Airbags",
    "airbags": "Airbags",
    # Dimensions
    "length": "Length",
    "width": "Width",
    "height": "Height",
    "wheelbase": "Wheelbase",
    "mm": "mm",
    "ground clearance": "Ground Clearance",
    "boot space": "Boot Space",
    "kerb weight": "Kerb Weight",
    # Power & Torque
    "hp": "hp",
    "ps": "PS",
    "kw": "kW",
    "max power": "Max Power",
    "nm": "Nm",
    "max torque": "Max Torque",
    # Transmission
    "manual": "Manual",
    "automatic": "Automatic",
    "mt": "MT",
    "at": "AT",
    "dct": "DCT",
    "ivt": "IVT",
    "cvt": "CVT",
    "torque converter": "Torque Converter",
    "dsg": "DSG",
    # Infotainment & Technology
    "touch screen": "Touch Screen",
    "android auto": "Android Auto",
    "apple carplay": "Apple CarPlay",
    "navigation": "Navigation",
    "display": "Display",
    "bluetooth": "Bluetooth",
    "infotainment": "Infotainment",
    # Camera & Sensors
    "rear camera": "Rear Camera",
    "surround view": "Surround View",
    "360 camera": "360 Camera",
    "parking sensor": "Parking Sensors",
    "parking sensors": "Parking Sensors",
    # Drive Modes
    "sport mode": "Sport Mode",
    "eco mode": "Eco Mode",
    "normal": "Normal Mode",
    "mud": "Mud Mode",
    "sand": "Sand Mode",
    "snow": "Snow Mode",
    "drive modes": "Drive Modes",
    # Safety Technologies
    "abs": "ABS",
    "ebd": "EBD",
    "esc": "ESC",
    "vsm": "VSM",
    "hac": "HAC",
    "hdc": "HDC",
    "tpms": "TPMS",
    "disc brakes": "Disc Brakes",
    # Technical Specifications
    "suspension": "Suspension",
    "mcpherson strut": "McPherson Strut",
    "ctba": "CTBA",
    "multi-link": "Multi-link"
}


@dataclass
class EnrichedBrochureChunk:
    """Dataclass representing a text chunk enriched with structured metadata."""
    chunk_id: str
    brand: str
    model: str
    year: int
    document_name: str
    page_number: int
    section: str
    text: str
    word_count: int
    manufacturer: str
    vehicle: str
    source_file: str
    source_type: str
    keywords: List[str]
    subsection: str
    metadata_version: str


CASE_SENSITIVE_KEYWORDS = {
    "MT", "AT", "DCT", "IVT", "CVT", "DSG", "HEV", "EV",
    "ABS", "EBD", "ESC", "VSM", "HAC", "HDC", "TPMS", "ADAS",
    "PS", "kW", "Nm"
}


def extract_keywords(text: str) -> List[str]:
    """Extracts meaningful unique automotive terms using word-boundary regexes."""
    extracted = set()

    for search_term, display_term in KEYWORD_MAPPINGS.items():
        if display_term in CASE_SENSITIVE_KEYWORDS:
            # Case-sensitive match for abbreviations
            pattern = r"\b" + re.escape(display_term) + r"\b"
            if re.search(pattern, text):
                extracted.add(display_term)
        else:
            # Case-insensitive match for normal terms
            sanitized_term = re.escape(search_term).replace(r"\ ", r"\s+")
            pattern = r"\b" + sanitized_term + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                extracted.add(display_term)

    return sorted(list(extracted))


def infer_subsection(section: str, text: str) -> str:
    """Infers the subsection category based on main section and content keyword rules."""
    text_lower = text.lower()

    if section == "Safety":
        if any(w in text_lower for w in ["adas", "smartsense", "lane", "cruise", "collision", "lka", "lfa", "scc", "fca"]):
            return "ADAS"
        if any(w in text_lower for w in ["airbag", "curtain"]):
            return "Airbags"
        if any(w in text_lower for w in ["abs", "ebd", "esc", "vsm", "brake", "disc"]):
            return "Braking"

    elif section == "Technology":
        if any(w in text_lower for w in ["infotainment", "touch screen", "screen", "bose", "jbl", "sound", "speaker", "audio", "music"]):
            return "Infotainment"
        if any(w in text_lower for w in ["android auto", "apple carplay", "bluelink", "alexa", "wireless", "connectivity", "bluetooth", "telematics"]):
            return "Connectivity"
        if any(w in text_lower for w in ["digital cluster", "instrument", "mid", "tft", "multi display"]):
            return "Digital Cluster"

    elif section == "Engine":
        if any(w in text_lower for w in ["hybrid", "hev"]):
            return "Hybrid"
        if any(w in text_lower for w in ["petrol", "gdi", "mstallion"]):
            return "Petrol"
        if any(w in text_lower for w in ["diesel", "crdi", "mhawk"]):
            return "Diesel"
        if any(w in text_lower for w in ["transmission", "gearbox", "mt", "at", "dct", "cvt", "ivt", "clutch", "manual", "automatic"]):
            return "Transmission"

    return "General"


def validate_chunk(chunk: EnrichedBrochureChunk) -> bool:
    """Validates that all required fields are present, properly typed, and non-empty."""
    required_strs = [
        chunk.chunk_id, chunk.brand, chunk.model, chunk.document_name,
        chunk.section, chunk.text, chunk.manufacturer, chunk.vehicle,
        chunk.source_file, chunk.source_type, chunk.subsection,
        chunk.metadata_version
    ]
    if any(not isinstance(s, str) or not s for s in required_strs):
        return False

    if not isinstance(chunk.year, int) or chunk.year <= 0:
        return False
    if not isinstance(chunk.page_number, int) or chunk.page_number < 0:
        return False
    if not isinstance(chunk.word_count, int) or chunk.word_count < 0:
        return False

    if not isinstance(chunk.keywords, list):
        return False
    if any(not isinstance(kw, str) or not kw for kw in chunk.keywords):
        return False

    return True


class MetadataEnricher:
    """Enriches vehicle brochure chunk dictionaries with structured metadata."""

    def __init__(self, enriched_dir: Union[str, Path]) -> None:
        """Initializes the enricher with the output enriched directory."""
        self.enriched_dir = Path(enriched_dir)

    def enrich_chunk(self, chunk_data: Dict[str, Any]) -> EnrichedBrochureChunk:
        """Enriches a single chunk dict with automotive keywords and subsections."""
        brand = chunk_data.get("brand", "Unknown")
        model = chunk_data.get("model", "Unknown")
        doc_name = chunk_data.get("document_name", "Unknown")
        section = chunk_data.get("section", "Miscellaneous")
        text = chunk_data.get("text", "")

        keywords = extract_keywords(text)
        subsection = infer_subsection(section, text)

        return EnrichedBrochureChunk(
            chunk_id=chunk_data.get("chunk_id", ""),
            brand=brand,
            model=model,
            year=chunk_data.get("year", 2025),
            document_name=doc_name,
            page_number=chunk_data.get("page_number", 0),
            section=section,
            text=text,
            word_count=chunk_data.get("word_count", 0),
            manufacturer=brand,
            vehicle=model,
            source_file=doc_name,
            source_type="pdf",
            keywords=keywords,
            subsection=subsection,
            metadata_version="1.0.0"
        )

    def enrich_document_chunks(self, chunk_file_path: Union[str, Path]) -> Path:
        """Loads chunks, enriches each one, validates schemas, and saves JSON."""
        path = Path(chunk_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
        except Exception as e:
            logger.error("Failed to parse chunk JSON %s: %s", path.name, e)
            raise ValueError(f"Malformed chunk JSON {path.name}: {e}") from e

        enriched_chunks = []
        for idx, chunk_dict in enumerate(chunks_data):
            enriched = self.enrich_chunk(chunk_dict)
            if not validate_chunk(enriched):
                raise ValueError(
                    f"Chunk at index {idx} in {path.name} failed validation."
                )
            enriched_chunks.append(asdict(enriched))

        destination_path = self.enriched_dir / path.name
        safe_json_write(enriched_chunks, destination_path)
        logger.info(
            "Saved %d enriched chunks to %s",
            len(enriched_chunks),
            destination_path.name
        )
        return destination_path
