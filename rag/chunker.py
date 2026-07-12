"""Document Chunking Module.

Splits parsed brochure text from JSON files into structured, page-bounded
chunks with section detection heuristics.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from rag.utils import safe_json_write

logger = logging.getLogger(__name__)

# Heuristics keywords for 14 core sections (Miscellaneous is the fallback)
SECTION_KEYWORDS: Dict[str, List[str]] = {
    "Overview": [
        "overview", "introduction", "welcome", "about", "concept", "legacy",
        "design language", "statement", "presence", "versatile", "imagine",
        "key highlights", "sculpted for greatness", "bold and dynamic"
    ],
    "Exterior": [
        "exterior", "grille", "led headlamp", "alloy", "roof rail", "design",
        "stance", "silhouette", "spoiler", "muffler", "skid plate", "garnish",
        "door handle", "turn signal", "tail lamp", "drl", "fog lamp",
        "body cladding", "wheels", "all-new wheel", "diamond cut alloys"
    ],
    "Interior": [
        "interior", "cabin", "seats", "cockpit", "seating", "upholstery",
        "dashboard", "steering wheel", "comfort", "spacious", "armrest",
        "cushion", "ventilated", "dual tone", "leather", "leatherette",
        "noble brown", "haze navy", "ambient light", "glovebox", "pedal",
        "sunvisor", "headrest", "crashpad", "gear knob", "door trim"
    ],
    "Safety": [
        "safety", "airbag", "abs", "ebd", "esc", "vsm", "hac", "adas",
        "collision", "lane keeping", "blind-spot", "parking sensor",
        "disc brakes", "hill descent", "immobilizer", "isofix", "smartsense",
        "assist", "warning", "stability control", "seatbelt", "child seat",
        "brake", "active safety", "passive safety"
    ],
    "Engine": [
        "engine", "displacement", "torque", "power", "ps @", "nm @", "valves",
        "petrol", "diesel", "cylinder", "gdi", "crdi", "transmission",
        "manual", "automatic", "mt", "at", "dct", "dohc", "hybrid", "hev",
        "fuel tank", "cc", "ltr", "1.5 l", "2.5l"
    ],
    "Performance": [
        "performance", "drive modes", "traction", "paddle shifters", "sport",
        "eco", "normal", "mud", "sand", "snow", "dynamic driving", "speed",
        "acceleration", "handling", "exhilarating"
    ],
    "Technology": [
        "technology", "infotainment", "screen", "digital cluster", "bluelink",
        "connectivity", "android auto", "apple carplay", "wireless charger",
        "bose", "sound system", "smart key", "voice enabled", "sunroof",
        "alexa", "navigation", "ota", "gps", "speaker", "audio", "display"
    ],
    "Dimensions": [
        "dimensions", "length", "width", "height", "wheelbase",
        "ground clearance", "boot space", "capacity", "fuel tank capacity",
        "unit : mm", "mm"
    ],
    "Specifications": [
        "specifications", "technical specifications", "spec sheet", "specs",
        "suspension", "brakes", "tyre", "tire", "strut", "ctba", "axle",
        "front suspension", "rear suspension", "disc brake", "tyre size"
    ],
    "Variants": [
        "variants", "variant", "executive", "prestige", "platinum",
        "signature", "corporate", "knight", "trim", "standard", "grade"
    ],
    "Features": [
        "features", "key features", "equipment", "comfort and convenience",
        "standard features"
    ],
    "Colors": [
        "colours", "colors", "monotone", "dual tone", "starry night",
        "emerald", "white", "black", "grey", "silver", "red", "blue",
        "paint", "finish", "shade", "body color"
    ],
    "Warranty": [
        "warranty", "roadside assistance", "roadside", "shield of trust",
        "maintenance", "package", "kilometers", "basic warranty",
        "roadside support", "yrs / ul kms"
    ],
    "Pricing": [
        "price", "pricing", "ex-showroom", "cost", "tariff", "lakhs", "lakh",
        "rs", "rs."
    ]
}


@dataclass
class BrochureChunk:
    """Dataclass representing a structured section-aware text chunk."""
    chunk_id: str
    brand: str
    model: str
    year: int
    document_name: str
    page_number: int
    section: str
    text: str
    word_count: int


def split_into_sentences(text: str) -> List[str]:
    """Splits text block into sentences based on punctuation."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
    return [s.strip() for s in sentence_endings.split(text) if s.strip()]


def split_large_paragraph(paragraph: str, max_words: int = 350) -> List[str]:
    """Splits a single paragraph if it exceeds the maximum word count limit."""
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]

    sentences = split_into_sentences(paragraph)
    sub_paras = []
    current_sentences: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_word_count + s_words > max_words and current_sentences:
            sub_paras.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_word_count = s_words
        else:
            current_sentences.append(sentence)
            current_word_count += s_words

    if current_sentences:
        sub_paras.append(" ".join(current_sentences))

    return sub_paras


def split_into_paragraphs(text: str) -> List[str]:
    """Splits text into paragraphs, reconstructing wrapped lines."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    paragraphs = []
    current_para = []
    bullet_prefixes = ("-", "*", "•", "–", "—", "I", "^", "#")

    for line in lines:
        if not current_para:
            current_para.append(line)
            continue

        prev_line = current_para[-1]
        should_split = False
        is_bullet = line.startswith(bullet_prefixes) or (
            line[0].isdigit() and "." in line[:3]
        )

        if is_bullet:
            should_split = True
        elif prev_line and prev_line[-1] in {".", "!", "?", ":", ";"}:
            if line[0].isupper() or line[0].isdigit():
                should_split = True

        if should_split:
            paragraphs.append(" ".join(current_para))
            current_para = [line]
        else:
            current_para.append(line)

    if current_para:
        paragraphs.append(" ".join(current_para))

    return paragraphs


def group_paragraphs_into_chunks(
    paragraphs: List[str], min_words: int = 250, max_words: int = 450
) -> List[str]:
    """Greedily groups paragraphs into chunks of 250-450 words."""
    chunks = []
    current_chunk_paras = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_word_count + para_words > max_words:
            if current_word_count >= min_words:
                chunks.append("\n\n".join(current_chunk_paras))
                current_chunk_paras = [para]
                current_word_count = para_words
            else:
                if current_chunk_paras:
                    chunks.append("\n\n".join(current_chunk_paras))
                current_chunk_paras = [para]
                current_word_count = para_words
        else:
            current_chunk_paras.append(para)
            current_word_count += para_words

    if current_chunk_paras:
        chunks.append("\n\n".join(current_chunk_paras))

    return chunks


def detect_section(text: str) -> str:
    """Detects the category/section of a text block using heuristics."""
    scores = {sec: 0 for sec in SECTION_KEYWORDS.keys()}
    text_lower = text.lower()

    # 1. Heading Matching Heuristic
    first_lines = [line.strip() for line in text.split("\n") if line.strip()][:3]
    for line in first_lines:
        if len(line) < 60:
            line_lower = line.lower()
            for sec in SECTION_KEYWORDS.keys():
                if sec.lower() in line_lower:
                    scores[sec] += 15

    # 2. Keyword Heuristics
    for sec, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'(s|es)?\b'
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                scores[sec] += matches * 2

    # Choose section with highest score
    max_sec = "Miscellaneous"
    max_score = 0
    for sec, score in scores.items():
        if score > max_score:
            max_score = score
            max_sec = sec

    return max_sec


def generate_chunk_id(
    brand: str, model: str, page_number: int, chunk_index: int
) -> str:
    """Generates a unique chunk ID: <brand>_<model>_<page>_<chunk>."""
    clean_brand = brand.lower().strip().replace(" ", "_").replace("-", "_")
    clean_model = model.lower().strip().replace(" ", "_").replace("-", "_")
    return f"{clean_brand}_{clean_model}_{page_number:03d}_{chunk_index:02d}"


class BrochureChunker:
    """Orchestrates structured, section-aware chunking of parsed brochures."""

    def __init__(self, chunks_dir: Union[str, Path]) -> None:
        """Initializes the chunker with the output chunks directory."""
        self.chunks_dir = Path(chunks_dir)

    def chunk_document(self, document_path: Union[str, Path]) -> Path:
        """Loads a processed document JSON, chunks it, and saves the file."""
        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"Processed document not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to parse JSON file %s: %s", path.name, e)
            raise ValueError(f"Malformed JSON file {path.name}: {e}") from e

        brand = data.get("brand", "Unknown")
        model = data.get("model", "Unknown")
        year = data.get("year", 2025)
        doc_name = data.get("document_name", path.name)
        pages = data.get("pages", [])

        all_chunks = []
        for page in pages:
            page_chunks = self.chunk_page(brand, model, year, doc_name, page)
            all_chunks.extend(page_chunks)

        return self.save_chunks(all_chunks, brand, model)

    def chunk_page(
        self,
        brand: str,
        model: str,
        year: int,
        doc_name: str,
        page: Dict[str, Any]
    ) -> List[BrochureChunk]:
        """Chunks a single page from a brochure document."""
        page_num = page.get("page_number", 0)
        page_text = page.get("text", "").strip()
        if not page_text:
            return []

        raw_paras = split_into_paragraphs(page_text)
        processed_paras = []
        for para in raw_paras:
            processed_paras.extend(split_large_paragraph(para))

        chunk_texts = group_paragraphs_into_chunks(processed_paras)
        chunks = []

        for idx, text in enumerate(chunk_texts, start=1):
            section = detect_section(text)
            chunk_id = generate_chunk_id(brand, model, page_num, idx)
            word_count = len(text.split())
            chunks.append(
                BrochureChunk(
                    chunk_id=chunk_id,
                    brand=brand,
                    model=model,
                    year=year,
                    document_name=doc_name,
                    page_number=page_num,
                    section=section,
                    text=text,
                    word_count=word_count
                )
            )

        return chunks

    def save_chunks(
        self, chunks: List[BrochureChunk], brand: str, model: str
    ) -> Path:
        """Saves chunks list as a pretty-printed JSON file."""
        clean_brand = brand.strip().replace(" ", "_")
        clean_model = model.strip().replace(" ", "_")
        filename = f"{clean_brand}_{clean_model}_chunks.json"
        destination_path = self.chunks_dir / filename

        chunks_data = [asdict(c) for c in chunks]
        safe_json_write(chunks_data, destination_path)
        logger.info("Saved %d chunks to %s", len(chunks), destination_path.name)
        return destination_path
