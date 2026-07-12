import json
import logging
from pathlib import Path
import shutil
import sys

import config
from rag.bm25_index import BM25Indexer
from rag.chunker import BrochureChunker
from rag.embeddings import EmbeddingGenerator
from rag.metadata import MetadataEnricher
from rag.parser import PDFParser
from rag.utils import safe_directory_create
from rag.vectorstore import FAISSVectorStore

# Setup logging directory and files
safe_directory_create(config.LOG_DIR)
log_filepath = Path(config.LOG_DIR) / "ingestion.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ingest")


def main() -> None:
    """Main orchestration loop for brochure ingestion."""
    logger.info("Starting ingestion pipeline.")
    
    # Ensure a clean rebuild of the vectorstore
    vectorstore_path = Path(config.VECTORSTORE_DIR)
    if vectorstore_path.exists():
        shutil.rmtree(vectorstore_path)
    safe_directory_create(vectorstore_path)
    
    # Initialize parser, chunker, enricher, embeddings, and vector store
    parser = PDFParser(
        brochure_dir=config.BROCHURE_DIR,
        processed_dir=config.PROCESSED_DIR
    )
    chunker = BrochureChunker(chunks_dir=config.CHUNKS_DIR)
    enricher = MetadataEnricher(enriched_dir=config.ENRICHED_DIR)
    embedding_generator = EmbeddingGenerator(model_name=config.EMBEDDING_MODEL)
    vector_store = FAISSVectorStore(store_dir=config.VECTORSTORE_DIR)
    bm25_indexer = BM25Indexer(index_path=vectorstore_path / "bm25.pkl")
    
    # Discover brochures
    try:
        brochure_paths = parser.discover_brochures()
    except Exception as e:
        logger.error("Failed to discover brochures: %s", e)
        print(f"Error discovering brochures: {e}")
        return

    total_files = len(brochure_paths)
    successful_count = 0
    failed_count = 0
    errors_list = []
    all_texts = []

    for path in brochure_paths:
        logger.info("Processing brochure: %s", path.name)
        try:
            # 1. Parse, Chunk, and Enrich
            document = parser.parse_pdf(path)
            json_path = parser.save_json(document)
            chunks_path = chunker.chunk_document(json_path)
            enriched_path = enricher.enrich_document_chunks(chunks_path)
            
            # 2. Load Enriched Chunks and generate embeddings
            with open(enriched_path, "r", encoding="utf-8") as f:
                enriched_chunks = json.load(f)
                
            if enriched_chunks:
                texts = [c["text"] for c in enriched_chunks]
                all_texts.extend(texts)
                embeddings = embedding_generator.generate_embeddings(texts)
                
                metadata_list = []
                for chunk in enriched_chunks:
                    metadata_list.append({
                        "chunk_id": chunk["chunk_id"],
                        "brand": chunk["brand"],
                        "model": chunk["model"],
                        "year": chunk["year"],
                        "page_number": chunk["page_number"],
                        "section": chunk["section"],
                        "subsection": chunk["subsection"],
                        "keywords": chunk["keywords"],
                        "source_file": chunk["source_file"],
                        "text": chunk["text"]
                    })
                
                vector_store.add_chunks(embeddings, metadata_list)
            
            successful_count += 1
        except Exception as e:
            logger.error("Error processing brochure %s: %s", path.name, e)
            failed_count += 1
            errors_list.append((path.name, str(e)))

    # Save the consolidated indices
    if successful_count > 0:
        vector_store.save()
        bm25_indexer.build_index(all_texts)
        bm25_indexer.save()

    logger.info("Completion: Brochure ingestion completed.")
    
    # Print summary to the console
    print("\n" + "=" * 50)
    print("INGESTION WORKFLOW SUMMARY")
    print("=" * 50)
    print(f"Total discovered PDFs: {total_files}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed / Skipped:       {failed_count}")
    
    if failed_count > 0:
        print("\nErrors encountered:")
        for filename, error in errors_list:
            print(f" - {filename}: {error}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
