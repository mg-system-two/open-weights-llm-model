#!/usr/bin/env python3
"""
materials_papers_ingest.py
--------------------------
Ingests retrieved papers into ArcadeDB.

For each paper in the manifest:
  1. Extract text from the PDF
  2. Check text quality — skip if garbled or too short
  3. Chunk into semantic pieces
  4. Embed each chunk via nomic-embed-text (Ollama)
  5. Store chunks and paper metadata in ArcadeDB
  6. Link chunks to their paper via CHUNK_OF edges

Usage:
    python materials_papers_ingest.py
    python materials_papers_ingest.py --domain cmc
    python materials_papers_ingest.py --manifest data/papers/manifest.json
    python materials_papers_ingest.py --limit 10   # ingest first 10 papers only

Requires:
    pip install pypdf httpx python-dotenv arcadedb-python
    Ollama running with nomic-embed-text pulled
    ArcadeDB running on localhost:2480
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path

import httpx
from dotenv import load_dotenv


load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE      = os.getenv("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL      = "nomic-embed-text"
MANIFEST_PATH    = os.getenv("MANIFEST_PATH", "data/papers/manifest.json")

# Chunking
CHUNK_SIZE       = 800    # target characters per chunk
CHUNK_OVERLAP    = 100    # overlap between chunks
MIN_CHUNK_CHARS  = 100    # discard chunks shorter than this

# Quality gate
MIN_PAGE_CHARS   = 100    # pages with fewer chars are likely scanned/empty
MIN_WORD_RATIO   = 0.5    # ratio of real words to total tokens (garble detection)

# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

def get_embedding(text: str, client: httpx.Client) -> list:
    """
    Get a nomic-embed-text embedding from Ollama.
    Returns a list of floats, or empty list on failure.
    """
    try:
        response = client.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        log.error(f"Embedding failed: {e}")
        return []

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using pypdf.
    Returns the full text string, or empty string on failure.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if len(text.strip()) >= MIN_PAGE_CHARS:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        log.warning(f"PDF extraction failed for {pdf_path}: {e}")
        return ""

def text_quality_ok(text: str) -> bool:
    """
    Basic quality check on extracted text.
    Returns False if the text looks garbled or too short.
    """
    if len(text.strip()) < 500:
        return False

    # Check word ratio — garbled text has lots of short non-word tokens
    tokens = text.split()
    if not tokens:
        return False

    real_words = sum(1 for t in tokens if t.isalpha() and len(t) > 1)
    ratio = real_words / len(tokens)

    return ratio >= MIN_WORD_RATIO

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, paper_id: str, domain: str) -> list:
    """
    Split text into overlapping chunks.
    Returns list of dicts with chunk_id, text, paper_id, domain, chunk_index.
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # Try to break at a sentence boundary
        if end < len(text):
            # Look for a full stop, newline, or space near the end
            for boundary in [". ", "\n", " "]:
                pos = text.rfind(boundary, start + CHUNK_SIZE // 2, end)
                if pos != -1:
                    end = pos + len(boundary)
                    break

        chunk_text = text[start:end].strip()

        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunk_id = hashlib.md5(
                f"{paper_id}:{index}".encode()
            ).hexdigest()

            chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text,
                "paper_id":    paper_id,
                "domain":      domain,
                "chunk_index": index,
            })
            index += 1

        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break

    return chunks

# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def ingest_paper(entry: dict, db, client: httpx.Client) -> bool:
    """
    Ingest a single paper from the manifest.
    Returns True if successfully ingested, False if skipped or failed.
    """
    title     = entry.get("title", "unknown")
    local_path = entry.get("local_path", "")
    domain    = entry.get("domain", "")
    paper_id  = entry.get("doi") or entry.get("arxiv_id") or hashlib.md5(
        title.encode()
    ).hexdigest()[:12]

    # Skip if already ingested
    if db.chunk_exists(f"{paper_id}:0"):
        log.info(f"  Already ingested: {title[:60]}")
        return False

    # Check PDF exists
    if not local_path or not Path(local_path).exists():
        log.warning(f"  PDF not found: {local_path}")
        return False

    # Extract text
    text = extract_text(local_path)
    if not text:
        log.warning(f"  No text extracted: {title[:60]}")
        return False

    # Quality check
    if not text_quality_ok(text):
        log.warning(f"  Text quality check failed: {title[:60]}")
        return False

    # Store paper metadata
    paper_rid = db.upsert_paper(entry)
    if not paper_rid:
        log.error(f"  Failed to store paper: {title[:60]}")
        return False

    # Store authors
    for author_name in entry.get("authors", []):
        if author_name.strip():
            author_rid = db.get_or_create_vertex("Author", author_name.strip())
            if author_rid and paper_rid:
                db.create_edge("AUTHORED_BY", paper_rid, author_rid)

    # Chunk and embed
    chunks = chunk_text(text, paper_id, domain)
    if not chunks:
        log.warning(f"  No chunks produced: {title[:60]}")
        return False

    log.info(f"  {len(chunks)} chunks — embedding and storing...")

    stored = 0
    for chunk in chunks:
        embedding = get_embedding(chunk["text"], client)
        if not embedding:
            log.warning(f"  Embedding failed for chunk {chunk['chunk_index']}")
            continue

        chunk_rid = db.insert_chunk(
            chunk_id    = chunk["chunk_id"],
            text        = chunk["text"],
            embedding   = embedding,
            paper_id    = chunk["paper_id"],
            domain      = chunk["domain"],
            chunk_index = chunk["chunk_index"],
        )

        if chunk_rid:
            db.link_chunk_to_paper(chunk_rid, paper_rid)
            stored += 1

    log.info(f"  Stored {stored}/{len(chunks)} chunks for: {title[:60]}")
    return stored > 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest retrieved papers into ArcadeDB."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=MANIFEST_PATH,
        help=f"Path to manifest.json (default: {MANIFEST_PATH})",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Only ingest papers from this domain (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to ingest (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter by domain if specified
    entries = list(manifest.values())
    if args.domain:
        entries = [e for e in entries if e.get("domain") == args.domain]
        log.info(f"Filtered to domain '{args.domain}': {len(entries)} papers")

    # Apply limit
    if args.limit:
        entries = entries[:args.limit]

    log.info(f"Papers to process: {len(entries)}")

    # Connect to ArcadeDB
    from arcadedb_client import MaterialsDB
    db = MaterialsDB()
    db.setup()

    # Ingest
    ingested  = 0
    skipped   = 0
    failed    = 0

    with httpx.Client() as client:
        for i, entry in enumerate(entries, 1):
            title = entry.get("title", "unknown")[:60]
            log.info(f"\n[{i}/{len(entries)}] {title}")

            try:
                success = ingest_paper(entry, db, client)
                if success:
                    ingested += 1
                else:
                    skipped += 1
            except Exception as e:
                log.error(f"  Unexpected error: {e}")
                failed += 1

    # Final stats
    log.info(f"\n{'='*60}")
    log.info(f"Ingestion complete.")
    log.info(f"  Ingested : {ingested}")
    log.info(f"  Skipped  : {skipped}")
    log.info(f"  Failed   : {failed}")
    log.info(f"\nDatabase stats:")
    for k, v in db.stats().items():
        log.info(f"  {k:15s} {v}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
