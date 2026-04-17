#!/usr/bin/env python3
"""
materials_graph_extract.py
--------------------------
Extracts entities and relations from ingested paper chunks and populates
the knowledge graph in ArcadeDB.

Uses qwen2.5:0.5b for fast, low-RAM entity extraction.
Uses qwen3.5:9b only for the query interface (materials_query.py).

Usage:
    python materials_graph_extract.py --domain cmc --limit 10
    python materials_graph_extract.py --domain all
    python materials_graph_extract.py

Requires:
    ArcadeDB running on localhost:2480
    Ollama running with qwen2.5:0.5b pulled
"""

import os
import json
import logging
import argparse
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE    = os.getenv("OLLAMA_BASE", "http://localhost:11434")
EXTRACT_MODEL  = "qwen2.5:0.5b"   # small, fast, low RAM
TIMEOUT        = 60                # seconds per extraction call

# ---------------------------------------------------------------------------
# Extraction prompt — kept deliberately simple for the 0.5b model
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "You are a materials science entity extractor. "
    "Respond ONLY with valid JSON. No explanation. No markdown."
)

EXTRACTION_PROMPT = """\
Extract materials science entities from the text below.

Return this exact JSON structure:
{
  "materials": ["list of material names as strings"],
  "processes": ["list of process names as strings"],
  "applications": ["list of application names as strings"]
}

If nothing is found for a category use an empty list.
Only include items explicitly mentioned in the text.

TEXT:
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_str_list(value) -> list:
    """
    Safely convert whatever the LLM returned for a list field into
    a clean list of strings. Handles strings, dicts, None, nested lists.
    """
    if not value:
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
        elif isinstance(item, dict):
            # model sometimes returns {"name": "..."} instead of "..."
            s = str(item.get("name") or item.get("value") or next(iter(item.values()), "")).strip()
        else:
            s = str(item).strip()
        if s:
            result.append(s)
    return result

# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str, client: httpx.Client) -> dict:
    """
    Call the small model to extract entities from a text chunk.
    Returns dict with materials, processes, applications lists.
    Returns empty dict on failure.
    """
    prompt = EXTRACTION_PROMPT + text[:1500]

    try:
        response = client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model":  EXTRACT_MODEL,
                "prompt": prompt,
                "system": EXTRACTION_SYSTEM,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 256,
                },
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # Strip markdown fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        # Find the JSON object
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {}

        return json.loads(raw[start:end])

    except json.JSONDecodeError as e:
        log.debug(f"JSON parse error: {e}")
        return {}
    except Exception as e:
        log.warning(f"Extraction failed: {e}")
        return {}

# ---------------------------------------------------------------------------
# Graph population
# ---------------------------------------------------------------------------

def populate_graph(entities: dict, chunk_rid: str, db) -> int:
    """
    Create entity nodes and MENTIONS edges from extracted entities.
    Returns count of nodes created or found.
    """
    count = 0

    for name in to_str_list(entities.get("materials", [])):
        rid = db.get_or_create_vertex("Material", name)
        if rid and chunk_rid:
            db.create_edge("MENTIONS", chunk_rid, rid)
            count += 1

    for name in to_str_list(entities.get("processes", [])):
        rid = db.get_or_create_vertex("Process", name)
        if rid and chunk_rid:
            db.create_edge("MENTIONS", chunk_rid, rid)
            count += 1

    for name in to_str_list(entities.get("applications", [])):
        rid = db.get_or_create_vertex("Application", name)
        if rid and chunk_rid:
            db.create_edge("MENTIONS", chunk_rid, rid)
            count += 1

    return count

# ---------------------------------------------------------------------------
# Chunk retrieval — simple and fast, no per-chunk edge checking
# ---------------------------------------------------------------------------

def get_chunks(db, domain: str = None, limit: int = None) -> list:
    domain_filter = f"AND domain = '{domain}'" if domain else ""
    limit_clause  = f"LIMIT {limit}" if limit else "LIMIT 40000"
    query = f"""
        SELECT @rid, chunk_id, text, paper_id, domain
        FROM Chunk
        WHERE chunk_id IS NOT NULL
        {domain_filter}
        {limit_clause}
    """
    try:
        return db.sql(query)
    except Exception as e:
        log.error(f"Failed to fetch chunks: {e}")
        return []

def get_paper_rid(db, paper_id: str) -> str:
    try:
        result = db.sql(
            f"SELECT @rid FROM Paper WHERE paper_id = {json.dumps(paper_id)} LIMIT 1"
        )
        return result[0].get("@rid") if result else None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract entities from chunks into the knowledge graph."
    )
    parser.add_argument("--domain", "-d", type=str, default=None,
                        help="Only process this domain (default: all)")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Maximum chunks to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Report progress every N chunks (default: 50)")
    return parser.parse_args()


def main():
    args = parse_args()

    import logging as _l
    _l.disable(_l.CRITICAL)
    from arcadedb_client import MaterialsDB
    db = MaterialsDB()
    db._ensure_database()
    _l.disable(_l.NOTSET)

    stats = db.stats()
    log.info(f"Connected — {stats.get('Chunk',0):,} chunks, "
             f"{stats.get('Paper',0):,} papers")
    log.info(f"Extraction model: {EXTRACT_MODEL}")

    log.info("Fetching chunks...")
    chunks = get_chunks(db, domain=args.domain, limit=args.limit)
    log.info(f"Chunks to process: {len(chunks)}")

    if not chunks:
        log.info("Nothing to process.")
        return

    processed = failed = entities_n = 0

    with httpx.Client() as client:
        for i, chunk in enumerate(chunks, 1):
            chunk_rid = chunk.get("@rid")
            text      = chunk.get("text", "")
            paper_id  = chunk.get("paper_id", "")

            if not chunk_rid or not text:
                continue

            entities = extract_entities(text, client)

            if not entities:
                failed += 1
            else:
                n = populate_graph(entities, chunk_rid, db)
                entities_n += n
                processed += 1

            if i % args.batch_size == 0 or i == len(chunks):
                s = db.stats()
                log.info(
                    f"[{i}/{len(chunks)}] "
                    f"ok={processed} fail={failed} "
                    f"materials={s.get('Material',0)} "
                    f"processes={s.get('Process',0)} "
                    f"apps={s.get('Application',0)}"
                )

    log.info(f"\n{'='*60}")
    log.info(f"Extraction complete.")
    log.info(f"  Processed : {processed}")
    log.info(f"  Failed    : {failed}")
    log.info(f"  Entities  : {entities_n}")
    for k, v in db.stats().items():
        log.info(f"  {k:15s} {v:,}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
