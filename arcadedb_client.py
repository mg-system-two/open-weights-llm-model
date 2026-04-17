#!/usr/bin/env python3
"""
arcadedb_client.py
------------------
Wrapper around arcadedb-python for the materials knowledge system.
Provides a clean interface for ingestion, graph extraction, and querying.

Usage:
    from arcadedb_client import MaterialsDB
    db = MaterialsDB()
    db.setup()   # creates database and schema if not exists
    print(db.stats())
"""

import os
import json
import logging

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCADEDB_HOST      = os.getenv("ARCADEDB_HOST", "localhost")
ARCADEDB_PORT      = int(os.getenv("ARCADEDB_PORT", "2480"))
ARCADEDB_USER      = os.getenv("ARCADEDB_USER", "root")
ARCADEDB_PASSWORD  = os.getenv("ARCADEDB_PASSWORD", "materials2026")
ARCADEDB_DATABASE  = os.getenv("ARCADEDB_DATABASE", "materials")

EMBEDDING_DIMENSIONS = 768   # nomic-embed-text output size


# ---------------------------------------------------------------------------
# MaterialsDB
# ---------------------------------------------------------------------------

class MaterialsDB:
    """Client for the materials knowledge graph database."""

    def __init__(self):
        from arcadedb_python import SyncClient, DatabaseDao

        self._DatabaseDao = DatabaseDao

        self._client = SyncClient(
            host=ARCADEDB_HOST,
            port=ARCADEDB_PORT,
            username=ARCADEDB_USER,
            password=ARCADEDB_PASSWORD,
        )
        self._db = None
        log.info(f"MaterialsDB client initialised ({ARCADEDB_HOST}:{ARCADEDB_PORT})")

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def setup(self):
        """Create the database and schema if they don't already exist."""
        self._ensure_database()
        self._create_schema()
        log.info("Database setup complete.")

    def _ensure_database(self):
        """Create the materials database if it doesn't exist."""
        existing = self._DatabaseDao.list_databases(self._client)
        if ARCADEDB_DATABASE not in existing:
            self._db = self._DatabaseDao.create(self._client, ARCADEDB_DATABASE)
            log.info(f"Created database: {ARCADEDB_DATABASE}")
        else:
            self._db = self._DatabaseDao(self._client, ARCADEDB_DATABASE)
            log.info(f"Connected to existing database: {ARCADEDB_DATABASE}")

    def _create_schema(self):
        """Create vertex types, edge types, properties, and indexes."""
        cmds = [
            # Vertex types
            "CREATE VERTEX TYPE Chunk",
            "CREATE PROPERTY Chunk.chunk_id STRING",
            "CREATE PROPERTY Chunk.text STRING",
            "CREATE PROPERTY Chunk.paper_id STRING",
            "CREATE PROPERTY Chunk.domain STRING",
            "CREATE PROPERTY Chunk.chunk_index INTEGER",
            "CREATE PROPERTY Chunk.embedding ARRAY_OF_FLOATS",

            "CREATE VERTEX TYPE Paper",
            "CREATE PROPERTY Paper.paper_id STRING",
            "CREATE PROPERTY Paper.title STRING",
            "CREATE PROPERTY Paper.abstract STRING",
            "CREATE PROPERTY Paper.doi STRING",
            "CREATE PROPERTY Paper.arxiv_id STRING",
            "CREATE PROPERTY Paper.published STRING",
            "CREATE PROPERTY Paper.domain STRING",
            "CREATE PROPERTY Paper.source STRING",
            "CREATE PROPERTY Paper.local_path STRING",

            "CREATE VERTEX TYPE Author",
            "CREATE PROPERTY Author.name STRING",

            "CREATE VERTEX TYPE Material",
            "CREATE PROPERTY Material.name STRING",
            "CREATE PROPERTY Material.formula STRING",
            "CREATE PROPERTY Material.domain STRING",

            "CREATE VERTEX TYPE Property",
            "CREATE PROPERTY Property.name STRING",
            "CREATE PROPERTY Property.value STRING",
            "CREATE PROPERTY Property.unit STRING",

            "CREATE VERTEX TYPE Process",
            "CREATE PROPERTY Process.name STRING",
            "CREATE PROPERTY Process.domain STRING",

            "CREATE VERTEX TYPE Application",
            "CREATE PROPERTY Application.name STRING",
            "CREATE PROPERTY Application.domain STRING",

            # Edge types
            "CREATE EDGE TYPE CHUNK_OF",
            "CREATE EDGE TYPE AUTHORED_BY",
            "CREATE EDGE TYPE MENTIONS",
            "CREATE EDGE TYPE HAS_PROPERTY",
            "CREATE EDGE TYPE MANUFACTURED_BY",
            "CREATE EDGE TYPE CITES",

            # Indexes
            "CREATE INDEX ON Chunk(chunk_id)",
            "CREATE INDEX ON Paper(paper_id)",
            "CREATE INDEX ON Paper(doi)",
            "CREATE INDEX ON Author(name)",
            "CREATE INDEX ON Material(name)",
        ]

        for cmd in cmds:
            try:
                self._db.query("sql", cmd, is_command=True)
            except Exception as e:
                log.debug(f"Schema cmd note: {cmd[:60]} — {e}")

        # Vector index on Chunk.embedding
        try:
            self._db.create_vector_index(
                "Chunk", "embedding", dimensions=EMBEDDING_DIMENSIONS
            )
            log.info("Vector index created on Chunk.embedding")
        except Exception as e:
            log.debug(f"Vector index note: {e}")

        log.info("Schema ready.")

    # -----------------------------------------------------------------------
    # Chunk operations
    # -----------------------------------------------------------------------

    def insert_chunk(self, chunk_id, text, embedding, paper_id, domain, chunk_index):
        """Insert a text chunk with its embedding. Returns record ID or None."""
        try:
            content = json.dumps({
                "chunk_id": chunk_id,
                "text": text,
                "embedding": embedding,
                "paper_id": paper_id,
                "domain": domain,
                "chunk_index": chunk_index,
            })
            result = self._db.query(
                "sql",
                f"INSERT INTO Chunk CONTENT {content}",
                is_command=True,
            )
            return result[0].get("@rid") if result else None
        except Exception as e:
            log.error(f"insert_chunk failed: {e}")
            return None

    def vector_search(self, query_embedding, top_k=10, domain=None):
        """Find the top_k most similar chunks to query_embedding."""
        try:
            results = self._db.vector_search(
                type_name="Chunk",
                embedding_field="embedding",
                query_embedding=query_embedding,
                top_k=top_k * 2 if domain else top_k,
            )
            if domain:
                results = [r for r in results if r.get("domain") == domain][:top_k]
            return results
        except Exception as e:
            log.error(f"vector_search failed: {e}")
            return []

    def chunk_exists(self, chunk_id):
        """Check whether a chunk already exists."""
        try:
            result = self._db.query(
                "sql",
                f"SELECT chunk_id FROM Chunk WHERE chunk_id = {json.dumps(chunk_id)} LIMIT 1",
            )
            return len(result) > 0
        except Exception as e:
            log.error(f"chunk_exists failed: {e}")
            return False

    # -----------------------------------------------------------------------
    # Paper operations
    # -----------------------------------------------------------------------

    def upsert_paper(self, paper):
        """Insert or update a paper vertex. Returns record ID or None."""
        paper_id = paper.get("doi") or paper.get("arxiv_id") or paper.get("title", "")[:40]
        try:
            existing = self._db.query(
                "sql",
                f"SELECT @rid FROM Paper WHERE paper_id = {json.dumps(paper_id)} LIMIT 1",
            )
            if existing:
                return existing[0].get("@rid")

            content = json.dumps({
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "doi": paper.get("doi", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "published": paper.get("published", ""),
                "domain": paper.get("domain", ""),
                "source": paper.get("source", ""),
                "local_path": paper.get("local_path", ""),
            })
            result = self._db.query(
                "sql",
                f"INSERT INTO Paper CONTENT {content}",
                is_command=True,
            )
            return result[0].get("@rid") if result else None
        except Exception as e:
            log.error(f"upsert_paper failed: {e}")
            return None

    # -----------------------------------------------------------------------
    # Graph operations
    # -----------------------------------------------------------------------

    def get_or_create_vertex(self, vertex_type, name, extra=None):
        """Get an existing vertex by name, or create it. Returns record ID."""
        try:
            existing = self._db.query(
                "sql",
                f"SELECT @rid FROM {vertex_type} WHERE name = {json.dumps(name)} LIMIT 1",
            )
            if existing:
                return existing[0].get("@rid")

            content = {"name": name}
            if extra:
                content.update(extra)
            result = self._db.query(
                "sql",
                f"INSERT INTO {vertex_type} CONTENT {json.dumps(content)}",
                is_command=True,
            )
            return result[0].get("@rid") if result else None
        except Exception as e:
            log.error(f"get_or_create_vertex({vertex_type}, {name}) failed: {e}")
            return None

    def create_edge(self, edge_type, from_rid, to_rid, properties=None):
        """Create a directed edge between two vertices. Returns True on success."""
        try:
            props = f"CONTENT {json.dumps(properties)}" if properties else ""
            self._db.query(
                "sql",
                f"CREATE EDGE {edge_type} FROM {from_rid} TO {to_rid} {props}",
                is_command=True,
            )
            return True
        except Exception as e:
            log.error(f"create_edge({edge_type}) failed: {e}")
            return False

    def link_chunk_to_paper(self, chunk_rid, paper_rid):
        """Create a CHUNK_OF edge from a chunk to its paper."""
        return self.create_edge("CHUNK_OF", chunk_rid, paper_rid)

    # -----------------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------------

    def sql(self, query, is_command=False):
        """Run a raw SQL query."""
        try:
            return self._db.query("sql", query, is_command=is_command)
        except Exception as e:
            log.error(f"SQL failed: {e}\n  {query[:100]}")
            return []

    def cypher(self, query):
        """Run a raw Cypher query."""
        try:
            return self._db.query("opencypher", query)
        except Exception as e:
            log.error(f"Cypher failed: {e}\n  {query[:100]}")
            return []

    def count(self, vertex_type):
        """Count records in a vertex type."""
        result = self.sql(f"SELECT count(*) FROM {vertex_type}")
        return result[0].get("count(*)", 0) if result else 0

    def stats(self):
        """Return record counts for all vertex types."""
        types = ["Chunk", "Paper", "Author", "Material", "Property", "Process", "Application"]
        return {t: self.count(t) for t in types}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    db = MaterialsDB()
    db.setup()

    print("\nDatabase stats:")
    for k, v in db.stats().items():
        print(f"  {k:15s} {v}")

    print("\narcadedb_client.py — OK")
