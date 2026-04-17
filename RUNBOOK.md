# Materials Knowledge System — Runbook

One-shot setup from a fresh machine to a running RAG + knowledge graph system.

**Stack:** Ollama (LLM + embeddings) · ArcadeDB (graph + vector DB) · Python 3.12

---

## 1. System prerequisites

```bash
# Ubuntu/Debian
apt-get update && apt-get install -y python3.12 python3.12-venv python3-pip curl git docker.io
```

---

## 2. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # starts on localhost:11434

# Pull required models
ollama pull nomic-embed-text   # embeddings (768-dim), used by ingest + query
ollama pull qwen2.5:0.5b       # entity extraction — fast, low RAM
ollama pull qwen3.5:9b         # query synthesis — needs ~8 GB VRAM or ~16 GB RAM
```

Verify:
```bash
curl http://localhost:11434/api/tags
```

---

## 3. ArcadeDB

```bash
docker run -d \
  --name arcadedb \
  -p 2480:2480 \
  -p 2424:2424 \
  -e ARCADEDB_SERVER_ROOT_PASSWORD=materials2026 \
  arcadedata/arcadedb:latest
```

Verify (should return server info):
```bash
curl -u root:materials2026 http://localhost:2480/api/v1/ready
```

The Python client creates the `materials` database and full schema automatically on first run — no manual DB setup needed.

---

## 4. Python environment

```bash
cd /root/materials-knowledge
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# spaCy English model (needed if using spacy-llm features)
python -m spacy download en_core_web_sm
```

---

## 5. Environment variables

Create `.env` in the project root (gitignored):

```bash
cat > .env << 'EOF'
ARCADEDB_HOST=localhost
ARCADEDB_PORT=2480
ARCADEDB_USER=root
ARCADEDB_PASSWORD=materials2026
ARCADEDB_DATABASE=materials

OLLAMA_BASE=http://localhost:11434

# Optional — enables Semantic Scholar paper search (free key at semanticscholar.org)
SEMANTIC_SCHOLAR_API_KEY=

MANIFEST_PATH=data/papers/manifest.json
EOF
```

---

## 6. Paper retrieval

Downloads PDFs from arXiv and Semantic Scholar into `data/papers/<domain>/` and writes `data/papers/manifest.json`.

```bash
# One domain at a time (recommended — arXiv rate-limits at 3s/req)
python materials_papers_retrieval.py --domain cmc    --limit 30
python materials_papers_retrieval.py --domain tbc    --limit 30
python materials_papers_retrieval.py --domain 2d_materials --limit 30
python materials_papers_retrieval.py --domain ald    --limit 20
python materials_papers_retrieval.py --domain armour --limit 20
python materials_papers_retrieval.py --domain biomaterials   --limit 20
python materials_papers_retrieval.py --domain energy_storage --limit 20
python materials_papers_retrieval.py --domain extreme_environments --limit 20
python materials_papers_retrieval.py --domain hydrogen       --limit 20
python materials_papers_retrieval.py --domain rubber_polymers --limit 20
python materials_papers_retrieval.py --domain tmd            --limit 20
python materials_papers_retrieval.py --domain vitrimers      --limit 20

# Or everything in one go (slow — expect 30-60 min)
python materials_papers_retrieval.py --domain all --limit 20
```

Output: `data/papers/manifest.json` + PDFs in domain subdirectories.

---

## 7. Ingest papers into ArcadeDB

Extracts text from PDFs, chunks it, embeds via `nomic-embed-text`, stores chunks + paper metadata + author graph in ArcadeDB.

```bash
# All domains
python materials_papers_ingest.py

# Single domain
python materials_papers_ingest.py --domain cmc

# Test with a small batch first
python materials_papers_ingest.py --limit 5
```

Idempotent — already-ingested papers are skipped.

---

## 8. Graph entity extraction

Runs `qwen2.5:0.5b` over each stored chunk to extract Materials, Properties, Processes, Applications and writes graph edges into ArcadeDB.

```bash
python materials_graph_extract.py --domain cmc
python materials_graph_extract.py           # all domains
```

This is the slow step (~1-2s per chunk on CPU). Run per-domain so you can resume if interrupted.

---

## 9. Query

Interactive or single-shot Q&A using `qwen3.5:9b` + vector search over ArcadeDB.

```bash
# Interactive REPL
python materials_query.py

# Single question
python materials_query.py --question "What are the failure modes of SiC/SiC CMCs?"

# Restrict to a domain
python materials_query.py --domain cmc --top-k 15
```

---

## 10. Verify the database

```bash
python arcadedb_client.py
```

Prints record counts for all vertex types (Chunk, Paper, Author, Material, Property, Process, Application).

---

## Full pipeline order

```
ollama serve
docker start arcadedb          # or docker run (step 3)
source venv/bin/activate

python materials_papers_retrieval.py --domain all --limit 20
python materials_papers_ingest.py
python materials_graph_extract.py
python materials_query.py
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` on port 11434 | `ollama serve` isn't running |
| `Connection refused` on port 2480 | ArcadeDB container stopped — `docker start arcadedb` |
| Embedding returns `[]` | `ollama pull nomic-embed-text` not done yet |
| `Text quality check failed` on most papers | PDFs are scanned images — need OCR pipeline (not included) |
| `qwen3.5:9b` OOM | Reduce `--top-k` or swap for `qwen2.5:7b` in `materials_query.py` |
| ArcadeDB schema errors on setup | Safe to ignore — errors fire when types/indexes already exist |
