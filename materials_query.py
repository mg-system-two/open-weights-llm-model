#!/usr/bin/env python3
"""
materials_query.py
------------------
Interactive query interface for the materials knowledge system.
Takes a plain English question, retrieves relevant chunks from ArcadeDB
via vector search, and synthesises an answer using Qwen 3.5 9B.

Usage:
    python materials_query.py
    python materials_query.py --question "What are the failure modes of SiC/SiC CMCs?"
    python materials_query.py --domain cmc
    python materials_query.py --top-k 15

Requires:
    ArcadeDB running on localhost:2480
    Ollama running with qwen3.5:9b and nomic-embed-text pulled
"""

import os
import sys
import json
import argparse
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE   = os.getenv("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL   = "nomic-embed-text"
LLM_MODEL     = "qwen3.5:9b"
DEFAULT_TOP_K = 10

SYSTEM_PROMPT = """You are a materials science research assistant with access to
a curated corpus of peer-reviewed papers. Your job is to answer questions
accurately and concisely, grounding every claim in the provided source passages.

Rules:
- Only use information from the provided passages
- If the passages do not contain enough information to answer, say so clearly
- Cite the paper title when making specific claims
- Use precise scientific language appropriate for a materials science audience
- Do not speculate beyond what the sources support
"""

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(text: str, client: httpx.Client) -> list:
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
# LLM call — handles qwen3.5:9b thinking model
# ---------------------------------------------------------------------------

def ask_llm(question: str, context: str, client: httpx.Client) -> str:
    prompt = f"""Here are relevant passages from the materials science literature:

{context}

---

Based on these passages, please answer the following question:

{question}"""

    try:
        response = client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048,
                }
            },
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()

        # qwen3.5:9b is a thinking model — answer may be in 'thinking' field
        answer = data.get("response", "").strip()
        if not answer:
            answer = data.get("thinking", "").strip()

        return answer or "No response generated."

    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return f"Error generating response: {e}"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(question: str, db, client: httpx.Client,
                    top_k: int = DEFAULT_TOP_K, domain: str = None) -> list:
    embedding = get_embedding(question, client)
    if not embedding:
        return []
    return db.vector_search(embedding, top_k=top_k, domain=domain)

def format_context(chunks: list) -> str:
    if not chunks:
        return "No relevant passages found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        domain = chunk.get("domain", "")
        text   = chunk.get("text", "")
        parts.append(f"[{i}] [{domain}]\n{text.strip()}")
    return "\n\n".join(parts)

def format_sources(chunks: list) -> str:
    if not chunks:
        return ""
    seen = set()
    sources = []
    for chunk in chunks:
        paper_id = chunk.get("paper_id", "")
        domain   = chunk.get("domain", "")
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            sources.append(f"  · [{domain}] {paper_id}")
    return "\n".join(sources)

# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------

def query(question: str, db, client: httpx.Client,
          top_k: int = DEFAULT_TOP_K, domain: str = None,
          verbose: bool = False) -> str:

    print(f"\nSearching knowledge base...", end=" ", flush=True)
    chunks = retrieve_chunks(question, db, client, top_k=top_k, domain=domain)
    print(f"{len(chunks)} passages retrieved.")

    if not chunks:
        return "No relevant passages found in the knowledge base for this question."

    if verbose:
        print("\n--- Retrieved passages ---")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] domain={c.get('domain')} paper={c.get('paper_id','')[:40]}")
            print(c.get('text','')[:200] + "...")
        print("--- End passages ---\n")

    context = format_context(chunks)

    print("Generating answer...", end=" ", flush=True)
    answer = ask_llm(question, context, client)
    print("done.\n")

    sources = format_sources(chunks)

    output  = f"\n{'='*60}\n"
    output += f"Q: {question}\n"
    output += f"{'='*60}\n\n"
    output += answer
    output += f"\n\n{'─'*60}\n"
    output += f"Sources ({len(set(c.get('paper_id') for c in chunks))} papers):\n"
    output += sources
    output += f"\n{'─'*60}\n"

    return output

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the materials science knowledge base."
    )
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--domain", "-d", type=str, default=None,
                        help="Filter to a specific domain (e.g. cmc, tbc, hydrogen)")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of passages to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show retrieved passages before the answer")
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
    print(f"\nMaterials Knowledge System")
    print(f"  Chunks : {stats.get('Chunk', 0):,}")
    print(f"  Papers : {stats.get('Paper', 0):,}")
    print(f"  Authors: {stats.get('Author', 0):,}")
    if args.domain:
        print(f"  Filter : domain = {args.domain}")
    print()

    with httpx.Client() as client:

        if args.question:
            result = query(
                args.question, db, client,
                top_k=args.top_k,
                domain=args.domain,
                verbose=args.verbose,
            )
            print(result)

        else:
            print("Interactive mode — type your question and press Enter.")
            print("Commands: 'quit' to exit, 'domain <n>' to filter, 'verbose' to toggle\n")

            verbose = args.verbose
            domain  = args.domain

            while True:
                try:
                    question = input("Q: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye.")
                    break

                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye.")
                    break
                if question.lower().startswith("domain "):
                    domain = question.split(" ", 1)[1].strip() or None
                    print(f"Domain filter: {domain or 'none'}")
                    continue
                if question.lower() == "verbose":
                    verbose = not verbose
                    print(f"Verbose: {'on' if verbose else 'off'}")
                    continue
                if question.lower() == "stats":
                    for k, v in db.stats().items():
                        print(f"  {k:15s} {v:,}")
                    continue

                result = query(
                    question, db, client,
                    top_k=args.top_k,
                    domain=domain,
                    verbose=verbose,
                )
                print(result)


if __name__ == "__main__":
    main()
