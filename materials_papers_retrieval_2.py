#!/usr/bin/env python3
"""
materials_papers_retrieval_2.py
--------------------------------
Retrieves open-access research papers for TMD and 2D materials domains.
Adds to the existing manifest without touching previously downloaded papers.

Usage:
    python materials_papers_retrieval_2.py --limit 60

Requires:
    pip install httpx python-dotenv

Environment variables (in .env file):
    SEMANTIC_SCHOLAR_API_KEY=your_key_here   # optional, enables S2 search
"""

import os
import json
import time
import argparse
import logging
import hashlib
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

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
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
S2_BASE    = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api/query"

ARXIV_DELAY    = 10.0  # conservative — arXiv rate limits aggressively
S2_DELAY       = 2.0   # S2 unauthenticated buffer
DOWNLOAD_DELAY = 2.0   # polite delay between PDF downloads
REQUEST_TIMEOUT = 30   # seconds

# ---------------------------------------------------------------------------
# Domains — TMD and 2D materials ONLY
# ---------------------------------------------------------------------------

DOMAINS = {
    "tmd": {
        "name": "Transition Metal Dichalcogenides",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND abs:molybdenum AND abs:dichalcogenide",
            "cat:cond-mat.mes-hall AND abs:transition AND abs:metal AND abs:dichalcogenide",
            "cat:cond-mat.mtrl-sci AND abs:molybdenum AND abs:disulfide",
            "cat:cond-mat.mes-hall AND abs:monolayer AND abs:tungsten AND abs:disulfide",
            "cat:cond-mat.mes-hall AND abs:valleytronics AND abs:monolayer",
            "cat:cond-mat.mtrl-sci AND abs:tungsten AND abs:diselenide",
        ],
        "s2_queries": [
            "transition metal dichalcogenide MoS2 properties applications",
            "monolayer molybdenum disulfide bandgap photovoltaics",
            "tungsten disulfide hydrogen evolution reaction catalysis",
            "molybdenum diselenide tungsten diselenide valleytronics",
            "transition metal dichalcogenide solid lubrication",
            "tungsten disulfide molybdenum disulfide photodetector",
        ],
        "filter_terms": [
            "molybdenum disulfide", "molybdenum diselenide",
            "tungsten disulfide", "tungsten diselenide",
            "transition metal dichalcogenide", "dichalcogenide",
            "monolayer semiconductor", "valleytronics",
            "valley polarisation", "valley polarization",
            "hydrogen evolution reaction", "photodetector",
            "solid lubrication", "direct bandgap",
            "MoS2", "MoSe2", "WS2", "WSe2",
        ],
    },
    "2d_materials": {
        "name": "2D Materials",
        "arxiv_queries": [
            "cat:cond-mat.mes-hall AND abs:two-dimensional AND abs:material",
            "cat:cond-mat.mtrl-sci AND abs:van AND abs:der AND abs:Waals AND abs:heterostructure",
            "cat:cond-mat.mes-hall AND abs:graphene AND abs:heterostructure",
            "cat:cond-mat.mtrl-sci AND abs:hexagonal AND abs:boron AND abs:nitride",
            "cat:cond-mat.mes-hall AND abs:atomically AND abs:thin AND abs:monolayer",
            "cat:cond-mat.mtrl-sci AND abs:MXene AND abs:two-dimensional",
        ],
        "s2_queries": [
            "2D materials beyond graphene monolayer properties",
            "van der Waals heterostructure two-dimensional material",
            "hexagonal boron nitride h-BN 2D material electronics",
            "graphene two-dimensional material applications",
            "2D material synthesis characterisation optoelectronics",
            "MXene two-dimensional material energy applications",
        ],
        "filter_terms": [
            "two-dimensional material", "2D material",
            "graphene", "monolayer", "few-layer",
            "atomically thin", "van der Waals",
            "heterostructure", "hexagonal boron nitride",
            "boron nitride", "black phosphorus",
            "silicene", "stanene", "MXene",
            "beyond graphene", "flatland",
        ],
    },
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def doi_to_filename(doi: str) -> str:
    return doi.replace("/", "_").replace(":", "_") + ".pdf"


def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"


def is_relevant(title: str, abstract: str, filter_terms: list) -> bool:
    title_lower    = title.lower()
    abstract_lower = abstract.lower()
    for term in filter_terms:
        if term.lower() in title_lower:
            return True
    matches = sum(1 for term in filter_terms if term.lower() in abstract_lower)
    return matches >= 2


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict, manifest_path: Path):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# arXiv retrieval
# ---------------------------------------------------------------------------

def search_arxiv(query: str, limit: int, client: httpx.Client) -> list:
    params = {
        "search_query": query,
        "max_results":  limit,
        "sortBy":       "relevance",
        "sortOrder":    "descending",
    }
    try:
        log.info(f"arXiv search: {query[:70]}...")
        response = client.get(ARXIV_BASE, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning(f"arXiv request failed: {e}")
        log.info("Rate limited — waiting 15 seconds before continuing...")
        time.sleep(15)
        return []

    time.sleep(ARXIV_DELAY)

    ns = {
        "atom":  "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        log.warning(f"arXiv XML parse error: {e}")
        return []

    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el   = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el      = entry.find("atom:id", ns)
        if title_el is None or id_el is None:
            continue

        title    = title_el.text.strip().replace("\n", " ")
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
        arxiv_id = id_el.text.strip().split("/abs/")[-1]

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None
        ]
        pub_el = entry.find("atom:published", ns)
        doi_el = entry.find("arxiv:doi", ns)

        papers.append({
            "title":     title,
            "abstract":  abstract,
            "authors":   authors,
            "arxiv_id":  arxiv_id,
            "pdf_url":   pdf_url,
            "published": pub_el.text[:10] if pub_el is not None else "",
            "doi":       doi_el.text.strip() if doi_el is not None else "",
            "source":    "arxiv",
        })

    log.info(f"  → {len(papers)} results from arXiv")
    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar retrieval
# ---------------------------------------------------------------------------

def search_semantic_scholar(query: str, limit: int, client: httpx.Client) -> list:
    if not SEMANTIC_SCHOLAR_API_KEY:
        log.debug("No Semantic Scholar API key — skipping S2 search")
        return []

    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params  = {
        "query":  query,
        "limit":  limit,
        "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
    }
    try:
        log.info(f"Semantic Scholar search: {query[:70]}...")
        response = client.get(
            f"{S2_BASE}/paper/search",
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning(f"Semantic Scholar request failed: {e}")
        log.info("Rate limited — waiting 15 seconds before continuing...")
        time.sleep(15)
        return []

    time.sleep(S2_DELAY)

    papers = []
    for item in response.json().get("data", []):
        oa = item.get("openAccessPdf")
        if not oa:
            continue
        pdf_url = oa.get("url", "")
        if not pdf_url:
            continue
        ext = item.get("externalIds", {})
        papers.append({
            "title":     item.get("title", ""),
            "abstract":  item.get("abstract", "") or "",
            "authors":   [a.get("name", "") for a in item.get("authors", [])],
            "arxiv_id":  ext.get("ArXiv", ""),
            "pdf_url":   pdf_url,
            "published": str(item.get("year", "")),
            "doi":       ext.get("DOI", ""),
            "source":    "semantic_scholar",
        })

    log.info(f"  → {len(papers)} open-access results from Semantic Scholar")
    return papers


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

def download_pdf(pdf_url: str, destination: Path, client: httpx.Client) -> bool:
    try:
        log.info(f"  Downloading: {pdf_url[:70]}...")
        response = client.get(pdf_url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type and len(response.content) < 1000:
            log.warning("  Response doesn't look like a PDF — skipping")
            return False
        destination.write_bytes(response.content)
        log.info(f"  Saved: {destination.name} ({len(response.content)//1024} KB)")
        time.sleep(DOWNLOAD_DELAY)
        return True
    except httpx.HTTPError as e:
        log.warning(f"  Download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main retrieval logic
# ---------------------------------------------------------------------------

def retrieve_domain(
    domain_key: str,
    domain_config: dict,
    output_dir: Path,
    limit: int,
    client: httpx.Client,
) -> list:
    domain_dir    = output_dir / domain_key
    domain_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest      = load_manifest(manifest_path)

    log.info(f"\n{'='*60}")
    log.info(f"Domain: {domain_config['name']}")
    log.info(f"Output: {domain_dir}")
    log.info(f"{'='*60}")

    candidates  = []
    seen_titles = set()

    for query in domain_config["arxiv_queries"]:
        for p in search_arxiv(query, limit, client):
            key = p["title"].lower()[:60]
            if key not in seen_titles:
                seen_titles.add(key)
                candidates.append(p)

    for query in domain_config["s2_queries"]:
        for p in search_semantic_scholar(query, limit, client):
            key = p["title"].lower()[:60]
            if key not in seen_titles:
                seen_titles.add(key)
                candidates.append(p)

    log.info(f"\nTotal candidates after deduplication: {len(candidates)}")

    relevant = [
        p for p in candidates
        if is_relevant(p["title"], p["abstract"], domain_config["filter_terms"])
    ]
    log.info(f"Relevant after filtering: {len(relevant)}")

    relevant.sort(key=lambda p: p.get("published", ""), reverse=True)

    downloaded = []
    for paper in relevant[:limit]:
        title        = paper["title"]
        doi          = paper.get("doi", "")
        arxiv_id     = paper.get("arxiv_id", "")
        manifest_key = doi or arxiv_id or hashlib.md5(title.encode()).hexdigest()[:12]

        if manifest_key in manifest:
            log.info(f"  Already downloaded: {title[:60]}...")
            continue

        if doi:
            filename = doi_to_filename(doi)
        elif arxiv_id:
            filename = arxiv_id.replace("/", "_") + ".pdf"
        else:
            filename = url_to_filename(paper["pdf_url"])

        destination = domain_dir / filename

        if download_pdf(paper["pdf_url"], destination, client):
            entry = {
                "title":        title,
                "authors":      paper["authors"],
                "published":    paper.get("published", ""),
                "doi":          doi,
                "arxiv_id":     arxiv_id,
                "source":       paper["source"],
                "pdf_url":      paper["pdf_url"],
                "local_path":   str(destination),
                "domain":       domain_key,
                "domain_name":  domain_config["name"],
                "abstract":     paper["abstract"],
                "retrieved_at": datetime.utcnow().isoformat(),
            }
            manifest[manifest_key] = entry
            downloaded.append(entry)
            save_manifest(manifest, manifest_path)

    log.info(f"\nRetrieved {len(downloaded)} new papers for {domain_config['name']}")
    return downloaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve open-access TMD and 2D materials papers."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum papers to download per domain (default: 60).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/papers",
        help="Output directory for PDFs (default: data/papers).",
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Materials Papers Retrieval — TMD and 2D Materials")
    log.info(f"Output directory : {output_dir}")
    log.info(f"Limit per domain : {args.limit}")
    log.info(
        f"Semantic Scholar : {'ENABLED' if SEMANTIC_SCHOLAR_API_KEY else 'DISABLED (arXiv only)'}"
    )

    with httpx.Client(
        headers={"User-Agent": "materials-knowledge-system/1.0 (research project)"},
        follow_redirects=True,
    ) as client:
        total = 0
        for domain_key, domain_config in DOMAINS.items():
            total += len(retrieve_domain(
                domain_key=domain_key,
                domain_config=domain_config,
                output_dir=output_dir,
                limit=args.limit,
                client=client,
            ))

    log.info(f"\n{'='*60}")
    log.info(f"Retrieval complete. Total papers downloaded: {total}")
    log.info(f"Manifest saved to: {output_dir / 'manifest.json'}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
