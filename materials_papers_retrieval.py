#!/usr/bin/env python3
"""
materials_papers_retrieval.py
-----------------------------
Harvests open-access research papers from arXiv and Semantic Scholar
for the materials science knowledge graph project.
 
Usage:
    python materials_papers_retrieval.py --domain cmc --limit 10
    python materials_papers_retrieval.py --domain all --limit 10
    python materials_papers_retrieval.py --domain tbc --limit 20 --output ~/materials-knowledge/data/papers
 
Requires:
    pip install httpx python-dotenv
 
Environment variables (in .env file):
    SEMANTIC_SCHOLAR_API_KEY=your_key_here   # optional, enables S2 search
"""
 
import os
import sys
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
S2_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api/query"
 
# Rate limit delays (seconds)
ARXIV_DELAY = 3.0          # arXiv asks for >= 3s between requests
S2_DELAY = 1.0             # S2 free tier: 1 req/sec
DOWNLOAD_DELAY = 2.0       # polite delay between PDF downloads
 
REQUEST_TIMEOUT = 30       # seconds
 
# ---------------------------------------------------------------------------
# Domain definitions
# Each domain has a list of search queries for arXiv and Semantic Scholar.
# arXiv category cond-mat.mtrl-sci covers most materials science papers.
# ---------------------------------------------------------------------------
 
DOMAINS = {
    "cmc": {
        "name": "Ceramic Matrix Composites",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:ceramic AND ti:composite",
            "cat:cond-mat.mtrl-sci AND abs:SiC AND abs:ceramic AND abs:aerospace",
            "cat:cond-mat.mtrl-sci AND abs:ceramic AND abs:matrix AND abs:turbine",
        ],
        "s2_queries": [
            "ceramic matrix composites aerospace high temperature",
            "SiC/SiC composite turbine aeroengine",
            "CMC manufacturing aerospace",
        ],
        "filter_terms": [
            "ceramic", "composite", "CMC", "SiC", "turbine",
            "aerospace", "high temperature", "matrix"
        ],
    },
    "tbc": {
        "name": "Thermal Barrier Coatings",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:thermal AND ti:barrier AND ti:coating",
            "cat:cond-mat.mtrl-sci AND abs:TBC AND abs:turbine",
            "cat:cond-mat.mtrl-sci AND abs:thermal AND abs:barrier AND abs:aerospace",
        ],
        "s2_queries": [
            "thermal barrier coating turbine high temperature",
            "TBC aerospace multilayer coating",
            "thermal barrier coating wear resistance hypersonic",
        ],
        "filter_terms": [
            "thermal barrier", "TBC", "coating", "turbine",
            "thermal insulation", "hypersonic", "zirconia"
        ],
    },
    "hydrogen": {
        "name": "Hydrogen Storage Transport Testing",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:hydrogen AND ti:storage",
            "cat:cond-mat.mtrl-sci AND abs:hydrogen AND abs:permeation AND abs:coating",
            "cat:cond-mat.mtrl-sci AND abs:hydrogen AND abs:storage AND abs:lightweight",
        ],
        "s2_queries": [
            "hydrogen storage materials lightweight",
            "hydrogen permeation resistant coating",
            "hydrogen storage metamaterials safety",
        ],
        "filter_terms": [
            "hydrogen", "storage", "permeation", "H2",
            "fuel cell", "net-zero", "lightweight"
        ],
    },
    "vitrimers": {
        "name": "Vitrimers and Reprocessable Polymers",
        "arxiv_queries": [
            "cat:cond-mat.soft AND ti:vitrimer",
            "cat:cond-mat.soft AND abs:vitrimer AND abs:recyclable",
            "cat:cond-mat.mtrl-sci AND abs:reprocessable AND abs:polymer",
        ],
        "s2_queries": [
            "vitrimer reprocessable polymer composite",
            "self-healing recyclable thermoset vitrimer",
            "vitrimer high temperature aerospace",
        ],
        "filter_terms": [
            "vitrimer", "reprocessable", "recyclable", "self-healing",
            "thermoset", "dynamic covalent", "circular"
        ],
    },
    "ald": {
        "name": "Atomic Layer Deposition and Plasma Deposition",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:atomic AND ti:layer AND ti:deposition",
            "cat:cond-mat.mtrl-sci AND abs:ALD AND abs:thin AND abs:film",
            "cat:cond-mat.mtrl-sci AND abs:plasma AND abs:deposition AND abs:coating",
        ],
        "s2_queries": [
            "atomic layer deposition thin film coating",
            "ALD hydrogen resistant coating",
            "plasma deposition nanoscale electronics sensor",
        ],
        "filter_terms": [
            "atomic layer deposition", "ALD", "plasma deposition",
            "thin film", "nanoscale", "coating", "CVD"
        ],
    },
    "energy_storage": {
        "name": "Energy Storage and Future Transport",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:energy AND ti:storage AND ti:material",
            "cat:cond-mat.mtrl-sci AND abs:thermal AND abs:energy AND abs:storage",
            "cat:cond-mat.mtrl-sci AND abs:corrosion AND abs:high AND abs:temperature",
        ],
        "s2_queries": [
            "advanced materials energy storage thermal",
            "high temperature corrosion resistance coating energy",
            "thermal energy storage materials durability",
        ],
        "filter_terms": [
            "energy storage", "thermal storage", "corrosion",
            "battery", "electrode", "electrolyte", "transport"
        ],
    },
    "armour": {
        "name": "Lightweight Armour and Protection Systems",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:armour",
            "cat:cond-mat.mtrl-sci AND abs:impact AND abs:composite AND abs:armour",
            "cat:cond-mat.mtrl-sci AND abs:energy AND abs:absorbing AND abs:composite",
        ],
        "s2_queries": [
            "lightweight armour composite impact resistant",
            "hybrid armour structure energy absorbing",
            "ceramic composite armour defence",
        ],
        "filter_terms": [
            "armour", "armor", "impact", "ballistic", "protection",
            "energy absorbing", "composite", "defence", "defense"
        ],
    },
    "rubber_polymers": {
        "name": "Rubber and Polymer Materials",
        "arxiv_queries": [
            "cat:cond-mat.soft AND ti:elastomer AND ti:aerospace",
            "cat:cond-mat.soft AND abs:rubber AND abs:fatigue",
            "cat:cond-mat.soft AND abs:polymer AND abs:dynamic AND abs:loading",
        ],
        "s2_queries": [
            "advanced elastomer aerospace seal fatigue",
            "polymer dynamic loading harsh chemical conditions",
            "rubber polymer characterisation mechanical",
        ],
        "filter_terms": [
            "elastomer", "rubber", "polymer", "fatigue",
            "dynamic loading", "seal", "viscoelastic"
        ],
    },
    "extreme_environments": {
        "name": "Extreme Environments and Extreme Cold",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:extreme AND ti:temperature",
            "cat:cond-mat.mtrl-sci AND abs:cryogenic AND abs:material",
            "cat:cond-mat.mtrl-sci AND abs:ultra AND abs:high AND abs:temperature",
        ],
        "s2_queries": [
            "materials extreme environment ultra high temperature",
            "cryogenic materials mechanical properties",
            "engineered surface coating extreme corrosion",
        ],
        "filter_terms": [
            "extreme", "cryogenic", "ultra-high temperature", "UHTC",
            "corrosive", "mechanical loading", "thermal shock"
        ],
    },
    "biomaterials": {
        "name": "Biomaterials and Mycology",
        "arxiv_queries": [
            "cat:cond-mat.mtrl-sci AND ti:mycelium AND ti:composite",
            "cat:cond-mat.mtrl-sci AND abs:bio AND abs:derived AND abs:material",
            "cat:cond-mat.mtrl-sci AND abs:fungal AND abs:material",
        ],
        "s2_queries": [
            "mycelium composite sustainable lightweight material",
            "bio-derived material construction insulation",
            "biopolymer engineered sustainable low carbon",
        ],
        "filter_terms": [
            "mycelium", "fungal", "bio-derived", "biopolymer",
            "sustainable", "low-carbon", "regenerative"
        ],
    },
}
 
# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
 
def doi_to_filename(doi: str) -> str:
    """Convert a DOI to a safe filename."""
    return doi.replace("/", "_").replace(":", "_") + ".pdf"
 
 
def url_to_filename(url: str) -> str:
    """Generate a filename from a URL using a short hash."""
    return hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"
 
 
def is_relevant(title: str, abstract: str, filter_terms: list) -> bool:
    """
    Check whether a paper is relevant to the domain by looking for
    filter terms in the title and abstract. At least one term must appear.
    Title matches are weighted higher — if one appears in the title, pass.
    """
    title_lower = title.lower()
    abstract_lower = abstract.lower()
 
    for term in filter_terms:
        if term.lower() in title_lower:
            return True
 
    matches = sum(1 for term in filter_terms if term.lower() in abstract_lower)
    return matches >= 2
 
 
def load_manifest(manifest_path: Path) -> dict:
    """Load existing manifest or return empty dict."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}
 
 
def save_manifest(manifest: dict, manifest_path: Path):
    """Save manifest to disk."""
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
 
 
# ---------------------------------------------------------------------------
# arXiv retrieval
# ---------------------------------------------------------------------------
 
def search_arxiv(query: str, limit: int, client: httpx.Client) -> list:
    """
    Search arXiv and return a list of paper dicts.
    Each dict has: title, abstract, authors, arxiv_id, pdf_url, published
    """
    params = {
        "search_query": query,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
 
    try:
        log.info(f"arXiv search: {query[:60]}...")
        response = client.get(ARXIV_BASE, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning(f"arXiv request failed: {e}")
        return []
 
    time.sleep(ARXIV_DELAY)
 
    # Parse Atom XML
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
 
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        log.warning(f"arXiv XML parse error: {e}")
        return []
 
    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
 
        if title_el is None or id_el is None:
            continue
 
        title = title_el.text.strip().replace("\n", " ")
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
        arxiv_url = id_el.text.strip()
        arxiv_id = arxiv_url.split("/abs/")[-1]
 
        # Find PDF link
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
 
        # Authors
        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None
        ]
 
        # Published date
        pub_el = entry.find("atom:published", ns)
        published = pub_el.text[:10] if pub_el is not None else ""
 
        # DOI if present
        doi_el = entry.find("arxiv:doi", ns)
        doi = doi_el.text.strip() if doi_el is not None else ""
 
        papers.append({
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "published": published,
            "doi": doi,
            "source": "arxiv",
        })
 
    log.info(f"  → {len(papers)} results from arXiv")
    return papers
 
 
# ---------------------------------------------------------------------------
# Semantic Scholar retrieval
# ---------------------------------------------------------------------------
 
def search_semantic_scholar(query: str, limit: int, client: httpx.Client) -> list:
    """
    Search Semantic Scholar and return a list of paper dicts.
    Requires SEMANTIC_SCHOLAR_API_KEY to be set.
    """
    if not SEMANTIC_SCHOLAR_API_KEY:
        log.debug("No Semantic Scholar API key — skipping S2 search")
        return []
 
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
    }
 
    try:
        log.info(f"Semantic Scholar search: {query[:60]}...")
        response = client.get(
            f"{S2_BASE}/paper/search",
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning(f"Semantic Scholar request failed: {e}")
        return []
 
    time.sleep(S2_DELAY)
 
    data = response.json()
    papers = []
 
    for item in data.get("data", []):
        open_access = item.get("openAccessPdf")
        if not open_access:
            continue  # skip papers without open access PDF
 
        pdf_url = open_access.get("url", "")
        if not pdf_url:
            continue
 
        external_ids = item.get("externalIds", {})
        doi = external_ids.get("DOI", "")
        arxiv_id = external_ids.get("ArXiv", "")
 
        authors = [
            a.get("name", "")
            for a in item.get("authors", [])
        ]
 
        papers.append({
            "title": item.get("title", ""),
            "abstract": item.get("abstract", "") or "",
            "authors": authors,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "published": str(item.get("year", "")),
            "doi": doi,
            "source": "semantic_scholar",
        })
 
    log.info(f"  → {len(papers)} open-access results from Semantic Scholar")
    return papers
 
 
# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------
 
def download_pdf(pdf_url: str, destination: Path, client: httpx.Client) -> bool:
    """
    Download a PDF to destination path.
    Returns True on success, False on failure.
    """
    try:
        log.info(f"  Downloading: {pdf_url[:70]}...")
        response = client.get(pdf_url, timeout=60, follow_redirects=True)
        response.raise_for_status()
 
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type and len(response.content) < 1000:
            log.warning(f"  Response doesn't look like a PDF — skipping")
            return False
 
        destination.write_bytes(response.content)
        size_kb = len(response.content) // 1024
        log.info(f"  Saved: {destination.name} ({size_kb} KB)")
        time.sleep(DOWNLOAD_DELAY)
        return True
 
    except httpx.HTTPError as e:
        log.warning(f"  Download failed: {e}")
        return False
 
 
# ---------------------------------------------------------------------------
# Main harvesting logic
# ---------------------------------------------------------------------------
 
def harvest_domain(
    domain_key: str,
    domain_config: dict,
    output_dir: Path,
    limit: int,
    client: httpx.Client,
) -> list:
    """
    Harvest papers for a single domain.
    Returns list of manifest entries for successfully downloaded papers.
    """
    domain_dir = output_dir / domain_key
    domain_dir.mkdir(parents=True, exist_ok=True)
 
    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
 
    log.info(f"\n{'='*60}")
    log.info(f"Domain: {domain_config['name']}")
    log.info(f"Output: {domain_dir}")
    log.info(f"{'='*60}")
 
    # Collect candidates from all sources
    candidates = []
    seen_titles = set()
 
    # arXiv searches
    for query in domain_config["arxiv_queries"]:
        papers = search_arxiv(query, limit, client)
        for p in papers:
            key = p["title"].lower()[:60]
            if key not in seen_titles:
                seen_titles.add(key)
                candidates.append(p)
 
    # Semantic Scholar searches
    for query in domain_config["s2_queries"]:
        papers = search_semantic_scholar(query, limit, client)
        for p in papers:
            key = p["title"].lower()[:60]
            if key not in seen_titles:
                seen_titles.add(key)
                candidates.append(p)
 
    log.info(f"\nTotal candidates after deduplication: {len(candidates)}")
 
    # Filter for relevance
    relevant = [
        p for p in candidates
        if is_relevant(p["title"], p["abstract"], domain_config["filter_terms"])
    ]
    log.info(f"Relevant after filtering: {len(relevant)}")
 
    # Sort by recency — prefer newer papers
    relevant.sort(key=lambda p: p.get("published", ""), reverse=True)
 
    # Download up to limit papers
    downloaded = []
    for paper in relevant[:limit]:
        title = paper["title"]
 
        # Check if already in manifest
        doi = paper.get("doi", "")
        arxiv_id = paper.get("arxiv_id", "")
        manifest_key = doi or arxiv_id or hashlib.md5(title.encode()).hexdigest()[:12]
 
        if manifest_key in manifest:
            log.info(f"  Already downloaded: {title[:60]}...")
            continue
 
        # Determine filename
        if doi:
            filename = doi_to_filename(doi)
        elif arxiv_id:
            filename = arxiv_id.replace("/", "_") + ".pdf"
        else:
            filename = url_to_filename(paper["pdf_url"])
 
        destination = domain_dir / filename
 
        # Download
        success = download_pdf(paper["pdf_url"], destination, client)
 
        if success:
            entry = {
                "title": title,
                "authors": paper["authors"],
                "published": paper.get("published", ""),
                "doi": doi,
                "arxiv_id": arxiv_id,
                "source": paper["source"],
                "pdf_url": paper["pdf_url"],
                "local_path": str(destination),
                "domain": domain_key,
                "domain_name": domain_config["name"],
                "abstract": paper["abstract"],
                "harvested_at": datetime.utcnow().isoformat(),
            }
            manifest[manifest_key] = entry
            downloaded.append(entry)
 
            # Save manifest after each successful download
            save_manifest(manifest, manifest_path)
 
    log.info(f"\nDownloaded {len(downloaded)} new papers for {domain_config['name']}")
    return downloaded
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Harvest open-access materials science papers."
    )
    parser.add_argument(
        "--domain",
        choices=list(DOMAINS.keys()) + ["all"],
        default="cmc",
        help="Domain to harvest (default: cmc). Use 'all' for all domains.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum papers to download per domain (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/papers",
        help="Output directory for PDFs (default: data/papers).",
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
 
    log.info(f"Materials Papers Retrieval")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Limit per domain: {args.limit}")
 
    if SEMANTIC_SCHOLAR_API_KEY:
        log.info(f"Semantic Scholar: ENABLED (key loaded)")
    else:
        log.info(f"Semantic Scholar: DISABLED (no API key — arXiv only)")
 
    domains_to_run = DOMAINS if args.domain == "all" else {args.domain: DOMAINS[args.domain]}
 
    with httpx.Client(
        headers={"User-Agent": "materials-knowledge-system/1.0 (research project)"},
        follow_redirects=True,
    ) as client:
        total_downloaded = 0
        for domain_key, domain_config in domains_to_run.items():
            results = harvest_domain(
                domain_key=domain_key,
                domain_config=domain_config,
                output_dir=output_dir,
                limit=args.limit,
                client=client,
            )
            total_downloaded += len(results)
 
    log.info(f"\n{'='*60}")
    log.info(f"Harvest complete. Total papers downloaded: {total_downloaded}")
    log.info(f"Manifest saved to: {output_dir / 'manifest.json'}")
    log.info(f"{'='*60}")
 
 
if __name__ == "__main__":
    main()
 
