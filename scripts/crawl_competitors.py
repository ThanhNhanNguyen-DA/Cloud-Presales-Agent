"""
Crawl public data from Vietnamese cloud competitors (VNG, Viettel, FPT, CMC, Bizfly).

Design principle — "only correct data": this crawler NEVER fabricates figures. For
every page it fetches it stores exactly what the page returned (source URL, fetch
timestamp, page title, cleaned text, and any HTML tables found verbatim). Fields it
cannot parse with confidence are left empty rather than guessed. What you get is a
faithful record of the real public pages, which you can review before trusting.

It is polite by default: it honours each site's robots.txt, sends an identifying
User-Agent, waits between requests, and only follows same-domain links whose URL
looks like a product / pricing / service page (depth 1 from the homepage).

Outputs (default → ./data/competitors/):
  * <provider>.json   — one record per crawled page (title, text, tables, url, time)
  * summary.csv       — one row per page (provider, url, title, #tables, text length)

Optionally, with --ingest, the cleaned page text is chunked, embedded (local Qwen),
and inserted into the `cloud_documents` table (category "Competitor") for RAG.

Usage:
    # Crawl all providers, write files only (review first — recommended)
    python -m scripts.crawl_competitors

    # Only some providers, more pages each
    python -m scripts.crawl_competitors --providers vng fpt --max-pages 12

    # Crawl AND ingest the text into the RAG store
    python -m scripts.crawl_competitors --ingest

Requires: requests, beautifulsoup4 (in requirements.txt). --ingest additionally needs
a configured .env (SUPABASE_*) and downloads the embedding model on first use.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# Public homepages of Vietnamese cloud providers (non-bigtech competitors).
PROVIDERS: Dict[str, str] = {
    "vng": "https://vngcloud.vn/",
    "viettel": "https://viettelcloud.vn/",
    "fpt": "https://fptcloud.com/",
    "cmc": "https://cmccloud.vn/",
    "bizfly": "https://bizflycloud.vn/",
}

# A same-domain link is followed only if its URL path contains one of these slugs,
# matched on whole hyphen-delimited tokens (so "gia" matches .../bang-gia but NOT
# .../giao-duc). This keeps the crawl on product/pricing/service pages.
LINK_KEYWORDS = (
    "bang-gia", "gia", "pricing", "price",
    "san-pham", "product", "products", "dich-vu", "service", "services",
    "cloud-server", "server", "storage", "luu-tru", "database",
    "kubernetes", "k8s", "cdn", "solution", "solutions", "giai-phap",
)

# Any URL whose path contains one of these (news, blog, careers, PR…) is rejected,
# even if it also matches a keyword — this kept CMC news pages out of the crawl.
EXCLUDE_KEYWORDS = (
    "tin-tuc", "blog", "news", "khuyen-mai", "tuyen-dung", "su-kien",
    "event", "events", "ve-chung-toi", "about", "lien-he", "contact",
    "careers", "bao-chi", "press", "webinar", "case-study",
)

USER_AGENT = (
    "cloud-presales-agent competitor-research bot "
    "(+internal presales tooling; contact: presales team)"
)
HEADERS = {"User-Agent": USER_AGENT}


@dataclass
class Page:
    provider: str
    url: str
    title: str
    fetched_at: str
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)  # tables[t][row][cell]


def _robots_for(base_url: str) -> RobotFileParser:
    """Load and parse robots.txt for the site of `base_url` (fail-open on error)."""
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        resp = requests.get(robots_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp.allow_all = True
    except Exception:
        rp.allow_all = True  # if robots.txt can't be read, don't block ourselves
    return rp


def _fetch(url: str, timeout: float = 15.0) -> str | None:
    """GET `url`; return HTML text on a 200 HTML response, else None."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as exc:
        print(f"    ✗ fetch error {url}: {exc}")
        return None
    if resp.status_code != 200:
        print(f"    ✗ HTTP {resp.status_code} {url}")
        return None
    if "html" not in resp.headers.get("content-type", "").lower():
        return None
    return resp.text


def _clean_text(soup: BeautifulSoup) -> str:
    """Extract readable body text, dropping scripts/styles/nav/header/footer."""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _extract_tables(soup: BeautifulSoup) -> List[List[List[str]]]:
    """Return every HTML table verbatim as rows of cell text (nothing invented)."""
    tables: List[List[List[str]]] = []
    for table in soup.find_all("table"):
        rows: List[List[str]] = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _path_tokens(url: str) -> List[str]:
    """Hyphen/slash-delimited tokens of the URL path, lowercased."""
    return [t for t in re.split(r"[/\-_.]+", urlparse(url).path.lower()) if t]


def _phrase_in(tokens: List[str], phrase: str) -> bool:
    """True if `phrase` (its own tokens) appears as a contiguous run in `tokens`."""
    kw = phrase.split("-")
    n = len(kw)
    return any(tokens[i:i + n] == kw for i in range(len(tokens) - n + 1))


def _url_is_relevant(url: str) -> bool:
    """A product/pricing/service page: matches a keyword and no exclusion, on whole tokens."""
    tokens = _path_tokens(url)
    if not tokens:
        return False  # bare domain / homepage is handled separately
    if any(_phrase_in(tokens, ex) for ex in EXCLUDE_KEYWORDS):
        return False
    return any(_phrase_in(tokens, kw) for kw in LINK_KEYWORDS)


def _discover_links(soup: BeautifulSoup, base_url: str, domain: str) -> List[str]:
    """Same-domain product/pricing/service links (token-matched, news/blog excluded)."""
    found: List[str] = []
    seen: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0].rstrip("/")
        if not href or href in seen:
            continue
        parsed = urlparse(href)
        if parsed.netloc != domain or parsed.scheme not in ("http", "https"):
            continue
        if _url_is_relevant(href):
            seen.add(href)
            found.append(href)
    return found


def crawl_provider(name: str, homepage: str, max_pages: int, delay: float) -> List[Page]:
    """Crawl a provider's homepage + up to `max_pages` product/pricing pages (depth 1)."""
    domain = urlparse(homepage).netloc
    rp = _robots_for(homepage)
    print(f"\n▶ {name} ({domain})")

    def allowed(url: str) -> bool:
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    pages: List[Page] = []
    visited: Set[str] = set()

    # 1) homepage
    queue: List[str] = [homepage.rstrip("/")]
    home_html = _fetch(homepage) if allowed(homepage) else None
    if home_html:
        soup = BeautifulSoup(home_html, "html.parser")
        pages.append(Page(name, homepage, (soup.title.string or "").strip() if soup.title else "",
                          datetime.now(timezone.utc).isoformat(), _clean_text(soup), _extract_tables(soup)))
        visited.add(homepage.rstrip("/"))
        queue = _discover_links(soup, homepage, domain)
        print(f"    ✓ homepage · discovered {len(queue)} candidate page(s)")

    # 2) discovered product/pricing pages (depth 1)
    for url in queue:
        if len([p for p in pages if p.url != homepage]) >= max_pages:
            break
        if url in visited:
            continue
        visited.add(url)
        if not allowed(url):
            print(f"    ⚠ robots.txt disallows {url} — skipped")
            continue
        time.sleep(delay)
        html = _fetch(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = _clean_text(soup)
        if len(text) < 120:  # near-empty / JS-only page — nothing correct to keep
            continue
        pages.append(Page(name, url, (soup.title.string or "").strip() if soup.title else "",
                          datetime.now(timezone.utc).isoformat(), text, _extract_tables(soup)))
        print(f"    ✓ {url}  ({len(text)} chars, {len(pages[-1].tables)} table(s))")

    print(f"  → {len(pages)} page(s) kept for {name}")
    return pages


def write_outputs(all_pages: Dict[str, List[Page]], out_dir: Path) -> None:
    """Write per-provider JSON and a combined summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pages in all_pages.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps([asdict(p) for p in pages], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = out_dir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["provider", "url", "title", "num_tables", "text_length", "fetched_at"])
        for name, pages in all_pages.items():
            for p in pages:
                writer.writerow([p.provider, p.url, p.title, len(p.tables), len(p.text), p.fetched_at])
    total = sum(len(p) for p in all_pages.values())
    print(f"\n✅ Wrote {total} page(s) to {out_dir}/  (per-provider JSON + summary.csv)")


def ingest_to_rag(all_pages: Dict[str, List[Page]], chunk_size: int, overlap: int) -> None:
    """Chunk + embed page text and insert into cloud_documents (category 'Competitor')."""
    # Imported lazily so file-only crawls don't require .env / the embedding model.
    from app.db.supabase_client import supabase
    from app.tools.embedding_tool import embed_text
    from scripts.text_chunking import chunk_text

    inserted = 0
    for name, pages in all_pages.items():
        for p in pages:
            chunks = chunk_text(p.text, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                supabase.table("cloud_documents").insert({
                    "service_name": f"{name} (competitor)",
                    "category": "Competitor",
                    "content": chunk,
                    "embedding": embed_text(chunk),
                    "metadata": {
                        "source": p.url,
                        "type": "competitor_crawl",
                        "provider": name,
                        "title": p.title,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    },
                }).execute()
                inserted += 1
    print(f"✅ Ingested {inserted} chunk(s) into cloud_documents (category 'Competitor').")


def load_from_files(providers: List[str], out_dir: Path) -> Dict[str, List[Page]]:
    """
    Load previously crawled pages from <out_dir>/<provider>.json, dropping news/blog
    pages (per EXCLUDE_KEYWORDS) and JS-empty shells (< 120 chars, e.g. "Loading...").
    Lets you ingest the good pages without re-crawling.
    """
    all_pages: Dict[str, List[Page]] = {}
    total_dropped = 0
    for name in providers:
        fp = out_dir / f"{name}.json"
        if not fp.exists():
            print(f"  ⚠ {fp} not found — skipped")
            all_pages[name] = []
            continue
        records = json.loads(fp.read_text(encoding="utf-8"))
        kept: List[Page] = []
        for rec in records:
            tokens = _path_tokens(rec["url"])
            if tokens and any(_phrase_in(tokens, ex) for ex in EXCLUDE_KEYWORDS):
                total_dropped += 1
                continue
            if len((rec.get("text") or "").strip()) < 120:  # JS-empty shell
                total_dropped += 1
                continue
            kept.append(Page(
                provider=rec["provider"], url=rec["url"], title=rec.get("title", ""),
                fetched_at=rec.get("fetched_at", ""), text=rec.get("text", ""),
                tables=rec.get("tables", []),
            ))
        all_pages[name] = kept
        print(f"  {name}: kept {len(kept)}, dropped {len(records) - len(kept)}")
    print(f"  → total dropped (news/empty): {total_dropped}")
    return all_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl public data from Vietnamese cloud competitors.")
    parser.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS),
                        help=f"Subset to crawl (default: all — {', '.join(sorted(PROVIDERS))}).")
    parser.add_argument("--max-pages", type=int, default=8, help="Max product/pricing pages per provider (default 8).")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds to wait between requests (default 1.5).")
    parser.add_argument("--out", default="data/competitors", help="Output directory (default data/competitors).")
    parser.add_argument("--ingest", action="store_true", help="Also insert page text into cloud_documents (RAG).")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--from-files", action="store_true",
                        help="Skip crawling; load & filter saved data/competitors/*.json (drops news/empty), then --ingest.")
    args = parser.parse_args()

    targets = args.providers or sorted(PROVIDERS)

    if args.from_files:
        all_pages = load_from_files(targets, Path(args.out))
        if args.ingest:
            ingest_to_rag(all_pages, args.chunk_size, args.overlap)
        else:
            kept = sum(len(p) for p in all_pages.values())
            print(f"\nLoaded {kept} page(s) from files. Add --ingest to insert into cloud_documents.")
        return

    all_pages: Dict[str, List[Page]] = {}
    for name in targets:
        all_pages[name] = crawl_provider(name, PROVIDERS[name], args.max_pages, args.delay)

    write_outputs(all_pages, Path(args.out))
    if args.ingest:
        ingest_to_rag(all_pages, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
