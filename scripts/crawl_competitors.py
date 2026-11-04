"""
Crawl public product & pricing pages of Vietnamese cloud competitors.

Design principle — "only correct data": this crawler NEVER fabricates figures. For every
page it stores exactly what the page returned (source URL, fetch time, title, cleaned
text, and HTML tables verbatim). Nothing is inferred or filled in.

Polite by default: honours robots.txt (and skips sites whose robots.txt reserves AI use via
Content-Signal), sends an identifying User-Agent, waits between requests, and only fetches
same-domain product / pricing pages.

Page discovery (v2), per provider:
  * the provider's sitemap.xml (when it has one), filtered by a provider-specific URL regex;
  * links on the homepage and on "service list" seed pages, filtered the same way;
  * news / blog / events / careers pages are always excluded, and Vietnamese pages are
    preferred (an /en/ page is skipped when its Vietnamese twin is also queued).

Cleaning:
  * menu / footer lines that repeat on most pages of a provider are removed;
  * pages whose text is still tiny are reported as JS-rendered (needs a headless browser).

Outputs (default → ./data/competitors/):
  * <provider>.json — one record per kept page;   summary.csv — one row per page.

With --ingest, each page becomes text chunks + one chunk per pricing-table row group (the
table header is repeated in every chunk), embedded with the local Qwen model and written to
`cloud_documents` with category "Competitor". Re-ingesting a provider first deletes its
previous Competitor rows, so the store never holds two crawls of the same site.

Usage:
    python -m scripts.crawl_competitors                       # crawl all, write files (review!)
    python -m scripts.crawl_competitors --providers vnpt fpt  # subset
    python -m scripts.crawl_competitors --from-files --ingest # ingest the reviewed files
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "cloud-presales-agent competitor-research bot "
    "(+internal presales tooling; contact: presales team)"
)
HEADERS = {"User-Agent": USER_AGENT}

# Always excluded, whatever the provider (news, blog, marketing, account pages…).
EXCLUDE_RE = re.compile(
    r"/(tin-tuc|news|blog|blogs|events?|su-kien|khuyen-mai|promotions?|tuyen-dung|careers?|"
    r"tac-gia|ve-chung-toi|about|lien-he|contacts?|bao-chi|press|webinar|case-stud(y|ies)|"
    r"dang-nhap|dang-ky|login|register|account|search|tim-kiem|tai-lieu)(/|$|-)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Provider:
    name: str
    homepage: str
    include: str                     # regex a URL path must match to be fetched
    sitemap: str | None = None
    seeds: tuple[str, ...] = ()      # listing pages whose links are followed (depth 1)
    role: str = "competitor"         # "competitor" or "own" (our own product site)
    sitemap_only: bool = False       # accept homepage/seed links only if also in the sitemap
    note: str = ""


PROVIDERS: dict[str, Provider] = {p.name: p for p in (
    Provider("vng", "https://vngcloud.vn/", r"^/(en/)?(product|solution)/[^/]+$",
             sitemap="https://vngcloud.vn/sitemap.xml"),
    Provider("fpt", "https://fptcloud.com/", r"^/(product/[^/]+|bang-gia|san-pham)$",
             seeds=("https://fptcloud.com/san-pham",), note="no sitemap"),
    Provider("bizfly", "https://bizflycloud.vn/",
             r"^/(?!tong-dai-ai|ai-xu-ly-ho-so|voice-bot|call-center|qc-assistant|business-email|"
             r"email-transaction)[a-z0-9-]+(/bang-gia|/[a-z0-9-]+)?$",
             sitemap="https://bizflycloud.vn/sitemap.xml",
             seeds=("https://bizflycloud.vn/danh-sach-dich-vu",), sitemap_only=True),
    Provider("vnpt", "https://cloud.vnpt.vn/", r"^/dich-vu/[^/]+$",
             sitemap="https://cloud.vnpt.vn/sitemap.xml",
             note="cloud.vnpt.vn (vnptcloud.vn reserves AI use via Content-Signal and is not crawled)"),
    Provider("viettelidc", "https://viettelidc.com.vn/",
             r"^/(dich-vu(/[^/]+)?|dich-vu-[a-z-]+|CloudServer|StartCloud|VirtualPrivateCloud)$",
             sitemap="https://viettelidc.com.vn/sitemap.xml"),
    Provider("viettel", "https://viettelcloud.vn/", r"^/(products?|san-pham)(/[^/]+)?$",
             sitemap="https://viettelcloud.vn/sitemap.xml",
             note="site is rendered client-side; plain HTTP returns an empty shell"),
    Provider("cmc", "https://cmccloud.vn/", r"^/dich-vu(/[^/]+)?$",
             sitemap="https://cmccloud.vn/sitemap.xml", role="own",
             note="our own product site — ingested as category 'Product', not 'Competitor'"),
)}
DEFAULT_PROVIDERS = [p for p, cfg in PROVIDERS.items() if cfg.role == "competitor"]

MIN_PAGE_CHARS = 300          # below this (after cleaning) the page is treated as JS-only
BOILERPLATE_SHARE = 0.5       # a line on ≥ 50% of a provider's pages is menu/footer noise


@dataclass
class Page:
    provider: str
    url: str
    title: str
    fetched_at: str
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)  # tables[t][row][cell]


# --------------------------------------------------------------------------- fetching
def _robots_for(base_url: str) -> tuple[RobotFileParser, bool]:
    """Parse robots.txt; also report whether it reserves AI input/training (Content-Signal)."""
    parsed = urlparse(base_url)
    rp = RobotFileParser()
    ai_reserved = False
    try:
        resp = requests.get(f"{parsed.scheme}://{parsed.netloc}/robots.txt", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            signals = " ".join(re.findall(r"(?im)^content-signal:\s*(.+)$", resp.text)).lower()
            ai_reserved = "ai-input=no" in signals or "ai-train=no" in signals
        else:
            rp.allow_all = True
    except Exception:
        rp.allow_all = True
    return rp, ai_reserved


def _fetch(url: str, timeout: float = 20.0) -> str | None:
    """GET `url`; return HTML on a 200 HTML response, else None."""
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
    resp.encoding = resp.apparent_encoding if not resp.encoding else resp.encoding
    return resp.text


def _sitemap_urls(url: str, depth: int = 0) -> list[str]:
    """All page URLs of a sitemap (follows nested sitemap indexes, 2 levels)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
    except Exception:
        return []
    if resp.status_code != 200:
        return []
    out: list[str] = []
    for loc in re.findall(r"<loc>\s*([^<]+?)\s*</loc>", resp.text):
        if loc.endswith(".xml"):
            if depth < 2:
                out += _sitemap_urls(loc, depth + 1)
        else:
            out.append(loc)
    return out


# --------------------------------------------------------------------------- parsing
def _clean_text(soup: BeautifulSoup) -> str:
    """Readable body text, dropping scripts/styles/nav/header/footer."""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg", "form"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines()]
    return join_broken_lines([ln for ln in lines if ln])


def join_broken_lines(lines: list[str]) -> str:
    """
    Re-join lines that HTML inline tags split mid-sentence ("Dòng máy chủ giá" + "rẻ đáp
    ứng…"): a line starting with a lowercase letter continues the previous one, unless the
    previous line ends a sentence. Price cards ("116.820" / "đ/tháng") are joined the same way.
    """
    out: list[str] = []
    for line in (re.sub(r"[ \t\u00a0\u200b]{2,}", " ", ln).strip() for ln in lines):
        if not line:
            continue
        if out and line[:1].islower() and not out[-1].endswith((".", "!", "?", ":")):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def _extract_tables(soup: BeautifulSoup) -> list[list[list[str]]]:
    """Every HTML table verbatim as rows of cell text (nothing invented)."""
    tables: list[list[list[str]]] = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s{2,}", " ", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def normalize_url(url: str) -> str:
    return url.split("#")[0].split("?")[0].rstrip("/")


def url_is_candidate(url: str, provider: Provider) -> bool:
    """Same domain, matches the provider's include regex, not a news/blog/account page."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc.lower().removeprefix("www.") != urlparse(provider.homepage).netloc.lower().removeprefix("www."):
        return False
    path = parsed.path.rstrip("/") or "/"
    if EXCLUDE_RE.search(path + "/"):
        return False
    return re.search(provider.include, path) is not None


def prefer_vietnamese(urls: list[str]) -> list[str]:
    """Drop an /en/... URL when the same path without /en/ is also present."""
    present = set(urls)
    out = []
    for u in urls:
        p = urlparse(u)
        if p.path.startswith("/en/"):
            twin = f"{p.scheme}://{p.netloc}{p.path[3:]}"
            if twin in present:
                continue
        out.append(u)
    return out


def strip_boilerplate(pages: list[Page], share: float = BOILERPLATE_SHARE) -> int:
    """
    Remove lines that repeat on at least `share` of a provider's pages (menus, footers,
    cookie banners). Needs ≥ 3 pages to judge. Returns the number of distinct lines removed.
    """
    if len(pages) < 3:
        return 0
    counts = Counter(line for p in pages for line in set(p.text.splitlines()))
    threshold = max(2, int(len(pages) * share))
    noise = {line for line, n in counts.items() if n >= threshold and len(line) < 200}
    for p in pages:
        p.text = "\n".join(ln for ln in p.text.splitlines() if ln not in noise)
    return len(noise)


# --------------------------------------------------------------------------- crawling
def discover(provider: Provider, home_soup: BeautifulSoup | None) -> list[str]:
    """Candidate URLs from sitemap + homepage + seed pages, filtered and de-duplicated."""
    found: list[str] = []
    in_sitemap: set[str] = set()
    if provider.sitemap:
        in_sitemap = {normalize_url(u) for u in _sitemap_urls(provider.sitemap)}
        found += sorted(in_sitemap)
    if home_soup is not None:
        found += [urljoin(provider.homepage, a["href"]) for a in home_soup.find_all("a", href=True)]
    for seed in provider.seeds:
        html = _fetch(seed)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            found += [seed] + [urljoin(seed, a["href"]) for a in soup.find_all("a", href=True)]
    seen: set[str] = set()
    out = []
    for url in map(normalize_url, found):
        if provider.sitemap_only and url not in in_sitemap:
            continue
        if url not in seen and url_is_candidate(url, provider):
            seen.add(url)
            out.append(url)
    # Pricing pages first: they hold the facts presales needs most.
    out = prefer_vietnamese(out)
    return sorted(out, key=lambda u: (0 if re.search(r"bang-gia|pricing|price", u) else 1, u))


def crawl_provider(provider: Provider, max_pages: int, delay: float) -> tuple[list[Page], dict]:
    """Crawl one provider. Returns (kept pages, stats)."""
    print(f"\n▶ {provider.name} ({urlparse(provider.homepage).netloc})")
    stats = {"provider": provider.name, "candidates": 0, "fetched": 0, "kept": 0,
             "js_only": 0, "robots_blocked": 0, "boilerplate_lines": 0, "skipped": ""}
    rp, ai_reserved = _robots_for(provider.homepage)
    if ai_reserved:
        stats["skipped"] = "robots.txt Content-Signal reserves AI use"
        print("    ⚠ skipped: robots.txt Content-Signal reserves AI use")
        return [], stats

    def allowed(url: str) -> bool:
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    pages: list[Page] = []
    home_html = _fetch(provider.homepage) if allowed(provider.homepage) else None
    home_soup = BeautifulSoup(home_html, "html.parser") if home_html else None
    queue = discover(provider, home_soup)
    stats["candidates"] = len(queue)
    print(f"    {len(queue)} candidate page(s)")

    for url in queue:
        if len(pages) >= max_pages:
            break
        if not allowed(url):
            stats["robots_blocked"] += 1
            continue
        time.sleep(delay)
        html = _fetch(url)
        if not html:
            continue
        stats["fetched"] += 1
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        tables = _extract_tables(soup)
        text = _clean_text(soup)
        if len(text) < MIN_PAGE_CHARS and not tables:
            stats["js_only"] += 1
            continue
        pages.append(Page(provider.name, url, title, datetime.now(UTC).isoformat(), text, tables))

    stats["boilerplate_lines"] = strip_boilerplate(pages)
    pages = [p for p in pages if len(p.text) >= MIN_PAGE_CHARS or p.tables]
    stats["kept"] = len(pages)
    for p in pages:
        print(f"    ✓ {p.url}  ({len(p.text)} chars, {len(p.tables)} table(s))")
    if stats["js_only"] and not pages:
        stats["skipped"] = "client-side rendered (needs a headless browser)"
    print(f"  → kept {len(pages)} page(s)  [{stats}]")
    return pages, stats


def write_outputs(all_pages: dict[str, list[Page]], out_dir: Path) -> None:
    """Per-provider JSON and a combined summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pages in all_pages.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps([asdict(p) for p in pages], ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["provider", "url", "title", "num_tables", "text_length", "fetched_at"])
        for pages in all_pages.values():
            for p in pages:
                writer.writerow([p.provider, p.url, p.title, len(p.tables), len(p.text), p.fetched_at])
    print(f"\n✅ Wrote {sum(len(p) for p in all_pages.values())} page(s) to {out_dir}/")


def load_from_files(providers: list[str], out_dir: Path) -> dict[str, list[Page]]:
    """Reload saved pages (re-applying the current URL filters) without re-crawling."""
    all_pages: dict[str, list[Page]] = {}
    for name in providers:
        fp = out_dir / f"{name}.json"
        records = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []
        pages = [Page(**rec) for rec in records]
        for page in pages:  # re-apply cleaning that may postdate the saved crawl
            page.text = join_broken_lines(page.text.splitlines())
            page.tables = [[[re.sub(r"\s{2,}", " ", c).strip() for c in row] for row in t] for t in page.tables]
        kept = [p for p in pages
                if (url_is_candidate(p.url, PROVIDERS[name]) or normalize_url(p.url) ==
                    normalize_url(PROVIDERS[name].homepage))
                and (len(p.text) >= MIN_PAGE_CHARS or p.tables)]
        all_pages[name] = kept
        print(f"  {name}: kept {len(kept)} of {len(pages)}")
    return all_pages


# --------------------------------------------------------------------------- ingestion
def page_service_name(page: Page) -> str:
    """'<provider> - <product>' from the page title (before the first separator)."""
    head = re.split(r"\s[|–—-]\s", page.title or "")[0].strip() or urlparse(page.url).path.strip("/")
    return f"{page.provider} - {head}"[:100]


def table_chunks(table: list[list[str]], heading: str, rows_per_chunk: int = 12) -> list[str]:
    """Render a table as text chunks of `rows_per_chunk` rows, each repeating the header row."""
    if not table:
        return []
    header, body = table[0], table[1:] or []
    if not body:
        return [f"{heading}\n" + " | ".join(header)]
    out = []
    for i in range(0, len(body), rows_per_chunk):
        lines = [" | ".join(header), " | ".join("---" for _ in header)]
        lines += [" | ".join(row) for row in body[i:i + rows_per_chunk]]
        out.append(f"{heading}\n" + "\n".join(lines))
    return out


def page_chunks(page: Page, chunk_size: int, overlap: int) -> list[str]:
    """Text chunks (prefixed with provider + title for retrieval) plus table chunks."""
    from scripts.text_chunking import chunk_text

    heading = f"{page_service_name(page)} ({page.url})"
    chunks = [f"{heading}\n{c}" for c in chunk_text(page.text, chunk_size=chunk_size, overlap=overlap)]
    for t_idx, table in enumerate(page.tables):
        chunks += table_chunks(table, f"{heading} — bảng {t_idx + 1}")
    return chunks


def ingest_to_rag(all_pages: dict[str, list[Page]], chunk_size: int, overlap: int) -> None:
    """Replace each provider's previous crawl rows in cloud_documents with this crawl."""
    from app.db.supabase_client import supabase
    from app.tools.embedding_tool import embed_text

    for name, pages in all_pages.items():
        if not pages:
            print(f"  {name}: nothing to ingest")
            continue
        cfg = PROVIDERS[name]
        category = "Product" if cfg.role == "own" else "Competitor"
        deleted = (supabase.table("cloud_documents").delete()
                   .eq("metadata->>type", "competitor_crawl").eq("metadata->>provider", name)
                   .execute().data or [])
        rows = []
        for page in pages:
            kind = "pricing" if re.search(r"bang-gia|pricing|price", page.url) else "product"
            for idx, chunk in enumerate(page_chunks(page, chunk_size, overlap)):
                rows.append({
                    "service_name": page_service_name(page),
                    "category": category,
                    "content": chunk,
                    "embedding": embed_text(chunk),
                    "metadata": {
                        "source": page.url, "type": "competitor_crawl", "provider": name,
                        "role": cfg.role, "page_kind": kind, "title": page.title,
                        "fetched_at": page.fetched_at, "chunk_index": idx,
                    },
                })
        for i in range(0, len(rows), 100):
            supabase.table("cloud_documents").insert(rows[i:i + 100]).execute()
        print(f"  {name}: replaced {len(deleted)} old row(s) with {len(rows)} chunk(s) "
              f"(category {category!r})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl public data from Vietnamese cloud providers.")
    parser.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS),
                        help=f"Default: all competitors ({', '.join(DEFAULT_PROVIDERS)}); "
                             "'cmc' (own site) only when named.")
    parser.add_argument("--max-pages", type=int, default=60, help="Max pages per provider (default 60).")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default 1.0).")
    parser.add_argument("--out", default="data/competitors")
    parser.add_argument("--ingest", action="store_true", help="Insert into cloud_documents (RAG).")
    parser.add_argument("--from-files", action="store_true", help="Skip crawling; use saved JSON files.")
    parser.add_argument("--list", action="store_true", help="Only print the candidate URLs (no page fetches).")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()
    targets = args.providers or DEFAULT_PROVIDERS

    if args.list:
        for name in targets:
            cfg = PROVIDERS[name]
            html = _fetch(cfg.homepage)
            urls = discover(cfg, BeautifulSoup(html, "html.parser") if html else None)
            print(f"{name}: {len(urls)} candidate(s)")
            for u in urls:
                print("   ", u)
        return

    if args.from_files:
        all_pages = load_from_files(targets, Path(args.out))
    else:
        all_pages, report = {}, []
        for name in targets:
            all_pages[name], stats = crawl_provider(PROVIDERS[name], args.max_pages, args.delay)
            report.append(stats)
        write_outputs(all_pages, Path(args.out))
        (Path(args.out) / "crawl_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.ingest:
        ingest_to_rag(all_pages, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
