"""Polite, breadth-first web crawler. Fetches pages, strips boilerplate, and
yields clean text documents for the rest of the pipeline. Designed for building
a multilingual corpus from seed URLs or a local directory of HTML/text files."""
from __future__ import annotations
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass
class RawDoc:
    """A single fetched document before chunking."""
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


_BOILERPLATE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript")


def extract_text(html: str) -> str:
    """Pull readable text out of an HTML page, dropping nav/script/style chrome."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def _same_host(seed: str, candidate: str) -> bool:
    return urlparse(seed).netloc == urlparse(candidate).netloc


def _links(base_url: str, html: str) -> Iterator[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if href:
            yield urljoin(base_url, href)


def crawl_web(seeds: Iterable[str], max_pages: int = 100, max_depth: int = 2,
              same_host_only: bool = True, delay_s: float = 0.2,
              timeout_s: float = 10.0) -> Iterator[RawDoc]:
    """Breadth-first crawl from `seeds`, staying on-host by default. Yields a
    RawDoc per fetched page. `delay_s` throttles requests to stay polite."""
    seeds = list(seeds)
    seen: Set[str] = set()
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    fetched = 0

    with httpx.Client(follow_redirects=True, timeout=timeout_s,
                      headers={"User-Agent": "BabelCrawler/1.0"}) as http:
        while queue and fetched < max_pages:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            try:
                resp = http.get(url)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue
            if "text/html" not in resp.headers.get("content-type", ""):
                continue

            text = extract_text(resp.text)
            fetched += 1
            yield RawDoc(text=text, metadata={"source": url, "depth": depth})

            if depth < max_depth:
                for link in _links(url, resp.text):
                    if link not in seen and (not same_host_only or _same_host(seeds[0], link)):
                        queue.append((link, depth + 1))
            time.sleep(delay_s)


def crawl_directory(path: str, glob: str = "**/*") -> Iterator[RawDoc]:
    """Load .html/.htm/.txt/.md files from a local directory as RawDocs."""
    root = Path(path)
    for fp in sorted(root.glob(glob)):
        if fp.suffix.lower() not in {".html", ".htm", ".txt", ".md"}:
            continue
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        text = extract_text(raw) if fp.suffix.lower() in {".html", ".htm"} else raw
        yield RawDoc(text=text, metadata={"source": str(fp), "file_type": fp.suffix.lstrip(".")})
