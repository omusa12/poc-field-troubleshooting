#!/usr/bin/env python3
"""
download_docs.py - Idempotent downloader for the DOCSIS RAG corpus.

Run:
    python download_docs.py

Behavior:
- Skips files that already exist (safe to re-run)
- Downloads PDFs to data/raw/tier1/ and data/raw/tier2/
- Scrapes web pages to data/raw/web/ as clean markdown with YAML frontmatter
- Handles failures gracefully — logs and continues
- Prints manual download instructions for auth-gated sources
"""
import re
import sys
import time
import traceback
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    AUTH_REQUIRED_SOURCES,
    PDF_SOURCES,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    TIER1_DIR,
    TIER2_DIR,
    WEB_DIR,
    WEB_SOURCES,
)


def ensure_dirs() -> None:
    for directory in [TIER1_DIR, TIER2_DIR, WEB_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("[setup] Data directories ready")


def download_pdf(label: str, url: str, dest: Path) -> str:
    """Download a PDF. Returns 'downloaded', 'skipped', or 'failed'."""
    if dest.exists():
        print(f"  [skip] {label}")
        return "skipped"

    print(f"  [download] {label}")
    print(f"    {url}")
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
            print(f"    [warn] Unexpected content-type: {content_type} — saving anyway")

        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)

        print(f"    [ok] {dest.name} ({total / 1024:.0f} KB)")
        return "downloaded"

    except requests.exceptions.HTTPError as e:
        print(f"    [error] HTTP {e.response.status_code} — {e}")
        if e.response.status_code == 404:
            print(f"    [note] File not found at this URL. Check for updated URL.")
        return "failed"
    except requests.exceptions.ConnectionError:
        print(f"    [error] Connection failed — check network")
        return "failed"
    except requests.exceptions.Timeout:
        print(f"    [error] Timeout after {REQUEST_TIMEOUT}s")
        return "failed"
    except Exception:
        traceback.print_exc()
        return "failed"


def scrape_web_page(label: str, url: str, slug: str, extra_meta: dict) -> str:
    """
    Fetch a web page, convert to markdown, save with YAML frontmatter.
    Returns 'saved', 'skipped', or 'failed'.

    The YAML frontmatter block is parsed by ingest.py to populate
    LangChain Document.metadata fields for source citations.
    """
    dest = WEB_DIR / f"{slug}.md"

    if dest.exists():
        print(f"  [skip] {label}")
        return "skipped"

    print(f"  [scrape] {label}")
    print(f"    {url}")
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise elements
        for tag in soup.find_all(["nav", "footer", "script", "style", "header",
                                   "aside", "noscript", "form", "button", "iframe"]):
            tag.decompose()

        # Prefer main content area over full body
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=lambda c: c and "content" in c.lower())
            or soup.find("div", id=lambda i: i and "content" in i.lower())
            or soup.body
            or soup
        )

        raw_md = md(str(main), heading_style="ATX", bullets="-")
        clean_md = re.sub(r"\n{3,}", "\n\n", raw_md).strip()

        if len(clean_md.split()) < 50:
            print(f"    [warn] Very little content extracted ({len(clean_md.split())} words) — saving anyway")

        # Build YAML frontmatter block
        meta_lines = ["---", f"source: {url}", f"label: {label}"]
        for key, val in extra_meta.items():
            meta_lines.append(f"{key}: {val}")
        meta_lines.append("---")

        content = "\n".join(meta_lines) + "\n\n" + clean_md

        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"    [ok] {slug}.md (~{len(clean_md.split())} words)")
        return "saved"

    except requests.exceptions.HTTPError as e:
        print(f"    [error] HTTP {e.response.status_code}")
        if e.response.status_code == 403:
            print(f"    [note] 403 Forbidden — site may block scrapers.")
            print(f"           Manual fallback: copy page text into {dest}")
        return "failed"
    except requests.exceptions.ConnectionError:
        print(f"    [error] Connection failed")
        return "failed"
    except requests.exceptions.Timeout:
        print(f"    [error] Timeout after {REQUEST_TIMEOUT}s")
        return "failed"
    except Exception:
        traceback.print_exc()
        return "failed"


def print_auth_instructions() -> None:
    if not AUTH_REQUIRED_SOURCES:
        return
    print("\n" + "=" * 65)
    print("MANUAL DOWNLOAD REQUIRED — AUTH-GATED SOURCES")
    print("These documents require membership/login to download.")
    print("If you have access, download them and place in data/raw/tier1/")
    print("=" * 65)
    for i, src in enumerate(AUTH_REQUIRED_SOURCES, 1):
        print(f"\n{i}. {src['label']}")
        print(f"   URL: {src['url']}")
        print(f"   Action: {src['instructions']}")
    print()


def main() -> None:
    print("\n=== DOCSIS RAG Corpus Downloader ===\n")
    ensure_dirs()

    # ── PDFs ──────────────────────────────────────────────────────────────────
    print("\n[PDFs] Tier 1 & 2 documents...")
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for label, url, dest, _tier in PDF_SOURCES:
        result = download_pdf(label, url, dest)
        counts[result] += 1
        time.sleep(0.5)

    print(f"\n[PDFs] {counts['downloaded']} downloaded, "
          f"{counts['skipped']} skipped, {counts['failed']} failed")

    # ── Web scraping ───────────────────────────────────────────────────────────
    print("\n[Web] Tier 3 web content...")
    web_counts = {"saved": 0, "skipped": 0, "failed": 0}
    for src in WEB_SOURCES:
        extra = {k: v for k, v in src.items() if k not in ("label", "url", "slug")}
        result = scrape_web_page(src["label"], src["url"], src["slug"], extra)
        web_counts[result] += 1
        time.sleep(1.0)

    print(f"\n[Web] {web_counts['saved']} saved, "
          f"{web_counts['skipped']} skipped, {web_counts['failed']} failed")

    # ── Auth-required notice ───────────────────────────────────────────────────
    print_auth_instructions()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_pdfs = counts["downloaded"] + counts["skipped"]
    total_web  = web_counts["saved"] + web_counts["skipped"]
    print("=" * 65)
    print("DOWNLOAD COMPLETE")
    print(f"  PDFs available:        {total_pdfs} ({counts['failed']} failed)")
    print(f"  Web pages available:   {total_web} ({web_counts['failed']} failed)")
    print(f"  Auth-gated (manual):   {len(AUTH_REQUIRED_SOURCES)}")
    print("\nNext step:  python ingest.py  — build the FAISS index")
    print("=" * 65)


if __name__ == "__main__":
    main()
