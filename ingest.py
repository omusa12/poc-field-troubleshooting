#!/usr/bin/env python3
"""
ingest.py - Build FAISS vector index from the DOCSIS corpus.

Run:
    python ingest.py

Pipeline:
    1. Load PDFs (PyPDFLoader) from tier1/ and tier2/
    2. Load web markdown files (custom frontmatter parser) from web/
    3. Chunk with RecursiveCharacterTextSplitter
    4. Embed with HuggingFace all-MiniLM-L6-v2 (local, no API key)
    5. Build and save FAISS index to faiss_index/

Re-running this script overwrites the existing index.
"""
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_PRIMARY,
    EMBEDDING_MODEL_VOYAGE,
    FAISS_INDEX_DIR,
    METADATA_RULES,
    TIER1_DIR,
    TIER2_DIR,
    USE_VOYAGE_EMBEDDINGS,
    WEB_DIR,
)

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_metadata_for_file(file_path: Path) -> dict:
    """
    Derive metadata from filename using METADATA_RULES.
    First matching rule wins. Falls back to generic tags.
    """
    meta = {
        "source": str(file_path),
        "doc_type": "document",
        "docsis_version": "3.0/3.1/4.0",
        "device_model": "generic",
        "topic": "DOCSIS",
    }
    stem = file_path.stem
    for pattern, tags in METADATA_RULES.items():
        if pattern.lower() in stem.lower():
            meta.update(tags)
            break
    return meta


def parse_markdown_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML-style frontmatter written by download_docs.py.
    Returns (metadata_dict, body_content).
    """
    meta = {}
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            for line in content[3:end].strip().splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
            return meta, content[end + 4:].strip()
    return meta, content


def load_pdfs(directories: list[Path]) -> list[Document]:
    """Load all PDFs from given directories using LangChain PyPDFLoader."""
    pdf_files = []
    for directory in directories:
        if directory.exists():
            pdf_files.extend(sorted(directory.glob("*.pdf")))

    if not pdf_files:
        print("[warn] No PDFs found. Run download_docs.py first.")
        return []

    print(f"\n[PDFs] Loading {len(pdf_files)} files...")
    documents = []
    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            base_meta = get_metadata_for_file(pdf_path)
            for page in pages:
                page.metadata.update(base_meta)
                # Normalize page number key
                if "page" in page.metadata:
                    page.metadata["page_number"] = page.metadata["page"] + 1
            documents.extend(pages)
        except Exception as e:
            print(f"\n  [error] {pdf_path.name}: {e}")

    print(f"[PDFs] {len(documents)} pages from {len(pdf_files)} files")
    return documents


def load_web_markdown(web_dir: Path) -> list[Document]:
    """Load web markdown files, parsing YAML frontmatter into Document metadata."""
    if not web_dir.exists():
        print("[warn] No web/ directory. Run download_docs.py first.")
        return []

    md_files = sorted(web_dir.glob("*.md"))
    if not md_files:
        print("[warn] No markdown files in web/. Run download_docs.py first.")
        return []

    print(f"\n[Web] Loading {len(md_files)} markdown files...")
    documents = []
    for md_path in tqdm(md_files, desc="Web"):
        try:
            raw = md_path.read_text(encoding="utf-8")
            frontmatter, body = parse_markdown_frontmatter(raw)

            if len(body.strip()) < 100:
                print(f"\n  [skip] {md_path.name} — too short")
                continue

            meta = {
                "source":        frontmatter.get("source", str(md_path)),
                "label":         frontmatter.get("label", md_path.stem),
                "doc_type":      "web_article",
                "docsis_version": frontmatter.get("docsis_version", "3.0/3.1/4.0"),
                "topic":         frontmatter.get("topic", "DOCSIS"),
                "device_model":  frontmatter.get("device_model", "generic"),
                "file_name":     md_path.name,
            }
            documents.append(Document(page_content=body, metadata=meta))
        except Exception as e:
            print(f"\n  [error] {md_path.name}: {e}")

    print(f"[Web] {len(documents)} documents loaded")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks. Metadata is preserved on every chunk."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    print(f"\n[Chunking] Splitting {len(documents)} documents...")
    chunks = splitter.split_documents(documents)

    # Drop chunks that are mostly whitespace
    meaningful = [c for c in chunks if len(c.page_content.strip()) > 50]
    discarded = len(chunks) - len(meaningful)

    print(f"[Chunking] {len(meaningful)} chunks ({discarded} too-short discarded)")
    return meaningful


def build_embeddings():
    """Initialize embedding model per config settings."""
    if USE_VOYAGE_EMBEDDINGS:
        try:
            import streamlit as st
            import os
            from langchain_voyageai import VoyageAIEmbeddings
            key = st.secrets.get("VOYAGE_API_KEY", "") or os.environ.get("VOYAGE_API_KEY", "")
            print(f"[Embeddings] Voyage AI: {EMBEDDING_MODEL_VOYAGE}")
            return VoyageAIEmbeddings(voyage_api_key=key, model=EMBEDDING_MODEL_VOYAGE)
        except Exception as e:
            print(f"[warn] Voyage AI unavailable ({e}), falling back to HuggingFace")

    print(f"[Embeddings] HuggingFace: {EMBEDDING_MODEL_PRIMARY}")
    print("[Embeddings] First run downloads the model (~90MB)...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PRIMARY,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss_index(chunks: list[Document], embeddings) -> None:
    """Build FAISS index from chunks and save to disk."""
    print(f"\n[FAISS] Building index for {len(chunks)} chunks...")
    print("[FAISS] This takes 2–5 minutes on first run...")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    print(f"[FAISS] Saved to {FAISS_INDEX_DIR}/")


def print_stats(pdf_docs: list, web_docs: list, chunks: list) -> None:
    print("\n" + "=" * 65)
    print("INGESTION COMPLETE")
    print("=" * 65)
    print(f"  PDF pages loaded:      {len(pdf_docs):>6,}")
    print(f"  Web pages loaded:      {len(web_docs):>6,}")
    print(f"  Total source docs:     {len(pdf_docs) + len(web_docs):>6,}")
    print(f"  Chunks in index:       {len(chunks):>6,}")
    print(f"  Chunk size / overlap:  {CHUNK_SIZE} chars / {CHUNK_OVERLAP} chars")
    print(f"  FAISS index at:        {FAISS_INDEX_DIR}/")
    print("\nNext step:  streamlit run app.py")
    print("=" * 65)


def main() -> None:
    print("\n=== DOCSIS RAG Ingestion Pipeline ===\n")

    pdf_docs = load_pdfs([TIER1_DIR, TIER2_DIR])
    web_docs = load_web_markdown(WEB_DIR)
    all_docs = pdf_docs + web_docs

    if not all_docs:
        print("\n[error] No documents found. Run python download_docs.py first.")
        sys.exit(1)

    chunks = chunk_documents(all_docs)
    if not chunks:
        print("\n[error] No meaningful chunks produced.")
        sys.exit(1)

    embeddings = build_embeddings()
    build_faiss_index(chunks, embeddings)
    print_stats(pdf_docs, web_docs, chunks)


if __name__ == "__main__":
    main()
