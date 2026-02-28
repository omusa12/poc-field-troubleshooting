"""
app.py - DOCSIS Troubleshooting Assistant (Streamlit)

Architecture:
    LangChain FAISS  →  vector similarity search (retrieval)
    Anthropic SDK    →  LLM call with streaming
    Streamlit        →  chat UI, auth gate, sidebar

Run:
    streamlit run app.py

Requires .streamlit/secrets.toml with:
    ANTHROPIC_API_KEY = "sk-ant-..."
    ACCESS_CODE = "your-demo-code"
"""
import sys
from pathlib import Path

import re

import anthropic
import streamlit as st

# LangChain — used only for FAISS retrieval
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
    EMBEDDING_MODEL_PRIMARY,
    EMBEDDING_MODEL_VOYAGE,
    EXAMPLE_QUESTIONS,
    FAISS_INDEX_DIR,
    RETRIEVAL_MIN_RELEVANT_CHUNKS,
    RETRIEVAL_K,
    RETRIEVAL_WEAK_DISTANCE_THRESHOLD,
    SYSTEM_PROMPT_PATH,
    USE_VOYAGE_EMBEDDINGS,
)

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Hitron branding CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .hitron-header {
        background: linear-gradient(90deg, #0C2340 0%, #00A4E4 100%);
        padding: 14px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .hitron-header h1 {
        color: white !important;
        font-size: 1.45rem;
        margin: 0;
        line-height: 1.3;
    }
    .hitron-header p {
        color: #cce8f7;
        font-size: 0.82rem;
        margin: 5px 0 0 0;
    }
    .source-citation {
        font-size: 0.8rem;
        color: #555;
        border-left: 3px solid #00A4E4;
        padding: 4px 8px;
        margin: 3px 0;
        background: #f7fbff;
        border-radius: 0 4px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Cached resource loaders ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def _load_hf_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PRIMARY,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Loading DOCSIS knowledge base...")
def load_vectorstore():
    """Load FAISS index once per session. Returns None if index doesn't exist."""
    if not FAISS_INDEX_DIR.exists():
        return None

    if USE_VOYAGE_EMBEDDINGS:
        try:
            from langchain_voyageai import VoyageAIEmbeddings
            embeddings = VoyageAIEmbeddings(
                voyage_api_key=st.secrets["VOYAGE_API_KEY"],
                model=EMBEDDING_MODEL_VOYAGE,
            )
        except Exception:
            st.warning("Voyage AI unavailable — using HuggingFace embeddings.")
            embeddings = _load_hf_embeddings()
    else:
        embeddings = _load_hf_embeddings()

    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # Safe: we built this index via ingest.py
    )


@st.cache_data(show_spinner=False)
def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a DOCSIS network troubleshooting expert. "
        "Provide accurate, cited guidance based on retrieved context."
    )


def get_chunk_count() -> int:
    try:
        vs = load_vectorstore()
        return vs.index.ntotal if vs else 0
    except Exception:
        return 0


def assess_retrieval_quality(
    sources: list[dict],
    distance_strategy: str,
    retrieved_count: int,
) -> dict:
    """
    Classify retrieval quality from FAISS distance scores.

    Retrieval is marked weak when too few high-quality chunks remain after
    threshold filtering.
    """
    if not sources:
        return {
            "is_weak": True,
            "reason": "No documentation chunks passed the source-quality threshold.",
            "relevant_count": 0,
            "total_count": 0,
            "retrieved_count": retrieved_count,
            "distance_strategy": distance_strategy,
        }

    relevant_count = len(sources)
    is_weak = len(sources) < RETRIEVAL_MIN_RELEVANT_CHUNKS
    if is_weak:
        reason = "Too few high-quality chunks were available after filtering."
    else:
        reason = "Retrieved context quality is acceptable after filtering."

    return {
        "is_weak": is_weak,
        "reason": reason,
        "relevant_count": relevant_count,
        "total_count": len(sources),
        "retrieved_count": retrieved_count,
        "distance_strategy": distance_strategy,
    }


# ─── Authentication gate ──────────────────────────────────────────────────────
def render_auth_gate() -> None:
    st.markdown(
        """
        <div class="hitron-header">
            <h1>🔧 DOCSIS Troubleshooting Assistant</h1>
            <p>Powered by Hitron Technologies | Field Diagnostics POC</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.subheader("Access Required")
        password = st.text_input(
            "Enter access code:",
            type="password",
            placeholder="Enter your demo access code",
        )
        if st.button("Sign In", type="primary", use_container_width=True):
            try:
                if password == st.secrets["ACCESS_CODE"]:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid access code. Contact your Hitron representative.")
            except KeyError:
                st.error("Configuration error: ACCESS_CODE not set in secrets.toml")


# ─── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve_context(query: str, vectorstore) -> tuple[str, list[dict], dict]:
    """
    Run FAISS similarity search.
    Returns (context_text_for_llm, sources_list_for_ui, retrieval_quality).
    """
    if vectorstore is None:
        retrieval_quality = assess_retrieval_quality(
            [],
            distance_strategy="unknown",
            retrieved_count=0,
        )
        return "", [], retrieval_quality

    results = vectorstore.similarity_search_with_score(query, k=RETRIEVAL_K)
    distance_strategy = str(getattr(vectorstore, "distance_strategy", "unknown")).split(".")[-1]
    retrieved_count = len(results)

    # Keep only chunks that pass the configured quality threshold.
    filtered_results = [
        (doc, float(score))
        for (doc, score) in results
        if float(score) <= RETRIEVAL_WEAK_DISTANCE_THRESHOLD
    ]

    context_parts = []
    sources = []

    for i, (doc, score) in enumerate(filtered_results):
        meta = doc.metadata
        label        = meta.get("label", meta.get("source", "Unknown source"))
        docsis_ver   = meta.get("docsis_version", "")
        page_num     = meta.get("page_number", meta.get("page", ""))
        topic        = meta.get("topic", "")

        # Build labeled context block for the LLM
        header = f"[Source {i+1}: {label}"
        if docsis_ver:
            header += f" | DOCSIS {docsis_ver}"
        if topic:
            header += f" | {topic}"
        if page_num:
            header += f" | Page {page_num}"
        header += "]"

        context_parts.append(f"{header}\n{doc.page_content}")

        sources.append({
            "label":        label,
            "source_url":   meta.get("source", ""),
            "doc_type":     meta.get("doc_type", ""),
            "docsis_version": docsis_ver,
            "page":         page_num,
            "topic":        topic,
            "distance":     score,
            "distance_strategy": distance_strategy,
            "snippet":      doc.page_content[:250] + "…" if len(doc.page_content) > 250 else doc.page_content,
        })

    retrieval_quality = assess_retrieval_quality(
        sources,
        distance_strategy,
        retrieved_count=retrieved_count,
    )
    return "\n\n---\n\n".join(context_parts), sources, retrieval_quality


# ─── LLM streaming ────────────────────────────────────────────────────────────
def stream_claude_response(
    user_query: str,
    context: str,
    chat_history: list[dict],
    system_prompt: str,
    retrieval_quality: dict | None = None,
):
    """
    Generator that yields Claude response tokens for st.write_stream().

    Uses Anthropic SDK directly (not LangChain) for reliable Streamlit streaming.
    LangChain ConversationalRetrievalChain streaming is fragile in Streamlit;
    the Anthropic SDK's text_stream iterator works cleanly with st.write_stream.
    """
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    if context:
        user_content = (
            f"Retrieved documentation context:\n\n{context}\n\n"
            f"---\n\nField technician query: {user_query}"
        )
    else:
        user_content = (
            f"Field technician query: {user_query}\n\n"
            "(No relevant documentation was retrieved from the knowledge base. "
            "Answer from your embedded DOCSIS expertise and note this limitation.)"
        )

    if retrieval_quality and retrieval_quality.get("is_weak"):
        user_content += (
            "\n\nRetrieval quality signal: WEAK.\n"
            f"- Retrieved chunks: {retrieval_quality.get('retrieved_count', 0)}\n"
            f"- High-quality chunks kept: {retrieval_quality.get('total_count', 0)}\n"
            "Instruction: Start your response with exactly "
            "\"Evidence quality: insufficient evidence from retrieved documentation.\" "
            "Then provide cautious, high-level troubleshooting steps and avoid "
            "specific claims that are not directly supported by the retrieved text."
        )

    # Include last 6 turns of history to support follow-up questions
    messages = []
    for turn in chat_history[-6:]:
        content = turn["content"]
        if turn["role"] == "assistant":
            # Keep source markers out of model history; they are UI-only metadata.
            content = _clean_citations(content)
        messages.append({"role": turn["role"], "content": content})
    messages.append({"role": "user", "content": user_content})

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=CLAUDE_TEMPERATURE,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ─── Source citation UI ───────────────────────────────────────────────────────
def render_retrieval_quality_warning(retrieval_quality: dict | None) -> None:
    if not retrieval_quality or not retrieval_quality.get("is_weak"):
        return

    st.warning(
        "Evidence quality warning: retrieved documentation matches are weak. "
        f"{retrieval_quality.get('reason', '')} "
        "Treat the response as provisional and verify against primary docs."
    )


def render_source_citations(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)} chunks retrieved)", expanded=False):
        for i, src in enumerate(sources):
            label       = src["label"]
            docsis_ver  = src.get("docsis_version", "")
            page        = src.get("page", "")
            snippet     = src.get("snippet", "")
            url         = src.get("source_url", "")

            header = f"**{i + 1}. {label}**"
            if docsis_ver:
                header += f" | DOCSIS {docsis_ver}"
            if page:
                header += f" | Page {page}"

            st.markdown(header)
            if url and url.startswith("http"):
                st.markdown(f"[View source]({url})")

            st.markdown(
                f'<div class="source-citation">{snippet}</div>',
                unsafe_allow_html=True,
            )

            if i < len(sources) - 1:
                st.markdown("---")


# ─── Inline citation helpers ──────────────────────────────────────────────────
def _extract_source_nums(text: str) -> list[int]:
    """Return unique, ordered source numbers from [Src:N] markers in text."""
    seen, result = set(), []
    for m in re.findall(r'\[(?:src|Src):\s*(\d+)\]', text):
        n = int(m)
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def _clean_citations(text: str) -> str:
    """Strip all [Src:N] markers from text."""
    return re.sub(r"\s*\[(?:src|Src):\s*\d+\]", "", text)


def _parse_sections(text: str) -> list[dict]:
    """
    Split response on markdown headers (## / ###).
    Returns list of {'content': str, 'source_nums': list[int]}.
    """
    header_re = re.compile(r'^(#{1,4}[^\n]+)$', re.MULTILINE)
    parts = header_re.split(text)
    # parts: [pre-header-text, header1, after1, header2, after2, ...]

    sections = []
    if parts[0].strip():
        sections.append({
            'content': parts[0],
            'source_nums': _extract_source_nums(parts[0]),
        })

    i = 1
    while i < len(parts):
        header = parts[i]
        after = parts[i + 1] if i + 1 < len(parts) else ''
        content = header + '\n' + after
        sections.append({
            'content': content,
            'source_nums': _extract_source_nums(content),
        })
        i += 2

    return sections


def render_response_with_inline_sources(response_text: str, sources: list[dict]) -> None:
    """
    Render assistant response with a collapsible source expander after each
    section that contains [Src:N] markers.  Falls back to the flat source
    list at the bottom when no inline markers are found (e.g. no RAG context).
    """
    sections = _parse_sections(response_text)
    has_inline = any(s['source_nums'] for s in sections)

    for section in sections:
        st.markdown(_clean_citations(section['content']))

        used = [
            (n, sources[n - 1])
            for n in section['source_nums']
            if sources and 1 <= n <= len(sources)
        ]
        if used:
            label = "📎 1 source" if len(used) == 1 else f"📎 {len(used)} sources"
            with st.expander(label, expanded=False):
                for idx, (n, src) in enumerate(used):
                    s_label     = src['label']
                    docsis_ver  = src.get('docsis_version', '')
                    page        = src.get('page', '')
                    snippet     = src.get('snippet', '')
                    url         = src.get('source_url', '')

                    header = f"**{n}. {s_label}**"
                    if docsis_ver:
                        header += f" | DOCSIS {docsis_ver}"
                    if page:
                        header += f" | Page {page}"
                    st.markdown(header)
                    if url and url.startswith("http"):
                        st.markdown(f"[View source]({url})")
                    if snippet:
                        st.markdown(
                            f'<div class="source-citation">{snippet}</div>',
                            unsafe_allow_html=True,
                        )
                    if idx < len(used) - 1:
                        st.markdown("---")

    if not has_inline:
        render_source_citations(sources)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(vectorstore) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="background:#0C2340;padding:12px 16px;border-radius:8px;margin-bottom:16px;">
                <p style="color:#00A4E4;font-weight:700;font-size:1.05rem;margin:0;">
                    Hitron Technologies
                </p>
                <p style="color:#cce8f7;font-size:0.78rem;margin:4px 0 0 0;">
                    DOCSIS Diagnostics POC
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**System**")
        chunk_count = get_chunk_count()
        st.caption(f"Model: {CLAUDE_MODEL}")
        st.caption(f"Embedding: all-MiniLM-L6-v2")
        st.caption(f"Knowledge base: {chunk_count:,} chunks")
        st.caption(f"Retrieval: top-{RETRIEVAL_K} per query")

        if vectorstore is None:
            st.warning(
                "Knowledge base not loaded.\n\n"
                "Run:\n```\npython download_docs.py\npython ingest.py\n```"
            )

        st.divider()

        st.markdown("**Try asking:**")
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            label = example[:60] + "…" if len(example) > 60 else example
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_example = example

        st.divider()

        if st.button("🔄 Reset conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources_history = {}
            st.session_state.retrieval_quality_history = {}
            st.rerun()

        st.divider()
        st.markdown(
            "<p style='font-size:0.7rem;color:#aaa;'>POC — not for production use. "
            "Consult official CableLabs and Hitron documentation for field decisions.</p>",
            unsafe_allow_html=True,
        )


# ─── Main chat interface ──────────────────────────────────────────────────────
def render_chat(vectorstore, system_prompt: str) -> None:
    st.markdown(
        """
        <div class="hitron-header">
            <h1>🔧 DOCSIS Troubleshooting Assistant</h1>
            <p>Powered by Hitron Technologies | Field Diagnostics POC</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = {}
    if "retrieval_quality_history" not in st.session_state:
        st.session_state.retrieval_quality_history = {}

    # Render conversation history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_retrieval_quality_warning(
                    st.session_state.retrieval_quality_history.get(i)
                )
                render_response_with_inline_sources(
                    msg["content"],
                    st.session_state.sources_history.get(i, []),
                )
            else:
                st.markdown(msg["content"])

    # Handle example question buttons from sidebar
    pending = st.session_state.pop("pending_example", None)

    # Chat input
    user_input = st.chat_input(
        "Describe the DOCSIS issue or paste diagnostic readings…"
    )

    query = pending or user_input
    if not query:
        return

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve context
    with st.spinner("Searching knowledge base…"):
        context, sources, retrieval_quality = retrieve_context(query, vectorstore)

    # Stream assistant response
    with st.chat_message("assistant"):
        render_retrieval_quality_warning(retrieval_quality)

        response_placeholder = st.empty()
        response_chunks = []
        for chunk in stream_claude_response(
            user_query=query,
            context=context,
            chat_history=st.session_state.messages[:-1],
            system_prompt=system_prompt,
            retrieval_quality=retrieval_quality,
        ):
            response_chunks.append(chunk)
            response_placeholder.markdown(_clean_citations("".join(response_chunks)))

        response_text = "".join(response_chunks)
        response_placeholder.empty()

        # Index at which the assistant message will be stored
        assistant_idx = len(st.session_state.messages)
        render_response_with_inline_sources(response_text, sources)

    # Persist to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.sources_history[assistant_idx] = sources
    st.session_state.retrieval_quality_history[assistant_idx] = retrieval_quality


# ─── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    if not st.session_state.get("authenticated", False):
        render_auth_gate()
        st.stop()

    vectorstore   = load_vectorstore()
    system_prompt = load_system_prompt()

    render_sidebar(vectorstore)
    render_chat(vectorstore, system_prompt)


if __name__ == "__main__":
    main()
