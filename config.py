"""
config.py - Central configuration for DOCSIS Troubleshooting Assistant POC.

All tunable constants live here to avoid magic numbers scattered across modules.
Every other module imports from this file.
"""
from pathlib import Path

# ─── Project root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent

# ─── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
TIER1_DIR       = RAW_DIR / "tier1"
TIER2_DIR       = RAW_DIR / "tier2"
WEB_DIR         = RAW_DIR / "web"
FAISS_INDEX_DIR = ROOT_DIR / "faiss_index"
PROMPTS_DIR     = ROOT_DIR / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

# ─── Embedding model ──────────────────────────────────────────────────────────
# Primary: free, local, no API key required (~90MB download on first run)
EMBEDDING_MODEL_PRIMARY = "sentence-transformers/all-MiniLM-L6-v2"
# Optional: Voyage AI (higher quality, requires VOYAGE_API_KEY + re-ingest)
EMBEDDING_MODEL_VOYAGE  = "voyage-3-large"
USE_VOYAGE_EMBEDDINGS   = False  # Toggle to True to use Voyage AI

# ─── LLM ──────────────────────────────────────────────────────────────────────
CLAUDE_MODEL       = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS  = 2048
CLAUDE_TEMPERATURE = 0.1  # Low for diagnostic consistency

# ─── RAG / Retrieval ──────────────────────────────────────────────────────────
RETRIEVAL_K   = 5    # Number of chunks to retrieve per query
CHUNK_SIZE    = 400  # Characters per chunk
CHUNK_OVERLAP = 50   # Overlap between adjacent chunks
# FAISS returns a distance score (lower is better). With normalized embeddings
# and L2 distance, larger values indicate weaker semantic match.
RETRIEVAL_WEAK_DISTANCE_THRESHOLD = 1.20
RETRIEVAL_MIN_RELEVANT_CHUNKS = 2

# ─── Web scraping ─────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── Document sources ─────────────────────────────────────────────────────────
# Each tuple: (label, url, dest_path, tier)
PDF_SOURCES = [
    # Tier 1 — Core DOCSIS technical references (all publicly downloadable)
    (
        "PNMP v03 - DOCSIS 3.0 PNM Best Practices (CableLabs CM-GL-PNMP-V03)",
        "https://volpefirm.com/wp-content/uploads/2017/01/CM-GL-PNMP-V03-160725.pdf",
        TIER1_DIR / "CM-GL-PNMP-V03.pdf",
        "tier1",
    ),
    (
        "DOCSIS 3.1 Operational Integration and PNM Tools (Volpe Firm)",
        "https://volpefirm.com/wp-content/uploads/2011/11/Volpe_DOCSIS3.1_Operation_Integration_and_PNM_Tools_v2_final.pdf",
        TIER1_DIR / "Volpe_DOCSIS31_PNM_Tools.pdf",
        "tier1",
    ),
    (
        "DOCSIS 3.1 Physical Layer Specification (CM-SP-PHYv3.1) — OFDM, MER, PLC, Profiles",
        "https://volpefirm.com/wp-content/uploads/2017/01/CM-SP-PHYv3.1-I08-151210.pdf",
        TIER1_DIR / "CM-SP-PHYv3.1.pdf",
        "tier1",
    ),
    (
        "DOCSIS 3.0 Physical Layer Specification (CM-SP-PHYv3.0) — SC-QAM Reference",
        "https://volpefirm.com/wp-content/uploads/2017/01/CM-SP-PHYv3.0-I08-090121.pdf",
        TIER1_DIR / "CM-SP-PHYv3.0.pdf",
        "tier1",
    ),
    (
        "ZCorum DOCSIS 3.1 PNM Toolbox — Practical Field Guide",
        "https://www.zcorum.com/wp-content/uploads/DOCSIS-3.1-and-the-PNM-Toolbox-The-Future-of-Plant-Maintenance-is-Here.pdf",
        TIER1_DIR / "ZCorum_DOCSIS31_PNM_Toolbox.pdf",
        "tier1",
    ),
    # Tier 2 — Hitron product manuals
    (
        "Hitron CODA User Manual (DOCSIS 3.1 Cable Modem)",
        "https://us.hitrontech.com/wp-content/uploads/2021/05/CODA_UM_003.pdf",
        TIER2_DIR / "CODA_UM.pdf",
        "tier2",
    ),
    (
        "Hitron CODA56 User Manual (Multi-Gigabit DOCSIS 3.1)",
        "https://us.hitrontech.com/wp-content/uploads/2023/01/CODA56-UM.pdf",
        TIER2_DIR / "CODA56_UM.pdf",
        "tier2",
    ),
]

# Web pages to scrape — saved as markdown with YAML frontmatter
WEB_SOURCES = [
    {
        "label": "CableLabs PNM Technology Overview",
        "url": "https://cablelabs.com/technologies/proactive-network-maintenance",
        "slug": "cablelabs_pnm_overview",
        "docsis_version": "3.0/3.1/4.0",
        "topic": "PNM overview",
    },
    {
        "label": "CableLabs DOCSIS 4.0 Technology",
        "url": "https://cablelabs.com/technologies/docsis-4-0-technology",
        "slug": "cablelabs_docsis40",
        "docsis_version": "4.0",
        "topic": "DOCSIS 4.0 FDD FDX overview",
    },
    {
        "label": "CableLabs Blog: The Evolution of DOCSIS PNM",
        "url": "https://cablelabs.com/blog/the-evolution-of-docsis-proactive-network-maintenance",
        "slug": "cablelabs_pnm_evolution",
        "docsis_version": "3.0/3.1/4.0",
        "topic": "PNM history MIB data management objects",
    },
    {
        "label": "CableLabs Blog: Tooling Up for DOCSIS 4.0",
        "url": "https://cablelabs.com/blog/tooling-up-for-docsis-technology",
        "slug": "cablelabs_docsis40_tooling",
        "docsis_version": "4.0",
        "topic": "DOCSIS 4.0 readiness WG5",
    },
    {
        "label": "ZCorum: What is DOCSIS PNM and Why You Need It",
        "url": "https://blog.zcorum.com/how-docsis-pnm-works-and-why-you-need-it",
        "slug": "zcorum_pnm_explainer",
        "docsis_version": "3.0/3.1",
        "topic": "PNM pre-equalization fault localization",
    },
    {
        "label": "AOI Guide to DOCSIS 4.0",
        "url": "https://ao-inc.com/html-resources/guide-to-docsis-4-0/",
        "slug": "aoi_docsis40_guide",
        "docsis_version": "4.0",
        "topic": "DOCSIS 4.0 deployment operator considerations",
    },
    {
        "label": "Hitron ProPulse Press Release",
        "url": "https://us.hitrontech.com/press-releases/hitron-launches-hitron-propulse/",
        "slug": "hitron_propulse",
        "docsis_version": "3.1",
        "topic": "ProPulse PNM platform",
        "device_model": "ProPulse",
    },
    {
        "label": "SCTE Blog: DOCSIS Technology Is Ready",
        "url": "https://scte.org/blog/docsis-technology-is-ready/",
        "slug": "scte_docsis_ready",
        "docsis_version": "4.0",
        "topic": "DOCSIS 4.0 readiness SCTE 300 migration",
    },
]

# Auth-required sources — cannot auto-download, print instructions instead
AUTH_REQUIRED_SOURCES = [
    {
        "label": "CableLabs DOCSIS 4.0 Tools & Readiness (Cable Operator Preparations)",
        "url": "https://account.cablelabs.com/server/alfresco/bc2527a6-42f4-4ddb-83ac-62b656f0d0be",
        "instructions": (
            "Requires CableLabs account. Log in at account.cablelabs.com, "
            "search for 'DOCSIS 4.0 Tools Readiness', download PDF, "
            "save to data/raw/tier1/CableLabs_D40_Readiness.pdf"
        ),
    },
    {
        "label": "CableLabs PNM Best Practices Primer: HFC Networks (DOCSIS 3.1) — CM-GL-PNM-3.1",
        "url": "https://www.cablelabs.com/specifications/CM-GL-PNM-3.1",
        "instructions": (
            "Requires CableLabs account. Log in at account.cablelabs.com, "
            "search for CM-GL-PNM-3.1, download PDF, "
            "save to data/raw/tier1/CM-GL-PNM-3.1.pdf"
        ),
    },
    {
        "label": "SCTE 280: Understanding & Troubleshooting Cable RF Spectrum (Downstream)",
        "url": "https://account.scte.org/standards/library/",
        "instructions": (
            "Requires SCTE membership. Log in at account.scte.org, "
            "search SCTE 280, download PDF, "
            "save to data/raw/tier1/SCTE_280.pdf"
        ),
    },
    {
        "label": "SCTE 294: Understanding & Troubleshooting Cable Upstream RF Spectrum",
        "url": "https://account.scte.org/standards/library/",
        "instructions": (
            "Requires SCTE membership. Log in at account.scte.org, "
            "search SCTE 294, download PDF, "
            "save to data/raw/tier1/SCTE_294.pdf"
        ),
    },
]

# ─── Metadata tagging rules ────────────────────────────────────────────────────
# Maps filename stem patterns → metadata fields applied during ingestion.
# First matching rule wins.
METADATA_RULES = {
    "CM-GL-PNMP-V03":         {"docsis_version": "3.0",       "doc_type": "best_practices",  "topic": "PNM diagnostics"},
    "Volpe_DOCSIS31":         {"docsis_version": "3.1",       "doc_type": "technical_guide", "topic": "PNM OFDM integration"},
    "CM-SP-PHYv3.1":          {"docsis_version": "3.1",       "doc_type": "specification",   "topic": "OFDM MER PLC profiles"},
    "CM-SP-PHYv3.0":          {"docsis_version": "3.0",       "doc_type": "specification",   "topic": "SC-QAM physical layer"},
    "ZCorum_DOCSIS31":        {"docsis_version": "3.1",       "doc_type": "field_guide",     "topic": "PNM tools pre-equalization"},
    "CODA56":                 {"docsis_version": "3.1",       "doc_type": "product_manual",  "topic": "hardware LED codes", "device_model": "CODA56"},
    "CODA_UM":                {"docsis_version": "3.1",       "doc_type": "product_manual",  "topic": "hardware LED codes", "device_model": "CODA"},
    "CableLabs_D40":          {"docsis_version": "4.0",       "doc_type": "best_practices",  "topic": "DOCSIS 4.0 readiness"},
    "CM-GL-PNM-3.1":          {"docsis_version": "3.1",       "doc_type": "best_practices",  "topic": "PNM OFDM diagnostics"},
    "SCTE_280":               {"docsis_version": "3.0/3.1",   "doc_type": "standard",        "topic": "RF spectrum downstream troubleshooting"},
    "SCTE_294":               {"docsis_version": "3.0/3.1",   "doc_type": "standard",        "topic": "RF spectrum upstream troubleshooting"},
    "cablelabs_pnm":          {"docsis_version": "3.0/3.1/4.0", "doc_type": "web_article",   "topic": "PNM"},
    "cablelabs_docsis40":     {"docsis_version": "4.0",       "doc_type": "web_article",     "topic": "DOCSIS 4.0 FDD FDX"},
    "zcorum_pnm":             {"docsis_version": "3.0/3.1",   "doc_type": "web_article",     "topic": "PNM pre-equalization fault localization"},
    "aoi_docsis40":           {"docsis_version": "4.0",       "doc_type": "web_article",     "topic": "DOCSIS 4.0 deployment"},
    "hitron_propulse":        {"docsis_version": "3.1",       "doc_type": "press_release",   "topic": "ProPulse PNM", "device_model": "ProPulse"},
    "scte_docsis_ready":      {"docsis_version": "4.0",       "doc_type": "web_article",     "topic": "DOCSIS 4.0 readiness migration"},
}

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
APP_TITLE    = "DOCSIS Troubleshooting Assistant"
APP_SUBTITLE = "Powered by Hitron Technologies | Field Diagnostics POC"
APP_ICON     = ":wrench:"

EXAMPLE_QUESTIONS = [
    "My CODA56 downstream power is -18 dBmV on all channels with SNR at 27 dB. What's wrong?",
    "Three modems in the same node have T3 timeouts and upstream power averaging +50 dBmV. What should we check first?",
    "DOCSIS 3.1 OFDM channel shows MER of 28 dB and the modem is profile flapping 5 times per hour. What does this indicate?",
    "PLC lock is lost intermittently on our CODA gateway. What causes PLC lock failure?",
    "After a rain event, 12 modems in a tap cluster went offline. How do I diagnose ingress vs. physical damage?",
    "Explain pre-equalization coefficients and how to use them for fault localization.",
    "What's the difference between FDD and FDX in DOCSIS 4.0?",
]
