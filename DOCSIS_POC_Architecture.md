# DOCSIS Troubleshooting Assistant — Proof of Concept

## Purpose

A functional demo deployable in days, not weeks. Hitron C-suite executives log into a branded chatbot, paste or describe DOCSIS diagnostic readings, and receive grounded troubleshooting guidance citing real industry documentation. The goal is to demonstrate the value proposition — not production readiness.

---

## 1. POC Architecture

### Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | Streamlit | Python-native, chat UI out of the box, deploy free on Streamlit Community Cloud or a $5/mo DigitalOcean droplet |
| **LLM** | Anthropic Claude (Sonnet 4.5 via API) | Best reasoning for technical diagnostics, tool-calling support, cost-effective for demo volume |
| **RAG / Embeddings** | LangChain + FAISS (local vector store) | Zero infrastructure cost, no managed vector DB needed, runs in-process |
| **Embedding Model** | `voyage-3-large` via Voyage AI API or `all-MiniLM-L6-v2` via sentence-transformers (free, local) | Voyage for quality, MiniLM for zero-cost |
| **Auth** | Streamlit `secrets.toml` with password gate OR Streamlit native `st.experimental_user` | Simple password page — sufficient for C-suite demo, not production |
| **Deployment** | Streamlit Community Cloud (free) or DigitalOcean App Platform ($5/mo) | One-click deploy from GitHub, HTTPS included |

### How It Works

```
Executive opens URL → Password gate → Chat interface
    ↓
User describes issue or pastes readings
    ↓
LangChain orchestrator:
  1. Embed query → FAISS similarity search → retrieve top-5 chunks
  2. Build prompt: system instructions + retrieved context + user query
  3. Call Claude Sonnet with diagnostic persona + rules summary
  4. Stream response with source citations
    ↓
Response displayed with cited sources
```

### Simplified Rules (Prompt-Embedded for POC)

Instead of a separate rules engine (production feature), the POC embeds key diagnostic thresholds directly in the system prompt:

```
DOCSIS 3.0 SC-QAM:
- DS power: -15 to +15 dBmV (optimal -7 to +7)
- US power: +35 to +51 dBmV (warn above +48)
- SNR: >30 dB good, 25-30 marginal, <25 fail
- Uncorrectables: >0.1% of total codewords = investigate

DOCSIS 3.1 OFDM:
- DS OFDM power: -15 to +15 dBmV
- MER: >35 dB good, 30-35 marginal, <30 fail
- PLC lock: must be locked for channel operation
- Profile flapping: >3 changes/hour = plant noise

Severity scoring:
- Offline/loss of lock: CRITICAL
- T4 timeouts: HIGH
- High US power (>48 dBmV) + T3: HIGH (return path)
- Low MER + rising errors: MEDIUM (ingress/damage)
- Provisioning failure: MEDIUM
```

This gives the chatbot consistent diagnostic behavior without building a separate rules engine.

---

## 2. RAG Data Sources — What's Publicly Available

### TIER 1: Directly Downloadable (High Value)

| # | Document | Source URL | Format | Content Value |
|---|----------|-----------|--------|--------------|
| 1 | **PNM Best Practices: HFC Networks (DOCSIS 3.0)** CM-GL-PNMP-V03-160725 | `volpefirm.com/wp-content/uploads/2017/01/CM-GL-PNMP-V03-160725.pdf` | PDF ~200pp | **Core diagnostic reference.** Pre-equalization analysis, micro-reflection detection, fault localization techniques, downstream/upstream troubleshooting. This is the DOCSIS 3.0 "Galactic Guide." |
| 2 | **DOCSIS 4.0 Tools & Readiness: Cable Operator Preparations** | `account.cablelabs.com/server/alfresco/bc2527a6-42f4-4ddb-83ac-62b656f0d0be` | PDF | Network readiness assessment, PNM-driven repair, drop/premise evaluation, DOCSIS 4.0 transition planning. |
| 3 | **DOCSIS 4.0 PHY Specification** (PHYv4.0) | CableLabs specifications library (public download with account) | PDF ~300pp | Section 9 = PNM requirements. Channel parameters, modulation, error correction, measurement objects. |
| 4 | **DOCSIS 3.1 Operational Integration and PNM Tools** | `volpefirm.com/wp-content/uploads/2011/11/Volpe_DOCSIS3.1_Operation_Integration_and_PNM_Tools_v2_final.pdf` | PDF | DOCSIS 3.1 OFDM troubleshooting, profile management, MER interpretation, operational integration. |
| 5 | **DOCSIS 3.0 Troubleshooting (SCTE Presentations)** | SlideShare: Volpe Firm SCTE Piedmont + Blacksburg presentations | PDF/PPT | Field troubleshooting workflows, channel bonding issues, upstream impairment identification, downstream problems catalog. |

### TIER 2: Hitron Product Documentation (Publicly Available)

| # | Document | Source URL | Format | Content Value |
|---|----------|-----------|--------|--------------|
| 6 | **CODA User Manual** (DOCSIS 3.1 cable modem) | `us.hitrontech.com/wp-content/uploads/2021/05/CODA_UM_003.pdf` | PDF | LED codes, setup, troubleshooting FAQ, specifications. Consumer-facing but covers technician-relevant LED diagnostics. |
| 7 | **CODA56 User Manual** (Multi-Gigabit DOCSIS 3.1) | `us.hitrontech.com/wp-content/uploads/2023/01/CODA56-UM.pdf` | PDF | 2.5G Ethernet modem — LED codes, connection troubleshooting, specs. |
| 8 | **CODA-4582/4682/4782 Gateway Manual** | ManualsLib / manuals.plus (publicly indexed) | PDF | Combined modem-router gateway — Wi-Fi + DOCSIS troubleshooting, admin interface, LED status. |
| 9 | **CGNV5 Gateway Manual** (DOCSIS 3.0 eMTA) | usermanual.wiki (publicly indexed) | PDF | DOCSIS 3.0 voice gateway — pairing, setup, troubleshooting, voice diagnostics. |
| 10 | **CGN-DP3 Quick Start Guide** (DOCSIS 3.1 Meter) | manuals.plus (publicly indexed) | PDF | The handheld meter that ProPulse integrates with — indicator lights, setup, reset functions. |
| 11 | **Hitron Product Pages** (all DOCSIS models) | `us.hitrontech.com/service-providers/docsis-modems-gateways/` | Web scrape | Specifications, feature lists, compatibility across full CODA/CGNV5 portfolio. |

### TIER 3: Industry Context & Troubleshooting Guides (Web Content)

| # | Document | Source | Content Value |
|---|----------|--------|--------------|
| 12 | **CableLabs PNM Technology Page** | `cablelabs.com/technologies/proactive-network-maintenance` | PNM overview, DCCF/XCCF frameworks, Wi-Fi PNM |
| 13 | **CableLabs DOCSIS 4.0 Technology Page** | `cablelabs.com/technologies/docsis-4-0-technology` | Spec overview, FDX vs FDD comparison, capacity specs |
| 14 | **CableLabs Blog: Evolution of DOCSIS PNM** | `cablelabs.com/blog/the-evolution-of-docsis-proactive-network-maintenance` | PNM history, MIB data, management objects, troubleshooting indicators |
| 15 | **CableLabs Blog: Tooling Up for DOCSIS 4.0** | `cablelabs.com/blog/tooling-up-for-docsis-technology` | WG5 report summary, readiness preparations, known deployment issues |
| 16 | **What is DOCSIS PNM and Why You Need It** (ZCorum) | `blog.zcorum.com/how-docsis-pnm-works-and-why-you-need-it` | Excellent technician-level explanation of pre-equalization, PNM coefficients, fault localization |
| 17 | **SCTE 280: Understanding & Troubleshooting Cable RF Spectrum** (description + excerpts) | `account.scte.org/standards/library/catalog/scte-280-...` | Full band capture, RF impairment identification, connector/cable/amplifier failure patterns |
| 18 | **SCTE 294: Understanding & Troubleshooting Cable Upstream RF Spectrum** (description) | `account.scte.org/standards/library/catalog/scte-294-...` | Upstream funneling, noise/ingress, frequency split impacts, amplifier considerations |
| 19 | **Hitron Modem LED Guides** (PC Guide, HomeOwner, RouterFreak) | Various consumer tech sites | LED meaning catalog across Hitron models — DS/US/Online/LAN/Wi-Fi/USB status interpretation |
| 20 | **AOI Guide to DOCSIS 4.0** | `ao-inc.com/html-resources/guide-to-docsis-4-0/` | Non-technical DOCSIS 4.0 overview, deployment stages, operator considerations |
| 21 | **SCTE Blog: DOCSIS Technology Is Ready** (Feb 2026) | `scte.org/blog/docsis-technology-is-ready/` | Current state of DOCSIS 4.0 readiness, SCTE 300 migration guide, PMA/PNM maturity |
| 22 | **Hitron ProPulse Press Release** (Sep 2025) | `us.hitrontech.com/press-releases/hitron-launches-hitron-propulse/` | Product announcement, capabilities claimed, CableLabs collaboration context |
| 23 | **Hitron Aprecomm Partnership** (Feb 2025) | Press release | AI-embedded CPE, customer-facing AI, deflection/truck roll metrics |

### Estimated RAG Corpus Size

- **Tier 1 (core technical):** ~500-800 pages → ~1,500-2,500 chunks at 300 tokens each
- **Tier 2 (Hitron product):** ~150-250 pages → ~500-800 chunks
- **Tier 3 (web content):** ~80-120 pages equivalent → ~300-500 chunks
- **Total:** ~2,300-3,800 chunks — well within FAISS local capacity

---

## 3. What Is NOT Included — Gaps for Production

This is critical to be transparent about with Hitron executives. The POC demonstrates the *interaction model* and *diagnostic logic* but does NOT include:

### Data Gaps

| Missing Source | Why It Matters | How to Get It |
|---------------|----------------|---------------|
| **CableLabs Galactic Guide (CM-GL-PNM-2023 / PNM-HFC v02 Jan 2026)** | The updated comprehensive PNM reference. The POC uses the 2016 DOCSIS 3.0 version; the 2023/2026 versions cover 3.1 and 4.0 PNM in depth. | Hitron vendor membership with CableLabs provides access. |
| **CableLabs AIOps workstream outputs** | RAG models, agentic AI patterns, Expert LLM benchmarks — the foundation Hitron's production system should align with. | Available through PNM Working Group participation. |
| **SCTE 280 & 294 full documents** | Complete RF spectrum troubleshooting guides (downstream + upstream). POC only has descriptions and excerpts. | SCTE membership or purchase ($200-400 each). |
| **SCTE 300: Best Practices for Migrating to DOCSIS 4.0** | The definitive migration guide — critical for 4.0 troubleshooting context. | SCTE membership. |
| **Hitron service-provider documentation** | Internal technician guides, provisioning procedures, model-specific field SOPs — far more detailed than consumer manuals. | Hitron internal. These are the highest-value documents. |
| **Hitron firmware release notes** | Known issues, per-version quirks, workarounds — critical for model-specific diagnostics. | Hitron internal. |
| **Operator-specific threshold values** | Each MSO tunes power/SNR/MER acceptable ranges differently. POC uses CableLabs spec defaults. | Operator partnerships. |
| **Historical resolved tickets** | Real-world case data for pattern matching and diagnostic accuracy. Even 50-200 cases dramatically improve relevance. | Hitron/operator partnerships. |
| **CODA60V DOCSIS 4.0 specific documentation** | The CODA60V is Hitron's only certified D4.0 device. No public technical docs exist yet for technicians. | Hitron internal — device is in trials. |
| **CGN-DP3M meter technical documentation** | Detailed meter capabilities, measurement outputs, API integration points for ProMeter app. | Hitron internal. |

### Capability Gaps

| Missing Capability | What the POC Does Instead | Production Requirement |
|-------------------|--------------------------|----------------------|
| **Deterministic rules engine** | Thresholds embedded in system prompt — consistent but not auditable or testable as a separate component | Separate Lambda-based rules engine with versioned threshold configs |
| **Structured readings parser** | User types/pastes readings, LLM extracts values — works but fragile | Formal parser accepting ProMeter/HitronCloud data formats |
| **Escalation packet generation** | LLM summarizes case in chat — not structured | NOC-ready JSON/PDF packets with standardized fields |
| **Repair verification** | User can describe post-fix readings — LLM compares informally | Automated comparison of pre/post measurements against pass criteria |
| **ProMeter/HitronCloud integration** | None — standalone web chatbot | API integration to receive readings and push results |
| **Case data persistence** | Chat history only within session | DynamoDB or equivalent with structured case schemas |
| **Multi-tenant operator configs** | Single threshold set | Per-operator threshold profiles |
| **Guardrails & safety** | Basic system prompt boundaries | Input validation, PII redaction, output filtering, confirmation gates |

---

## 4. Deployment Plan

### Phase 1: Data Collection & Ingestion (Day 1-2)

```bash
# 1. Download all Tier 1 PDFs
mkdir -p data/raw/cablelabs data/raw/hitron data/raw/scte data/raw/industry

# 2. Download Tier 2 Hitron manuals
wget -O data/raw/hitron/CODA_manual.pdf "https://us.hitrontech.com/wp-content/uploads/2021/05/CODA_UM_003.pdf"
wget -O data/raw/hitron/CODA56_manual.pdf "https://us.hitrontech.com/wp-content/uploads/2023/01/CODA56-UM.pdf"
# ... additional manuals

# 3. Scrape Tier 3 web content
# Use requests + BeautifulSoup for CableLabs/SCTE blog posts
# Save as markdown files for clean chunking

# 4. Process and chunk
python ingest.py --chunk-size 400 --chunk-overlap 50 --metadata-tag
# Tags: docsis_version, doc_type, source, device_model, topic
```

### Phase 2: RAG Pipeline Build (Day 2-3)

```python
# Core pipeline
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # or VoyageAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain

# Embedding + vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Retrieval chain with Claude
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)
```

### Phase 3: Streamlit App (Day 3-4)

```python
# app.py - core structure
import streamlit as st

# Password gate
if "authenticated" not in st.session_state:
    password = st.text_input("Enter access code:", type="password")
    if password == st.secrets["ACCESS_CODE"]:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Invalid access code")
    st.stop()

# Chat interface
st.title("🔧 DOCSIS Troubleshooting Assistant")
st.caption("Powered by Hitron Technologies | Demo")

# ... chat loop with streaming responses and source citations
```

### Phase 4: Deploy (Day 4-5)

**Option A: Streamlit Community Cloud (Free)**
- Push to GitHub (private repo)
- Connect Streamlit Cloud → one-click deploy
- Add secrets (API key, access code) in Streamlit dashboard
- URL: `https://hitron-docsis-assistant.streamlit.app`

**Option B: DigitalOcean App Platform ($5/mo)**
- More control, custom domain possible
- `hitron-demo.yourdomain.com`
- Docker-based, auto-deploys from GitHub

### Phase 5: Test & Polish (Day 5-7)

- Run 20-30 test scenarios across DOCSIS 3.0/3.1/4.0
- Verify citations point to real source documents
- Tune retrieval (k value, chunk size, overlap)
- Add Hitron branding (logo, colors: #00A4E4, #0C2340)
- Create 5-10 "try asking" example prompts for executives

---

## 5. Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| Anthropic API (Claude Sonnet, ~50 demo conversations/mo) | ~$5-15 |
| Streamlit Community Cloud hosting | Free |
| Voyage AI embeddings (if used instead of local) | ~$2-5 |
| Domain (optional, e.g. `demo.hitron-ai.com`) | ~$1 |
| **Total** | **~$5-20/month** |

---

## 6. What Executives Will See

A branded chat interface where they can:

1. **Describe a scenario:** "A technician reports the CODA-5519E downstream power is at -18 dBmV across all channels with SNR at 28 dB"
2. **Get a grounded diagnosis:** The system identifies out-of-spec downstream power, correlates with low SNR, suggests attenuation issue (damaged cable, bad splitter, or amplifier failure), and cites CableLabs PNM best practices.
3. **Ask follow-up questions:** "What should the technician check first?" — the system provides a prioritized action list.
4. **See source citations:** Every recommendation traces back to a specific document.

### What the POC Proves

- AI can parse unstructured field descriptions into diagnostic evaluations
- RAG grounding produces accurate, citable guidance — not hallucinated advice
- The interaction model (conversational intake → structured diagnosis → cited guidance) works
- The system spans DOCSIS 3.0/3.1/4.0 vocabulary correctly
- CableLabs/SCTE standards are an effective RAG foundation

### What the POC Does NOT Prove

- Production-scale performance (this is single-user demo)
- Integration with ProMeter/HitronCloud (standalone chatbot)
- Deterministic consistency (no separate rules engine)
- Operator-specific tuning (generic thresholds only)
- Real-world accuracy (no validation against actual field cases)

---

## 7. Next Step After Demo

If the C-suite demo lands well, the immediate next step is the Week 1 Discovery phase from the main proposal — which produces the scoped production requirements, integration plan, and success criteria that the POC intentionally deferred.
