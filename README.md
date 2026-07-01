# Telecom RAG Demo — Hybrid Retrieval for 3GPP Specification Reasoning

> A production-inspired Retrieval-Augmented Generation (RAG) pipeline for reasoning over 3GPP telecom specifications — built with open tools, zero proprietary code.

**Author:** Prasad Narayan Kachawar  
**Role:** Senior AI Consultant · AI/GenAI Platforms & Telecom Systems  
**LinkedIn:** [linkedin.com/in/prasadkachawar](https://linkedin.com/in/prasadkachawar)

---

## What this demonstrates

This repo shows the core architecture of a **hybrid RAG system** I designed for 3GPP specification reasoning in a production telecom environment. The production system reduced spec query resolution time from ~45 minutes of manual search to under 30 seconds.

The code here uses only free, open-source tools so you can run it locally and explore the concepts — no proprietary data or infrastructure required.

---

## Architecture overview

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Hybrid Retriever        │
│                                 │
│  ┌─────────────┐  ┌──────────┐ │
│  │   Semantic  │  │ Keyword  │ │
│  │   (FAISS)   │  │  (BM25)  │ │
│  └──────┬──────┘  └────┬─────┘ │
│         │   α·s + (1-α)·k      │
│         └────────┬─────────────┘
│                  │ Fused score  │
└──────────────────┼──────────────┘
                   │
                   ▼
          Top-K Retrieved Chunks
          (3GPP TS/TR excerpts)
                   │
                   ▼
        ┌──────────────────────┐
        │    LLM Grounding     │
        │  (Claude via API)    │
        │  Evidence-constrained│
        └──────────────────────┘
                   │
                   ▼
        Auditable, Cited Answer
```

**Key design decisions:**
- **Hybrid retrieval** combines dense (semantic) and sparse (BM25 keyword) search — because telecom engineers query both ways: conceptually ("what happens during handover") and precisely ("PREAMBLE_TRANSMISSION_COUNTER")
- **Configurable alpha** lets you tune the semantic vs keyword balance per use case
- **Grounding prompt** constrains LLM to retrieved evidence only — no hallucination beyond what's retrieved
- **Delta chunking** pattern (described in code comments) handles spec versioning in production

---

## Tech stack

| Component | Tool used here | Production equivalent |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Fine-tuned telecom encoder |
| Vector store | `faiss-cpu` | AWS S3 + custom vector index |
| Keyword search | `rank-bm25` | Elasticsearch BM25 |
| Graph context | *(described in comments)* | Amazon Neptune |
| LLM grounding | `anthropic` (Claude) | AWS Bedrock multi-LLM router |
| Output | `rich` terminal | FastAPI + frontend |

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/prasadkachawar/telecom-rag-demo.git
cd telecom-rag-demo
```

### 2. Install dependencies

```bash
pip install sentence-transformers faiss-cpu rank-bm25 anthropic rich
```

### 3. (Optional) Set your Anthropic API key

For full LLM-grounded responses via Claude. Without this, the demo runs in retrieval-only mode — still shows the hybrid scoring and ranking.

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 4. Run

```bash
python rag_telecom_demo.py
```

The script runs 5 demo queries against the corpus, prints a ranked retrieval table with semantic / keyword / combined scores, and (if API key is set) generates a grounded Claude response citing the exact spec and section.

Then it enters interactive mode — type any 3GPP question.

---

## Example output

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Retrieved context for: What happens during a Random Access procedure?    │
├──────┬──────────┬─────────┬───────────────────────┬──────────┬──────────┤
│ Rank │ Spec     │ Section │ Title                 │ Semantic │ Combined │
├──────┼──────────┼─────────┼───────────────────────┼──────────┼──────────┤
│  1   │ TS 38.321│ 5.4     │ Random Access         │  0.891   │  0.847   │
│  2   │ TS 38.321│ 5.15    │ Scheduling Request    │  0.712   │  0.634   │
│  3   │ TS 38.300│ 6.1     │ NR Architecture       │  0.623   │  0.580   │
└──────┴──────────┴─────────┴───────────────────────┴──────────┴──────────┘

Retrieval: 18ms · alpha=0.6

╭─ Grounded answer ──────────────────────────────────────────────────────────╮
│ According to TS 38.321 §5.4, the Random Access procedure begins when the   │
│ MAC layer initiates it. The UE selects a Random Access preamble and        │
│ transmits it on PRACH. The gNB responds with a Random Access Response      │
│ (RAR) within the RAR window. If no RAR is received, the UE increments      │
│ the preamble transmission counter and retransmits after a backoff period.  │
╰────────────────────────────────────────────────────────────────────────────╯
```

---

## Corpus

The demo uses a small illustrative corpus of real 3GPP specification summaries covering:

- **TS 38.300** — NR Architecture Overview, CU-DU Split
- **TS 38.321** — Random Access Procedure, Scheduling Request
- **TS 38.331** — RRC Connection Setup, RRC Reconfiguration
- **TS 23.501** — 5GC Architecture, PDU Session Establishment

In production, this scales to thousands of chunked TS/TR documents with version tracking.

---

## Extending this

Want to try it on real 3GPP specs? The documents are freely available:

1. Download any TS/TR from [3gpp.org/ftp/Specs](https://www.3gpp.org/ftp/Specs/archive/)
2. Convert to text with `pdfplumber` or `pymupdf`
3. Chunk by section heading and replace the `CORPUS` list in the script
4. Re-run — the hybrid retriever works on any corpus size

---

## Related writing

I wrote about the design decisions behind this system — why naive RAG fails for telecom specs and how the hybrid + graph approach solves it:

**[How I Built a Hybrid RAG System for 3GPP Specification Reasoning](#)** *http://prasadkachawar.netlify.app/*

---

## About

Prasad Kachawar is a Senior AI Consultant with 9+ years of experience building production AI and telecom systems. This demo reflects architectural patterns from real production work — shared here as a public, reproducible reference.

3 patents filed · Pune, India · [prasadkachawar@gmail.com](mailto:prasadkachawar@gmail.com)
