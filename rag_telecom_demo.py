"""
=============================================================
Hybrid RAG Demo — Telecom Specification Reasoning
=============================================================
Author : Prasad Narayan Kachawar
GitHub : github.com/prasadkachawar
LinkedIn: linkedin.com/in/prasadkachawar

Demonstrates the core architecture of a production RAG system
for 3GPP specification reasoning — using only open/free tools
so you can run it locally.

No proprietary code or data. Concepts reflect real production
patterns from my work at Radisys India.

Stack used here (free / local):
  - sentence-transformers  → embeddings
  - faiss-cpu              → vector store
  - rank-bm25              → keyword search (BM25)
  - anthropic              → LLM grounding (Claude)
  - rich                   → pretty terminal output

Install:
  pip install sentence-transformers faiss-cpu rank-bm25 anthropic rich
=============================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

# ── Sample 3GPP Knowledge Corpus (public spec summaries) ─────────────────────
# In production this would be 1000s of chunked 3GPP TS/TR documents
# Here we use a small illustrative corpus of real spec concepts

CORPUS = [
    {
        "id": "ts38300-1",
        "spec": "TS 38.300",
        "section": "6.1",
        "title": "NR Architecture Overview",
        "text": (
            "The 5G NR architecture consists of the gNB which provides the NR user plane "
            "and control plane protocol terminations towards the UE. The gNB can be split "
            "into a gNB-CU (Central Unit) and one or more gNB-DU (Distributed Units). "
            "The interface between gNB-CU and gNB-DU is called the F1 interface."
        ),
    },
    {
        "id": "ts38300-2",
        "spec": "TS 38.300",
        "section": "6.2",
        "title": "CU-DU Split",
        "text": (
            "The gNB-CU hosts the RRC, SDAP and PDCP protocols of the gNB. "
            "The gNB-DU hosts the RLC, MAC and PHY layers of the gNB. "
            "The gNB-CU-CP handles the control plane while gNB-CU-UP handles the user plane. "
            "Split Option 7-2x (also called Split 6) places the lower PHY in the RU."
        ),
    },
    {
        "id": "ts38321-1",
        "spec": "TS 38.321",
        "section": "5.4",
        "title": "Random Access Procedure",
        "text": (
            "The Random Access procedure is initiated by the MAC layer. "
            "The UE selects a Random Access preamble and transmits it on PRACH. "
            "The gNB responds with a Random Access Response (RAR) within the RAR window. "
            "If no RAR is received, the UE increments the preamble transmission counter "
            "and retransmits after a backoff period."
        ),
    },
    {
        "id": "ts38321-2",
        "spec": "TS 38.321",
        "section": "5.15",
        "title": "Scheduling Request",
        "text": (
            "A Scheduling Request (SR) is used by the UE to request UL-SCH resources. "
            "The UE transmits an SR on PUCCH when it has data to send but no UL grant. "
            "If the SR fails (no grant received), the UE can retransmit up to "
            "sr-TransMax times before triggering a Random Access procedure."
        ),
    },
    {
        "id": "ts38331-1",
        "spec": "TS 38.331",
        "section": "5.3.3",
        "title": "RRC Connection Setup",
        "text": (
            "The RRC Connection Establishment procedure allows a UE in RRC_IDLE to "
            "transition to RRC_CONNECTED state. The UE sends an RRCSetupRequest message "
            "to the gNB. Upon reception, the gNB responds with RRCSetup containing "
            "radio resource configuration. The UE then sends RRCSetupComplete."
        ),
    },
    {
        "id": "ts38331-2",
        "spec": "TS 38.331",
        "section": "5.3.8",
        "title": "RRC Reconfiguration",
        "text": (
            "RRC Reconfiguration is used to modify an RRC connection. It can be used "
            "for handover, radio bearer establishment/modification/release, and measurement "
            "configuration. The UE responds with RRCReconfigurationComplete upon success. "
            "If the UE cannot comply, it triggers an RRC re-establishment."
        ),
    },
    {
        "id": "ts23501-1",
        "spec": "TS 23.501",
        "section": "4.2",
        "title": "5GC Architecture",
        "text": (
            "The 5G Core Network (5GC) uses a Service Based Architecture (SBA). "
            "Key NFs include AMF (Access and Mobility Management), SMF (Session Management), "
            "UPF (User Plane Function), PCF (Policy Control), and UDM (Unified Data Management). "
            "NFs communicate via Service Based Interfaces (SBI) using HTTP/2 and JSON."
        ),
    },
    {
        "id": "ts23501-2",
        "spec": "TS 23.501",
        "section": "5.6",
        "title": "PDU Session Establishment",
        "text": (
            "A PDU Session provides connectivity between UE and a Data Network. "
            "The UE triggers PDU Session Establishment via NAS signalling to AMF. "
            "AMF selects an SMF, which selects a UPF and allocates an IP address. "
            "The PDU Session can be of type IPv4, IPv6, IPv4v6, Ethernet, or Unstructured."
        ),
    },
]


# ── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class Document:
    id: str
    spec: str
    section: str
    title: str
    text: str
    embedding: np.ndarray = field(default=None, repr=False)


@dataclass
class RetrievalResult:
    doc: Document
    semantic_score: float
    keyword_score: float
    combined_score: float


# ── Embedder ──────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        console.print(f"[dim]Loading embedding model: {model_name}[/dim]")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


# ── Vector Store (FAISS) ──────────────────────────────────────────────────────
class VectorStore:
    def __init__(self, docs: List[Document], embedder: Embedder):
        import faiss
        self.docs = docs
        self.dim = 384  # all-MiniLM-L6-v2 output dim
        self.index = faiss.IndexFlatIP(self.dim)  # inner product = cosine on normalized vecs

        console.print("[dim]Building FAISS index...[/dim]")
        embeddings = embedder.encode([d.text for d in docs])
        for i, doc in enumerate(docs):
            doc.embedding = embeddings[i]
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), top_k
        )
        return [(self.docs[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]


# ── Keyword Store (BM25) ──────────────────────────────────────────────────────
class KeywordStore:
    def __init__(self, docs: List[Document]):
        from rank_bm25 import BM25Okapi
        tokenized = [d.text.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.docs = docs

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1.0
        return [(self.docs[i], float(scores[i]) / max_score) for i in top_indices]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Combines semantic (dense) and keyword (sparse) retrieval with
    a configurable alpha weighting — mirroring the production approach.

    alpha=1.0 → pure semantic
    alpha=0.0 → pure keyword
    alpha=0.6 → production default (semantic-leaning hybrid)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embedder: Embedder,
        alpha: float = 0.6,
    ):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.embedder = embedder
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        # Dense retrieval
        q_emb = self.embedder.encode([query])
        semantic_results = dict(self.vector_store.search(q_emb, top_k=6))

        # Sparse retrieval
        keyword_results = dict(self.keyword_store.search(query, top_k=6))

        # Merge and score
        all_docs = set(list(semantic_results.keys()) + list(keyword_results.keys()))
        scored = []
        for doc in all_docs:
            s_score = semantic_results.get(doc, 0.0)
            k_score = keyword_results.get(doc, 0.0)
            combined = self.alpha * s_score + (1 - self.alpha) * k_score
            scored.append(RetrievalResult(
                doc=doc,
                semantic_score=s_score,
                keyword_score=k_score,
                combined_score=combined,
            ))

        scored.sort(key=lambda x: x.combined_score, reverse=True)
        return scored[:top_k]


# ── LLM Grounding (Claude via Anthropic API) ─────────────────────────────────
class LLMGrounder:
    """
    Generates grounded responses strictly constrained to retrieved evidence.
    Mirrors production pattern: no hallucination beyond what's retrieved.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, query: str, results: List[RetrievalResult]) -> str:
        context_blocks = []
        for i, r in enumerate(results, 1):
            context_blocks.append(
                f"[Source {i}: {r.doc.spec} §{r.doc.section} — {r.doc.title}]\n{r.doc.text}"
            )
        context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a 3GPP specification expert assistant. "
            "Answer the user's question using ONLY the provided specification excerpts. "
            "Always cite the spec and section number (e.g. TS 38.300 §6.1). "
            "If the answer cannot be found in the provided context, say so explicitly. "
            "Be concise, accurate, and technically precise."
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Specification context:\n{context}\n\n"
            "Answer based strictly on the above context:"
        )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text


# ── RAG Pipeline ──────────────────────────────────────────────────────────────
class TelecomRAGPipeline:
    def __init__(self):
        console.print(Panel.fit(
            "[bold blue]Telecom RAG Pipeline[/bold blue]\n"
            "[dim]Hybrid retrieval + LLM grounding for 3GPP spec reasoning[/dim]",
            border_style="blue"
        ))

        self.embedder = Embedder()
        docs = [Document(**d) for d in CORPUS]
        self.vector_store = VectorStore(docs, self.embedder)
        self.keyword_store = KeywordStore(docs)
        self.retriever = HybridRetriever(self.vector_store, self.keyword_store, self.embedder)

        if os.environ.get("ANTHROPIC_API_KEY"):
            self.grounder = LLMGrounder()
            console.print("[green]Claude LLM grounder ready[/green]")
        else:
            self.grounder = None
            console.print("[yellow]No ANTHROPIC_API_KEY found — running in retrieval-only mode[/yellow]")

    def query(self, question: str, alpha: float = 0.6) -> dict:
        self.retriever.alpha = alpha
        t0 = time.time()
        results = self.retriever.retrieve(question, top_k=3)
        retrieval_ms = (time.time() - t0) * 1000

        # Show retrieval results table
        table = Table(title=f"Retrieved context for: [italic]{question}[/italic]", show_lines=True)
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Spec", style="cyan", width=10)
        table.add_column("Section", width=8)
        table.add_column("Title", width=22)
        table.add_column("Semantic", justify="right", width=9)
        table.add_column("Keyword", justify="right", width=9)
        table.add_column("Combined", justify="right", style="bold green", width=9)

        for i, r in enumerate(results, 1):
            table.add_row(
                str(i),
                r.doc.spec,
                r.doc.section,
                r.doc.title,
                f"{r.semantic_score:.3f}",
                f"{r.keyword_score:.3f}",
                f"{r.combined_score:.3f}",
            )
        console.print(table)
        console.print(f"[dim]Retrieval: {retrieval_ms:.0f}ms · alpha={alpha}[/dim]\n")

        answer = None
        if self.grounder:
            console.print("[dim]Generating grounded response...[/dim]")
            t1 = time.time()
            answer = self.grounder.generate(question, results)
            grounding_ms = (time.time() - t1) * 1000
            console.print(Panel(
                answer,
                title="[bold]Grounded answer[/bold]",
                border_style="green",
            ))
            console.print(f"[dim]LLM grounding: {grounding_ms:.0f}ms[/dim]\n")
        else:
            console.print(Panel(
                results[0].doc.text,
                title=f"[bold]Top result: {results[0].doc.spec} §{results[0].doc.section}[/bold]",
                border_style="blue",
            ))

        return {"results": results, "answer": answer}


# ── Demo Queries ──────────────────────────────────────────────────────────────
DEMO_QUERIES = [
    "What happens during a Random Access procedure in 5G NR?",
    "How is the CU-DU split defined in 3GPP?",
    "What is the difference between gNB-CU-CP and gNB-CU-UP?",
    "Explain PDU session establishment in 5GC",
    "What triggers an RRC re-establishment?",
]


def main():
    pipeline = TelecomRAGPipeline()
    console.print("\n[bold]Running demo queries...[/bold]\n")

    for q in DEMO_QUERIES:
        console.rule(f"[bold blue]Query[/bold blue]")
        pipeline.query(q)
        console.print()

    # Interactive mode
    console.print(Panel.fit(
        "[bold]Interactive mode[/bold]\nType your 3GPP question or 'quit' to exit.",
        border_style="dim"
    ))
    while True:
        question = console.input("[bold blue]> [/bold blue]").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            pipeline.query(question)


if __name__ == "__main__":
    main()
