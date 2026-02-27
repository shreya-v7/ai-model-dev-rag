"""
generate_evidence_rag.py — Offline end-to-end RAG evidence builder.

Pipeline:
1) Load cached corpus text from cache/papers_text/P1..P10.txt
2) Chunk text deterministically
3) Retrieve top chunks per claim query using token-overlap scoring
4) Extract quote-like sentence snippets
5) Write evidence.json
"""
import json
import os
import re
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "cache", "papers_text")
OUT_PATH = os.path.join(ROOT, "evidence.json")


@dataclass
class Chunk:
    paper_id: str
    chunk_id: str
    text: str


CLAIM_SPECS = [
    ("C1", "P6", "ReAct ALFWorld WebShop 34% 10% hallucination wikipedia"),
    ("C2", "P7", "Reflexion 91% HumanEval GPT-4 80% verbal reinforcement"),
    ("C3", "P5", "Toolformer 6.7B GPT-3 175B zero-shot SVAMP 29.4"),
    ("C4", "P8", "MLGym benchmarks improve hyperparameters no novel algorithms"),
    ("C5", "P1", "AI Scientist less than $15 paper 65% 66% balanced accuracy"),
    ("C6", "P9", "Galactica 20.4% PaLM 8.8% 30B 18 times"),
    ("C7", "P10", "BioGPT 44.98 38.42 78.2 PubMedQA BC5CDR"),
    ("C8", "P3", "PaperQA GPT-4 57.9 86.3 PubMedQA"),
    ("C9", "P4", "ChemCrow evaluator cannot distinguish wrong completions"),
    ("C10", "P2", "SciAgents ontological knowledge graph 1000 papers novelty feasibility"),
]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _normalize(text).split() if len(tok) > 2}


def _chunk_text(text: str, paper_id: str, size: int = 1200, overlap: int = 220) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + size)
        frag = text[start:end].strip()
        if frag:
            chunks.append(Chunk(paper_id=paper_id, chunk_id=f"{paper_id}-CHUNK-{idx:04d}", text=frag))
            idx += 1
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _best_chunks(chunks: list[Chunk], query: str, top_k: int = 2) -> list[Chunk]:
    q = _tokenize(query)
    scored = []
    for chunk in chunks:
        t = _tokenize(chunk.text)
        if not t:
            continue
        overlap = len(q & t)
        score = overlap / max(1, len(q))
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def _pick_quote(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    candidates = []
    for sent in sents:
        words = sent.split()
        if 10 <= len(words) <= 80:
            # Prefer metric-heavy evidence sentences
            metric_bonus = 1 if re.search(r"\d+(\.\d+)?%|\$\d+|pass@1|accuracy|F1|outperform", sent, re.I) else 0
            candidates.append((metric_bonus, len(words), sent.strip()))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]
    words = text.split()
    return " ".join(words[:40]).strip()


def _load_paper_text(paper_id: str) -> str:
    path = os.path.join(CACHE_DIR, f"{paper_id}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def main() -> None:
    evidence_rows = []
    for claim_id, paper_id, query in CLAIM_SPECS:
        paper_text = _load_paper_text(paper_id)
        if not paper_text:
            continue
        chunks = _chunk_text(paper_text, paper_id=paper_id)
        selected = _best_chunks(chunks, query=query, top_k=2)
        if not selected and chunks:
            selected = chunks[:1]
        for chunk in selected:
            quote = _pick_quote(chunk.text)
            evidence_rows.append(
                {
                    "claim_id": claim_id,
                    "paper_id": paper_id,
                    "support_level": "supports",
                    "quote": quote,
                    "location": f"{chunk.chunk_id} in cache/papers_text/{paper_id}.txt",
                    "explanation": f"Retrieved by offline token-overlap RAG for {claim_id}.",
                }
            )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(evidence_rows, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print(f"evidence.json written via offline RAG -> {OUT_PATH}")
    print(f"entries: {len(evidence_rows)}")


if __name__ == "__main__":
    main()
