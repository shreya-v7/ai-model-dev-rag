"""Streamlit portal for robust multi-document RAG."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import TOP_K_PER_CLAIM, UI_MAX_DOCS, UI_MIN_DOCS, bootstrap_runtime_dirs
from src.models import QAAnswer, SynthesisResult
from src.pipeline import CorpusIndex, answer_question, build_index_from_uploads, synthesize_topic

st.set_page_config(page_title="Robust RAG Portal", layout="wide")


def _init_state() -> None:
    if "corpus" not in st.session_state:
        st.session_state.corpus = None
    if "synthesis" not in st.session_state:
        st.session_state.synthesis = None
    if "qa_results" not in st.session_state:
        st.session_state.qa_results = []
    if "last_llm_call_ts" not in st.session_state:
        st.session_state.last_llm_call_ts = 0.0


def _enforce_app_rate_limit(seconds: float = 1.0) -> None:
    elapsed = time.time() - st.session_state.last_llm_call_ts
    if elapsed < seconds:
        time.sleep(seconds - elapsed)
    st.session_state.last_llm_call_ts = time.time()


def _save_run_artifacts(synthesis: SynthesisResult, qa_results: list[QAAnswer]) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / f"portal_run_{ts}.json"
    artifact = {
        "synthesis": synthesis.model_dump(),
        "qa_results": [item.model_dump() for item in qa_results],
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(artifact_path)


def _render_security_banner() -> None:
    st.info(
        "This portal applies: upload limits, file type checks, prompt-injection line filtering, "
        "retrieval-only answering, quote-grounded references, and request pacing/rate limiting."
    )


def _render_synthesis(synthesis: SynthesisResult) -> None:
    st.subheader("Synthesis Output")
    st.markdown(f"**Topic summary:** {synthesis.topic_summary}")
    st.markdown("**Claims and evidence**")
    for idx, claim in enumerate(synthesis.claims, start=1):
        st.markdown(f"{idx}. **Claim:** {claim.claim} (`confidence={claim.confidence}`)")
        for evidence in claim.evidences:
            st.markdown(f"- Evidence: {evidence.statement}")
            for ref in evidence.references:
                st.code(
                    f"[{ref.doc_id} | {ref.chunk_id}] {ref.quote}",
                    language="text",
                )
    if synthesis.unresolved_questions:
        st.markdown("**Unresolved questions**")
        for question in synthesis.unresolved_questions:
            st.markdown(f"- {question}")


def _render_answers(answers: list[QAAnswer]) -> None:
    if not answers:
        return
    st.subheader("Question Answers")
    for answer in answers:
        st.markdown(f"**Q:** {answer.question}")
        st.markdown(f"**A:** {answer.answer}")
        st.markdown(f"**Uncertainty:** {answer.uncertainty}")
        if answer.references:
            st.markdown("**References**")
            for ref in answer.references:
                st.code(f"[{ref.doc_id} | {ref.chunk_id}] {ref.quote}", language="text")
        else:
            st.warning("No verifiable references were found for this answer.")


def main() -> None:
    bootstrap_runtime_dirs()
    _init_state()
    st.title("Robust RAG Document Portal")
    _render_security_banner()
    offline_mode = st.checkbox(
        "Offline deterministic mode (no API keys)",
        value=False,
        help="Uses a deterministic mock LLM and hash embeddings for offline demo runs.",
    )
    if offline_mode:
        os.environ["OFFLINE_MODE"] = "1"
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBED_PROVIDER"] = "hash"

    with st.expander("How this portal works", expanded=False):
        st.write(
            "- You set expected document count (recommended 10-20).\n"
            "- Upload exactly that many docs.\n"
            "- Docs are chunked and embedded for retrieval.\n"
            "- LLM receives only retrieved chunk context.\n"
            "- Answers must include references (`doc_id`, `chunk_id`, quote).\n"
            f"- Current retrieval depth per query: top-{TOP_K_PER_CLAIM} chunks.\n"
            "- Optional offline mode runs without API keys for demos."
        )

    expected_docs = st.number_input(
        "Enter number of docs to upload",
        min_value=UI_MIN_DOCS,
        max_value=UI_MAX_DOCS,
        value=UI_MIN_DOCS,
        step=1,
    )
    uploads = st.file_uploader(
        "Upload documents (pdf/txt/md/rst/json/csv)",
        type=["pdf", "txt", "md", "rst", "json", "csv"],
        accept_multiple_files=True,
    )
    topic = st.text_input("Analysis objective/topic", value="Synthesize key claims and evidence.")

    if st.button("Build RAG Index + Run Synthesis", use_container_width=True):
        if len(uploads) != expected_docs:
            st.error(f"Please upload exactly {expected_docs} documents. Current: {len(uploads)}.")
        else:
            with st.spinner("Building index and synthesizing..."):
                try:
                    _enforce_app_rate_limit(1.0)
                    corpus: CorpusIndex = build_index_from_uploads(
                        uploads,
                        max_documents=expected_docs,
                    )
                    synthesis = synthesize_topic(corpus, topic=topic)
                    st.session_state.corpus = corpus
                    st.session_state.synthesis = synthesis
                    st.session_state.qa_results = []
                    st.success(
                        "RAG index ready. "
                        f"Filtered {corpus.injection_lines_filtered} suspicious "
                        "lines during ingestion."
                    )
                except Exception as exc:
                    st.exception(exc)

    if st.session_state.synthesis is not None:
        _render_synthesis(st.session_state.synthesis)

    st.divider()
    st.subheader("Ask Questions Over Uploaded Corpus")
    st.caption(
        "Answers are generated from retrieved chunks "
        "and include references when verifiable."
    )
    questions_blob = st.text_area(
        "Enter one or more questions (one per line)",
        placeholder="What is the main conclusion?\nWhich assumptions are most uncertain?",
    )

    if st.button("Answer Questions", use_container_width=True):
        corpus = st.session_state.corpus
        if corpus is None:
            st.error("Build the RAG index first.")
        else:
            questions = [line.strip() for line in questions_blob.splitlines() if line.strip()]
            if not questions:
                st.error("Enter at least one question.")
            else:
                results: list[QAAnswer] = []
                for question in questions:
                    with st.spinner(f"Answering: {question}"):
                        try:
                            _enforce_app_rate_limit(1.0)
                            results.append(answer_question(corpus, question))
                        except Exception as exc:
                            st.error(f"Failed on question: {question}")
                            st.exception(exc)
                st.session_state.qa_results = results
                if st.session_state.synthesis is not None:
                    artifact_path = _save_run_artifacts(st.session_state.synthesis, results)
                    st.success(f"Saved run artifacts to: {artifact_path}")

    _render_answers(st.session_state.qa_results)


if __name__ == "__main__":
    main()
