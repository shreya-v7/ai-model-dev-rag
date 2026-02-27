"""Streamlit portal for robust multi-document RAG."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import UI_MAX_DOCS, UI_MIN_DOCS, bootstrap_runtime_dirs
from src.llm_client import get_llm_client
from src.models import QAAnswer, SynthesisResult
from src.pipeline import (
    CorpusIndex,
    answer_question,
    build_index_from_paths,
    build_index_from_uploads,
    synthesize_topic,
)
from src.prompts import SYSTEM_PROMPT, build_reference_judge_prompt
from src.security import quote_supported_by_chunk

st.set_page_config(page_title="RAG Learning Portal", layout="wide")
DEFAULT_TOPIC = "Synthesize key claims and evidence."
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
log = logging.getLogger("rag.portal")


def _init_state() -> None:
    st.session_state.setdefault("corpus", None)
    st.session_state.setdefault("synthesis", None)
    st.session_state.setdefault("qa_results", [])
    st.session_state.setdefault("last_llm_call_ts", 0.0)
    st.session_state.setdefault("last_report_path", "")
    st.session_state.setdefault("verifier_result", None)
    st.session_state.setdefault("activity_logs", [])
    st.session_state.setdefault(
        "pipeline_stats",
        {
            "documents": 0,
            "chunks": 0,
            "filtered_lines": 0,
            "claims": 0,
            "answers": 0,
        },
    )


def _apply_ui_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
        [data-testid="stMetricValue"] {font-size: 1.6rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 0.2rem;}
        .stTabs [data-baseweb="tab"] {height: 2.2rem; padding: 0.3rem 0.8rem;}
        .stAlert {padding-top: 0.45rem; padding-bottom: 0.45rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _log_event(message: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    event = f"{ts} | {message}"
    st.session_state.activity_logs.append(event)
    log.info(message)


def _is_survey_path(path: str) -> bool:
    return Path(path).name.upper().startswith("S")


def _is_paper_path(path: str) -> bool:
    return Path(path).name.upper().startswith("P")


def _enforce_app_rate_limit(seconds: float = 1.0) -> None:
    elapsed = time.time() - st.session_state.last_llm_call_ts
    if elapsed < seconds:
        time.sleep(seconds - elapsed)
    st.session_state.last_llm_call_ts = time.time()


def _fallback_judge(quote: str, chunk_text: str, rule_supported: bool) -> dict:
    if rule_supported:
        return {
            "used": True,
            "method": "offline_fallback",
            "verdict": "valid",
            "score": 5,
            "reasoning": "Exact quote match found in chunk text.",
        }
    quote_words = set(word.lower() for word in quote.split() if word.strip())
    chunk_words = set(word.lower() for word in chunk_text.split() if word.strip())
    overlap = len(quote_words & chunk_words)
    denom = max(1, len(quote_words))
    ratio = overlap / denom
    if ratio >= 0.6:
        verdict = "weak"
        score = 3
    else:
        verdict = "invalid"
        score = 1
    return {
        "used": True,
        "method": "offline_fallback",
        "verdict": verdict,
        "score": score,
        "reasoning": f"Token overlap ratio={ratio:.2f}.",
    }


def _judge_reference(
    *,
    parent_text: str,
    source_type: str,
    quote: str,
    chunk_text: str,
    judge_policy: str,
    enable_llm_judge: bool,
    rule_supported: bool,
) -> dict:
    should_use_llm = False
    if judge_policy.startswith("A"):
        should_use_llm = True
    elif judge_policy.startswith("B"):
        should_use_llm = enable_llm_judge
    elif judge_policy.startswith("C"):
        should_use_llm = True

    if not should_use_llm:
        return {
            "used": False,
            "method": "rule_only",
            "verdict": "valid" if rule_supported else "invalid",
            "score": 5 if rule_supported else 1,
            "reasoning": "LLM judge disabled for this export mode.",
        }

    try:
        llm = get_llm_client()
        prompt = build_reference_judge_prompt(
            parent_text=parent_text,
            source_type=source_type,
            quote=quote,
            chunk_text=chunk_text,
        )
        payload = llm.generate_json(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=250)
        verdict = str(payload.get("verdict", "weak")).lower()
        score = int(payload.get("score", 3))
        reasoning = str(payload.get("reasoning", "No reasoning provided by judge."))
        if verdict not in {"valid", "weak", "invalid"}:
            verdict = "weak"
        return {
            "used": True,
            "method": "llm_judge",
            "verdict": verdict,
            "score": max(1, min(5, score)),
            "reasoning": reasoning,
        }
    except Exception as exc:
        if judge_policy.startswith("C"):
            fallback = _fallback_judge(
                quote=quote,
                chunk_text=chunk_text,
                rule_supported=rule_supported,
            )
            fallback["reasoning"] = f"{fallback['reasoning']} Fallback reason: {exc}"
            return fallback
        return {
            "used": False,
            "method": "llm_judge_error",
            "verdict": "weak" if rule_supported else "invalid",
            "score": 3 if rule_supported else 1,
            "reasoning": f"LLM judge failed: {exc}",
        }


def _build_comprehensive_report(
    *,
    index: CorpusIndex,
    synthesis: SynthesisResult,
    qa_results: list[QAAnswer],
    topic: str,
    run_mode: str,
    judge_policy: str,
    enable_llm_judge: bool,
) -> dict:
    chunk_lookup = {chunk.chunk_id: chunk for chunk in index.chunks}
    validations: list[dict] = []

    for claim_idx, claim in enumerate(synthesis.claims, start=1):
        for evidence_idx, evidence in enumerate(claim.evidences, start=1):
            for ref in evidence.references:
                chunk = chunk_lookup.get(ref.chunk_id)
                chunk_text = chunk.text if chunk else ""
                rule_supported = bool(chunk) and quote_supported_by_chunk(ref.quote, chunk_text)
                judge = _judge_reference(
                    parent_text=evidence.statement,
                    source_type="synthesis_evidence",
                    quote=ref.quote,
                    chunk_text=chunk_text,
                    judge_policy=judge_policy,
                    enable_llm_judge=enable_llm_judge,
                    rule_supported=rule_supported,
                )
                validations.append(
                    {
                        "source": "synthesis",
                        "claim_index": claim_idx,
                        "evidence_index": evidence_idx,
                        "parent_text": evidence.statement,
                        "doc_id": ref.doc_id,
                        "chunk_id": ref.chunk_id,
                        "quote": ref.quote,
                        "rule_supported": rule_supported,
                        "judge": judge,
                    }
                )

    for qa_idx, answer in enumerate(qa_results, start=1):
        for ref in answer.references:
            chunk = chunk_lookup.get(ref.chunk_id)
            chunk_text = chunk.text if chunk else ""
            rule_supported = bool(chunk) and quote_supported_by_chunk(ref.quote, chunk_text)
            judge = _judge_reference(
                parent_text=answer.answer,
                source_type="qa_answer",
                quote=ref.quote,
                chunk_text=chunk_text,
                judge_policy=judge_policy,
                enable_llm_judge=enable_llm_judge,
                rule_supported=rule_supported,
            )
            validations.append(
                {
                    "source": "qa",
                    "qa_index": qa_idx,
                    "question": answer.question,
                    "parent_text": answer.answer,
                    "doc_id": ref.doc_id,
                    "chunk_id": ref.chunk_id,
                    "quote": ref.quote,
                    "rule_supported": rule_supported,
                    "judge": judge,
                }
            )

    verdict_counts = {"valid": 0, "weak": 0, "invalid": 0}
    for item in validations:
        verdict = item["judge"]["verdict"]
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    report = {
        "metadata": {
            "generated_at_utc": datetime.utcnow().isoformat(),
            "topic": topic,
            "run_mode": run_mode,
            "judge_policy": judge_policy,
            "llm_judge_enabled": enable_llm_judge,
            "documents_count": len(index.documents),
            "chunks_count": len(index.chunks),
            "qa_questions_count": len(qa_results),
            "injection_lines_filtered": index.injection_lines_filtered,
        },
        "synthesis": synthesis.model_dump(),
        "qa_results": [item.model_dump() for item in qa_results],
        "reference_validations": validations,
        "quality_summary": {
            "total_references": len(validations),
            "rule_supported_count": sum(1 for item in validations if item["rule_supported"]),
            "judge_verdict_counts": verdict_counts,
        },
    }
    return report


def _export_comprehensive_report(report: dict) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"comprehensive_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(report_path)


def _load_markdown(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f"File not found: `{path}`"
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _render_sidebar() -> tuple[int, list, str, str]:
    st.sidebar.title("Portal Controls")
    st.sidebar.caption("Minimal controls only.")

    mode = st.sidebar.radio(
        "Run mode",
        options=["Online (API)", "Offline demo (no keys)"],
        index=0,
        help="Offline mode uses deterministic mock output for classroom demos.",
    )
    data_source = st.sidebar.radio(
        "Data source",
        options=["Upload PDFs", "Use docs/*.pdf"],
        index=1,
    )
    offline_mode = mode.startswith("Offline")
    if offline_mode:
        os.environ["OFFLINE_MODE"] = "1"
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBED_PROVIDER"] = "hash"
    else:
        os.environ["OFFLINE_MODE"] = "0"

    st.sidebar.divider()
    expected_docs = UI_MIN_DOCS
    uploads = []
    if data_source == "Upload PDFs":
        expected_docs = st.sidebar.number_input(
            "How many documents?",
            min_value=UI_MIN_DOCS,
            max_value=UI_MAX_DOCS,
            value=UI_MIN_DOCS,
            step=1,
        )
        uploads = st.sidebar.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "rst", "json", "csv"],
            accept_multiple_files=True,
        )
    return expected_docs, uploads, mode, data_source


def _render_home_tab(run_mode: str) -> None:
    st.subheader("Home")
    st.caption("Overview and guidance.")
    st.info(
        "This portal reads uploaded documents, chunks text, retrieves relevant evidence, "
        "synthesizes grounded claims, and answers questions with references."
    )
    st.markdown("**What this app does**")
    st.markdown(
        "- Builds a retrieval index from your files\n"
        "- Produces topic synthesis with claim/evidence structure\n"
        "- Answers questions over the same corpus\n"
        "- Exports a comprehensive report with reference validation"
    )
    st.markdown("**Current mode**")
    st.write(run_mode)
    st.markdown("**Fixed topic used for synthesis**")
    st.code(DEFAULT_TOPIC, language="text")


def _render_workspace_tab(
    *,
    run_mode: str,
    show_activity_panel: bool,
) -> None:
    st.subheader("Functions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Index", "Ready" if st.session_state.corpus else "Not ready")
    with col2:
        st.metric("Chunks", st.session_state.pipeline_stats["chunks"])
    with col3:
        st.metric("Claims", st.session_state.pipeline_stats["claims"])

    if show_activity_panel:
        with st.expander("Pipeline activity", expanded=False):
            if st.session_state.activity_logs:
                st.code("\n".join(st.session_state.activity_logs[-30:]), language="text")
            else:
                st.caption("No activity yet.")

    synthesis = st.session_state.synthesis
    if synthesis is None:
        st.info("Use sidebar files + 'Build Index + Run Synthesis' to begin.")
        return

    judge_policy = st.radio(
        "Reference Judge Mode",
        options=[
            "A - Always LLM Judge",
            "B - Optional LLM Judge",
            "C - LLM with Offline Fallback",
        ],
        horizontal=True,
    )
    enable_llm_judge = True
    if judge_policy.startswith("B"):
        enable_llm_judge = st.checkbox("Enable LLM judge for this export", value=False)

    st.markdown(f"**Summary:** {synthesis.topic_summary}")
    for idx, claim in enumerate(synthesis.claims, start=1):
        with st.expander(f"Claim {idx}: {claim.claim}", expanded=False):
            st.caption(f"Confidence: {claim.confidence}")
            for evidence in claim.evidences:
                st.write(f"- {evidence.statement}")
                for ref in evidence.references:
                    st.code(f"[{ref.doc_id} | {ref.chunk_id}] {ref.quote}", language="text")
    if synthesis.unresolved_questions:
        with st.expander("Unresolved questions", expanded=False):
            for question in synthesis.unresolved_questions:
                st.write(f"- {question}")

    if st.button("Export Comprehensive Report", use_container_width=True):
        if st.session_state.corpus is None:
            st.error("Build the index before exporting report.")
        else:
            report = _build_comprehensive_report(
                index=st.session_state.corpus,
                synthesis=synthesis,
                qa_results=st.session_state.qa_results,
                topic=DEFAULT_TOPIC,
                run_mode=run_mode,
                judge_policy=judge_policy,
                enable_llm_judge=enable_llm_judge,
            )
            path = _export_comprehensive_report(report)
            st.session_state.last_report_path = path
            st.success(f"Comprehensive report exported: {path}")
    if st.session_state.last_report_path:
        st.caption(f"Latest report: {st.session_state.last_report_path}")


def _render_qa_tab(show_activity_panel: bool) -> None:
    st.subheader("Ask Questions")
    st.caption("One question per line. Keep questions short and specific.")
    questions_blob = st.text_area(
        "Questions",
        placeholder="What is the main conclusion?\nWhich assumption is most uncertain?",
        height=120,
    )
    ask_clicked = st.button("Answer Questions", use_container_width=True)

    if ask_clicked:
        corpus = st.session_state.corpus
        if corpus is None:
            st.error("Build the index first from the sidebar.")
        else:
            questions = [line.strip() for line in questions_blob.splitlines() if line.strip()]
            if not questions:
                st.error("Please enter at least one question.")
            else:
                results: list[QAAnswer] = []
                _log_event(f"Q&A started for {len(questions)} question(s).")
                qa_status = st.status("Answering questions...", expanded=show_activity_panel)
                for question in questions:
                    qa_status.write(f"Question: {question}")
                    try:
                        _enforce_app_rate_limit(1.0)
                        results.append(answer_question(corpus, question))
                    except Exception as exc:  # pragma: no cover - streamlit UI guard
                        _log_event(f"Q&A failed for question: {question}. Error: {exc}")
                        st.error(f"Failed: {question}")
                        st.exception(exc)
                qa_status.update(label="Q&A completed.", state="complete")
                _log_event("Q&A completed.")
                st.session_state.qa_results = results
                st.session_state.pipeline_stats["answers"] = len(results)

    if not st.session_state.qa_results:
        st.info("No answers yet.")
        return

    for idx, answer in enumerate(st.session_state.qa_results, start=1):
        with st.expander(f"Q{idx}: {answer.question}", expanded=False):
            st.write(f"**Answer:** {answer.answer}")
            st.write(f"**Uncertainty:** {answer.uncertainty}")
            if answer.references:
                for ref in answer.references:
                    st.code(f"[{ref.doc_id} | {ref.chunk_id}] {ref.quote}", language="text")
            else:
                st.warning("No verifiable references for this answer.")


def _render_verifier_tab() -> None:
    st.subheader("Survey Verifier")
    st.caption("Choose which survey (S1-S4) most accurately reflects papers (P1-P10).")
    question = st.text_area(
        "Verification question",
        placeholder=(
            "Which survey most accurately captures contributions and limitations from P1-P10, "
            "and why?"
        ),
        height=100,
    )
    if st.button("Evaluate S1-S4 against P1-P10", use_container_width=True):
        index = st.session_state.corpus
        if index is None:
            st.error("Build the index first.")
            return
        if not question.strip():
            st.error("Enter a verification question.")
            return

        _log_event("Survey verifier started.")
        with st.status("Verifier running...", expanded=True) as status:
            status.write("Step 1/3: Retrieving survey and paper evidence")
            retrieved = index.retriever.search(question, top_k=60)
            survey_chunks = [
                item for item in retrieved if _is_survey_path(item.chunk.source_path)
            ][:20]
            paper_chunks = [
                item for item in retrieved if _is_paper_path(item.chunk.source_path)
            ][:30]
            if not survey_chunks or not paper_chunks:
                st.error("Could not gather enough survey/paper chunks from retrieval.")
                status.update(label="Verifier failed.", state="error")
                return

            def render_chunks(items: list, title: str) -> str:
                blocks: list[str] = []
                for idx, item in enumerate(items, start=1):
                    c = item.chunk
                    blocks.append(
                        f"[{title}-{idx}] doc_id={c.doc_id} "
                        f"chunk_id={c.chunk_id} source={Path(c.source_path).name}\n"
                        f"text={c.text}"
                    )
                return "\n\n".join(blocks)

            survey_context = render_chunks(survey_chunks, "S")
            paper_context = render_chunks(paper_chunks, "P")

            status.write("Step 2/3: Running comparative judge")
            prompt = f"""
You are an expert technical reviewer.
Goal: determine which survey among S1-S4 best matches evidence from papers P1-P10.
Be strict and technically accurate.

QUESTION:
{question}

SURVEY_CONTEXTS:
{survey_context}

PAPER_CONTEXTS:
{paper_context}

Return strict JSON:
{{
  "best_survey": "S1|S2|S3|S4|none",
  "confidence": "low|medium|high",
  "ranking": [
    {{
      "survey": "S1",
      "score": 0,
      "justification": "short technical reason",
      "supporting_paper_refs": [{{"doc_id":"", "chunk_id":"", "quote":""}}],
      "survey_refs": [{{"doc_id":"", "chunk_id":"", "quote":""}}]
    }}
  ],
  "final_answer": "clear final recommendation"
}}
""".strip()

            try:
                llm = get_llm_client()
                payload = llm.generate_json(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=1800)
            except Exception as exc:
                st.error(f"Verifier failed: {exc}")
                status.update(label="Verifier failed.", state="error")
                _log_event(f"Survey verifier failed: {exc}")
                return

            status.write("Step 3/3: Validating returned references")
            chunk_lookup = {chunk.chunk_id: chunk for chunk in index.chunks}
            for row in payload.get("ranking", []):
                for ref_key in ("supporting_paper_refs", "survey_refs"):
                    refs = row.get(ref_key, [])
                    valid_refs = []
                    for ref in refs:
                        chunk = chunk_lookup.get(ref.get("chunk_id", ""))
                        quote = str(ref.get("quote", ""))
                        if chunk and quote_supported_by_chunk(quote, chunk.text):
                            valid_refs.append(ref)
                    row[ref_key] = valid_refs

            st.session_state.verifier_result = payload
            status.update(label="Verifier completed.", state="complete")
            _log_event("Survey verifier completed.")

    result = st.session_state.verifier_result
    if not result:
        st.info("No verifier result yet.")
        return

    st.markdown(f"**Best survey:** {result.get('best_survey', 'unknown')}")
    st.markdown(f"**Confidence:** {result.get('confidence', 'unknown')}")
    st.markdown(f"**Recommendation:** {result.get('final_answer', '')}")
    for row in result.get("ranking", []):
        survey = row.get("survey", "unknown")
        score = row.get("score", 0)
        with st.expander(f"{survey} (score={score})", expanded=False):
            st.write(row.get("justification", ""))
            st.markdown("Paper references")
            for ref in row.get("supporting_paper_refs", []):
                st.code(
                    f"[{ref.get('doc_id')} | {ref.get('chunk_id')}] {ref.get('quote')}",
                    language="text",
                )
            st.markdown("Survey references")
            for ref in row.get("survey_refs", []):
                st.code(
                    f"[{ref.get('doc_id')} | {ref.get('chunk_id')}] {ref.get('quote')}",
                    language="text",
                )
    st.download_button(
        "Download verifier result JSON",
        data=json.dumps(result, indent=2, ensure_ascii=True),
        file_name="survey_verifier_result.json",
        mime="application/json",
    )


def _render_system_design_tab() -> None:
    st.subheader("Interactive System Design")
    st.caption("Pick any component to learn what it does and how data moves through the system.")

    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            user -> ui;
            ui -> ingest;
            ingest -> retrieve;
            retrieve -> llm;
            llm -> grounding;
            grounding -> output;
            output -> user;
        }
        """
    )

    components: dict[str, dict[str, str]] = {
        "UI Layer": {
            "files": "src/ui/streamlit_app.py, src/main.py",
            "role": "Collects input and shows output in a simple flow.",
            "operations": "Upload docs, trigger synthesis, ask Q&A.",
        },
        "Ingestion Layer": {
            "files": "src/ingest.py, src/security.py",
            "role": "Reads documents, validates file rules, sanitizes suspicious lines.",
            "operations": "Type checks, size checks, chunk creation.",
        },
        "Retrieval Layer": {
            "files": "src/retrieval.py",
            "role": "Finds the most relevant chunks for a topic or question.",
            "operations": "Embeddings + similarity ranking (online or offline hash).",
        },
        "LLM Layer": {
            "files": "src/llm_client.py, src/prompts.py",
            "role": "Builds prompts and calls configured model provider.",
            "operations": "Online providers (Grok/Azure) or offline deterministic mock.",
        },
        "Grounding Layer": {
            "files": "src/grounding.py, src/pipeline.py",
            "role": "Keeps only references whose quote appears in retrieved text.",
            "operations": "Reference filtering and uncertainty signaling.",
        },
        "Ops and Delivery": {
            "files": ".github/workflows/ci.yml, .github/workflows/release.yml, Dockerfile",
            "role": "Automates lint, test, security checks, and builds deployable image.",
            "operations": "CI quality gates, release image publishing.",
        },
    }

    choice = st.selectbox("Choose a component", list(components.keys()))
    data = components[choice]
    st.markdown(f"**Role:** {data['role']}")
    st.markdown(f"**Operations:** {data['operations']}")
    st.code(data["files"], language="text")

    st.divider()
    st.markdown("### Project Docs Explorer")
    docs = {
        "README": "README.md",
        "Deployment": "docs/deployment.md",
        "Security": "SECURITY.md",
        "Contributing": "CONTRIBUTING.md",
    }
    selected_doc = st.radio("Open document", list(docs.keys()), horizontal=True)
    st.markdown(_load_markdown(docs[selected_doc]))


def main() -> None:
    bootstrap_runtime_dirs()
    _init_state()
    _apply_ui_theme()
    st.title("RAG Learning Portal")
    st.caption("Guided workspace for synthesis, evidence tracking, and Q&A.")

    expected_docs, uploads, run_mode, data_source = _render_sidebar()
    show_activity_panel = True
    st.info(
        f"Mode: **{run_mode}** | Flow: **Upload -> Synthesize -> Ask Questions -> Export**"
    )

    if st.button("Build Index + Run Synthesis", use_container_width=True):
        if data_source == "Upload PDFs" and len(uploads) != expected_docs:
            st.error(f"Please upload exactly {expected_docs} documents. Current: {len(uploads)}.")
        else:
            st.session_state.activity_logs = []
            try:
                _log_event("Run started.")
                run_status = st.status("Pipeline started...", expanded=show_activity_panel)
                run_status.write("Step 1/4: Validating and reading uploads")
                _enforce_app_rate_limit(1.0)

                run_status.write("Step 2/4: Chunking documents and building index")
                if data_source == "Use docs/*.pdf":
                    doc_paths = sorted(str(path.resolve()) for path in Path("docs").glob("*.pdf"))
                    if not doc_paths:
                        raise ValueError("No PDF files found in docs/ folder.")
                    corpus = build_index_from_paths(doc_paths)
                else:
                    corpus = build_index_from_uploads(
                        uploads,
                        max_documents=expected_docs,
                    )
                st.session_state.pipeline_stats["documents"] = len(corpus.documents)
                st.session_state.pipeline_stats["chunks"] = len(corpus.chunks)
                st.session_state.pipeline_stats["filtered_lines"] = corpus.injection_lines_filtered
                _log_event(
                    "Index ready: "
                    f"documents={len(corpus.documents)}, chunks={len(corpus.chunks)}, "
                    f"filtered_lines={corpus.injection_lines_filtered}."
                )

                run_status.write("Step 3/4: Retrieving relevant chunks and synthesizing")
                synthesis = synthesize_topic(corpus, topic=DEFAULT_TOPIC)
                st.session_state.pipeline_stats["claims"] = len(synthesis.claims)
                _log_event(f"Synthesis completed: claims={len(synthesis.claims)}.")

                run_status.write("Step 4/4: Persisting state and preparing workspace")
                st.session_state.corpus = corpus
                st.session_state.synthesis = synthesis
                st.session_state.qa_results = []
                st.session_state.pipeline_stats["answers"] = 0
                run_status.update(label="Pipeline completed.", state="complete")
                _log_event("Run completed successfully.")
                st.success(
                    "Synthesis ready. "
                    f"Filtered {corpus.injection_lines_filtered} suspicious lines."
                )
            except Exception as exc:  # pragma: no cover - streamlit UI guard
                _log_event(f"Pipeline failed: {exc}")
                st.exception(exc)

    tab_home, tab_functions, tab_qa, tab_verify, tab_design = st.tabs(
        ["Home", "Functions", "Q&A", "Survey Verifier", "System Design"],
    )
    with tab_home:
        _render_home_tab(run_mode=run_mode)
    with tab_functions:
        _render_workspace_tab(
            run_mode=run_mode,
            show_activity_panel=show_activity_panel,
        )
    with tab_qa:
        _render_qa_tab(show_activity_panel=show_activity_panel)
    with tab_verify:
        _render_verifier_tab()
    with tab_design:
        _render_system_design_tab()


if __name__ == "__main__":
    main()
