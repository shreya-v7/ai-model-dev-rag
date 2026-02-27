# ai-model-dev-rag

Production-grade RAG pipeline and Streamlit portal with modular architecture, prompt-injection safeguards, deterministic offline mode, CI/CD quality gates, and test coverage enforcement.

## Key Features

- Modular service-oriented pipeline (`src/pipeline.py` + `RagService`).
- Robust document ingestion and chunking with upload validation.
- Retrieval grounding with quote verification (`src/grounding.py`).
- LLM abstraction with retries and provider routing (`grok`, `azure_openai`, `mock`).
- Deterministic offline mode for no-key demos (`LLM_PROVIDER=mock`, `EMBED_PROVIDER=hash`).
- Security-oriented prompt design and ingestion sanitization.
- CI/CD workflows for lint, tests, security scans, and container build.
- Test suite with `>=80%` coverage gate.

## Architecture

- `src/config.py`: environment-driven config and runtime bootstrap.
- `src/security.py`: upload policies, prompt-injection line filtering, quote checks.
- `src/ingest.py`: document loading + sanitization + chunk generation.
- `src/retrieval.py`: retrieval engine with sentence-transformer or hash embedder.
- `src/llm_client.py`: provider clients + deterministic offline mock.
- `src/grounding.py`: citation grounding enforcement.
- `src/pipeline.py`: corpus indexing and application service orchestration.
- `src/ui/streamlit_app.py`: Streamlit interface adapter.
- `src/main.py`: CLI interface adapter.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

## Run (Online)

```bash
python -m src.main \
  --topic "Synthesize the key carbon accounting assumptions across documents" \
  --docs /absolute/path/doc1.pdf /absolute/path/doc2.txt \
  --output outputs/synthesis_result.json
```

```bash
streamlit run src/ui/streamlit_app.py
```

## Run (Offline, No API Keys)

Use this mode for professor/evaluator demos where internet/API keys are unavailable.

### CLI Offline

```bash
python -m src.main \
  --offline \
  --topic "Explain the key assumptions" \
  --docs /absolute/path/doc1.txt /absolute/path/doc2.md \
  --output outputs/offline_result.json
```

### Streamlit Offline

```bash
streamlit run src/ui/streamlit_app.py
```

Then enable **Offline deterministic mode** in the UI.

## Testing and Quality Gates

```bash
ruff check src tests
pytest
pip-audit -r requirements.txt
bandit -r src -q
```

Coverage policy is enforced by pytest config:
- `--cov=src`
- `--cov-fail-under=80`

## CI/CD

- `.github/workflows/ci.yml`
  - lint (`ruff`)
  - tests with coverage gate (`pytest`)
  - dependency security scan (`pip-audit`)
  - static security scan (`bandit`)
  - Docker image build validation
- `.github/workflows/release.yml`
  - tag-triggered release image publish to GHCR

## Container

```bash
docker build -t ai-model-dev-rag:local .
docker run --rm ai-model-dev-rag:local
```

## Security Model

- Treats retrieved corpus text as untrusted.
- Filters suspicious instruction-like lines during ingestion.
- Uses strict system prompts to enforce instruction hierarchy.
- Validates quoted references against retrieved chunk text before final output.

See `SECURITY.md` for details.

## Operational Docs

- `docs/deployment.md`
- `SECURITY.md`
- `CONTRIBUTING.md`

## Assignment End-to-End Packaging

Generate rubric-oriented submission artifacts and ZIPs:

```bash
python scripts/run_assignment_pipeline.py \
  --andrewid <andrewid-lowercase> \
  --mode offline \
  --comparison-survey S2 \
  --paper-file /absolute/path/to/paper.docx
```

Outputs under `submission/`:
- `<andrewid>-code.zip`
- `<andrewid>-paper.zip` (when `--paper-file` is provided)
- timestamped code bundle folder containing `evidence.json`, `eval.json`, `prompts.md`, `README.md`
- run diagnostics in each bundle:
  - `run_trace.jsonl` (timestamp, pid, thread_id, step, status, detail)
  - `run_summary.json`
  - `rubric_report.json`
