# ai-model-dev-rag

Reusable RAG skeleton and Streamlit portal for ingesting 10-20 documents, synthesizing cross-document insights, separating claims vs evidence, and attaching references to source chunks.

## What this includes

- Shared, `.env`-driven LLM API configuration reused from `llm-carbon-footprint-research-portal`.
- Provider-agnostic LLM client (`grok` and `azure_openai`) with retry/backoff and throttling.
- Streamlit portal with controlled upload count input and upload limit enforcement.
- Document ingestion for `pdf`, `txt`, `md`, `rst`, `json`, `csv`.
- Chunking + embedding retrieval using `sentence-transformers`.
- Structured synthesis output with:
  - `topic_summary`
  - `claims[]`
  - `evidences[]`
  - `references[]` as `doc_id`, `chunk_id`, and direct `quote`
  - `unresolved_questions[]`
- Retrieval-augmented Q&A over the uploaded corpus with reference-backed answers.
- Prompt-injection hardening:
  - suspicious instruction-like lines are filtered during ingestion
  - model is instructed to treat document text as untrusted
  - references are post-validated against real chunk text

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your API keys.

## Run

### CLI

```bash
python -m src.main \
  --topic "Synthesize the key carbon accounting assumptions across these documents" \
  --docs /absolute/path/doc1.pdf /absolute/path/doc2.txt /absolute/path/doc3.md \
  --output outputs/synthesis_result.json
```

### Streamlit Portal

```bash
streamlit run src/ui/streamlit_app.py
```

Portal flow:
- set expected number of docs (`UI_MIN_DOCS` to `UI_MAX_DOCS`, default 10-20)
- upload exactly that many docs
- run synthesis
- ask one or more questions (one per line) for retrieval-grounded answers
- inspect references (`doc_id`, `chunk_id`, quote) in UI

## Output format

The output JSON is validated by Pydantic schema in `src/models.py`:

- `topic_summary: str`
- `claims: list[ClaimItem]`
  - `claim: str`
  - `confidence: low|medium|high`
  - `evidences: list[EvidenceItem]`
    - `statement: str`
    - `references: list[Reference]`
      - `doc_id: str`
      - `chunk_id: str`
      - `quote: str`
- `unresolved_questions: list[str]`

## Project structure

```
src/
  config.py       # env config + tunables
  llm_client.py   # provider abstraction and retries
  security.py     # upload validation + injection filtering + quote checks
  ingest.py       # document loading and chunking
  retrieval.py    # embeddings + cosine search
  prompts.py      # synthesis prompt templates
  models.py       # structured output schemas
  pipeline.py     # end-to-end orchestration
  main.py         # CLI entrypoint
  ui/
    streamlit_app.py
```

## Notes

- This is a skeleton intended to be extended.
- "Perfect" reference identification depends on source document quality and model behavior; this scaffold maximizes grounding by forcing quote-level references and validating that quotes exist in retrieved chunks.
