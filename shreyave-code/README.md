# Mini-Survey: LLM Agents for Scientific Research

## Submission Mode

**Replay Mode** — all PDF text was extracted locally using `strings` (no API keys required).
The `evidence.json` quotes are sourced from cached corpus text in `cache/papers_text/`.
The `paper.docx` is generated from `scripts/build_paper.js` using only local files.

## External Services Used

- **None at replay time.** Corpus PDFs were read via `strings` extraction.
- During authoring: Claude (claude-sonnet-4-6) via claude.ai interface for synthesis reasoning and critical analysis.
- The API keys in `.env.example` enable a full automated pipeline (see below), but are **NOT required** for replay.

## Cache Files Included

| File | Contents |
|------|----------|
| `cache/papers_text/P1.txt` … `P10.txt` | Cached corpus text used by quote verification |

## Exact Reproduction Command

```bash
# Replay mode — no API keys needed
pip install -r requirements.txt
npm install
python scripts/run_all.py --mode replay

# This will:
# 1. Verify evidence.json quotes against cache/papers_text/
# 2. Regenerate eval.json from evidence.json
# 3. Regenerate taxonomy_figure.png
# 4. Rebuild paper.docx from scripts/build_paper.js
# 5. Copy artifacts to outputs/offline/
```

Artifacts are also copied to separate spaces:
- Offline replay outputs: `outputs/offline/`
- Online/full outputs: `outputs/online/`

## Full Pipeline (requires API keys in .env)

```bash
cp .env.example .env
# Fill in GROK_API_KEY and AZURE_API_KEY
python scripts/run_all.py --mode full
```

This runs:
1. PDF text extraction
2. Quote retrieval using sentence-transformers embeddings (all-MiniLM-L6-v2)
3. Claim verification via Grok-3 or o4-mini
4. Automatic evidence.json and eval.json generation
5. Syncs artifacts to `outputs/online/`

## Minimal Submission Structure

```
.
├── paper.docx                    # Final paper (also in paper ZIP)
├── evidence.json                 # 20 verbatim-quote entries, 2 per claim
├── eval.json                     # Self-reported metrics
├── prompts.md                    # Prompt log
├── README.md                     # This file
├── .env.example                  # API key template (no real keys)
├── requirements.txt              # Python dependencies for scripts
├── package.json                  # Node dependency for docx builder
├── scripts/
│   ├── run_all.py                # Main entry point
│   ├── extract_text.py           # PDF strings extraction
│   ├── verify_quotes.py          # Verify evidence.json against cache
│   ├── generate_eval.py          # Auto-generate eval.json
│   ├── generate_figure.py        # Generate taxonomy_figure.png
│   └── build_paper.js            # Build paper.docx (Node.js)
└── cache/
    └── papers_text/              # P1.txt … P10.txt
```

## Word Count Check

| Section | Words |
|---------|-------|
| Literature Summary | 821 |

## Dependency Notes

```
# Node.js (for paper.docx generation)
npm install

# Python
pip install -r requirements.txt
```
