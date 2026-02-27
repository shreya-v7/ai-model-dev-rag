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
| `cache/papers_text/P1.txt` … `P10.txt` | Raw strings-extracted text from each corpus PDF |
| `cache/llm_outputs/evidence_draft.json` | Initial evidence draft (overwritten by verified version) |

## Exact Reproduction Command

```bash
# Replay mode — no API keys needed
python scripts/run_all.py --mode replay

# This will:
# 1. Verify evidence.json quotes against cache/papers_text/
# 2. Regenerate eval.json from evidence.json
# 3. Rebuild paper.docx from scripts/build_paper.js
# 4. Regenerate taxonomy_figure.png
# 5. Package both ZIPs
```

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

## Project Structure

```
.
├── paper.docx                    # Final paper (also in paper ZIP)
├── evidence.json                 # 20 verbatim-quote entries, 2 per claim
├── eval.json                     # Self-reported metrics
├── prompts.md                    # Major prompts log
├── README.md                     # This file
├── .env.example                  # API key template (no real keys)
├── requirements.txt              # Python dependencies
├── scripts/
│   ├── run_all.py                # Main entry point
│   ├── extract_text.py           # PDF strings extraction
│   ├── verify_quotes.py          # Verify evidence.json against cache
│   ├── generate_eval.py          # Auto-generate eval.json
│   ├── generate_figure.py        # Generate taxonomy_figure.png
│   └── build_paper.js            # Build paper.docx (Node.js)
└── cache/
    ├── papers_text/              # P1.txt … P10.txt
    └── llm_outputs/              # Cached LLM responses
```

## Word Counts (excluding References)

| Section | Words |
|---------|-------|
| Literature Summary | ~780 |
| Key Claims Table | ~420 |
| Future Directions | ~530 |
| Taxonomy | ~380 |
| Reflection | ~580 |
| **Total** | **~3461** |

## Dependency Notes

```
# Node.js (for paper.docx generation)
npm install docx  # in scripts/ directory

# Python
pip install sentence-transformers requests python-dotenv
```
