# Mini-Survey: LLM Agents for Scientific Research

## Pipeline Modes

- `replay`: deterministic rebuild from existing artifacts and cache
- `offline_rag`: end-to-end local RAG from cache (no API keys)
- `full`: refresh cache from PDFs, then run replay

## Important Scope Note

- `replay` is reproducibility mode.
- `offline_rag` is the true offline RAG flow in this repo:
  1) chunk cached corpus text,
  2) retrieve evidence with local scoring,
  3) generate `evidence.json`,
  4) verify quotes,
  5) regenerate `eval.json`,
  6) rebuild taxonomy figure and `paper.docx`.

## Run Commands

```bash
cd shreyave-code
pip install -r requirements.txt
npm install

# deterministic artifact rebuild
python scripts/run_all.py --mode replay

# end-to-end offline RAG (no API keys)
python scripts/run_all.py --mode offline_rag
```

## Online/Full Mode

```bash
cp .env.example .env
# set keys in .env
python scripts/run_all.py --mode full
```

## Outputs

- Offline artifacts: `outputs/offline/`
- Online artifacts: `outputs/online/`

## Cache Included

| File | Contents |
|------|----------|
| `cache/papers_text/P1.txt` ... `P10.txt` | Cached corpus text used by replay and offline RAG |

## Dependencies

```bash
# Python
pip install -r requirements.txt

# Node.js (paper.docx build)
npm install
```
