# Code Bundle Reproducibility

- Supported mode: `offline`
- External services:
  - `online`: configured LLM provider from `.env`
  - `offline`: deterministic mock + hash embeddings
- Cache files: generated artifacts in this bundle
  (`evidence.json`, `eval.json`, `verifier_result.json`)

Reproduce artifacts:
```bash
python scripts/run_assignment_pipeline.py --andrewid <andrewid> --mode offline --comparison-survey S2
```

If no scripts are used by grader, artifacts remain readable and reviewable as static JSON.
