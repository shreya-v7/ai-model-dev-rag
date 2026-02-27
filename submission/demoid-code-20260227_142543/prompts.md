# Prompt Log

## Prompt 1: Topic Synthesis
- Tool/model: configured project LLM client
- Purpose: generate structured synthesis (summary, claims, evidence, references)
- Prompt summary: topic-based synthesis over retrieved contexts with strict JSON schema

## Prompt 2: Survey Verifier
- Tool/model: configured project LLM client
- Purpose: rank S1-S4 against P1-P10 using explicit evidence
- Prompt text:
From S1, S2, S3, S4, which survey most accurately matches P1-P10 on methods, evaluation, and limitations? Rank all and justify with evidence.

## Prompt 3: Reference Validation
- Tool/model: configured project LLM client or fallback rule checks
- Purpose: validate whether referenced quote is grounded in retrieved chunk text

## Automation and Reproducibility Notes
- This pipeline supports offline/replay-friendly execution without grader API keys.
- Use `--mode offline` for deterministic local fallback.
