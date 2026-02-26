# Security Policy

## Scope

This project focuses on prompt-injection resilience and grounded-output guarantees for RAG workflows.

## Implemented Controls

- Upload restrictions:
  - extension allowlist
  - upload size limit
- Ingestion sanitization:
  - suspicious instruction-like lines are filtered
- Prompt hardening:
  - strict system instruction hierarchy
  - explicit directive that corpus text is untrusted
- Grounding checks:
  - every cited quote is verified against chunk text
  - unsupported references are removed

## Reporting Vulnerabilities

If you discover a security issue, do not post it publicly. Share details privately with maintainers and include:
- reproduction steps
- affected file/module
- potential impact
- suggested remediation
