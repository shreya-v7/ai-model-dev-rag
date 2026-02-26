# Deployment Guide

## CI Pipeline

`ci.yml` enforces:
- lint checks
- tests with coverage threshold
- dependency/static security scans
- container build verification

## Release Flow

1. Merge to default branch with all CI checks passing.
2. Create and push semantic tag (`vX.Y.Z`).
3. `release.yml` builds and publishes image to GHCR.

## Runtime Configuration

- Configure `.env` with provider and tuning values.
- For deterministic no-key demo:
  - `OFFLINE_MODE=1`
  - `LLM_PROVIDER=mock`
  - `EMBED_PROVIDER=hash`

## Rollback

- Re-deploy previous known-good image tag from GHCR.
- Keep release notes tied to tag versions for fast rollback decisions.
