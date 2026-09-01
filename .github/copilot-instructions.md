# Project instructions for Copilot

## Code Graph First (MANDATORY)
- Before analyzing or modifying ANY part of this project, first read `CODEGRAPH.md` at the repo root (the maintained code map: module index, Python/Go dual-backend mechanics, core data flows, entry-file jump table).
- Then use a code-graph approach (symbol usages/references, value-flow tracing, or an Explore subagent) to read the relevant call chain end-to-end before making changes. No blind repo-wide searches.
- Confirm whether the task is on the Python path (`api/` `rag/` `deepdoc/` `agent/`) or the Go path (`cmd/` `internal/`) before editing.
- See `AGENTS.md` for the full operating guide.

## How to run (minimum)
- Install:
  - python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt
- Run:
  - (fill) e.g. uvicorn app.main:app --reload
- Verify:
  - (fill) curl http://127.0.0.1:8000/health

## Project layout (what matters)
- app/: API entrypoints + routers
- services/: business logic
- configs/: config loading (.env)
- docs/: documents
- tests/: pytest

## Conventions
- Prefer small, incremental changes.
- Add logging for new flows.
- Add/adjust tests for behavior changes.
