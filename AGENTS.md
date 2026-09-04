# RAGFlow Instructions

Use this file as the local operating guide for the current codebase. Prefer the code and the current CLAUDE.md over any older convention or remembered project shape.

## Before Any Analysis — Code Graph First (MANDATORY)
- **Always read `CODEGRAPH.md` at the repo root FIRST** before analyzing, modifying, or reviewing any part of this project. It is the maintained code map: directory index, dual-backend (Python/Go) mechanics, the four core data flows, and a jump table to high-frequency entry files.
- After locating the involved modules via `CODEGRAPH.md`, read the relevant code's overall call chain with a code-graph approach (symbol usage/reference tracing, value-flow tracing, or an exploration subagent) — entry → middle layers → persistence/output — before drawing conclusions or making edits. Do not start with blind repo-wide searches.
- Always determine whether the task belongs to the Python path (`api/` `rag/` `deepdoc/` `agent/`) or the Go path (`cmd/` `internal/`) before editing; they are parallel implementations and changing the wrong one has no effect.
- If project structure changes significantly (new top-level dirs, backend convergence, port/service changes), update `CODEGRAPH.md` in the same change.

## CodeGraph Index Sync (MANDATORY)
- The local CodeGraph index lives in `.codegraph/` (machine-local, git-ignored). Whenever ANY code change has been made (edit, create, delete, rename), run `codegraph sync` at the repo root before the turn ends so the index stays current. This is mandatory after every change batch — do not skip it or defer it to the user.
- A lefthook `post-commit` job (`codegraph-sync`) also runs `codegraph sync -q` automatically on every commit as a safety net; the manual sync above is still required because analysis happens between commits.
- If `codegraph` is not on PATH (expected at `D:\Users\hongze01.zhang\AppData\Local\codegraph\current\bin\codegraph.cmd` on this machine), report it and continue; never block the task on a missing index.

## Git Workflow (MANDATORY)
- **All work happens on the `study` branch.** If it does not exist locally, create it (`git checkout -b study`). Never commit directly to `main` or any other branch.
- **Push only to `origin`** (https://github.com/Hz-186/ragflow.git), e.g. `git push origin study`. **Never push to `upstream`** (the infiniflow/ragflow source repo) or open PRs against it.
- Do not rebase/force-push shared branches without explicit user instruction.

## Core Stance
- Treat legacy code as liability, not as a compatibility target.
- Prefer deletion over shims, deprecated branches, wrapper APIs, and dual-track migration notes.
- If old and new implementations coexist, converge to one path unless an external contract forces compatibility.
- Remove dead tests, commented-out code, stale docs, and "move later" notes instead of preserving them.
- Reduce public surface area when a helper can be made private or internal.
- Keep refactors centered on the owning abstraction, not on adjacent compatibility layers.

## Current stack
- Backend: Python 3.13+, Quart-based API server, Peewee ORM, async workers.
- Frontend: React + TypeScript + Vite in `web/` (dual-backend Go/Python variant conventions: see `web/CLAUDE.md`).
- Go: the repository also has a substantial Go module for servers, ingestion, parser/runtime, CLI, and supporting services.
- Runtime services commonly include MySQL/PostgreSQL, Redis, MinIO, and Elasticsearch/Infinity/OpenSearch depending on configuration.

## Code Layout to Expect
- `api/`: Python API server entrypoints, blueprints, services, and database code.
- `rag/`: ingestion, retrieval, LLM integration, and graph RAG logic.
- `deepdoc/`: parsing and OCR.
- `agent/`: workflow canvas, components, tools, and templates.
- `cmd/`: Go entrypoints. `ragflow_main` is the main server/admin/ingestor binary surface; `ragflow-cli` is the CLI entrypoint.
- `internal/`: main Go application code. Important subtrees:
- `internal/agent/`: Go agent runtime, canvas execution, components, tool bindings, workflow helpers.
- `internal/cli/`: CLI parsing, HTTP transport, command execution, response formatting.
- `internal/dao/`: Go data-access layer and persistence-facing helpers.
- `internal/deepdoc/`: Go DeepDOC integrations, especially native-backed PDF/DOCX parsing.
- `internal/engine/`: search/index backends such as Elasticsearch and Infinity.
- `internal/entity/`: shared Go entities and model definitions.
- `internal/handler/`: HTTP handlers and route-facing request logic.
- `internal/ingestion/`: Go ingestion pipeline, canvas adapter, components, wiring, service orchestration.
- `internal/ingestion/component/`: stage implementations such as file/parser/chunker/tokenizer/extractor.
- `internal/ingestion/pipeline/`: DSL translation, canvas-driven execution, checkpoints, resume/run logic.
- `internal/parser/`: parser and chunk libraries used by ingestion and other Go paths.
- `internal/parser/parser/`: typed parse-result parsers for markdown/html/pdf/docx/xlsx/text and related families.
- `internal/parser/chunk/`: chunk operator library and DSL/typed execution helpers.
- `internal/service/`: higher-level business services used by handlers and server flows.
- `internal/storage/`: storage backends and in-memory test doubles.
- `internal/router/`: HTTP route registration.
- `internal/server/`: server bootstrap/config wiring.
- `internal/cpp/`: C++ sources used by native-backed Go features.
- `web/`: frontend application.
- `docker/`: local and production compose files.
- `sdk/` and `test/`: SDK and automated tests.

## 注释规范（强制）

本仓库的注释以「让没读过这段代码的人一遍看懂」为唯一目标，统一执行以下规则（范例见 `rag/nlp/__init__.py` 中的中文注释风格）：

1. **函数开头注释只写三样东西**：
   - 一句话说明这个函数是干什么的（用大白话，可以加一个「—— 某某器/某某工」的短比喻）；
   - **每个传入参数的含义，以及它「长得什么样子」**：用代码块/缩进给出真实的数据结构示例（例如 Go 的 map/slice/struct 字面量、Python 的 dict/list 示例），让读者不用跳去别处就能想象出数据的实际形态；
   - 返回值「长得什么样子」（同样给出真实结构示例）。
2. **步骤说明一律写在函数体内对应代码的旁边，且必须标注真实数据长相**：
   - 每一步做什么、为什么这么做，写成紧贴该步代码的行内/块内注释。**禁止**把一个函数的所有步骤集中堆在函数开头写成一大段「流程总览」。唯一例外：某一段逻辑本身技术含量很高、三言两语说不清，可以在那一小段代码上方多写几行把它讲透；
   - **行内必须带真实数据示例**：关键数据处理与流转步骤，必须直接在旁边用 `[]`（列表/切片）、`{}`（字典/JSON 对象）等具体结构展示输入、产出数据的真实长相（例如 `输入: [{"id": "c1", ...}]`，`输出: [[{"chunk_id": "c1", ...}]]`），让读者一眼看透数据如何流动与变形。
3. **语言要求**：注释用中文，通俗易懂；禁止「这个/那个」式指代、黑话、不加解释的专有名词堆砌。原有英文注释翻译成中文；如果直译后仍然难懂，就改写成能让人看懂的版本。**只改注释，绝不改动任何代码逻辑**。
4. **批量改写必须分批提交**：一次要改的注释太多时，先列一个待改函数清单（list），然后分多次编辑，每次只替换一部分，保证每次改动可审、可回滚。
5. **改完必须自查**：每个文件改完后，检查是否还有「函数头大段步骤说明」残留、是否有未翻译的英文注释、关键步骤是否已带上 `[]` / `{}` 真实数据结构示例、是否有被误改的代码；必要时用子代理（subagent）复查，发现问题继续修，直到达标。

## Working Rules
- When reviewing documentation or code, inspect the full affected path and report all verifiable findings in one review; do not return after only a few findings and expose further issues in later rounds.
- When handling review comments, independently verify each substantive claim against the current code or tests before accepting, rejecting, or acting on it.
- Before editing, inspect the nearest code path that actually owns the behavior.
- Keep changes small and local unless the task is explicitly a broader refactor.

## Commands
### Backend
```bash
uv sync --python 3.13 --all-extras
uv run python3 ragflow_deps/download_deps.py
docker compose -f docker/docker-compose-base.yml up -d
source .venv/bin/activate
export PYTHONPATH=$(pwd)
bash docker/launch_backend_service.sh
uv run pytest
ruff check
ruff format
```

### Go
```bash
uv run ragflow_deps/download_deps.py
bash build.sh --test ./path/to/package/...
bash build.sh --go
# or build specific binaries:
bash build.sh --all
```

## Validation Preference
- Run the narrowest relevant test, lint, or build command after a change.
- For backend changes, prefer targeted pytest or ruff checks over full-suite runs.
- For Go changes, prefer package-scoped `bash build.sh --test ...` first.
- Do not default to raw `go test`, `go build`, or IDE Run/Debug for Go in this repo. They often miss the required CGO flags and native static libraries (`office_oxide`, `pdfium-static`, `pdf_oxide`) that `build.sh` wires correctly.
- If Go native builds fail, inspect `build.sh` and `internal/development.md` before changing code. Common environment issues are missing downloaded native deps and missing `lld` on Linux.
