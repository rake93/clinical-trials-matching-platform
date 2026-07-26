# Copilot Instructions

## Repository Overview

Local-first clinical research AI monorepo with four Python modules, a Vite UI, and a legacy static UI:

| Module | Purpose | Runtime |
|---|---|---|
| `agentic-reasoning/` | Deterministic two-phase pipeline: GraphRAG retrieval → LLM synthesis | Python 3.12+ |
| `data-acquisition/` | Multi-source PDF fetcher (ClinicalTrials.gov, PubMed, bioRxiv, medRxiv); local storage by default, optional cloud providers | Python 3.12+ |
| `data-ingestion/` | Five document stages (OCR → embed) plus optional LLM triplet extraction into Neo4j | Python 3.11 via the root Makefile |
| `simple-ui/` | Legacy standalone HTML/JS/CSS frontend; no build step | Static |
| `palantir-blueprint/` | Vite + React + strict TypeScript UI using Blueprint components | Node 20 in CI |
| `core-llm-inference/` | SGLang production inference server (GPU, RTX 5080 / L4) | Python 3.12+ |

End-to-end flow: `data-acquisition` fetches PDFs → `data-ingestion` processes them into Qdrant vectors + Neo4j graph → `agentic-reasoning` queries both → `simple-ui` / `palantir-blueprint` visualises results.

Treat the root `Makefile`, `config/app.yaml`, `.github/workflows/ci.yml`, current source, and tests as authoritative. Some module READMEs and `docs_v2/` pages still describe removed module-local Makefiles, LangGraph/ReAct routing, a Next.js `platform-ui`, or a cloud-first default.

---

## Commands

**Makefile topology:** one Makefile at the repo root. Do not use module-local Makefiles. Use root namespaced targets (`reasoning-*`, `acquisition-*`, `ingestion-*`, `blueprint-*`, `inference-*`); `make help` lists the current surface.

### CI-equivalent validation

Python modules each own a top-level `src` package, so run lint/tests from inside the touched module. Running repo-wide `ruff check .` or `pytest` from the root causes cross-module import collisions. CI uses:

```bash
(cd agentic-reasoning && ruff check . && .venv/bin/python -m pytest -m "not integration" --tb=short)
(cd data-acquisition && ruff check . && .venv/bin/python -m pytest -m "not integration" --tb=short)
(cd data-ingestion && ruff check . && .venv/bin/python -m pytest -m "not integration" --tb=short)

# UI type-check + production build; package.json has no separate lint/test script
(cd palantir-blueprint && npm ci --legacy-peer-deps && npm run build)
```

### agentic-reasoning

```bash
# Setup (from repo root)
make reasoning-install              # create .venv if needed + pip install -e .

# Run
make reasoning-run                            # interactive CLI
make reasoning-run-query QUERY="..."          # single-shot query

# Explicit SGLang endpoint
make reasoning-sglang-run-query QUERY="..."   # SGLANG_BASE_URL=http://localhost:30000/v1

# Test
make reasoning-test                           # pytest tests/ -v
# Single test:
cd agentic-reasoning && .venv/bin/python -m pytest \
  tests/test_agent.py::TestPhase1Enforcement::test_graphrag_called_before_llm -v
```

### data-ingestion

```bash
# Setup (from repo root)
make ingestion-install                        # Python 3.11 .venv + requirements.txt

# Run pipeline
make ingestion-run                            # stages 1-6 (default N=2 PDFs)
make ingestion-run N=10 SKIP=ocr             # skip: ocr|convert|clean|chunk|vectorize|graph

# Test
make ingestion-test                           # pytest tests/ -v
cd data-ingestion && .venv/bin/python -m pytest \
  tests/test_vector_indexing.py::test_index_chunks_path_replaces_only_matching_scoped_document -v
make ingestion-test-processors               # python tests/test_processors.py (standalone script)
make ingestion-test-embedder                 # python tests/test_embedder.py
make ingestion-test-qdrant                   # python tests/test_qdrant.py

# Infrastructure
make ingestion-qdrant-up                     # start Qdrant (data-ingestion/infra/docker-compose.yaml)
make ingestion-neo4j-build                   # build knowledge graph from chunks

# Debugging
make ingestion-inspect                        # file counts at each pipeline stage
make ingestion-list-documents                # list tracked docs with UUIDs
make ingestion-compare-runs DOC=<uuid> EXEC1=<uuid> EXEC2=<uuid>
```

### data-acquisition

```bash
# Setup (from repo root)
make acquisition-install

# Fetch PDFs
make fetch SOURCE=medrxiv MAX_PDFS=10        # aliases acquisition-fetch

# Test (skips integration tests requiring cloud credentials)
make acquisition-test                         # pytest -m "not integration"
# Single test:
cd data-acquisition && .venv/bin/python -m pytest \
  tests/storage/test_local_fallback.py::TestLocalFallbackConfig::test_local_config -v
```

### Benchmarking

```bash
make benchmark-all                            # full evaluation harness (RUN_DIR auto-generated)
make benchmark-retrieval                      # Recall@K, NDCG, MRR, HitRate
make benchmark-reasoning                      # two-phase agent evaluation (20 golden queries)
make benchmark-report RUN_DIR=benchmarking/results/<run>

# Full deterministic end-to-end run (wipe → ingest → KG → all benchmarks → manifest.json):
make deterministic-run
make deterministic-run BENCH_PDF=data/pdfs/my.pdf BENCH_RUNS=5

# Reranker override (blank = disabled):
make benchmark-retrieval RERANKER_MODEL=""
```

### palantir-blueprint (React UI)

```bash
make blueprint-install                        # npm install
make blueprint-dev                            # Vite dev server on :5173 (hot-reload)
make blueprint-build                          # production build → palantir-blueprint/dist/
make blueprint-preview                        # build + preview on :4173
```

### core-llm-inference (SGLang — GPU only)

```bash
make inference-install                        # create .venv + torch cu124 + sglang[all]
make inference-serve                          # start detached SGLang server on :30000
make inference-serve-fg                       # foreground (Ctrl-C to stop)
make inference-stop                           # kill background server
make inference-status                         # health + loaded model + GPU stats
make inference-benchmark N=10                 # latency/throughput benchmark

# Lightweight unit-test environment (does not require SGLang/GPU)
cd core-llm-inference
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/unit/
.venv/bin/python -m pytest tests/unit/test_metrics.py::TestComputeMbu::test_known_values -v
```

### Infrastructure

```bash
make bootstrap                                # first-time: check deps + create .env.local
make up                                       # start Neo4j + Qdrant (docker-compose.local.yml)
make down
make validate                                 # check SGLang/LM Studio failover, Qdrant, Neo4j
make status                                   # running containers + artifact counts

make dev                                      # ★ start all services: reasoning API :8000, ingestion API :8001, blueprint UI :5173
make ingestion-api                            # ingestion FastAPI server on :8001 (standalone)
make reasoning-serve                          # reasoning FastAPI server on :8000 (standalone)
make simple-ui-serve                          # serve simple-ui on :3000
```

### Cross-module tests (root `tests/`)

```bash
make test-suite                               # pytest tests/ -v (uses agentic-reasoning .venv)
# Single test:
agentic-reasoning/.venv/bin/python -m pytest \
  tests/test_config_schema.py::TestIngestionConfig::test_chunking_params_sane -v
```

### API Servers

**agentic-reasoning** exposes a FastAPI server (`src/server.py`) on `:8000`:
- `POST /api/match` — Phase 1: hybrid GraphRAG retrieval → matches JSON
- `POST /api/synthesis` — Phase 2: LLM synthesis from cached evidence (can be called separately after `/api/match`)
- `POST /api/synthesis/stream` — Phase 2 SSE token stream keyed by the `evidenceId` returned from `/api/match`
- `GET /api/health` — primary/fallback synthesis-provider readiness without loading retrieval clients
- `GET /api/verify` — snippet by byte range from clean artifact
- `GET /api/stats` — latest benchmark run summary
- `GET /api/pdf/{doi_path}` — stream a raw PDF
- `GET /api/debug/heatmap` — sentence-level cosine similarity heatmap
- `GET /api/debug/subgraph/{entity}` — 1-hop Neo4j neighbourhood (D3 force graph)

The `Agent` is a lazy-init singleton inside the server (initialised on first request behind an `asyncio.Lock`). `/api/match` caches the 32 most recent evidence results; `/api/synthesis` reuses that evidence or retrieves on demand. Do not import or construct `Agent` at module load time.

**data-ingestion** exposes a FastAPI server (`src/api/server.py`) on `:8001`; `POST /api/ingest` streams pipeline progress and `/api/ingest/artifacts/*` serves intermediate outputs. Vite proxies `/api/ingest*` to `:8001` before forwarding other `/api*` requests to reasoning on `:8000`; proxy order in `palantir-blueprint/vite.config.ts` is significant.

---

## Architecture

### Configuration-First Design

Cross-module runtime policy is centralised in YAML. Python source supplies infrastructure and conservative defaults; do not duplicate configurable values in code.

- **`config/app.yaml`** — single non-secret source of truth: agent model/prompt/params, GraphRAG config, acquisition sources/storage, ingestion settings
- **`.env.local`** — ports, URLs, and secrets (gitignored; copy from `.env.local.example`)

`agentic-reasoning/src/config.py` validates its section with Pydantic v2. Acquisition and ingestion use section-specific dictionary loaders; root `tests/test_config_schema.py` enforces cross-module structural invariants. All loaders recursively expand `${VAR}` values after loading `.env.local`, without overwriting environment variables already set by the caller.

### Execution Model (agentic-reasoning)

**Deterministic two-phase pipeline — not dynamic agent routing:**

- **Phase 1 (mandatory):** `GraphRAGTool.cached_execute(query)` always runs before the LLM sees anything. No LLM routing decision.
- **Phase 2 (conditional):** If evidence was found, the LLM synthesises an answer grounded exclusively in that evidence. If `found=False`, a fixed "No evidence found" string is returned without invoking the LLM.

The `Agent` class (`src/agent.py`) is the sole reasoning entry point. It holds one `GraphRAGTool` instance. The strict system prompt in `config/app.yaml` prohibits parametric memory use when evidence is available.

### LLM Backends (agentic-reasoning)

`llm_factory.build_llm()` routes on the `provider/model-name` prefix in `config/app.yaml`:

| Prefix | Default URL | Override env var |
|---|---|---|
| `lmstudio/` | `http://localhost:1234/v1` | `LM_STUDIO_BASE_URL` |
| `ollama/` | `http://localhost:11434` | `LLM_BASE_URL` or `OLLAMA_BASE_URL` |
| `sglang/` | `http://localhost:30000/v1` | `SGLANG_BASE_URL` |

The checked-in policy uses `sglang/Qwen/Qwen2.5-7B-Instruct` as primary and `lmstudio/${LLM_MODEL}` as explicit fallback. Both providers share a 4,096-token synthesis envelope: `model_params.max_tokens` reserves output, `prompt_safety_margin_tokens` absorbs tokenizer variance, and `Agent` deterministically fits ranked evidence into the remaining prompt budget without mutating cached provenance. Synthesis health-checks providers before invocation and retries the fallback only before visible output; provider failures become typed HTTP/SSE errors. Run `make validate` to check the same primary/fallback path plus Qdrant and Neo4j.

### GraphRAGTool

`src/tools/graphrag.py` performs hybrid retrieval:
1. Select exactly one named target from `config.app.yaml`: `patient_context` for the active upload or `literature` when no upload is active.
2. Patient retrieval requires exact Qdrant `(scope, source)` filters and matching Neo4j `(scope, source_slug)` provenance; patient and literature evidence are never merged implicitly.
3. Reject bracketed filename mismatches before embedding, then encode with `BAAI/bge-small-en-v1.5` and reject hits below `min_relevance_score`.
4. Over-fetch and apply the configured CrossEncoder reranker before truncating to the request's bounded `top_k`.
5. Return vector/graph evidence plus retrieval context and a deterministic empty outcome. Successful `/api/match` calls receive an opaque evidence ID for context-safe synthesis.

All clients (Qdrant, Neo4j, embedder, reranker) are lazy-initialised on first use. `BaseTool.cached_execute()` uses canonical mapping keys so target/source/query/limit remain cache-isolated.

### Multi-Cloud Fallback (data-acquisition)

The CLI defaults to `--storage-mode local`, and the checked-in `data_acquisition.storage.providers.chain` contains only local storage at priority 1. AWS S3 and Azure Blob definitions remain available for an explicitly configured cloud chain; cloud mode requires matching `providers.routing`, chain entries, SDKs, and credentials.

`MultiCloudStorageManager` orders configured providers by priority, applies provider retry settings, tracks consecutive failures, skips degraded providers, and records the attempted fallback chain. Metadata sidecars use `{key_without_extension}.metadata.json`.

### Ingestion Pipeline (data-ingestion)

```
data/pdfs/raw/      →  [1] Surya OCR          →  data/artifacts/extract/
                    →  [2] SuryaConverter      →  data/artifacts/convert/
                    →  [3] TextCleaner + PII   →  data/artifacts/clean/
                    →  [4] MarkdownChunker     →  data/artifacts/chunk/
                    →  [5] MedicalVectorizer   →  Qdrant (localhost:6333)
                    →  [6] KG extraction LLM   →  Neo4j
```

`make ingestion-run` executes all six stages. The top-level `make ingest` runs stages 1-5 with `--skip-graph`, then invokes `ingestion-neo4j-build` explicitly. Stages 1-4 persist named outputs for inspection and content-addressed copies under `data/artifacts/{stage}/{hash[:2]}/`; `data/determinism.db` stores deterministic document IDs, random execution IDs, environment fingerprints, and SHA-256 stage hashes.

**PII redaction** (Stage 3): `TextCleaner` receives the full ingestion config. The checked-in policy enables Presidio for `EMAIL_ADDRESS` and Singapore NRIC/FIN (`[STFG]\d{7}[A-Z]`); recognisers, replacements, entities, and error behaviour all come from `data_ingestion.cleaning`. Be aware that `fail_safe_on_pii_error: true` logs critically and returns the original unredacted text.

**Chunking** (Stage 4): `MarkdownChunker` preserves header breadcrumbs and page boundaries, uses paragraph overlap only when splitting large sections, and avoids splitting atomic list/code blocks. Each persisted chunk carries `content`, `context`, `level`, `page_number`, `is_boilerplate`, `char_start`, and `char_end`; do not reintroduce legacy `parent_id`/`depth` fields without changing consumers and tests.

**Vector indexing** (Stage 5): `MedicalVectorizer` indexes only the current run's chunk artifacts, filters flagged boilerplate when configured, deletes existing points for the same `(source, scope)`, and uses deterministic UUIDv5 point IDs from collection/scope/source/chunk index.

**Embedding model**: `BAAI/bge-small-en-v1.5` (384 dimensions) must match exactly between ingestion and retrieval. Device selection is explicit in `config/app.yaml`, not auto-detected.

---

## Key Conventions

### Testing conventions

**Agent tests** (`agentic-reasoning/tests/`) mock both `GraphRAGTool` and the LLM — no Qdrant, Neo4j, or LLM backend is required to run them. The key invariant under test: Phase 1 (GraphRAG) is always called before the LLM, regardless of the outcome.

**Root cross-module tests** (`tests/`) use `agentic-reasoning`'s `.venv` and validate structural correctness:
- `tests/test_config_schema.py` — validates all four sections of `config/app.yaml` (required keys, provider chain ordering, temperature bounds, etc.). Any change to the config schema must pass this suite.
- The root `tests/conftest.py` exposes `app_config` and `repo_root` session fixtures.

**data-acquisition integration tests** are skipped automatically when cloud credentials are absent (`pytest.skip()`). Always run `make acquisition-test` (adds `-m "not integration"`) to avoid failures in local environments.

### Python Environments

- Each module has its own `.venv`; never share environments across modules.
- `agentic-reasoning` and `data-acquisition` use Python 3.12 editable installs (`pip install -e .`).
- `data-ingestion` uses a Python 3.11 `.venv` populated from `requirements.txt`; `core-llm-inference` uses Python 3.12.
- Never cross-import a sibling module's top-level `src` package. Agentic code uses package-relative imports; acquisition fetchers retain `try/except ImportError` shims where the same file supports package and direct-script execution.

### Adding a New Data Source (data-acquisition)

1. Add `data_acquisition.sources.{name}` to `config/app.yaml` following existing structure.
2. Register the fetcher class in `data-acquisition/scripts/fetch_pdfs.py` `_FETCHERS`.
3. Implement class inheriting `BaseFetcher`. Make `search()` `async` if it does HTTP calls — `fetch_pdfs.py` uses `inspect.iscoroutinefunction()` to route; do not call `asyncio.run()` inside a sync `search()` as the script already runs inside `asyncio.run()`.

### Adding a New Ingestion Processor

Inherit `BaseExtractor` from `src/extractors/base.py` (must return `{'content': str, 'metadata': dict}`). For clean/chunk stages, follow `TextCleaner`/`MarkdownChunker` pattern in `src/processors/`.

### Logging

The reasoning CLI uses `src/logging_handler.py` to write `agentic-reasoning/log/{execution_id}.json` plus `log/summary.jsonl`; server/runtime diagnostics use standard module loggers. Do not use `print()` for observability.

`data-ingestion` pipeline logs to `data/artifacts/ingestion.log`.

### Secrets and Env Vars

Copy `.env.local.example` → `.env.local` at repo root. Acquisition also reads `data-acquisition/.env` and `data-acquisition/.env.local` as progressively higher-priority local overrides; already-exported shell variables still win. Keep credentials out of YAML and source.

### Benchmarking

`benchmarking/` evaluators use the reasoning module's `.venv` (`BENCH_PYTHON` in Makefile). Golden query set: `benchmarking/golden/queries.json`. Results written to `benchmarking/results/run_{date}_{hash}/`. `make deterministic-run` is the canonical way to produce a reproducible, fully-annotated manifest.
