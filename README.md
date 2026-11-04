# Cloud Presales Agent

**An AI Agent for cloud presales — it routes each question to a specialist tool, fact-checks every number before it reaches the customer, and learns from human corrections.**

Built with **LangGraph** (orchestration), **Groq / GPT-OSS 120B** (LLM, Qwen3.8 27B fallback), **Qwen3-Embedding** (local vectorisation), **Supabase / pgvector** (vector + structured data), **FastAPI** and **Streamlit**.

---

## The problem

Presales engineers repeat the same technical answers — SLAs, security, VM sizing, backup cost, "what's your equivalent of AWS S3?" — much of it **numbers that go straight into a customer quote**. Generic chatbots make it worse: they hallucinate figures confidently, can't tell an SLA question from a pricing calc, and never learn. "Usually right" is not good enough.

## How it solves it

An AI agent behind one chat box:

- A **supervisor** classifies each question and routes it to the right specialist — cost/sizing, documentation lookup, or competitor comparison.
- Each specialist **must use real tools** (calculators, vector search, SQL); the model may not invent numbers.
- A **Grounding Grader** fact-checks every answer against the raw tool output *before it reaches the user* (see [Answer evaluation](#answer-evaluation)).
- A **self-learning loop** stores human corrections and applies them to similar future questions — no fine-tuning.

### Flow

```
                ┌──────────────────────────────────────────────────────┐
   question     │                     SUPERVISOR                        │
 ───────────────▶  1. retrieve_past_experiences()  (mandatory recall)   │
                │  2. classify intent → route                           │
                └───────┬───────────┬───────────┬───────────┬───────────┘
                        │           │           │           │
                  calculator     document    competitor   general
                        │           │           │           │
                        ▼           ▼           ▼           │
                 ┌─────────────────────────────────┐        │
                 │ specialist (tool calling)        │        │
                 │ appends raw_tool_output to state │        │
                 └───────────────┬─────────────────┘        │
                                 ▼                           │
                       ┌────────────────────┐               │
                       │  GROUNDING GRADER   │  facts match tool data?
                       └───┬────────────┬────┘               │
                    retry  │            │  pass / exhausted   ▼
             (≤1, SAME     │            └────────▶ END ◀──────┘
              specialist,  └──────────▶ calculator | document | competitor
              + feedback)
```

### Where the LLM is used

Called at **six** places in the request path (below), plus the RAGAS judge for the live scorecard — one extra call per answer (faithfulness; answer relevancy uses only embeddings). Vectorisation is **not** LLM work — embeddings run locally on Qwen, so the Groq key only pays for reasoning.

| # | Task | File | Temp | Mechanism |
|---|------|------|:----:|-----------|
| 1 | Intent routing | [`graph/supervisor.py`](app/graph/supervisor.py) | 0 | Prompt → one word |
| 2 | Calculation & sizing | [`agents/calculator_agent.py`](app/agents/calculator_agent.py) | 0 | Function calling |
| 3 | Document lookup (RAG) | [`agents/document_agent.py`](app/agents/document_agent.py) | 0 | Function calling |
| 4 | Competitor mapping | [`agents/competitor_agent.py`](app/agents/competitor_agent.py) | 0 | Function calling |
| 5 | General answer | [`graph/builder.py`](app/graph/builder.py) | 0.3 | Direct, no tools |
| 6 | Grounding grader | [`graders/self_rag_grader.py`](app/graders/self_rag_grader.py) | 0 | Prompt → PASS/FAIL |

All calls go through one wrapper, [`app/llm.py`](app/llm.py): an ordered model chain (`openai/gpt-oss-120b` → `qwen/qwen3.8-27b`) that switches model the moment one is rate-limited, and — if all are exhausted — waits out the limit (honouring `retry-after`) and retries, so a free-tier rate limit rides out instead of crashing. Every prompt follows the same recipe: **role · context · task · output format · anti-hallucination constraints**.

### Data (Supabase / pgvector)

| Table | Role |
|-------|------|
| `cloud_documents` | RAG corpus: `content` + `embedding VECTOR(512)` + metadata |
| `cloud_competitor_mapping` | AWS/Azure/GCP → local service mapping + migration difficulty |
| `cloud_service_pricing` | VM / storage pricing reference rows |
| `cloud_feedback_memory` | Episodic memory: query/response + `is_positive` + `user_correction` + embedding |
| `cloud_conversation_history` | Multi-turn memory: one row per message (`session_id`, `role`, `content`, `route`) |

Text is vectorised by **Qwen3-Embedding-0.6B**, truncated via Matryoshka to **512 dims** and L2-normalised; two RPC functions do cosine search (HNSW index). Documents are chunked **structure-aware** (DOCX by heading + tables, PDF per case study, recursive split for the rest), not blind fixed-length cuts.

### Retrieval pipeline

`question → embed_query (Qwen "query" prompt) → vector search, top 15 → cross-encoder rerank (BAAI/bge-reranker-v2-m3, fp16) → keep score ≥ 0.1 → top 4 with sources`

The LLM only chooses the query and an optional category/provider; counts and thresholds are fixed.
`search_cloud_documents` searches **our** docs and excludes crawled competitor pages;
`search_competitor_docs` (competitor agent) searches only those pages, per provider.
Measured with `scripts/eval_retrieval.py` on 60 generated questions (`scripts/build_retrieval_eval.py`):

| Retrieval | Hit@1 | Hit@3 | MRR@10 | ms/query |
|---|---|---|---|---|
| RPC through the old ivfflat index | 0.10 | 0.17 | 0.13 | 177 |
| exact vector search | 0.68 | 0.90 | 0.81 | 44 |
| + Qwen query prompt | 0.72 | 0.93 | 0.83 | 11 |
| + bge-reranker-v2-m3 (top 15, fp16) | **0.87** | 0.93 | **0.91** | ~180 |
| + mmarco-mMiniLMv2 (top 15, fp16) | 0.83 | **0.98** | 0.90 | ~45 |

`sql/migrations/003_retrieval.sql` replaces the ivfflat indexes with HNSW (the 0.17 row above). Until it is
run, the tools fall back to the old RPC signature and filter in Python.

---

## Answer evaluation

Two live checks on every answer, plus an offline batch.

**Live 1 — Grounding Grader (gate).** After a specialist produces an answer and *before it is shown to the user*, a separate LLM call cross-checks the answer against the raw tool/RAG output (each tool call is recorded with its arguments and result, and the grader also sees the user's question): numbers (cost, time, vCPU, RAM, IOPS, storage, SLA %), service names, certifications and SLA terms. It replies with JSON `{"verdict", "issues"}`. On a FAIL the **same specialist** runs once more with the issues in its prompt (routing and memory recall are not repeated, and the retry is graded against its own tool calls only); if it still fails, the answer is delivered with a clear ⚠️ warning — **never a silently wrong number**. If the grader itself can't run or answers unparseably, the answer gets a "could not verify" warning instead of looping. Answers with no tool data (general chat, clarifying questions) auto-pass. See [`app/graders/self_rag_grader.py`](app/graders/self_rag_grader.py).

**Live 2 — RAGAS scorecard.** Right *after* the answer is produced (so it never delays time-to-first-token), the two lightest **reference-free** metrics — Faithfulness (one judge call) and Answer Relevancy (local embeddings) — are scored (Context Precision stays in the offline batch) and returned with the answer (`ragas` field on `/chat`, a `ragas` SSE event on `/chat/stream`) and shown as a scorecard under the reply in the UI.

**Offline — RAGAS batch.** All six metrics in three groups, computed with the project's own Groq judge + local embeddings (no external `ragas` package):

| Group | Metrics |
|-------|---------|
| **Retrieval** (retriever / vector DB) | Context Precision · Context Recall |
| **Generation** (the LLM response) | Faithfulness · Answer Relevancy |
| **End-to-End** (vs golden answer) | Answer Correctness · Answer Similarity |

Defined in [`app/evaluation/metrics.py`](app/evaluation/metrics.py). Run on the labelled set:
```bash
python -m scripts.run_ragas_eval        # uses data/eval/ground_truth.json → data/eval/latest_ragas.json
```

---

## Get started

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), a free [Groq API key](https://console.groq.com/keys), a [Supabase project](https://app.supabase.com).

```bash
# 1. Install
uv venv --python python3.12 .venv                    # reuse an existing uv .venv if present
source .venv/bin/activate                           # Windows: .venv\Scripts\activate
uv pip sync requirements.lock                       # exact tested versions (app + tests)
# to upgrade: uv pip compile requirements-dev.txt -o requirements.lock --upgrade
test -f .env || cp .env.example .env                 # fill in GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY
```
> `.env`: `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B`, `EMBEDDING_DIM=512`. Optional `GROQ_MODELS=openai/gpt-oss-120b,qwen/qwen3.8-27b`. The embedding model (~1.2 GB) caches locally on first use — no key needed for it.

```bash
# 2. Database — run sql/schema.sql, sql/seed_data.sql, then sql/migrations/*.sql in order,
#    in the Supabase SQL Editor (python -m app.db.run_migration prints the exact SQL).
#    Creates 5 tables + 2 RPC + pgvector, the feedback review columns, and Row Level Security.
#    Use a secret / service_role key as SUPABASE_KEY (RLS blocks the anon key).

# 3. Load RAG data (pick any)
python -m scripts.ingest_documents                     # built-in sample docs (quick start)
python -m scripts.ingest_files ./docs --category General   # your PDF/DOCX/TXT/MD
python -m scripts.crawl_competitors                    # crawl VN competitors' product/pricing pages → data/competitors/ (review)
python -m scripts.crawl_competitors --from-files --ingest   # → cloud_documents, category "Competitor" (replaces the previous crawl)

# 4. Run both services (Ctrl+C stops both)
bash scripts/dev.sh

# Or run separately in two activated terminals:
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
python -m streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open the chat UI at **http://localhost:8501**, API docs at **http://localhost:8000/docs**,
and the liveness check at **http://localhost:8000/health**. `scripts/dev.sh` uses
`.venv/bin/python` directly, so Conda and shell activation are not required.

Quick check:
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "How long does a 10GB backup over 100Mbps take, and the monthly Standard cost?"}'
```

---

## API

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET`  | `/health` | — | `{"status":"ok"}` |
| `POST` | `/chat` | `{"query", "session_id"?}` | `{"route","response","ragas"}` |
| `POST` | `/chat/stream` | `{"query", "session_id"?}` | **SSE**: `progress` → `ttft` → `token`× (`reset` when a new answer message starts, e.g. a grader retry) → `ragas` → `final`; `error` if the graph fails |
| `POST` | `/feedback` | `{"query","response","is_positive","correction"?}` | `{"message"}` |
| `GET`  | `/admin/feedback?status=pending` | — (admin key) | `{"items":[…]}` |
| `POST` | `/admin/feedback/{id}/review` | `{"decision":"approve"\|"reject"}` (admin key) | `{"id","review_status"}` |

### Security

- Every endpoint except `/health` needs an **`X-API-Key`** header matching one of `API_KEYS`
  (`label:key`, comma-separated); `/admin/*` needs `ADMIN_API_KEY`. The server refuses to start
  without `API_KEYS` (set `AUTH_DISABLED=true` only for throwaway local runs).
- **Rate limits** per key: `RATE_LIMIT_PER_MINUTE` (chat, default 20) and
  `FEEDBACK_RATE_LIMIT_PER_MINUTE` (default 30); over the limit → `429` with `Retry-After`.
  In-memory, per process.
- **Input limits:** `query` ≤ 2000 chars, `correction` ≤ 2000, `session_id` = 1–100 of `[A-Za-z0-9_.:-]`.
  Conversations are namespaced per key, so one client can't read another's history.
- **Human corrections are reviewed before use.** A downvote correction is stored as `pending`
  (with the submitter's key label) and only reaches prompts after an admin approves it; it is
  then injected inside a delimited `<reviewed_corrections>` block as reference data, never as
  instructions. `FEEDBACK_AUTO_APPROVE=true` skips the review for demos.
- CORS is limited to `CORS_ORIGINS`. All tables have RLS enabled with no policies.

Pass a stable **`session_id`** to keep a conversation in context (multi-turn memory: recent turns are loaded before answering and saved after, so follow-ups like *"and the monthly cost?"* resolve). Omit it for a stateless call. The FastAPI app is fully **async** (graph via `ainvoke` / threaded stream; blocking DB & embedding calls offloaded with `asyncio.to_thread`).

## Testing

```bash
python -m pytest -q     # ~4 s; no .env needed — Groq/Supabase/embeddings are mocked
```
`tests/conftest.py` sets dummy credentials and blocks the real Supabase client and
embedding model, so a test that forgets a mock fails instead of writing to your project.
Covers the tools, each agent through the graph, the grounding grader (pass/fail/retry), the feedback loop, multi-turn memory, the async endpoints, the LLM rate-limit fallback/retry, the RAGAS metrics, and the chunkers.

## Logging

Every log line carries the request id (`[req=…]`); send `X-Request-ID` to set it, otherwise one
is generated and returned in the `X-Request-ID` response header. Each LLM call logs model,
latency and tokens, the supervisor logs the route, the grader its verdict, and each answered
question ends with one summary line:

```
INFO [req=demo-001] app.main: chat done route=calculator latency_s=9.555 llm_calls=5 llm_s=7.085 tokens_in=3337 tokens_out=1442 graded=True retries=0
```

`LOG_LEVEL` (default `INFO`) sets the verbosity. Graph nodes return partial state updates
(only the keys they own), and the graph is consumed with `ainvoke` / `astream`; the nodes
themselves are sync and run in LangGraph's worker threads.

## Project structure

```
app/
  config.py · llm.py           # env; Groq wrapper (model fallback, rate-limit wait, bad tool-call retry)
  main.py                      # FastAPI: /chat, /chat/stream, /feedback, /admin/feedback, /health
  security.py                  # API keys, admin key, rate limiting
  observability.py             # logging with request ids, per-request LLM metrics
  db/ · tools/                 # Supabase (lazy client); calc/RAG/competitor/memory/conversation tools
  agents/                      # tool_agent.py (shared tool loop) + 3 thin specialists (prompt + tools)
  graders/ · evaluation/       # grounding grader; RAGAS metric library
  graph/                       # state, supervisor (routing), builder
scripts/                       # ingest_documents/files, crawl_competitors, run_ragas_eval, chunkers
data/eval/ · sql/ · ui/ · tests/
```

## Extend it

Next steps: move from *answering* to *doing* (action-taking agents, e.g. draft & save a quote), and grow the labelled RAGAS ground-truth set with real presales queries.
