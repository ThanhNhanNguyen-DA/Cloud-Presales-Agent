# Cloud Presales Agent

**An AI Agent for cloud presales — it routes each question to a specialist tool, fact-checks every number before it reaches the customer, and learns from human corrections.**

Built with **LangGraph** (orchestration), **Groq / Llama 3.3 70B** (LLM), **Qwen3-Embedding** (local vectorisation), **Supabase / pgvector** (vector + structured data), **FastAPI** and **Streamlit**.

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
                       │  GROUNDING GRADER   │  numbers match?
                       └───┬────────────┬────┘               │
                    retry  │            │  pass / exhausted   ▼
              (≤1, back to │            └────────▶ END ◀──────┘
               supervisor) └──────────▶ supervisor
```

### Where the LLM is used

Called at **six** places in the request path (below), plus the RAGAS judge for the live scorecard — up to two extra calls per answer (context precision + faithfulness; answer relevancy uses only embeddings). Vectorisation is **not** LLM work — embeddings run locally on Qwen, so the Groq key only pays for reasoning.

| # | Task | File | Temp | Mechanism |
|---|------|------|:----:|-----------|
| 1 | Intent routing | [`graph/supervisor.py`](app/graph/supervisor.py) | 0 | Prompt → one word |
| 2 | Calculation & sizing | [`agents/calculator_agent.py`](app/agents/calculator_agent.py) | 0 | Function calling |
| 3 | Document lookup (RAG) | [`agents/document_agent.py`](app/agents/document_agent.py) | 0 | Function calling |
| 4 | Competitor mapping | [`agents/competitor_agent.py`](app/agents/competitor_agent.py) | 0 | Function calling |
| 5 | General answer | [`graph/builder.py`](app/graph/builder.py) | 0.3 | Direct, no tools |
| 6 | Grounding grader | [`graders/self_rag_grader.py`](app/graders/self_rag_grader.py) | 0 | Prompt → PASS/FAIL |

All calls go through one wrapper, [`app/llm.py`](app/llm.py): an ordered model chain (`llama-3.3-70b-versatile` → `llama-3.1-8b-instant`) that switches model the moment one is rate-limited, and — if all are exhausted — waits out the limit (honouring `retry-after`) and retries, so a free-tier rate limit rides out instead of crashing. Every prompt follows the same recipe: **role · context · task · output format · anti-hallucination constraints**.

### Data (Supabase / pgvector)

| Table | Role |
|-------|------|
| `cloud_documents` | RAG corpus: `content` + `embedding VECTOR(512)` + metadata |
| `cloud_competitor_mapping` | AWS/Azure/GCP → local service mapping + migration difficulty |
| `cloud_service_pricing` | VM / storage pricing reference rows |
| `cloud_feedback_memory` | Episodic memory: query/response + `is_positive` + `user_correction` + embedding |
| `cloud_conversation_history` | Multi-turn memory: one row per message (`session_id`, `role`, `content`, `route`) |

Text is vectorised by **Qwen3-Embedding-0.6B**, truncated via Matryoshka to **512 dims** and L2-normalised; two RPC functions do cosine search. Documents are chunked **structure-aware** (DOCX by heading + tables, PDF per case study, recursive split for the rest), not blind fixed-length cuts.

---

## Answer evaluation

Two live checks on every answer, plus an offline batch.

**Live 1 — Grounding Grader (gate).** After a specialist produces an answer and *before it is shown to the user*, a separate LLM call cross-checks the numbers in the answer (cost, time, vCPU, RAM, IOPS, storage) against the raw tool/RAG output. On a mismatch it retries once (back through the supervisor); if it still fails it delivers the answer with a clear ⚠️ warning to double-check — **never a silently wrong number**. Answers with no numbers (e.g. general chat) auto-pass. See [`app/graders/self_rag_grader.py`](app/graders/self_rag_grader.py).

**Live 2 — RAGAS scorecard.** Right *after* the answer is produced (so it never delays time-to-first-token), the three **reference-free** metrics — Context Precision, Faithfulness, Answer Relevancy — are scored and returned with the answer (`ragas` field on `/chat`, a `ragas` SSE event on `/chat/stream`) and shown as a scorecard under the reply in the UI.

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

**Prerequisites:** Python 3.12+, a free [Groq API key](https://console.groq.com/keys), a [Supabase project](https://app.supabase.com).

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY
```
> `.env`: `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B`, `EMBEDDING_DIM=512`. Optional `GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant`. The embedding model (~1.2 GB) caches locally on first use — no key needed for it.

```bash
# 2. Database — run sql/schema.sql then sql/seed_data.sql in the Supabase SQL Editor
#    (python -m app.db.run_migration prints the exact SQL). Creates 5 tables + 2 RPC + pgvector.

# 3. Load RAG data (pick any)
python -m scripts.ingest_documents                     # built-in sample docs (quick start)
python -m scripts.ingest_files ./docs --category General   # your PDF/DOCX/TXT/MD
python -m scripts.crawl_competitors --ingest           # crawl VN competitors → RAG store

# 4. Run (two terminals)
uvicorn app.main:app --reload --port 8000              # backend — docs at /docs
streamlit run ui/streamlit_app.py                      # chat UI
```

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
| `POST` | `/chat/stream` | `{"query", "session_id"?}` | **SSE**: `progress` → `ttft` → `token`× → `ragas` → `final` |
| `POST` | `/feedback` | `{"query","response","is_positive","correction"?}` | `{"message"}` |

Pass a stable **`session_id`** to keep a conversation in context (multi-turn memory: recent turns are loaded before answering and saved after, so follow-ups like *"and the monthly cost?"* resolve). Omit it for a stateless call. The FastAPI app is fully **async** (graph via `ainvoke` / threaded stream; blocking DB & embedding calls offloaded with `asyncio.to_thread`).

## Testing

```bash
pytest -q     # Groq + Supabase mocked; set any GROQ_API_KEY to allow imports
```
Covers the tools, each agent through the graph, the grounding grader (pass/fail/retry), the feedback loop, multi-turn memory, the async endpoints, the LLM rate-limit fallback/retry, the RAGAS metrics, and the chunkers.

## Project structure

```
app/
  config.py · llm.py           # env; Groq wrapper (fallback + retry)
  main.py                      # FastAPI: /chat, /chat/stream, /feedback, /health
  db/ · tools/ · agents/       # Supabase; calc/RAG/competitor/memory/conversation tools; 3 specialists
  graders/ · evaluation/       # grounding grader; RAGAS metric library
  graph/                       # state, supervisor (routing), builder
scripts/                       # ingest_documents/files, crawl_competitors, run_ragas_eval, chunkers
data/eval/ · sql/ · ui/ · tests/
```

## Extend it

Next steps: move from *answering* to *doing* (action-taking agents, e.g. draft & save a quote), and grow the labelled RAGAS ground-truth set with real presales queries.
