# ☁️ Cloud Presales Agent

**A multi-agent Agentic RAG assistant for cloud presales — it routes each question to a specialist, fact-checks every number before it reaches the customer, and learns from human corrections.**

Built with **LangGraph** (orchestration), **Groq / Llama 3.3 70B** (LLM), **Qwen3-Embedding** running locally (vectorisation), **Supabase / pgvector** (vector store + structured data), **FastAPI** (API) and **Streamlit** (UI, with a built-in RAGAS evaluation page).

---

## 😩 The Pain

Presales engineers answer the same technical questions over and over — SLAs, security posture, VM sizing, backup cost, "what's your equivalent of AWS S3?" — and each answer requires digging through multiple documents. Worse, much of it is **numbers**: a backup-duration estimate, a VM spec, a monthly cost. **A single wrong number goes straight into a customer quote.**

Off-the-shelf chatbots make this worse, not better. They **hallucinate figures** with total confidence, they can't tell an SLA question from a pricing calculation, and they never get better — the same mistake resurfaces next week. For presales, "usually right" is not good enough.

## ✅ The Solution

**Cloud Presales Agent** is a small "team of specialists" behind one chat box:

- A **supervisor** classifies every question and routes it to the right specialist — cost/sizing calculation, documentation lookup, or competitor comparison.
- Each specialist **must use real tools** (calculators, vector search, SQL) — the model is forbidden from inventing numbers.
- A **decoupled Grounding Grader** cross-checks the numbers in every answer against the raw tool output *before* the answer is shown, retrying or flagging any mismatch.
- A **self-learning loop** remembers human corrections and applies them to similar future questions — **no fine-tuning required**.

The result: answers that are routed correctly, grounded in real data, and that improve over time.

## 🔬 Proof

Everything below is implemented and in the repository:

| Claim | Evidence |
|-------|----------|
| Multi-agent routing to 4 specialists | [`app/graph/supervisor.py`](app/graph/supervisor.py), [`app/graph/builder.py`](app/graph/builder.py) |
| Numbers are never invented — tools do the math | [`app/tools/presales_tools.py`](app/tools/presales_tools.py) (backup/cost, VM sizing, IOPS) |
| Hallucinated figures are caught before the customer sees them | [`app/graders/self_rag_grader.py`](app/graders/self_rag_grader.py) |
| The system self-corrects from 👍/👎 feedback | [`app/tools/memory_tools.py`](app/tools/memory_tools.py), [`ui/streamlit_app.py`](ui/streamlit_app.py) |
| Real documents are chunked by structure, not blindly | [`scripts/docx_chunker.py`](scripts/docx_chunker.py) (headings + tables), [`scripts/pdf_casestudy_chunker.py`](scripts/pdf_casestudy_chunker.py) (per case study) |
| Competitor data is crawled from real sources, never fabricated | [`scripts/crawl_competitors.py`](scripts/crawl_competitors.py) |
| Quality is measured against an international standard (RAGAS) | Evaluation page on the sidebar: [`ui/pages/1_📊_Evaluation.py`](ui/pages/1_📊_Evaluation.py) |
| The service stays up when a model hits its rate limit | [`app/llm.py`](app/llm.py) — automatic model fallback |
| It runs without live credentials for testing | [`tests/`](tests/) — Groq/Supabase mocked, `pytest -q` |

### How it works

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
                 │ specialist agent (tool calling)  │        │
                 │ appends raw_tool_output to state │        │
                 └───────────────┬─────────────────┘        │
                                 ▼                           │
                       ┌────────────────────┐               │
                       │  GROUNDING GRADER   │               │
                       │  numbers match?     │               │
                       └───┬────────────┬────┘               │
                    retry  │            │  pass / exhausted   │
              (≤1, back to │            ▼                     ▼
               supervisor) │           END ◀──────────────────┘
                           └──────────▶ supervisor
```

### Where the LLM is used

The LLM (Groq / Llama 3.3 70B) is called at exactly **six** places. Vectorisation is **not** one of them — embeddings run locally on Qwen, so the Groq key only pays for reasoning.

| # | Task | File | Temperature | Mechanism |
|---|------|------|:-----------:|-----------|
| 1 | Intent routing | [`graph/supervisor.py`](app/graph/supervisor.py) | 0 | Prompt → one word |
| 2 | Calculation & sizing | [`agents/calculator_agent.py`](app/agents/calculator_agent.py) | 0 | Function calling + synthesis |
| 3 | Document lookup (RAG) | [`agents/document_agent.py`](app/agents/document_agent.py) | 0 | Function calling + synthesis |
| 4 | Competitor mapping | [`agents/competitor_agent.py`](app/agents/competitor_agent.py) | 0 | Function calling + synthesis |
| 5 | General answer | [`graph/builder.py`](app/graph/builder.py) | 0.3 | Direct, no tools |
| 6 | Grounding grader | [`graders/self_rag_grader.py`](app/graders/self_rag_grader.py) | 0 | Prompt → PASS/FAIL |

All six go through one wrapper, [`app/llm.py`](app/llm.py), which holds an ordered model chain (`llama-3.3-70b-versatile` → `llama-3.1-8b-instant`) and **switches model instantly when one is rate-limited** (the client's own slow retry loop is disabled).

### The data

| Table | Role |
|-------|------|
| `cloud_documents` | RAG corpus: `content` + `embedding VECTOR(512)` + `service_name`, `category`, `metadata` |
| `cloud_competitor_mapping` | AWS/Azure/GCP → local service mapping + migration difficulty |
| `cloud_service_pricing` | VM / storage pricing reference rows |
| `cloud_feedback_memory` | Episodic memory: query/response + `is_positive` + `user_correction` + `query_embedding` |
| `cloud_conversation_history` | Multi-turn chat memory: one row per message (`session_id`, `role`, `content`, `route`) so follow-up questions keep context |
| `cloud_interaction_logs` | Per-answer telemetry (written in the background): `route`, `grading_passed`, `retry_count`, `faithfulness`, `answer_relevancy`, `ttft`/`total` — feeds the monitoring dashboard |

Two RPC functions (`match_cloud_documents`, `match_past_experience`) do cosine-similarity search. Text is vectorised by **Qwen3-Embedding-0.6B**, truncated via Matryoshka to **512 dimensions** and L2-normalised. Documents are chunked by **three structure-aware strategies** (DOCX by heading with tables preserved; PDF per customer case study; recursive character splitting for the rest) rather than blind fixed-length cuts.

---

## 🚀 Get started

### Prerequisites
- **Python 3.10+**
- A **Groq API key** — free at https://console.groq.com/keys
- A **Supabase project** (free tier is fine) — https://app.supabase.com

### 1. Install
```bash
cd cloud-presales-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create your environment file and fill in real values:
```bash
cp .env.example .env
```
```env
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIM=512
# Optional — ordered fallback chain (primary first)
# GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant
```
> `SUPABASE_URL` must be the bare project URL (no `/rest/v1`). The embedding model (~1.2 GB) downloads and caches locally on first use. No provider key is needed for embeddings.

### 2. Create the database
Run [`sql/schema.sql`](sql/schema.sql) then [`sql/seed_data.sql`](sql/seed_data.sql) in the **Supabase SQL Editor** (the helper `python -m app.db.run_migration` prints the exact steps and SQL). This creates the 6 tables, 2 RPC functions, pgvector, and indexes.

### 3. Load data (pick any)
```bash
# a) Built-in sample documents — quick start
python -m scripts.ingest_documents

# b) Your own PDF / DOCX / TXT / MD (structure-aware chunking, auto-selected)
python -m scripts.ingest_files ./docs --category General

# c) Crawl Vietnamese cloud competitors → files for review (data/competitors/)
python -m scripts.crawl_competitors
#    …and also ingest the crawled text into the RAG store:
python -m scripts.crawl_competitors --ingest
```
> The crawler stores only what the pages actually return (source URL + timestamp + verbatim text/tables); fields it cannot parse are left empty — it never fabricates figures. It honours `robots.txt` and rate-limits itself.

### 4. Run (two terminals)
```bash
# Terminal 1 — backend
uvicorn app.main:app --reload --port 8000        # docs at http://localhost:8000/docs

# Terminal 2 — UI (chat + Evaluation page in the sidebar)
streamlit run ui/streamlit_app.py
```
Quick check:
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "How long does a 10GB backup over 100Mbps take, and the monthly Standard cost?"}'
```

---

## API reference

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET`  | `/health` | — | `{"status":"ok"}` |
| `POST` | `/chat` | `{"query":"...", "session_id"?:"..."}` | `{"route":"...","response":"..."}` |
| `POST` | `/chat/stream` | `{"query":"...", "session_id"?:"..."}` | **SSE**: `progress` per node → `ttft` → `token` per token → `final` |
| `POST` | `/feedback` | `{"query","response","is_positive","correction"?}` | `{"message":"..."}` |

> **Multi-turn memory:** pass a stable `session_id` on `/chat` and `/chat/stream` to keep a
> conversation in context — the backend loads the recent turns (sliding window) from the
> `cloud_conversation_history` table before answering and saves the new turn after, so
> follow-ups like *"and the monthly cost?"* or *"what about GCP?"* resolve correctly. Omit
> `session_id` for a stateless, single-shot call. The Streamlit UI generates one per session
> and starts a fresh one when you click **Clear chat history**.

## Evaluation & monitoring (automated)

Evaluation is **automated and decoupled from the answer**. After every response is sent, a
**background task** scores it (live RAGAS: faithfulness + answer relevancy) and appends a row
to `cloud_interaction_logs` — so scoring adds **zero latency** to the chat (the API is async;
see below). The Streamlit **📊 Evaluation** page ([`ui/pages/1_📊_Evaluation.py`](ui/pages/1_📊_Evaluation.py))
reads that table and shows a live monitoring dashboard: routing mix, grounding-grader
catch/retry rate (the hallucination signal), RAGAS trend over time, latency, and recent
interactions.

For deeper, ground-truth-based numbers, run the **offline batch evaluator** on a labelled set:
```bash
python -m scripts.run_ragas_eval            # uses data/eval/ground_truth.json
```
It runs the full graph per question and scores faithfulness, answer relevancy, answer
correctness, and context recall (project's Groq judge + local Qwen embeddings — no external
`ragas` package), prints a scorecard, and writes `data/eval/latest_ragas.json`, which the
dashboard displays.

### Async backend

The FastAPI app is fully `async`: the graph runs via `ainvoke`/a threaded `stream` so the
synchronous LLM, Supabase, and embedding calls never block the event loop; conversation
save/feedback are offloaded with `asyncio.to_thread`; and evaluation + logging run in a
background task after the answer is sent.

## Testing
```bash
pytest -q     # Groq + Supabase are mocked; set any GROQ_API_KEY to allow imports
```
Coverage: the calculation tools, each agent through the compiled graph, the grounding grader (pass/fail/retry), the end-to-end feedback loop, multi-turn conversation memory, the async FastAPI endpoints (with background evaluation), the background evaluation/logging path, and the DOCX/PDF chunkers.

## Project structure
```
cloud-presales-agent/
├── app/
│   ├── config.py                  # env (GROQ_API_KEY, GROQ_MODELS, SUPABASE_*, EMBEDDING_*)
│   ├── llm.py                     # Groq wrapper with automatic model fallback
│   ├── main.py                    # FastAPI: /chat, /chat/stream, /feedback, /health
│   ├── db/                        # Supabase client + migration helper
│   ├── tools/                     # calculation, RAG, competitor, memory, conversation, embedding
│   ├── agents/                    # calculator, document, competitor agents
│   ├── graders/                   # grounding (hallucination) grader
│   ├── evaluation/                # ragas_live (async metrics), interaction_log (bg write), log_store (read)
│   └── graph/                     # state, supervisor (routing), builder (graph)
├── scripts/
│   ├── ingest_documents.py        # built-in sample docs
│   ├── ingest_files.py            # your PDF/DOCX/TXT/MD (structure-aware)
│   ├── crawl_competitors.py       # crawl VN cloud competitors → files + optional RAG
│   ├── run_ragas_eval.py          # offline batch RAGAS on a labelled ground-truth set
│   ├── docx_chunker.py · pdf_casestudy_chunker.py · text_chunking.py
├── data/eval/                     # ground_truth.json (labelled set) + latest_ragas.json (batch output)
├── sql/                           # schema.sql (6 tables + 2 RPC) + seed_data.sql
├── ui/
│   ├── streamlit_app.py           # chat UI with 👍/👎 feedback
│   └── pages/1_📊_Evaluation.py   # RAGAS evaluation page (sidebar)
├── tests/                         # pytest suite (mocked)
└── requirements.txt · .env.example
```

## Extend it
Multi-turn conversation memory is now in place (see the note above). The natural next step is to move from *answering* to *doing*: add action-taking agents (e.g. draft and save a quote) on top of it, and compute live RAGAS scores from a labelled set of real presales queries. Contributions and issues welcome.
