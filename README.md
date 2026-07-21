# ☁️ Cloud Presales Agent

**A multi-agent Agentic RAG assistant for cloud presales — it routes each question to a specialist, fact-checks every number before it reaches the customer, and learns from human corrections.**

Built with **LangGraph** (orchestration), **Groq / Llama 3.3 70B** (LLM), **Qwen3-Embedding** running locally (vectorisation), **Supabase / pgvector** (vector store + structured data), **FastAPI** (API) and **Streamlit** (chat UI), plus an offline RAGAS evaluation script.

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
| Quality is measured against an international standard (RAGAS) | Offline batch evaluator: [`scripts/run_ragas_eval.py`](scripts/run_ragas_eval.py) + [`app/evaluation/metrics.py`](app/evaluation/metrics.py) |
| The service stays up when a model hits its rate limit | [`app/llm.py`](app/llm.py) — model fallback + wait-and-retry |
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

The LLM (Groq / Llama 3.3 70B) is called at exactly **six** places in the request path (a seventh, the RAGAS judge, runs only in the offline evaluator). Vectorisation is **not** one of them — embeddings run locally on Qwen, so the Groq key only pays for reasoning.

| # | Task | File | Temperature | Mechanism |
|---|------|------|:-----------:|-----------|
| 1 | Intent routing | [`graph/supervisor.py`](app/graph/supervisor.py) | 0 | Prompt → one word |
| 2 | Calculation & sizing | [`agents/calculator_agent.py`](app/agents/calculator_agent.py) | 0 | Function calling + synthesis |
| 3 | Document lookup (RAG) | [`agents/document_agent.py`](app/agents/document_agent.py) | 0 | Function calling + synthesis |
| 4 | Competitor mapping | [`agents/competitor_agent.py`](app/agents/competitor_agent.py) | 0 | Function calling + synthesis |
| 5 | General answer | [`graph/builder.py`](app/graph/builder.py) | 0.3 | Direct, no tools |
| 6 | Grounding grader | [`graders/self_rag_grader.py`](app/graders/self_rag_grader.py) | 0 | Prompt → PASS/FAIL |

All these calls go through one wrapper, [`app/llm.py`](app/llm.py), which holds an ordered model chain (`llama-3.3-70b-versatile` → `llama-3.1-8b-instant`): it **switches to the next model the moment one is rate-limited**, and if every model is exhausted it **waits out the limit** (honouring the server's `retry-after`) and retries — so a sustained free-tier rate limit rides out instead of crashing.

Every prompt in the system follows the same recipe — **role · context · task · output format · anti-hallucination constraints** — so outputs are consistent and easy to parse (e.g. the router returns one word, the grader returns `PASS`/`FAIL`, the specialists return a structured answer).

### The data

| Table | Role |
|-------|------|
| `cloud_documents` | RAG corpus: `content` + `embedding VECTOR(512)` + `service_name`, `category`, `metadata` |
| `cloud_competitor_mapping` | AWS/Azure/GCP → local service mapping + migration difficulty |
| `cloud_service_pricing` | VM / storage pricing reference rows |
| `cloud_feedback_memory` | Episodic memory: query/response + `is_positive` + `user_correction` + `query_embedding` |
| `cloud_conversation_history` | Multi-turn chat memory: one row per message (`session_id`, `role`, `content`, `route`) so follow-up questions keep context |

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
Run [`sql/schema.sql`](sql/schema.sql) then [`sql/seed_data.sql`](sql/seed_data.sql) in the **Supabase SQL Editor** (the helper `python -m app.db.run_migration` prints the exact steps and SQL). This creates the 5 tables, 2 RPC functions, pgvector, and indexes.

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

# Terminal 2 — chat UI
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

## Evaluation (RAGAS)

Quality is measured with **six RAGAS-style metrics in three groups** — computed with the
project's own Groq judge + local Qwen embeddings (no external `ragas` package):

| Group | Metric | What it measures | Needs ground truth? |
|-------|--------|------------------|:---:|
| **Retrieval** (the retriever / vector DB) | Context Precision | Signal-to-noise of the retrieved context — are the relevant chunks the ones that surfaced? | No |
| | Context Recall | Did retrieval fetch **everything** the correct answer needs? | Yes |
| **Generation** (the LLM response) | Faithfulness | Is the answer strictly grounded in the context (hallucination check)? | No |
| | Answer Relevancy | How directly does the answer address the question? | No |
| **End-to-End** (vs a golden answer) | Answer Correctness | Factual accuracy of the answer vs the ground truth. | Yes |
| | Answer Similarity | Semantic resemblance (embedding cosine) to the ground truth. | Yes |

All six are defined in one place: [`app/evaluation/metrics.py`](app/evaluation/metrics.py).

Evaluation is an **offline batch job** (it is not part of the request path — the chat API
just answers). Run it on the labelled ground-truth set:
```bash
python -m scripts.run_ragas_eval            # uses data/eval/ground_truth.json
```
It runs the full graph per question, scores all six metrics grouped by category, prints a
scorecard, and writes `data/eval/latest_ragas.json`. Edit `data/eval/ground_truth.json` to
change the labelled questions.

## Async backend

The FastAPI app is fully `async`: the graph runs via `ainvoke`/a threaded `stream` so the
synchronous LLM, Supabase, and embedding calls never block the event loop; conversation
save and feedback writes are offloaded with `asyncio.to_thread`.

## Testing
```bash
pytest -q     # Groq + Supabase are mocked; set any GROQ_API_KEY to allow imports
```
Coverage: the calculation tools, each agent through the compiled graph, the grounding grader (pass/fail/retry), the end-to-end feedback loop, multi-turn conversation memory, the async FastAPI endpoints, the LLM rate-limit fallback/retry, the RAGAS metric library, and the DOCX/PDF chunkers.

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
│   ├── evaluation/                # metrics (the 6 RAGAS-style metrics, for the batch evaluator)
│   └── graph/                     # state, supervisor (routing), builder (graph)
├── scripts/
│   ├── ingest_documents.py        # built-in sample docs
│   ├── ingest_files.py            # your PDF/DOCX/TXT/MD (structure-aware)
│   ├── crawl_competitors.py       # crawl VN cloud competitors → files + optional RAG
│   ├── run_ragas_eval.py          # offline batch RAGAS on a labelled ground-truth set
│   ├── docx_chunker.py · pdf_casestudy_chunker.py · text_chunking.py
├── data/eval/                     # ground_truth.json (labelled set) + latest_ragas.json (batch output)
├── sql/                           # schema.sql (5 tables + 2 RPC) + seed_data.sql
├── ui/
│   └── streamlit_app.py           # chat UI with 👍/👎 feedback
├── tests/                         # pytest suite (mocked)
└── requirements.txt · .env.example
```

## Extend it
Multi-turn conversation memory is now in place (see the note above). The natural next step is to move from *answering* to *doing*: add action-taking agents (e.g. draft and save a quote) on top of it, and grow the labelled RAGAS ground-truth set with real presales queries. Contributions and issues welcome.
