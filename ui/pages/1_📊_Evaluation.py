"""
Monitoring dashboard (sidebar) — live system quality from the interaction logs.

Every answer the backend produces is scored (live RAGAS) and logged asynchronously to
`cloud_interaction_logs`. This page reads that table and turns it into an operational
view: how questions are routed, how often the grounding grader catches a problem
(hallucination / retry rate), and how faithfulness & answer relevancy trend over time.

It also shows the latest offline batch-RAGAS run (data/eval/latest_ragas.json) if the
`scripts/run_ragas_eval.py` job has produced one.

Reads Supabase directly via a lightweight helper (no embedding model is loaded here).
"""

import json
import pathlib
import sys

import pandas as pd
import streamlit as st

# Make the repo root importable when Streamlit runs this page from the ui/ directory.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Evaluation · Monitoring", page_icon="📊", layout="wide")

st.title("📊 System Monitoring — live quality")
st.caption(
    "Read from cloud_interaction_logs: routing mix, grounding-grader catch/retry rate, "
    "and live RAGAS (faithfulness, answer relevancy) over time. Metrics are scored in the "
    "background after each answer, so they never slow the chat."
)


@st.cache_data(ttl=30)
def _fetch_logs(limit: int = 1000):
    """Load recent interaction logs (cached briefly so refreshes are cheap)."""
    try:
        from app.evaluation.log_store import load_recent_logs
        return load_recent_logs(limit), None
    except Exception as e:  # noqa: BLE001 - surface a friendly hint instead of a traceback
        return [], str(e)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


colf, colr = st.columns([3, 1])
with colr:
    if st.button("🔄 Refresh"):
        _fetch_logs.clear()
        st.rerun()

rows, err = _fetch_logs()

if err:
    st.error(
        "Could not read the interaction logs.\n\n"
        f"`{err}`\n\n"
        "Make sure the `cloud_interaction_logs` table exists (re-run the new part of "
        "`sql/schema.sql`) and that Supabase credentials are set in `.env`."
    )
    st.stop()

if not rows:
    st.info(
        "No interactions logged yet. Ask a few questions on the main chat page — each "
        "answer is scored and logged in the background, then appears here. If you have "
        "been chatting and still see nothing, confirm the `cloud_interaction_logs` table "
        "was created from `sql/schema.sql`."
    )
    st.stop()

df = pd.DataFrame(rows)
df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
df = df.sort_values("created_at")
for col in ["faithfulness", "answer_relevancy", "ttft_seconds", "total_seconds", "retry_count"]:
    if col in df:
        df[col] = _num(df[col])

# --- Scorecard -------------------------------------------------------------------------
total = len(df)
faiths = df["faithfulness"].dropna() if "faithfulness" in df else pd.Series(dtype=float)
rels = df["answer_relevancy"].dropna() if "answer_relevancy" in df else pd.Series(dtype=float)
graded = df["grading_passed"].dropna() if "grading_passed" in df else pd.Series(dtype=object)
pass_rate = (graded == True).mean() if len(graded) else None  # noqa: E712 - pandas mask
retried = (df["retry_count"].fillna(0) > 0).mean() if "retry_count" in df else None

st.subheader("Scorecard")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Interactions", total)
c2.metric("Avg faithfulness", f"{faiths.mean():.2f}" if len(faiths) else "—",
          help="LLM-judged fraction of claims supported by retrieved context (answers with context only).")
c3.metric("Avg answer relevancy", f"{rels.mean():.2f}" if len(rels) else "—",
          help="Question↔answer embedding similarity.")
c4.metric("Grounding pass rate", f"{pass_rate*100:.0f}%" if pass_rate is not None else "—",
          help="Share of graded answers the grounding grader passed (higher = fewer numeric mismatches).")
c5.metric("Answers needing retry", f"{retried*100:.0f}%" if retried is not None else "—",
          help="Share of answers that triggered a grading retry.")

st.divider()

# --- Routing mix & grounding outcome ---------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Routing mix")
    if "route" in df:
        route_counts = df["route"].fillna("unknown").value_counts()
        st.bar_chart(route_counts)
    else:
        st.caption("No route data.")
with right:
    st.subheader("Grounding grader outcome")
    if "grading_passed" in df:
        outcome = df["grading_passed"].map({True: "passed", False: "failed"}).fillna("not graded")
        st.bar_chart(outcome.value_counts())
    else:
        st.caption("No grading data.")

st.divider()

# --- RAGAS over time -------------------------------------------------------------------
st.subheader("Live RAGAS over time")
ts = df.dropna(subset=["created_at"]).set_index("created_at")
cols_present = [c for c in ["faithfulness", "answer_relevancy"] if c in ts and ts[c].notna().any()]
if cols_present:
    st.line_chart(ts[cols_present])
else:
    st.caption("Not enough scored answers with context yet to plot a trend.")

# --- Latency ---------------------------------------------------------------------------
if "ttft_seconds" in df and df["ttft_seconds"].notna().any():
    st.subheader("Latency (seconds)")
    la, lb = st.columns(2)
    la.metric("Avg TTFT", f"{df['ttft_seconds'].dropna().mean():.2f}s")
    if "total_seconds" in df and df["total_seconds"].notna().any():
        lb.metric("Avg total", f"{df['total_seconds'].dropna().mean():.2f}s")

st.divider()

# --- Recent interactions ---------------------------------------------------------------
st.subheader("Recent interactions")
show_cols = [c for c in ["created_at", "route", "user_query", "grading_passed",
                         "retry_count", "faithfulness", "answer_relevancy", "ttft_seconds"]
             if c in df]
recent = df.sort_values("created_at", ascending=False).head(50)[show_cols]
st.dataframe(recent, hide_index=True, use_container_width=True)

st.divider()

# --- Latest offline batch-RAGAS run ----------------------------------------------------
st.subheader("Offline batch RAGAS (ground-truth set)")
batch_path = ROOT / "data" / "eval" / "latest_ragas.json"
if batch_path.exists():
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    st.caption(f"Last run: {batch.get('run_at', 'unknown')} · {batch.get('num_samples', '?')} samples")
    agg = batch.get("aggregate", {})
    if agg:
        cols = st.columns(len(agg))
        for col, (metric, value) in zip(cols, agg.items()):
            col.metric(metric.replace("_", " ").title(),
                       f"{value:.2f}" if isinstance(value, (int, float)) else "—")
    if batch.get("per_sample"):
        with st.expander("Per-question results"):
            st.dataframe(pd.DataFrame(batch["per_sample"]), hide_index=True, use_container_width=True)
else:
    st.info(
        "No batch run found. Compute full-quality RAGAS on a labelled set with:\n\n"
        "```bash\npython -m scripts.run_ragas_eval\n```\n\n"
        "It writes `data/eval/latest_ragas.json`, which appears here."
    )

st.caption("Standard: RAGAS-style metrics (project's Groq judge + Qwen embeddings). Live scores are reference-free; batch adds ground-truth-based metrics.")
