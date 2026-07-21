"""
Evaluation page (sidebar) — offline batch results against a labelled ground-truth set.

Per-answer scoring now happens INLINE in the chat (a scorecard is shown right under each
answer), so this page is no longer a live monitor. Instead it displays the latest offline
batch run — the full six-metric suite, grouped into Retrieval / Generation / End-to-End —
produced by `scripts/run_ragas_eval.py` (which writes data/eval/latest_ragas.json).
"""

import json
import pathlib

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Evaluation · Batch RAGAS", page_icon="📊", layout="wide")

st.title("📊 Batch Evaluation — RAGAS")
st.caption(
    "The full six-metric suite on a labelled ground-truth set, grouped into three categories. "
    "Per-answer (reference-free) scores are shown live in the chat; this page shows the offline "
    "batch run, which adds the ground-truth metrics."
)

batch_path = ROOT / "data" / "eval" / "latest_ragas.json"

if not batch_path.exists():
    st.info(
        "No batch run found yet. Compute the six-metric suite on the labelled set with:\n\n"
        "```bash\npython -m scripts.run_ragas_eval\n```\n\n"
        "It runs the full agent graph per question and writes `data/eval/latest_ragas.json`, "
        "which appears here. Edit `data/eval/ground_truth.json` to change the labelled questions."
    )
    st.stop()

batch = json.loads(batch_path.read_text(encoding="utf-8"))
st.caption(f"Last run: {batch.get('run_at', 'unknown')} · {batch.get('num_samples', '?')} samples")

# --- Grouped scorecard (Retrieval / Generation / End-to-End) ---------------------------
GROUP_HELP = {
    "Retrieval": "Evaluates the retriever / vector DB.",
    "Generation": "Evaluates the LLM response.",
    "End-to-End": "Compares the answer against the golden ground truth.",
}
METRIC_HELP = {
    "context_precision": "Signal-to-noise of the retrieved context (relevant vs noise).",
    "context_recall": "Did retrieval fetch everything the correct answer needs?",
    "faithfulness": "Is the answer grounded in the context (hallucination check)?",
    "answer_relevancy": "How directly the answer addresses the question.",
    "answer_correctness": "Factual accuracy of the answer vs the ground truth.",
    "answer_similarity": "Semantic resemblance (embedding) to the ground truth.",
}

groups = batch.get("groups")
if groups:
    for group_name, metrics in groups.items():
        st.subheader(group_name)
        if GROUP_HELP.get(group_name):
            st.caption(GROUP_HELP[group_name])
        cols = st.columns(len(metrics))
        for col, (metric, value) in zip(cols, metrics.items()):
            col.metric(
                metric.replace("_", " ").title(),
                f"{value:.2f}" if isinstance(value, (int, float)) else "—",
                help=METRIC_HELP.get(metric),
            )
else:
    # Fallback for an older flat-format results file.
    agg = batch.get("aggregate", {})
    cols = st.columns(len(agg)) if agg else []
    for col, (metric, value) in zip(cols, agg.items()):
        col.metric(metric.replace("_", " ").title(),
                   f"{value:.2f}" if isinstance(value, (int, float)) else "—")

st.divider()

# --- Per-question detail ---------------------------------------------------------------
if batch.get("per_sample"):
    st.subheader("Per-question results")
    st.dataframe(pd.DataFrame(batch["per_sample"]), hide_index=True, use_container_width=True)

st.caption("RAGAS-style metrics computed with the project's Groq judge + local Qwen embeddings "
           "(no external `ragas` package). Re-run `python -m scripts.run_ragas_eval` to refresh.")
