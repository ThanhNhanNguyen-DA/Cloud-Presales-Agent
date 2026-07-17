"""
Evaluation page (sidebar) — RAG quality scored against the RAGAS framework.

RAGAS (Retrieval-Augmented Generation Assessment) is a widely used open standard for
evaluating RAG systems along four core metrics. This page presents a design-based
assessment of the Cloud Presales Agent against those metrics, with scores and tables,
plus the exact procedure to compute the metrics live on a labelled test set.

The scores here are an engineering assessment tied to the system's architecture — they
are NOT yet measured on a labelled dataset. The "How to run live RAGAS" section below
shows how to replace them with computed numbers.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evaluation · RAGAS", page_icon="📊", layout="centered")

st.title("📊 System Evaluation — RAGAS")
st.caption(
    "RAG quality scored against the RAGAS standard (Faithfulness, Answer Relevancy, "
    "Context Precision, Context Recall). Scores below are a design-based assessment, "
    "not yet measured on a labelled set — see methodology at the bottom."
)

# --- RAGAS metric assessment (0.0–1.0) -------------------------------------------------
METRICS = [
    {
        "Metric": "Faithfulness",
        "Score": 0.80,
        "Measures": "Is every claim in the answer grounded in the retrieved context (no hallucination)?",
        "Assessment": "A decoupled Grounding Grader cross-checks numeric claims against the raw tool/RAG "
                      "output and retries or warns on mismatch. Strong for numbers; textual claims "
                      "(SLA wording, legal terms) are not yet grounding-checked.",
    },
    {
        "Metric": "Answer Relevancy",
        "Score": 0.82,
        "Measures": "Does the answer actually address the user's question, without padding or drift?",
        "Assessment": "The supervisor routes each question to a focused specialist agent, and system "
                      "prompts force tool use and same-language replies — keeping answers on-topic.",
    },
    {
        "Metric": "Context Precision",
        "Score": 0.72,
        "Measures": "Are the retrieved chunks relevant, and ranked with the best ones first?",
        "Assessment": "Structure-aware chunking (DOCX by heading + tables, PDF per case study) raises "
                      "chunk coherence. No re-ranker yet, and the similarity threshold (0.6) is fixed.",
    },
    {
        "Metric": "Context Recall",
        "Score": 0.65,
        "Measures": "Does the retrieved context contain everything needed to answer fully?",
        "Assessment": "Limited mainly by corpus completeness (one document group still pending ingestion) "
                      "and a small top-k (match_count = 3). Expected to rise as the corpus is completed.",
    },
]


def rating(score: float) -> str:
    if score >= 0.85:
        return "🟢 Excellent"
    if score >= 0.75:
        return "🟡 Good"
    if score >= 0.65:
        return "🟠 Fair"
    return "🔴 Needs work"


overall = round(sum(m["Score"] for m in METRICS) / len(METRICS), 2)

# --- Live scores measured this session (shared via st.session_state) -------------------
st.subheader("🟢 Live scores — measured this session")
history = st.session_state.get("chat_history", [])
live = [m["ragas"] for m in history if m.get("role") == "assistant" and m.get("ragas")]
faiths = [r["faithfulness"] for r in live if isinstance(r.get("faithfulness"), (int, float))]
rels = [r["answer_relevancy"] for r in live if isinstance(r.get("answer_relevancy"), (int, float))]

if not live:
    st.info("No answers scored yet. Ask questions on the main chat page — each answer is "
            "scored live (faithfulness + answer relevancy) and averaged here.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Answers scored", len(live))
    c2.metric("Avg faithfulness", f"{sum(faiths) / len(faiths):.2f}" if faiths else "—",
              help="LLM-judged, over the answers that had retrieved context.")
    c3.metric("Avg answer relevancy", f"{sum(rels) / len(rels):.2f}" if rels else "—",
              help="Question↔answer embedding similarity.")
    st.caption("Reference-free RAGAS, computed live per answer. Context precision/recall "
               "need a labelled set and are shown as a design assessment below.")

st.divider()

# --- Design-based assessment (baseline across all four metrics) ------------------------
st.subheader("🎨 Design-based assessment")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Overall RAGAS score", f"{overall:.2f}", help="Unweighted mean of the four metrics (0–1).")
with col2:
    st.progress(overall, text=f"{rating(overall)} · {int(overall * 100)} / 100")

st.divider()

# --- Per-metric bars -------------------------------------------------------------------
st.subheader("Scores by metric")
for m in METRICS:
    st.progress(m["Score"], text=f"{m['Metric']} — {m['Score']:.2f}  ({rating(m['Score'])})")

# --- Detail table ----------------------------------------------------------------------
st.subheader("Scorecard")
df = pd.DataFrame(
    [
        {"Metric": m["Metric"], "Score": f"{m['Score']:.2f}", "Rating": rating(m["Score"]),
         "What it measures": m["Measures"], "Assessment": m["Assessment"]}
        for m in METRICS
    ]
)
st.dataframe(df, hide_index=True, use_container_width=True)

# --- Roadmap to raise the weakest metrics ---------------------------------------------
st.subheader("Priority actions to raise the score")
st.dataframe(
    pd.DataFrame([
        {"Target metric": "Context Recall", "Action": "Finish ingesting the pending document group; raise match_count.", "Impact": "High"},
        {"Target metric": "Context Precision", "Action": "Add a lightweight re-ranker over the top-k chunks.", "Impact": "Medium"},
        {"Target metric": "Faithfulness", "Action": "Extend the grader to check textual grounding, not only numbers.", "Impact": "Medium"},
    ]),
    hide_index=True, use_container_width=True,
)

# --- Methodology / how to compute live -------------------------------------------------
with st.expander("Methodology & how to run live RAGAS"):
    st.markdown(
        """
**How these scores were derived.** They are an engineering assessment mapped to each
RAGAS metric from the system's design (grounding grader, routing, chunking strategy,
corpus status), *not* numbers measured on a labelled dataset. Treat them as a relative
priority signal.

**How to replace them with computed numbers.** RAGAS metrics are computed from a small
evaluation set of `(question, ground_truth, retrieved_contexts, answer)` records, using
an LLM as judge:

```bash
pip install ragas datasets
```

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall,
)

# For each eval question: call the graph, capture the answer AND the retrieved chunks,
# and pair with a human-written ground-truth answer.
data = Dataset.from_dict({
    "question":  [...],
    "answer":    [...],   # final_response from the graph
    "contexts":  [...],   # list of retrieved chunk texts per question
    "ground_truth": [...],
})
result = evaluate(data, metrics=[faithfulness, answer_relevancy,
                                 context_precision, context_recall])
print(result)  # -> plug these numbers into METRICS above
```

Building the ~30–50 question labelled set from real presales queries is the main
remaining task; once it exists, this page can display live-computed scores.
"""
    )

st.caption("Standard: RAGAS · assessment dated 2026-07-17 · scores are design-based pending a labelled eval set.")
