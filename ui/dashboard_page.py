"""
Admin "Eval & Quality Dashboard": reads evals/results/*.json (written by evals/eval_*.py).
Protected by ADMIN_API_KEY unless AUTH_DISABLED=true (same rule as the API's /admin routes).
"""

import hmac
import os

import plotly.graph_objects as go
import streamlit as st

import eval_data as ed
from answer_rendering import format_answer_markdown

VERDICT_COLORS = {"PASS": "#2e9e5b", "FAIL": "#d64545", "UNVERIFIED": "#d99a00"}


def authorized() -> bool:
    if os.getenv("AUTH_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        return True
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        st.error("ADMIN_API_KEY chưa được cấu hình: trang quản trị bị khóa.")
        return False
    if st.session_state.get("admin_ok"):
        return True
    key = st.text_input("Admin API key", type="password")
    if key and hmac.compare_digest(key.encode(), expected.encode()):
        st.session_state.admin_ok = True
        st.rerun()
    if key:
        st.error("Sai khóa quản trị.")
    return False


@st.cache_data(ttl=30, show_spinner=False)
def load(directory: str) -> dict:
    return ed.load_runs(directory)


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


st.title("📊 Eval & Quality Dashboard")
if not authorized():
    st.stop()

# --------------------------------------------------------------------------- 0. end-to-end audit
audits = ed.load_audits()
if audits:
    latest = audits[-1]
    previous = audits[-2] if len(audits) > 1 else None
    st.subheader("🏁 Kiểm toán toàn hệ thống (giám khảo AI độc lập)")
    total_col, *layer_cols = st.columns(len(latest["layers"]) + 1)
    delta = (None if not previous or latest["overall"] is None or previous["overall"] is None
             else f"{latest['overall'] - previous['overall']:+.1f}")
    total_col.metric("Điểm tổng", "—" if latest["overall"] is None else f"{latest['overall']:.0f}/100", delta, border=True)
    for col, (key, layer) in zip(layer_cols, latest["layers"].items(), strict=True):
        col.metric(ed.LAYER_LABELS.get(key, key), "—" if layer["score"] is None else f"{layer['score']:.0f}", border=True)
    st.caption(f"{latest['run_at']} · giám khảo {latest['judge']} · chạy lại: `python -m evals.audit --refresh`")
    with st.expander("Vấn đề cần xử lý"):
        for key, layer in latest["layers"].items():
            for issue in layer["issues"]:
                st.markdown(f"- **{ed.LAYER_LABELS.get(key, key)}**: {issue}")
    st.divider()

runs = load(str(ed.RESULTS_DIR))
if not runs["e2e"]:
    runs = ed.demo_runs()
    st.info("Chưa có kết quả eval trong `evals/results/`: đang hiển thị **dữ liệu mẫu**. "
            "Chạy `python -m evals.eval_e2e --live --ragas` và `python -m evals.eval_grader --live` để có số liệu thật.",
            icon="ℹ️")

by_name = {r["name"]: r for r in runs["e2e"]}
names = list(by_name)
pick_current, pick_baseline, refresh = st.columns([2, 2, 1], vertical_alignment="bottom")
current = by_name[pick_current.selectbox("Lần chạy hiện tại (Current)", names, index=len(names) - 1)]
baseline = by_name[pick_baseline.selectbox("So sánh với (Baseline)", names, index=max(0, len(names) - 2))]
if refresh.button("🔄 Tải lại", width="stretch"):
    load.clear()
    st.rerun()
st.caption(f"Chế độ: **{current.get('mode', '?')}** · chạy lúc {current.get('run_at', '?')}")

# --------------------------------------------------------------------------- 1. KPIs
now, before = ed.kpis(current), ed.kpis(baseline)
cols = st.columns(4)
cols[0].metric("Tổng số test case", now["cases"], border=True)
for col, key, label in ((cols[1], "pass_rate", "E2E Pass Rate"), (cols[2], "recovery_rate", "Self-Correction Recovery")):
    delta = None if now[key] is None or before[key] is None else f"{now[key] - before[key]:+.0%}"
    col.metric(label, _pct(now[key]), delta, border=True)
p95, p95_before = now["p95_latency_s"], before["p95_latency_s"]
cols[3].metric("P95 Latency", "—" if p95 is None else f"{p95:.1f} s",
               None if p95 is None or p95_before is None else f"{p95 - p95_before:+.1f} s",
               delta_color="inverse", border=True)

# --------------------------------------------------------------------------- 2. radar + 3. sankey
radar_col, sankey_col = st.columns(2)
with radar_col:
    st.subheader("Chất lượng: Baseline vs Current")
    fig = go.Figure()
    missing = set()
    for label, run in (("Baseline", baseline), ("Current", current)):
        values = ed.radar_values(run, ed.grader_for(run, runs["grader"]))
        missing |= {k for k, v in values.items() if v is None}
        r = [values[a] or 0 for a in ed.RADAR_AXES]
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=[*ed.RADAR_AXES, ed.RADAR_AXES[0]], fill="toself", name=label))
    fig.update_layout(polar={"radialaxis": {"range": [0, 1], "tickformat": ".0%"}}, height=420,
                      margin={"l": 40, "r": 40, "t": 30, "b": 30})
    st.plotly_chart(fig, width="stretch")
    if missing:
        st.caption(f"Chưa có số liệu (vẽ = 0): {', '.join(sorted(missing))}. "
                   "RAGAS cần `--ragas`, Grader F1 cần một lần chạy `eval_grader`.")

with sankey_col:
    st.subheader("Luồng request: route → grader → kết quả")
    labels, source, target, value = ed.sankey(current["results"])
    fig = go.Figure(go.Sankey(node={"label": labels, "pad": 18, "thickness": 14},
                              link={"source": source, "target": target, "value": value}))
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- 4. threshold scatter
st.subheader("Hiệu chỉnh `RERANK_MIN_SCORE`: điểm rerank cao nhất vs Faithfulness")
points = ed.scatter_points(current["results"])
threshold = current.get("rerank_min_score", float(os.getenv("RERANK_MIN_SCORE", "0.1")))
if points:
    fig = go.Figure()
    for verdict in sorted({p["verdict"] for p in points}):
        group = [p for p in points if p["verdict"] == verdict]
        fig.add_trace(go.Scatter(x=[p["x"] for p in group], y=[p["y"] for p in group], mode="markers",
                                 name=f"Grader {verdict}", text=[p["id"] for p in group],
                                 marker={"size": 11, "color": VERDICT_COLORS.get(verdict, "#888")},
                                 hovertemplate="%{text}<br>rerank %{x:.2f} · faithfulness %{y:.2f}<extra></extra>"))
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"RERANK_MIN_SCORE = {threshold:g}")
    fig.update_layout(xaxis={"title": "top_rerank_score", "range": [0, 1]},
                      yaxis={"title": "RAGAS Faithfulness", "range": [0, 1.05]}, height=420,
                      margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig, width="stretch")
    st.caption("Điểm đỏ bên phải đường ngưỡng: passage được chấp nhận nhưng câu trả lời vẫn sai. "
               "Điểm xanh có Faithfulness cao sát bên trái: ngưỡng có thể đang loại passage tốt.")
else:
    st.caption("Chưa có ca nào có đủ `top_rerank_score` và Faithfulness (cần `--live --ragas`, reranker bật).")

# --------------------------------------------------------------------------- 5. explorer
st.subheader("Failure & Discrepancy Explorer")
selected = st.multiselect("Lọc", ed.FILTERS, default=list(ed.FILTERS))
rows = ed.explorer_rows(current["results"], set(selected))
if not rows:
    st.success("Không có ca nào khớp bộ lọc.", icon="✅")
else:
    event = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="explorer",
                         column_config={"faithfulness": st.column_config.NumberColumn(format="%.2f"),
                                        "top_rerank": st.column_config.NumberColumn(format="%.2f")})
    picked = event.selection.rows
    if not picked:
        st.caption("Bấm vào một dòng để xem chi tiết.")
    else:
        record = next(r for r in current["results"] if r["id"] == rows[picked[0]]["id"])
        with st.container(border=True):
            st.markdown(f"**{record['id']}** · route `{record.get('route')}` · grader "
                        f"{' → '.join(record.get('verdicts') or []) or '—'}")
            st.markdown(f"**Câu hỏi:** {record.get('question', '')}")
            if record.get("discrepancy"):
                st.warning(f"Lệch pha: {record['discrepancy']}")
            for issue in record.get("grader_issues") or []:
                st.error(issue, icon="🔎")
            for key, label in (("missing_facts", "Thiếu fact"), ("forbidden_hits", "Số/fact bị cấm"),
                               ("unsupported_numbers", "Số không có nguồn"), ("leaks", "Rò rỉ dữ liệu")):
                if record.get(key):
                    st.markdown(f"**{label}:** {record[key]}")
            with st.expander("tool(args) → result", expanded=False):
                st.code(record.get("raw_tool_output") or "(không gọi tool)", language="text", wrap_lines=True)
            st.markdown("**Câu trả lời của agent:**")
            st.markdown(format_answer_markdown(record.get("answer") or ""), unsafe_allow_html=True)
