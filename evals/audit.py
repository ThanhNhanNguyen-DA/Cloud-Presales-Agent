"""
End-to-end audit: one 0-100 score per layer and an overall score, judged by an independent LLM.

The judge is from a different model family than the system under test (Gemini judges a
GPT-OSS / Qwen system), so it does not grade its own relatives. Free-tier friendly: every judge
reply is cached on disk (evals/results/judge_cache.json), so an interrupted audit resumes
without spending quota again, and --max-judge-calls caps one run.

Layers (score = mean of the listed parts, each in 0..1, x100):
  L1 store       chunk quality (no junk / metadata-only chunks), no duplicate chunks, embedding
                 health (512-d, unit norm), index recall@10 (HNSW RPC vs exact scan), topic
                 coverage (Feature / SLA / Security / Compliance present)
  L2 retrieval   Hit@1, Hit@4, MRR@4 (evals/retrieval/ranking.json), keyword-stress Hit@4,
                 judge context precision (share of returned passages relevant to the question)
  L3 generation  judge per answer: faithful to the tool data, correct, complete, cites sources,
                 admits missing information (the judge's overall 0-10)
  L4 agents      routing accuracy, tool+argument accuracy, Grounding Grader accuracy
  L5 system      p95 latency (<= 8 s full marks, >= 28 s zero), tokens per answer (<= 6k full,
                 >= 20k zero), graph errors, prompt-injection checks
  L6 evaluation  judge-vs-grader agreement, judge-vs-deterministic-fact agreement, dataset size
                 (vs 100 cases), human-reviewed labels present (none yet: scored 0, on purpose)
Overall = weighted mean: L1 10%, L2 25%, L3 30%, L4 20%, L5 10%, L6 5%.

    python -m evals.audit --refresh          # re-run eval_e2e --live and eval_grader --live first
    python -m evals.audit                    # use the latest e2e_*/grader_* results
Inputs it reads: evals/results/e2e_*.json, grader_*.json (live runs), the live database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

from evals import common

# What a correct answer looks like per test category, when it is not "answer from the evidence".
# Without this the judge penalises intended refusals as "unhelpful".
BEHAVIOUR = {
    "competitor-unsupported": "Competitor comparison is deliberately NOT supported: the correct answer says so "
                              "plainly and invents no competitor facts. A polite refusal deserves full marks.",
    "negative": "The asked service/fact does not exist in the documentation: the correct answer says the "
                "information is not available and invents nothing.",
    "document": "Answer from the evidence only. The EXPECTED FACTS come from a test fixture: if the evidence does "
                "not contain them, saying the information is not available is the CORRECT answer (full marks for "
                "correctness and honesty); stating them anyway would be a hallucination.",
    "multi_intent": "Answer each part from the evidence; parts without evidence must be stated as not available.",
    "memory": "A human-REVIEWED correction (not shown in the evidence) says each project may have at most 1000 "
              "Object Storage buckets; using that fact is correct, not a hallucination. Any other figure is wrong.",
    "guardrail": "The input is invalid or incomplete: the correct answer asks for / explains the missing or "
                 "invalid value and states no computed figure.",
}
DEFAULT_BEHAVIOUR = "Answer the question from the evidence."

WEIGHTS = {"L1_store": 0.10, "L2_retrieval": 0.25, "L3_generation": 0.30,
           "L4_agents": 0.20, "L5_system": 0.10, "L6_evaluation": 0.05}
JUDGE_MODELS = ["gemini:gemini-3.5-flash", "gemini:gemini-3.5-flash-lite"]
CACHE = common.RESULTS / "judge_cache.json"


# --------------------------------------------------------------------------- judge
class Judge:
    """LLM judge with an on-disk cache and a call budget; replies must be JSON objects."""

    def __init__(self, llm, max_calls: int, cache_path: Path = CACHE):
        self.llm, self.max_calls, self.calls, self.cache_path = llm, max_calls, 0, cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.skipped = 0

    def ask(self, prompt: str) -> dict | None:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if self.calls >= self.max_calls:
            self.skipped += 1
            return None
        self.calls += 1
        try:
            raw = self.llm.invoke(prompt).content
            text = raw if isinstance(raw, str) else str(raw)
            data = json.loads(re.search(r"\{.*\}", text, re.S).group())
        except Exception as exc:  # noqa: BLE001 - quota / malformed: not cached, retried next run
            print(f"  judge failed: {str(exc)[:120]}", flush=True)
            self.skipped += 1
            return None
        self.cache[key] = data
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False))
        return data


CONTEXT_PROMPT = """You are a strict retrieval auditor for a cloud presales assistant (CMC Cloud documentation).
QUESTION: {question}

RETRIEVED PASSAGES:
{passages}

For each passage, decide whether it contains information that helps answer the QUESTION.
Reply with ONLY JSON: {{"relevant": [true/false for each passage, in order], "note": "<one short sentence>"}}"""

ANSWER_PROMPT = """You are a strict, independent auditor of a cloud presales AI assistant. Numbers in its answers go
into customer quotes, so be exacting. Judge the ANSWER using only the EVIDENCE (tool calls / retrieved
documents the assistant saw) and the EXPECTED FACTS from the test set.

QUESTION: {question}
EXPECTED FACTS (must appear, if any): {facts}
EXPECTED BEHAVIOUR: {behaviour}
EVIDENCE:
{evidence}

ANSWER:
{answer}

Score each criterion from 0.0 to 1.0:
- faithfulness: share of factual claims in the ANSWER supported by the EVIDENCE (1.0 if it makes no claims)
- correctness: facts and numbers are right (agree with EVIDENCE and EXPECTED FACTS)
- completeness: covers what the QUESTION asks (or clearly says what is missing)
- citations: names the source/service of document facts (null if the answer needs no documents)
- honesty: when evidence is missing or insufficient, says so instead of guessing (1.0 if not applicable)
Then an overall score from 0 to 10 for a presales engineer who will send this to a customer.

Reply with ONLY JSON: {{"faithfulness": x, "correctness": x, "completeness": x, "citations": x or null,
"honesty": x, "overall": n, "issues": ["<short concrete problem>", ...]}}"""


# --------------------------------------------------------------------------- helpers
def _latest(kind: str) -> Path | None:
    files = sorted(common.RESULTS.glob(f"{kind}_*.json"))
    return files[-1] if files else None


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _layer(parts: dict[str, float | None], metrics: dict, issues: list[str]) -> dict:
    present = {k: v for k, v in parts.items() if v is not None}
    score = round(100 * mean(present.values()), 1) if present else None
    return {"score": score, "parts": {k: (round(v, 3) if v is not None else None) for k, v in parts.items()},
            "metrics": metrics, "issues": issues}


def _fetch_corpus(supabase) -> list[dict]:
    rows, start = [], 0
    while True:
        batch = (supabase.table("cloud_documents").select("id,category,service_name,content,metadata,embedding")
                 .range(start, start + 499).execute().data or [])
        rows += batch
        if len(batch) < 500:
            return rows
        start += 500


# --------------------------------------------------------------------------- L1 store
def audit_store(rows: list[dict], rpc_top10, sample_queries: list[str], embed) -> dict:
    """rpc_top10(vector) -> list of ids from the production index; embed(text) -> vector."""
    import numpy as np

    n = len(rows)
    junk = [r for r in rows if len((r.get("content") or "").strip()) < 80
            or (re.search(r"›\s*Metadata\s*›", r.get("content") or "") and len(r["content"]) < 400)]
    counts: dict[str, int] = {}
    for r in rows:
        key = re.sub(r"\s+", " ", (r.get("content") or "")).strip().casefold()
        counts[key] = counts.get(key, 0) + 1
    dup_extra = sum(c - 1 for c in counts.values() if c > 1)
    vectors, bad_vec = [], 0
    for r in rows:
        emb = r.get("embedding")
        v = np.asarray(json.loads(emb) if isinstance(emb, str) else (emb or []), dtype=float)
        if v.shape != (512,) or abs(float(np.linalg.norm(v)) - 1.0) > 0.01:
            bad_vec += 1
            v = np.zeros(512)
        vectors.append(v)
    matrix, ids = np.stack(vectors), [r["id"] for r in rows]
    recalls = []
    for q in sample_queries:
        qv = np.asarray(embed(q))
        exact = {ids[j] for j in np.argsort(-(matrix @ qv))[:10]}
        recalls.append(len(exact & set(rpc_top10(qv))) / 10)
    wanted = {"Feature", "SLA", "Security", "Compliance"}
    present = {r.get("category") for r in rows} & wanted
    issues = []
    if junk:
        issues.append(f"{len(junk)} junk / metadata-only chunks")
    if dup_extra:
        issues.append(f"{dup_extra} duplicate chunks")
    if missing := sorted(wanted - present):
        issues.append(f"no documents in categories: {', '.join(missing)}")
    return _layer(
        {"chunk_quality": _clamp(1 - 10 * len(junk) / n) if n else None,
         "no_duplicates": _clamp(1 - 10 * dup_extra / n) if n else None,
         "embedding_health": 1 - bad_vec / n if n else None,
         "index_recall_at_10": mean(recalls) if recalls else None,
         "topic_coverage": len(present) / len(wanted)},
        {"chunks": n, "junk": len(junk), "duplicates": dup_extra, "bad_embeddings": bad_vec,
         "categories": sorted({str(r.get("category")) for r in rows}), "index_samples": len(recalls)},
        issues)


# --------------------------------------------------------------------------- L2 retrieval
def audit_retrieval(ranking: dict, stress_hit4: float | None, judged: list[dict | None], passages_per_q: list[int]) -> dict:
    m = ranking["metrics"]
    precisions = [sum(map(bool, j["relevant"][:k])) / k for j, k in zip(judged, passages_per_q, strict=True)
                  if j and k and isinstance(j.get("relevant"), list)]
    issues = [f"Hit@1 {m['hit_at_1']:.0%}: the right passage is often not ranked first"] if (m["hit_at_1"] or 0) < 0.9 else []
    if precisions and mean(precisions) < 0.6:
        issues.append(f"only {mean(precisions):.0%} of returned passages are relevant (noise reaches the agent)")
    return _layer({"hit_at_1": m["hit_at_1"], "hit_at_4": m["hit_at_k"], "mrr_at_4": m["mrr_at_k"],
                   "keyword_stress_hit_at_4": stress_hit4,
                   "context_precision": mean(precisions) if precisions else None},
                  {"queries": m["queries"], "judged_queries": len(precisions)}, issues)


# --------------------------------------------------------------------------- L3 generation
def audit_generation(records: list[dict], judged: list[dict | None]) -> dict:
    scored = [(r, j) for r, j in zip(records, judged, strict=True) if j and isinstance(j.get("overall"), (int, float))]
    issues = [f"{r['id']}: {issue}" for r, j in scored if j["overall"] < 7 for issue in (j.get("issues") or [])[:2]]

    def avg(key):
        vals = [j[key] for _, j in scored if isinstance(j.get(key), (int, float))]
        return mean(vals) if vals else None
    return _layer({"overall": mean(_clamp(j["overall"] / 10) for _, j in scored) if scored else None},
                  {"answers": len(records), "judged": len(scored),
                   **{k: (round(v, 3) if v is not None else None)
                      for k in ("faithfulness", "correctness", "completeness", "citations", "honesty")
                      for v in [avg(k)]}},
                  issues[:15])


# --------------------------------------------------------------------------- L4 agents
def audit_agents(e2e: dict, grader: dict | None) -> dict:
    t1, t3 = e2e["summary"]["tier1_routing"], e2e["summary"]["tier3_tools"]
    grader_acc = grader["summary"]["accuracy"] if grader else None
    issues = [f"wrong route: {r['id']} -> {r['route']} (want {r['expected_route']})"
              for r in e2e["results"] if not r["route_ok"]]
    issues += [f"wrong tool / arguments: {r['id']}" for r in e2e["results"] if not r["tool_ok"]]
    if grader and grader["summary"]["fnr"]:
        issues.append(f"grader misses {grader['summary']['fnr']:.0%} of wrong answers")
    return _layer({"routing": t1["accuracy"], "tools": t3["tool_accuracy"], "grader": grader_acc},
                  {"cases": len(e2e["results"]), "grader_fnr": grader and grader["summary"]["fnr"],
                   "grader_fpr": grader and grader["summary"]["fpr"]}, issues)


# --------------------------------------------------------------------------- L5 system
def audit_system(e2e: dict) -> dict:
    rs = e2e["results"]
    lat = [r["latency_s"] for r in rs]
    tokens = [r["tokens_in"] + r["tokens_out"] for r in rs]
    p95, tok = common.percentile(lat, 95), (mean(tokens) if tokens else 0)
    memory = e2e["summary"]["tier1_routing"].get("memory") or {}
    checks = [v for v in memory.values() if v is not None]
    errors = sum(bool(r["error"]) for r in rs)
    issues = []
    if p95 > 15:
        issues.append(f"p95 latency {p95:.1f} s")
    if tok > 10000:
        issues.append(f"{tok:,.0f} tokens per answer")
    if errors:
        issues.append(f"{errors} graph errors")
    return _layer({"latency_p95": _clamp(1 - (p95 - 8) / 20), "tokens_per_answer": _clamp(1 - (tok - 6000) / 14000),
                   "no_errors": 1 - errors / len(rs) if rs else None,
                   "injection_and_memory_checks": (sum(checks) / len(checks)) if checks else None},
                  {"p50_s": common.percentile(lat, 50), "p95_s": p95, "tokens_per_answer": round(tok),
                   "errors": errors, "memory_checks": memory}, issues)


# --------------------------------------------------------------------------- L6 evaluation system
def audit_evaluation(records: list[dict], judged: list[dict | None], n_cases: int, human_labels: bool) -> dict:
    pairs = [(r, j) for r, j in zip(records, judged, strict=True) if j and isinstance(j.get("faithfulness"), (int, float))]
    graded = [(r, j) for r, j in pairs if r.get("verdicts")]
    grader_agree = mean((r["verdicts"][-1] == "PASS") == (j["faithfulness"] >= 0.8) for r, j in graded) if graded else None
    factual = [(r, j) for r, j in pairs if r.get("n_facts") and isinstance(j.get("correctness"), (int, float))]
    facts_agree = mean((not r["missing_facts"]) == (j["correctness"] >= 0.8) for r, j in factual) if factual else None
    issues = []
    if not human_labels:
        issues.append("no human-reviewed labels: judge and grader are never checked against people")
    if n_cases < 100:
        issues.append(f"only {n_cases} end-to-end cases (target 100+, incl. unanswerable questions)")
    if grader_agree is not None and grader_agree < 0.8:
        issues.append(f"grader and judge disagree on {1 - grader_agree:.0%} of answers")
    return _layer({"judge_vs_grader": grader_agree, "judge_vs_fact_checks": facts_agree,
                   "dataset_size": min(1.0, n_cases / 100), "human_labels": 1.0 if human_labels else 0.0},
                  {"graded_pairs": len(graded), "fact_pairs": len(factual), "cases": n_cases}, issues)


def overall(layers: dict) -> float | None:
    present = {k: v["score"] for k, v in layers.items() if v["score"] is not None}
    if not present:
        return None
    weight = sum(WEIGHTS[k] for k in present)
    return round(sum(WEIGHTS[k] * s for k, s in present.items()) / weight, 1)


def markdown(report: dict) -> str:
    lines = [f"# Audit {report['run_at']}", "", f"**Overall: {report['overall']} / 100** "
             f"(judge: {report['judge']}, {report['judge_calls']} new calls, {report['judge_skipped']} not judged)", "",
             "| Layer | Score | Parts |", "|---|---|---|"]
    for name, layer in report["layers"].items():
        parts = ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} —" for k, v in layer["parts"].items())
        lines.append(f"| {name} | {layer['score'] if layer['score'] is not None else '—'} | {parts} |")
    for name, layer in report["layers"].items():
        if layer["issues"]:
            lines += ["", f"## {name}: issues", *[f"- {i}" for i in layer["issues"]]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- run
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="Run eval_e2e --live and eval_grader --live first.")
    parser.add_argument("--max-judge-calls", type=int, default=60)
    parser.add_argument("--context-queries", type=int, default=20, help="Retrieval queries judged for precision.")
    parser.add_argument("--out", default=common.default_out("audit"))
    args = parser.parse_args()

    if args.refresh:
        for mod in ("evals.eval_e2e", "evals.eval_grader"):
            subprocess.run([sys.executable, "-m", mod, "--live"], check=False)
    e2e_path, grader_path = _latest("e2e"), _latest("grader")
    if not e2e_path:
        parser.error("no evals/results/e2e_*.json: run with --refresh (or python -m evals.eval_e2e --live)")
    e2e = json.loads(e2e_path.read_text())
    grader = json.loads(grader_path.read_text()) if grader_path else None
    if e2e.get("mode") != "live":
        print("warning: the latest e2e run is OFFLINE (oracle answers): L3-L6 do not reflect the real system")

    import app.config  # noqa: F401 - loads .env
    from app.db.supabase_client import supabase
    from app.llm import get_chat_llm
    from app.tools.embedding_tool import embed_query
    from app.tools.rag_tools import CANDIDATES, document_scope, retrieve_documents
    from scripts import eval_lexical_stress
    from scripts.eval_retrieval_production import evaluate_queries

    judge = Judge(get_chat_llm(models=JUDGE_MODELS, rate_limit_max_retries=2), args.max_judge_calls)
    ranking_set = json.loads((Path(__file__).parent / "retrieval" / "ranking.json").read_text())

    print("L1 store …", flush=True)
    rows = _fetch_corpus(supabase)
    sample = [q["question"] for q in random.Random(7).sample(ranking_set, min(20, len(ranking_set)))]

    def rpc_top10(vector):
        return [d["id"] for d in supabase.rpc("match_cloud_documents", {
            "query_embedding": [float(x) for x in vector], "match_threshold": -1.0, "match_count": 10,
            "filter_category": None, "exclude_categories": None}).execute().data or []]
    layers = {"L1_store": audit_store(rows, rpc_top10, sample, embed_query)}

    print("L2 retrieval …", flush=True)
    ranking = evaluate_queries(ranking_set)
    stress_items = json.loads((Path(__file__).parent / "retrieval" / "lexical_stress.json").read_text())
    stress = eval_lexical_stress.run(stress_items, hybrid=True)
    judged_ctx, sizes = [], []
    for q in ranking_set[:args.context_queries]:
        docs = retrieve_documents(q["question"], **document_scope()).documents
        sizes.append(len(docs))
        if not docs:
            judged_ctx.append(None)
            continue
        passages = "\n\n".join(f"[{i}] {d['content'][:1200]}" for i, d in enumerate(docs, 1))
        judged_ctx.append(judge.ask(CONTEXT_PROMPT.format(question=q["question"], passages=passages)))
    layers["L2_retrieval"] = audit_retrieval(ranking, stress["hit_at_4"], judged_ctx, sizes)

    print("L3 generation …", flush=True)
    records = e2e["results"]
    judged_ans = [judge.ask(ANSWER_PROMPT.format(
        question=r["question"], facts=r.get("required_facts") or "none",
        behaviour=BEHAVIOUR.get(r.get("category"), DEFAULT_BEHAVIOUR),
        evidence=(r.get("raw_tool_output") or "(no tool was called)")[:20000], answer=r["answer"][:4000]))
        for r in records]
    layers["L3_generation"] = audit_generation(records, judged_ans)
    layers["L4_agents"] = audit_agents(e2e, grader)
    layers["L5_system"] = audit_system(e2e)
    layers["L6_evaluation"] = audit_evaluation(records, judged_ans, len(records), human_labels=False)

    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "judge": JUDGE_MODELS[0],
              "judge_calls": judge.calls, "judge_skipped": judge.skipped,
              "inputs": {"e2e": str(e2e_path), "grader": str(grader_path) if grader_path else None,
                         "candidates_per_channel": CANDIDATES},
              "overall": overall(layers), "weights": WEIGHTS, "layers": layers}
    common.write_report(args.out, report, markdown(report))
    print(markdown(report))
    if judge.skipped:
        print(f"{judge.skipped} items not judged (quota / budget): run again later to complete them from the cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
