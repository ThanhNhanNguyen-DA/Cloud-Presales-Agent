"""
Measure retrieval quality on data/eval/retrieval_eval.json (see build_retrieval_eval.py).

Compares, on the same questions:
  * rpc           — the production path: match_cloud_documents RPC (ivfflat index) with the
                    tool's defaults (threshold 0.6, top 3) and a wide variant (top 10, no threshold)
  * exact         — exact cosine search over all stored embeddings (what the index approximates)
  * exact+prompt  — same, but the QUESTION is embedded with Qwen3-Embedding's "query" prompt
  * <reranker>    — top-N candidates from the best first stage, re-scored by a cross-encoder

Metrics: Hit@1/3/5/10 (gold chunk in top-k), MRR@10, ServiceHit@3 (a chunk of the gold
service in the top 3), and mean latency per query.

    python -m scripts.eval_retrieval --rerankers bge mmarco qwen3 --candidates 30
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from app.db.supabase_client import supabase
from app.tools.embedding_tool import get_embedding_model

RERANKERS = {
    "bge": "BAAI/bge-reranker-v2-m3",                      # 568M, multilingual
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # 118M, multilingual
    "qwen3": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",        # 0.6B, same family as our embedder
}
CACHE = Path("data/eval/.corpus_cache.pkl")


def load_corpus(refresh: bool) -> list[dict]:
    if CACHE.exists() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    rows, start = [], 0
    while True:
        batch = (supabase.table("cloud_documents")
                 .select("id,service_name,category,content,embedding").range(start, start + 499)
                 .execute().data or [])
        rows += batch
        if len(batch) < 500:
            break
        start += 500
    for r in rows:
        r["embedding"] = np.asarray(json.loads(r["embedding"]), dtype=np.float32)
    CACHE.write_bytes(pickle.dumps(rows))
    return rows


def metrics(ranked: list[list[str]], items: list[dict], corpus_by_id: dict, seconds: float) -> dict:
    hits = {k: 0 for k in (1, 3, 5, 10)}
    mrr = service = 0.0
    for ids, item in zip(ranked, items, strict=True):
        if item["gold_id"] in ids[:10]:
            rank = ids.index(item["gold_id"]) + 1
            mrr += 1 / rank
            for k in hits:
                hits[k] += rank <= k
        if any(corpus_by_id[i]["service_name"] == item["gold_service"] for i in ids[:3] if i in corpus_by_id):
            service += 1
    n = len(items)
    return {**{f"hit@{k}": round(v / n, 3) for k, v in hits.items()},
            "mrr@10": round(mrr / n, 3), "service@3": round(service / n, 3),
            "ms/query": round(1000 * seconds / n, 1)}


def rpc_search(vectors: np.ndarray, threshold: float, count: int) -> tuple[list[list[str]], float]:
    out, t0 = [], time.perf_counter()
    for v in vectors:
        data = supabase.rpc("match_cloud_documents", {
            "query_embedding": v.tolist(), "match_threshold": threshold,
            "match_count": count, "filter_category": None}).execute().data or []
        out.append([d["id"] for d in data])
    return out, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/eval/retrieval_eval.json")
    parser.add_argument("--rerankers", nargs="*", default=["bge", "mmarco", "qwen3"], choices=list(RERANKERS))
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--refresh", action="store_true", help="Re-download the corpus from Supabase.")
    parser.add_argument("--out", default="data/eval/latest_retrieval.json")
    args = parser.parse_args()

    items = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    corpus = load_corpus(args.refresh)
    by_id = {r["id"]: r for r in corpus}
    ids = [r["id"] for r in corpus]
    matrix = np.stack([r["embedding"] for r in corpus])
    questions = [i["question"] for i in items]
    model = get_embedding_model()
    print(f"{len(items)} questions · {len(corpus)} chunks")

    results: dict[str, dict] = {}
    groups = sorted({i["category"] for i in items})

    def record(name: str, ranked: list[list[str]], seconds: float) -> None:
        results[name] = {"all": metrics(ranked, items, by_id, seconds)}
        for g in groups:
            idx = [k for k, it in enumerate(items) if it["category"] == g]
            results[name][g] = metrics([ranked[k] for k in idx], [items[k] for k in idx], by_id, seconds)
        print(f"{name:22s} {results[name]['all']}")

    t0 = time.perf_counter()
    plain = model.encode(questions, normalize_embeddings=True)
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    prompted = model.encode(questions, normalize_embeddings=True, prompt_name="query")
    t_prompt = time.perf_counter() - t0

    record("rpc (prod: 0.6, top3)", *rpc_search(plain, 0.6, 3))
    record("rpc (top10, no thr.)", *rpc_search(plain, -1.0, 10))

    def exact(vectors: np.ndarray, k: int) -> list[list[str]]:
        scores = vectors @ matrix.T
        return [[ids[j] for j in np.argsort(-row)[:k]] for row in scores]

    record("exact", exact(plain, 10), t_plain)
    first_stage = exact(prompted, args.candidates)
    record("exact+query-prompt", [r[:10] for r in first_stage], t_prompt)

    for name in args.rerankers:
        from sentence_transformers import CrossEncoder

        try:
            reranker = CrossEncoder(RERANKERS[name], max_length=512)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: could not load ({exc})")
            continue
        if reranker.tokenizer.pad_token is None:  # causal-LM based rerankers
            reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
            reranker.model.config.pad_token_id = reranker.tokenizer.pad_token_id
        ranked, t0 = [], time.perf_counter()
        for q, cand in zip(questions, first_stage, strict=True):
            scores = reranker.predict([(q, by_id[c]["content"]) for c in cand], batch_size=16)
            ranked.append([cand[j] for j in np.argsort(-np.asarray(scores))[:10]])
        record(f"+rerank {name}", ranked, time.perf_counter() - t0 + t_prompt)
        del reranker

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"✅ {args.out}")


if __name__ == "__main__":
    main()
