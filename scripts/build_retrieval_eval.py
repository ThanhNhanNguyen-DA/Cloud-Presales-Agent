"""
Build a retrieval evaluation set from the documents already in `cloud_documents`.

For a stratified random sample of chunks, the LLM writes ONE realistic presales question
that the chunk answers (in the chunk's language). The source chunk is the gold answer, so
retrieval quality can be measured without hand labelling (Hit@k / MRR in
scripts/eval_retrieval.py). Chunks that carry no answerable fact (menus, "Không áp dụng",
captions) are rejected by the LLM and skipped.

    python -m scripts.build_retrieval_eval --per-category Feature=45 CaseStudy=15 Competitor=30
    → data/eval/retrieval_eval.json   [{id, question, gold_id, gold_service, category, source}]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from app.db.supabase_client import supabase
from app.llm import get_chat_llm

PROMPT = """You build a retrieval test set for a cloud presales assistant.
Below is ONE chunk from the knowledge base. Write ONE question that a presales engineer or a
customer could realistically ask and that THIS chunk answers.

Rules:
- Write the question in the same language as the chunk (Vietnamese chunk -> Vietnamese question).
- Do not copy the chunk's heading verbatim; paraphrase like a real user would.
- Mention the product/service name when the chunk is about a specific product.
- If the chunk contains no concrete, answerable information (menu text, captions, "Không áp
  dụng", a bare title), answer with answerable=false.

CHUNK (service: {service}, category: {category}):
{content}

Reply with ONLY JSON: {{"answerable": true or false, "question": "..."}}"""


def fetch_rows(category: str) -> list[dict]:
    rows, start = [], 0
    while True:
        batch = (supabase.table("cloud_documents").select("id,service_name,category,content,metadata")
                 .eq("category", category).range(start, start + 499).execute().data or [])
        rows += batch
        if len(batch) < 500:
            return rows
        start += 500


def sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Spread the sample across source documents; skip tiny chunks."""
    usable = [r for r in rows if len(r["content"]) >= 300]
    by_source: dict[str, list[dict]] = {}
    for r in usable:
        by_source.setdefault((r["metadata"] or {}).get("source", "?"), []).append(r)
    for group in by_source.values():
        rng.shuffle(group)
    picked, sources = [], list(by_source)
    rng.shuffle(sources)
    while len(picked) < n and any(by_source.values()):
        for s in sources:
            if by_source[s] and len(picked) < n:
                picked.append(by_source[s].pop())
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", nargs="+", default=["Feature=45", "CaseStudy=15"],
                        help="CATEGORY=N pairs (oversampled ×1.4 to absorb rejected chunks).")
    parser.add_argument("--out", default="data/eval/retrieval_eval.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    llm = get_chat_llm(temperature=0.2)
    items = []
    for spec in args.per_category:
        category, n = spec.split("=")
        target = int(n)
        rows = sample(fetch_rows(category), int(target * 1.4) + 2, rng)
        kept = 0
        for row in rows:
            if kept >= target:
                break
            reply = llm.invoke(PROMPT.format(service=row["service_name"], category=category,
                                             content=row["content"][:1800])).content
            match = re.search(r"\{.*\}", reply if isinstance(reply, str) else str(reply), re.DOTALL)
            try:
                data = json.loads(match.group()) if match else {}
            except json.JSONDecodeError:
                data = {}
            if not data.get("answerable") or not str(data.get("question", "")).strip():
                continue
            kept += 1
            items.append({
                "id": f"{category.lower()}-{kept:03d}",
                "question": data["question"].strip(),
                "gold_id": row["id"],
                "gold_service": row["service_name"],
                "category": category,
                "source": (row["metadata"] or {}).get("source"),
            })
        print(f"{category}: {kept} question(s) from {len(rows)} sampled chunk(s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {len(items)} questions → {args.out}")


if __name__ == "__main__":
    main()
