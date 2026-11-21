"""
Manage reviewed canned answers (cloud_canned_answers, migration 006). Nothing is served until
a person approves it.

    python -m scripts.canned_answers draft                 # run the FAQ list through the agent;
                                                           # grader-passed answers -> status draft
    python -m scripts.canned_answers list [--status draft]
    python -m scripts.canned_answers show <id>             # full text, to review before approving
    python -m scripts.canned_answers approve <id> --reviewer "Tên" [--days 90]
    python -m scripts.canned_answers disable <id>
    python -m scripts.canned_answers add --question "..." --answer "..." [--route document] [--source "..."]
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime, timedelta

from app.db.supabase_client import supabase
from app.tools.embedding_tool import embed_query

TABLE = "cloud_canned_answers"

# Frequently asked, stable questions (no calculations, no prices that change).
FAQ_QUESTIONS = [
    "Bạn giúp được gì cho team presales?",
    "Hệ thống này có những chức năng nào?",
    "ELB hỗ trợ những thuật toán cân bằng tải nào?",
    "ELB hỗ trợ những giao thức listener nào?",
    "Cách tạo Auto Scaling Group như thế nào?",
    "Key Management Service dùng để làm gì?",
    "Virtual Private Cloud dùng để làm gì?",
    "Cách cấu hình VPN site-to-site?",
    "Kết nối VPC to VPC thực hiện thế nào?",
    "RDS for MySQL hỗ trợ những thao tác quản lý nào?",
    "Elastic IP là gì và dùng như thế nào?",
    "Các Availability Zone hiện có ở đâu?",
    "Cloud Kafka Service có những tính năng gì?",
    "Cloud Trace Service ghi lại những gì?",
    "Elastic GPU nên dùng trong những trường hợp nào?",
    "CMC Cloud DaaS phù hợp với doanh nghiệp nào?",
    "Terraform Service hỗ trợ những gì?",
    "Dịch vụ Elastic Load Balancer có những gói nào?",
]


def _sources(raw_tool_output: str | None) -> str | None:
    found = dict.fromkeys(re.findall(r"Source: (\S+)", raw_tool_output or ""))
    return ", ".join(found) or None


def insert(question: str, answer: str, route: str, source: str | None) -> str:
    row = supabase.table(TABLE).insert({
        "question": question, "answer": answer, "route": route, "source": source,
        "question_embedding": embed_query(question), "status": "draft",
    }).execute().data[0]
    return row["id"]


def draft(questions: list[str]) -> None:
    from app.graph.builder import build_graph

    existing = {r["question"] for r in supabase.table(TABLE).select("question").execute().data or []}
    graph = build_graph()
    for q in questions:
        if q in existing:
            print(f"skip (exists)   {q}")
            continue
        result = graph.invoke({"messages": [], "session_id": None, "user_query": q, "route": None,
                               "retrieved_experience": None, "raw_tool_output": None, "agent_response": None,
                               "final_response": None, "grading_passed": None, "retry_count": 0,
                               "grader_feedback": None})
        route, answer = result.get("route"), result.get("final_response") or ""
        ok = answer and route in ("document", "general") and (route == "general" or result.get("grading_passed") is True)
        if not ok:
            print(f"skip (not clean: route={route}, passed={result.get('grading_passed')})   {q}")
            continue
        print(f"draft {insert(q, answer, route, _sources(result.get('raw_tool_output')))}   {q}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("draft")
    ls = sub.add_parser("list")
    ls.add_argument("--status", choices=["draft", "approved", "disabled"])
    show = sub.add_parser("show")
    show.add_argument("id")
    ap = sub.add_parser("approve")
    ap.add_argument("id")
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--days", type=int, help="Expire after N days (default: never).")
    dis = sub.add_parser("disable")
    dis.add_argument("id")
    add = sub.add_parser("add")
    add.add_argument("--question", required=True)
    add.add_argument("--answer", required=True)
    add.add_argument("--route", default="document", choices=["calculator", "document", "general"])
    add.add_argument("--source")
    args = parser.parse_args()

    if args.cmd == "draft":
        draft(FAQ_QUESTIONS)
    elif args.cmd == "list":
        q = supabase.table(TABLE).select("id,status,route,hit_count,expires_at,question").order("created_at")
        for r in (q.eq("status", args.status) if args.status else q).execute().data or []:
            print(f"{r['id']}  {r['status']:8s} {r['route']:9s} hits={r['hit_count']:<3} "
                  f"exp={(r['expires_at'] or '-')[:10]}  {r['question']}")
    elif args.cmd == "show":
        r = supabase.table(TABLE).select("*").eq("id", args.id).execute().data
        if not r:
            raise SystemExit("not found")
        r = r[0]
        print(f"[{r['status']}] {r['question']}\nroute={r['route']} source={r['source']}\n\n{r['answer']}")
    elif args.cmd == "approve":
        now = datetime.now(UTC)
        update = {"status": "approved", "reviewed_by": args.reviewer, "approved_at": now.isoformat(),
                  "expires_at": (now + timedelta(days=args.days)).isoformat() if args.days else None}
        done = supabase.table(TABLE).update(update).eq("id", args.id).execute().data
        print("approved" if done else "not found")
    elif args.cmd == "disable":
        done = supabase.table(TABLE).update({"status": "disabled"}).eq("id", args.id).execute().data
        print("disabled" if done else "not found")
    elif args.cmd == "add":
        print("draft", insert(args.question, args.answer, args.route, args.source))


if __name__ == "__main__":
    main()
