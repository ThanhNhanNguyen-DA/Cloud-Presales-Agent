"""
Benchmark free LLMs for the orchestration roles of the agent (OpenAI-compatible APIs).

Suites (all prompts/tools are the production ones from the app):
  A. routing  — the supervisor's ROUTING_PROMPT on 54 labelled questions (vi/en, follow-ups
                with history, near-miss wording). Metric: accuracy.
  B. tools    — the specialists' tool schemas on 16 cases. Metric: right tool + right key
                arguments, or correctly NOT calling a tool when information is missing.
  C. grader   — the Grounding Grader prompt on 10 answer/raw-data pairs (5 correct, 5 with
                a wrong figure or invented fact). Metric: verdict accuracy.

Providers: groq (GROQ_API_KEY), gemini (GEMINI_API_KEY), hf (HUGGINGFACEHUB_API_TOKEN).

    python -m scripts.eval_orchestrator --models groq:openai/gpt-oss-120b gemini:gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from langchain_core.utils.function_calling import convert_to_openai_tool

import app.config  # noqa: F401 - loads .env
from app.agents.calculator_agent import SYSTEM_PROMPT as CALC_PROMPT
from app.agents.calculator_agent import TOOLS as CALC_TOOLS
from app.agents.competitor_agent import SYSTEM_PROMPT as COMP_PROMPT
from app.agents.competitor_agent import TOOLS as COMP_TOOLS
from app.agents.document_agent import SYSTEM_PROMPT as DOC_PROMPT
from app.agents.document_agent import TOOLS as DOC_TOOLS
from app.graders.self_rag_grader import GROUNDING_GRADER_PROMPT, parse_verdict
from app.graph.supervisor import ROUTING_PROMPT, parse_route

ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", 2.1),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY", 4.2),
    "hf": ("https://router.huggingface.co/v1/chat/completions", "HUGGINGFACEHUB_API_TOKEN", 0.5),
}

# --------------------------------------------------------------------------- suite A
H_BACKUP = [("user", "Backup 200GB qua đường 100Mbps mất bao lâu?"),
            ("assistant", "Khoảng 5,6 giờ với hiệu suất mạng 80%.")]
H_S3 = [("user", "AWS S3 tương đương dịch vụ nào bên mình?"),
        ("assistant", "AWS S3 tương đương Object Storage, mức độ chuyển đổi: Easy.")]
H_SLA = [("user", "SLA của Object Storage là bao nhiêu?"),
         ("assistant", "Object Storage cam kết uptime 99,9% mỗi tháng.")]
ROUTING_CASES = [
    # calculator
    ("Backup 500GB qua đường truyền 1Gbps mất bao lâu?", [], "calculator"),
    ("Chi phí lưu trữ 2TB ở tier Cool mỗi tháng là bao nhiêu?", [], "calculator"),
    ("Ứng dụng Java 3000 CCU cần bao nhiêu vCPU và RAM?", [], "calculator"),
    ("Database hiện 200GB, tăng 10%/tháng, sau 12 tháng cần bao nhiêu dung lượng và IOPS 5000 thì dùng loại ổ nào?", [], "calculator"),
    ("How long to back up 50 GB over 20 Mbps?", [], "calculator"),
    ("Size a VM for a Node.js app with 800 concurrent users.", [], "calculator"),
    ("Tính giúp mình thời gian sao lưu 1TB qua 300Mbps", [], "calculator"),
    ("Mình có 10 TB dữ liệu archive, mỗi tháng tốn bao nhiêu tiền?", [], "calculator"),
    ("Web PHP khoảng 1500 người truy cập đồng thời thì nên chọn cấu hình máy chủ nào?", [], "calculator"),
    ("Ước tính dung lượng DB sau 2 năm nếu hiện tại 80GB và tăng 5% mỗi tháng", [], "calculator"),
    ("Còn nếu dùng tier Archive thì sao?", H_BACKUP, "calculator"),
    ("Thế nếu băng thông là 1Gbps?", H_BACKUP, "calculator"),
    # document
    ("SLA uptime của Object Storage là bao nhiêu?", [], "document"),
    ("Cloud Server có chứng chỉ ISO 27001 không?", [], "document"),
    ("Dịch vụ có tuân thủ Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân không?", [], "document"),
    ("ELB hỗ trợ những thuật toán cân bằng tải nào?", [], "document"),
    ("Làm sao cấu hình Auto Scaling Group?", [], "document"),
    ("Is the platform PCI-DSS compliant?", [], "document"),
    ("What is the compensation if the SLA is breached?", [], "document"),
    ("RDS for MySQL có hỗ trợ backup tự động không?", [], "document"),
    ("Key Management Service dùng để làm gì?", [], "document"),
    ("Các Availability Zone hiện có ở đâu?", [], "document"),
    ("Còn Cloud Server thì cam kết bao nhiêu?", H_SLA, "document"),
    ("Tài liệu hướng dẫn tạo VPN site-to-site ở đâu?", [], "document"),
    # competitor
    ("AWS S3 tương đương dịch vụ nào bên mình?", [], "competitor"),
    ("Azure Blob Storage map sang sản phẩm nào?", [], "competitor"),
    ("Khách đang dùng GCP Compute Engine, chuyển sang mình thì dùng gì?", [], "competitor"),
    ("What's our equivalent of Amazon RDS?", [], "competitor"),
    ("So sánh EC2 với Cloud Server của mình", [], "competitor"),
    ("Migrate từ AWS Lambda sang thì khó không?", [], "competitor"),
    ("Liệt kê các dịch vụ AWS đã có mapping", [], "competitor"),
    ("Còn GCP thì sao?", H_S3, "competitor"),
    ("Azure thì tương đương là gì?", H_S3, "competitor"),
    ("Khách hỏi Azure SQL Database thì mình đề xuất dịch vụ nào?", [], "competitor"),
    ("Google Cloud Storage có dịch vụ tương ứng không?", [], "competitor"),
    ("Độ khó chuyển đổi từ AWS EKS là bao nhiêu?", [], "competitor"),
    ("Giá Cloud Server của Bizfly rẻ nhất là bao nhiêu?", [], "competitor"),
    ("VNPT Cloud có dịch vụ Kubernetes không?", [], "competitor"),
    ("So sánh Object Storage của mình với FPT Cloud", [], "competitor"),
    ("Viettel IDC cam kết SLA bao nhiêu cho Cloud Server?", [], "competitor"),
    ("VNG Cloud có GPU cloud không, giá thế nào?", [], "competitor"),
    ("Khách đang dùng vServer của VNG, mình đề xuất gì?", [], "competitor"),
    # general
    ("Xin chào", [], "general"),
    ("Bạn giúp được gì cho mình?", [], "general"),
    ("Cảm ơn nhé!", [], "general"),
    ("Hello, who are you?", [], "general"),
    ("Bạn là AI à?", [], "general"),
    ("Hôm nay thời tiết thế nào?", [], "general"),
    ("Ok, cảm ơn bạn", H_BACKUP, "general"),
    ("Viết giúp mình một bài thơ về mùa thu", [], "general"),
    ("Hệ thống này có những chuyên gia nào?", [], "general"),
    ("Tạm biệt", [], "general"),
    ("Good morning!", [], "general"),
    ("Bạn có nhớ mình vừa hỏi gì không?", H_SLA, "general"),
]

# --------------------------------------------------------------------------- suite B
TOOL_CASES = [
    # (system, tools, question, expected tool or None, {arg: expected value})
    ("calc", "Backup 100GB qua 200Mbps, lưu tier Cool thì mất bao lâu và tốn bao nhiêu mỗi tháng?",
     "calculate_backup_and_cost", {"data_volume_gb": 100, "bandwidth_mbps": 200, "storage_class": "cool"}),
    ("calc", "How long to back up 1 TB over 1 Gbps (Standard storage)?",
     "calculate_backup_and_cost", {"data_volume_gb": (1000, 1024), "bandwidth_mbps": 1000}),
    ("calc", "Ứng dụng Python 1200 CCU cần cấu hình VM thế nào?",
     "size_web_app_vm", {"ccu": 1200, "app_type": "python"}),
    ("calc", "Node.js app, 5000 concurrent users — what VM size?",
     "size_web_app_vm", {"ccu": 5000, "app_type": "nodejs"}),
    ("calc", "DB 300GB tăng 8%/tháng trong 18 tháng, IOPS yêu cầu 12000",
     "size_database_storage", {"current_size_gb": 300, "monthly_growth_rate_percent": 8, "months": 18, "required_iops": 12000}),
    ("calc", "Database 50GB, growth 20% per month for 6 months, needs 2500 IOPS",
     "size_database_storage", {"current_size_gb": 50, "monthly_growth_rate_percent": 20, "months": 6, "required_iops": 2500}),
    ("calc", "Mình muốn backup dữ liệu lên cloud thì mất bao lâu?", None, {}),   # missing volume/bandwidth
    ("calc", "Cần bao nhiêu RAM cho ứng dụng của tôi?", None, {}),               # missing CCU
    ("doc", "SLA uptime của Object Storage là bao nhiêu?", "search_cloud_documents", {}),
    ("doc", "Cloud Server có chứng chỉ PCI-DSS không?", "search_cloud_documents", {}),
    ("doc", "ELB hỗ trợ những giao thức listener nào?", "search_cloud_documents", {}),
    ("doc", "Does Key Management Service support key rotation?", "search_cloud_documents", {}),
    ("comp", "AWS S3 tương đương dịch vụ nào?", "find_competitor_mapping", {"competitor_service_name": "s3"}),
    ("comp", "What's the Azure Blob Storage equivalent?", "find_competitor_mapping", {"competitor_service_name": "blob"}),
    ("comp", "Liệt kê tất cả dịch vụ GCP đã có mapping", "list_all_competitor_services", {"competitor_name": "gcp"}),
    ("comp", "Amazon RDS chuyển sang thì dùng gì, khó không?", "find_competitor_mapping", {"competitor_service_name": "rds"}),
]
AGENTS = {"calc": (CALC_PROMPT, CALC_TOOLS), "doc": (DOC_PROMPT, DOC_TOOLS), "comp": (COMP_PROMPT, COMP_TOOLS)}

# --------------------------------------------------------------------------- suite C
RAW_BACKUP = ("calculate_backup_and_cost({'data_volume_gb': 10, 'bandwidth_mbps': 100}) -> "
              "{'calculated_monthly_cost_usd': 0.23, 'estimated_hours': 0.28, 'estimated_minutes': 16.7, "
              "'storage_class_used': 'Standard', 'network_efficiency': '80%'}")
RAW_VM = ("size_web_app_vm({'ccu': 500, 'app_type': 'java'}) -> {'recommended_vcpu': 4, "
          "'recommended_ram_gb': 12, 'app_type': 'java', 'target_ccu': 500}")
RAW_SLA = ("search_cloud_documents({'query': 'Object Storage SLA'}) -> [1] Service: Object Storage | "
           "Category: SLA\nObject Storage guarantees 99.9% monthly uptime. If not met, a 10% service credit applies.")
GRADER_CASES = [
    ("Backup 10GB qua 100Mbps?", RAW_BACKUP, "Backup 10 GB qua 100 Mbps mất khoảng **16,7 phút** (0,28 giờ); chi phí **0,23 USD/tháng**.", True),
    ("Backup 10GB qua 100Mbps?", RAW_BACKUP, "Backup 10 GB qua 100 Mbps mất khoảng **27 phút**; chi phí **0,23 USD/tháng**.", False),
    ("Backup 10GB qua 100Mbps?", RAW_BACKUP, "Mất ~16.7 phút, chi phí **2,3 USD/tháng** cho gói Standard.", False),
    ("Java 500 CCU?", RAW_VM, "Khuyến nghị **4 vCPU** và **12 GB RAM** cho 500 người dùng đồng thời (Java).", True),
    ("Java 500 CCU?", RAW_VM, "Khuyến nghị **8 vCPU** và **16 GB RAM** cho 500 người dùng đồng thời.", False),
    ("SLA Object Storage?", RAW_SLA, "Object Storage cam kết uptime **99,9%**/tháng; nếu vi phạm được bồi hoàn **10%** phí dịch vụ.", True),
    ("SLA Object Storage?", RAW_SLA, "Object Storage cam kết uptime **99,99%**/tháng và bồi hoàn 10%.", False),
    ("SLA Object Storage?", RAW_SLA, "Uptime 99,9%, bồi hoàn 10%. Dịch vụ cũng đạt chứng chỉ **ISO 27001** và **PCI-DSS**.", False),
    ("Java 500 CCU?", RAW_VM, "Với 500 CCU Java, cấu hình đề xuất là 4 vCPU / 12 GB RAM. Nên theo dõi tải thực tế và scale khi cần.", True),
    ("Backup 10GB qua 100Mbps?", RAW_BACKUP, "Thời gian ≈ 0,28 h (khoảng 17 phút), hiệu suất mạng giả định 80%, phí 0,23 USD mỗi tháng.", True),
]


# --------------------------------------------------------------------------- client
class Client:
    """Minimal OpenAI-compatible client with pacing and 429/5xx retries."""

    def __init__(self, provider: str, model: str) -> None:
        self.url, env, self.gap = ENDPOINTS[provider]
        self.key = os.environ[env]
        self.model = model
        self._last = 0.0
        self._lock = threading.Lock()

    def chat(self, messages: list[dict], tools: list | None = None, max_tokens: int = 1024) -> tuple[dict, float]:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        for attempt in range(6):
            with self._lock:
                wait = self.gap - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            t0 = time.perf_counter()
            try:
                r = requests.post(self.url, json=body, timeout=90,
                                  headers={"Authorization": f"Bearer {self.key}"})
            except requests.RequestException as exc:
                err = str(exc)
            else:
                if r.ok:
                    return r.json()["choices"][0]["message"], time.perf_counter() - t0
                err = f"{r.status_code} {r.text[:200]}"
                if r.status_code in (402, 429) and re.search(r"exceeded your current quota|credits|payment", r.text, re.I):
                    return {"content": "", "error": "quota exhausted: " + err}, time.perf_counter() - t0
                if r.status_code == 400 and "tool" in r.text.lower():
                    return {"content": "", "tool_calls": [], "error": err}, time.perf_counter() - t0
                if r.status_code not in (408, 429, 500, 502, 503, 504):
                    return {"content": "", "error": err}, time.perf_counter() - t0
            time.sleep(min(60, 5 * 2 ** attempt))
        return {"content": "", "error": err}, 0.0


def _history(turns):
    return [{"role": r, "content": c} for r, c in turns]


def run_routing(client: Client) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    from app.graph.supervisor import _format_history_for_routing

    ok, lat, errors, misses = 0, [], 0, []
    for q, hist, gold in ROUTING_CASES:
        msgs = [HumanMessage(c) if r == "user" else AIMessage(c) for r, c in hist]
        prompt = ROUTING_PROMPT.format(query=q, history_section=_format_history_for_routing(msgs))
        msg, s = client.chat([{"role": "user", "content": prompt}], max_tokens=512)
        lat.append(s)
        errors += bool(msg.get("error"))
        pred = parse_route(msg.get("content") or "")
        ok += pred == gold
        if pred != gold:
            misses.append(f"{q[:50]} → {pred} (want {gold})")
    return {"acc": round(ok / len(ROUTING_CASES), 3), "p50_s": round(statistics.median(lat), 2),
            "errors": errors, "misses": misses}


def _arg_ok(actual, expected) -> bool:
    if isinstance(expected, tuple):
        return any(_arg_ok(actual, e) for e in expected)
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - expected) < 1e-6
        except (TypeError, ValueError):
            return False
    return expected.lower() in str(actual).lower()


def run_tools(client: Client) -> dict:
    ok, errors, misses, lat = 0, 0, [], []
    for agent, q, want_tool, want_args in TOOL_CASES:
        prompt, tools = AGENTS[agent]
        schemas = [convert_to_openai_tool(t) for t in tools]
        msg, s = client.chat([{"role": "system", "content": prompt}, {"role": "user", "content": q}], tools=schemas)
        lat.append(s)
        errors += bool(msg.get("error"))
        calls = msg.get("tool_calls") or []
        if want_tool is None:
            good = not calls
        elif not calls:
            good = False
        else:
            fn = calls[0]["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            good = fn["name"] == want_tool and all(_arg_ok(args.get(k), v) for k, v in want_args.items())
        ok += good
        if not good:
            got = [(c["function"]["name"], c["function"].get("arguments")) for c in calls] or msg.get("error") or "no tool call"
            misses.append(f"{q[:45]} → {str(got)[:90]}")
    return {"acc": round(ok / len(TOOL_CASES), 3), "p50_s": round(statistics.median(lat), 2),
            "errors": errors, "misses": misses}


def run_grader(client: Client) -> dict:
    ok, lat, misses = 0, [], []
    for q, raw, answer, gold in GRADER_CASES:
        prompt = GROUNDING_GRADER_PROMPT.format(question=q, raw_output=raw, final_answer=answer)
        msg, s = client.chat([{"role": "user", "content": prompt}], max_tokens=1024)
        lat.append(s)
        passed, _ = parse_verdict(msg.get("content") or "")
        ok += passed is gold
        if passed is not gold:
            misses.append(f"{answer[:50]} → {passed} (want {gold})")
    return {"acc": round(ok / len(GRADER_CASES), 3), "p50_s": round(statistics.median(lat), 2), "misses": misses}


_save_lock = threading.Lock()
SUITES: list[str] = ["routing", "tools", "grader"]


def _save(path: str, spec: str, result: dict) -> None:
    """Merge one model's result into the output file right away (survives interruptions)."""
    with _save_lock:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        data[spec] = result
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(spec: str, out_path: str) -> tuple[str, dict]:
    provider, model = spec.split(":", 1)
    client = Client(provider, model)
    out = {}
    for name, fn in (("routing", run_routing), ("tools", run_tools), ("grader", run_grader)):
        if name not in SUITES:
            continue
        out[name] = fn(client)
        print(f"[{spec}] {name}: acc={out[name]['acc']} p50={out[name]['p50_s']}s "
              f"errors={out[name].get('errors', 0)}", flush=True)
        if out[name].get("errors", 0) >= len(ROUTING_CASES) // 2:
            print(f"[{spec}] too many API errors — skipping the remaining suites", flush=True)
            break
    _save(out_path, spec, out)
    return spec, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="provider:model, e.g. groq:openai/gpt-oss-120b")
    parser.add_argument("--out", default="data/eval/latest_orchestrator.json")
    parser.add_argument("--suites", nargs="+", default=["routing", "tools", "grader"],
                        choices=["routing", "tools", "grader"])
    args = parser.parse_args()
    SUITES[:] = args.suites
    # One thread per provider; models of the same provider run one after another (shared quota).
    by_provider: dict[str, list[str]] = {}
    for spec in args.models:
        by_provider.setdefault(spec.split(":")[0], []).append(spec)

    def run_provider(specs):
        return [evaluate(s, args.out) for s in specs]

    results = {}
    with ThreadPoolExecutor(len(by_provider)) as pool:
        for batch in pool.map(run_provider, by_provider.values()):
            results.update(dict(batch))
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmodel".ljust(44), "routing  tools  grader  route_p50")
    for spec, r in results.items():
        cells = [f"{r[k]['acc']:.2f}" if k in r else " -- " for k in ("routing", "tools", "grader")]
        first = next(iter(r.values()))
        print(spec.ljust(44), "     ".join(cells), f"   {first['p50_s']}s")


if __name__ == "__main__":
    main()
