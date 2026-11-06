"""
Turns the backend's stream events into short, plain-Vietnamese log lines for the
"Nhật ký hệ thống" panel. No Streamlit here, so it can be unit-tested.

Each entry is a dict: {"t": seconds since the question was sent, "icon": "🧭",
"text": "...", "level": "info" | "ok" | "warn" | "error"}.
"""

from __future__ import annotations

ROUTE_LABELS = {
    "calculator": ("🧮", "Tính toán chi phí / cấu hình"),
    "document": ("📚", "Tra cứu tài liệu sản phẩm"),
    "competitor": ("🆚", "So sánh với đối thủ"),
    "general": ("💬", "Trò chuyện chung"),
}

TOOL_LABELS = {
    "calculate_backup_and_cost": "máy tính thời gian sao lưu & chi phí lưu trữ",
    "size_web_app_vm": "công cụ ước tính cấu hình máy chủ",
    "size_database_storage": "công cụ ước tính dung lượng database",
    "search_cloud_documents": "kho tài liệu sản phẩm",
    "search_competitor_docs": "dữ liệu website đối thủ",
    "find_competitor_mapping": "bảng đối chiếu dịch vụ AWS/Azure/GCP",
    "list_all_competitor_services": "bảng đối chiếu dịch vụ AWS/Azure/GCP",
}


def route_badge(route: str | None) -> str:
    icon, label = ROUTE_LABELS.get(route or "", ("❔", "Không xác định"))
    return f"{icon} {label}"


def _entry(t: float, icon: str, text: str, level: str = "info") -> dict:
    return {"t": round(max(0.0, t), 1), "icon": icon, "text": text, "level": level}


def _tools_text(tools: list[str]) -> str:
    labels: list[str] = []
    for name in tools:
        label = TOOL_LABELS.get(name, name)
        if label not in labels:
            labels.append(label)
    return ", ".join(labels)


def entries_for_event(event: dict, t: float, ctx: dict | None = None) -> list[dict]:
    """
    Log lines for one stream event (most events produce zero or one line).

    `ctx` is a per-question dict the caller keeps between events; it remembers the route and
    the tools already announced, so a tool isn't reported twice and greetings get no score.
    """
    ctx = {} if ctx is None else ctx
    announced: set = ctx.setdefault("tools", set())
    etype = event.get("type")
    if etype == "progress":
        node, d = event.get("node"), event.get("detail") or {}
        if node == "tool_call":
            label = TOOL_LABELS.get(d.get("tool"), d.get("tool") or "công cụ")
            if label in announced:
                return []
            announced.add(label)
            return [_entry(t, "🔧", f"Đang dùng: {label}")]
        if node == "supervisor":
            ctx["route"] = d.get("route")
            icon, label = ROUTE_LABELS.get(d.get("route") or "", ("❔", "Không xác định"))
            out = [_entry(t, "🧭", f"Đã hiểu yêu cầu: **{label}**")]
            if d.get("used_corrections"):
                out.append(_entry(t, "📝", "Áp dụng góp ý đã được duyệt trước đây"))
            return out
        if node in ("calculator", "document", "competitor"):
            tools = d.get("tools") or []
            if not tools:
                return [_entry(t, "❓", "Chưa đủ thông tin để tính / tra cứu — sẽ hỏi lại bạn", "warn")]
            new = [name for name in tools if TOOL_LABELS.get(name, name) not in announced]
            announced.update(TOOL_LABELS.get(name, name) for name in tools)
            out = [_entry(t, "🔧", f"Đã dùng: {_tools_text(new)}")] if new else []
            if d.get("found_data") is False:
                out.append(_entry(t, "🔎", "Không tìm thấy dữ liệu phù hợp trong hệ thống", "warn"))
            return out
        if node == "general":
            return [_entry(t, "💬", "Trả lời trực tiếp, không cần tra cứu")]
        if node == "grader":
            verdict = d.get("verdict")
            return {
                "pass": [_entry(t, "✅", "Đã kiểm tra: số liệu khớp với dữ liệu gốc", "ok")],
                "skipped": [_entry(t, "ℹ️", "Không có số liệu cần đối chiếu")],
                "retry": [_entry(t, "⚠️", "Phát hiện số liệu chưa khớp — hệ thống tự soạn lại", "warn")],
                "warn": [_entry(t, "⚠️", "Vẫn còn chênh lệch — câu trả lời có kèm cảnh báo, nên kiểm tra lại", "warn")],
                "unverified": [_entry(t, "⚠️", "Không tự kiểm tra được — nên kiểm tra lại số liệu", "warn")],
            }.get(verdict, [])
        return []
    if etype == "ttft":
        return [_entry(t, "✍️", "Đang soạn câu trả lời…")]
    if etype == "reset":
        return [_entry(t, "🔁", "Đang viết lại câu trả lời cho chính xác hơn", "warn")]
    if etype == "ragas":
        if ctx.get("route") == "general":
            return []  # small talk: there is no data to score against
        parts = []
        if isinstance(event.get("faithfulness"), (int, float)):
            parts.append(f"bám sát dữ liệu {event['faithfulness']:.0%}")
        if isinstance(event.get("answer_relevancy"), (int, float)):
            parts.append(f"đúng trọng tâm {event['answer_relevancy']:.0%}")
        return [_entry(t, "📊", "Chấm điểm chất lượng: " + ", ".join(parts))] if parts else []
    if etype == "final":
        total = event.get("total")
        text = f"Hoàn tất sau {total:.1f} giây".replace(".", ",") if isinstance(total, (int, float)) else "Hoàn tất"
        return [_entry(t, "🏁", text, "ok")]
    if etype == "error":
        return [_entry(t, "❌", "Có lỗi khi trả lời — vui lòng thử lại", "error")]
    return []


def format_entry(entry: dict) -> str:
    """One markdown line: `+1,2s` 🧭 text."""
    seconds = f"{entry['t']:.1f}".replace(".", ",")
    return f"<span class='log-time'>+{seconds}s</span> {entry['icon']} {entry['text']}"
