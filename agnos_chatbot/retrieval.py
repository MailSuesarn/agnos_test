from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Module-level retrievers set by init_tools()
_THREADS_RET = None
_DISEASES_RET = None

def init_tools(threads_ret, diseases_ret) -> None:
    global _THREADS_RET, _DISEASES_RET
    _THREADS_RET = threads_ret
    _DISEASES_RET = diseases_ret

def build_numbered_context(threads_docs, diseases_docs) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]], bool]:
    parts: List[str] = []
    t_links: List[Tuple[str, str]] = []
    d_links: List[Tuple[str, str]] = []

    t_docs = [d for d in threads_docs if (getattr(d, "page_content", "") or "").strip()]
    if t_docs:
        parts.append("### Doctor Threads (T-series)")
        for i, d in enumerate(t_docs, start=1):
            tid = f"T{i}"
            url = (d.metadata.get("source") or "").strip()
            t_links.append((tid, url))
            title = (d.metadata.get("title") or "").strip()
            cat = (d.metadata.get("category") or "").strip()
            text = (d.page_content or "").strip()
            parts.append(f"[{tid}] Title: {title}\nCategory: {cat}\nDoctor says:\n{text}\n(Source: {url})")
    else:
        parts.append("### Doctor Threads (T-series)\n(none)")

    if diseases_docs:
        parts.append("\n### Disease References (D-series)")
        for j, d in enumerate(diseases_docs, start=1):
            did = f"D{j}"
            url = (d.metadata.get("source") or "").strip()
            d_links.append((did, url))
            title = (d.metadata.get("title") or "").strip()
            text = (d.page_content or "").strip()
            parts.append(f"[{did}] {title}\n{text}\n(Source: {url})")
    else:
        parts.append("\n### Disease References (D-series)\n(none)")

    ctx = "\n\n".join(parts)
    has_doctor = len(t_docs) > 0
    return ctx, t_links, d_links, has_doctor

@tool
def retrieve_medical_context(
    query: str,
    k_threads: Optional[int] = None,
    k_diseases: Optional[int] = None,
) -> str:
    """ดึงบริบทจากคลัง Agnos: กระทู้ที่แพทย์ตอบ (T#) และความรู้เกี่ยวกับโรค (D#).
    Returns JSON: { context, t_links, d_links, has_doctor }
    """
    if _THREADS_RET is None or _DISEASES_RET is None:
        return json.dumps({"context":"","t_links":[],"d_links":[],"has_doctor":"no","error":"retrievers_not_initialized"}, ensure_ascii=False)

    def _to_int(v, default):
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    k_threads  = _to_int(k_threads, 3)
    k_diseases = _to_int(k_diseases, 2)

    try:
        t_docs = _THREADS_RET.get_relevant_documents(query)[:k_threads]
        d_docs = _DISEASES_RET.get_relevant_documents(query)[:k_diseases]
        context_text, t_links, d_links, has_doctor = build_numbered_context(t_docs, d_docs)
        return json.dumps({
            "context": context_text,
            "t_links": t_links,
            "d_links": d_links,
            "has_doctor": "yes" if has_doctor else "no"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"context":"","t_links":[],"d_links":[],"has_doctor":"no","error":str(e)}, ensure_ascii=False)

def _sanitize_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    q = args.get("query", "")
    out["query"] = "" if q is None else str(q)

    def _maybe_int(x):
        if x is None:
            return None
        try:
            return int(x)
        except Exception:
            return None

    kt = _maybe_int(args.get("k_threads"))
    kd = _maybe_int(args.get("k_diseases"))
    if kt is not None:
        out["k_threads"] = kt
    if kd is not None:
        out["k_diseases"] = kd
    return out

def chat_once(
    system_prompt: str,
    user_text: str,
    history_messages: Optional[List[Dict[str, str]]],
    chat_model,
) -> str:
    """Single response with tool-calling allowing up to 3 tool turns."""
    history_messages = history_messages or []
    msgs: List[Any] = [SystemMessage(content=system_prompt)]
    for m in history_messages:
        role = m.get("role")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content))
        elif role in ("assistant", "ai"):
            msgs.append(AIMessage(content))
    msgs.append(HumanMessage(user_text))

    if not hasattr(chat_model, "bind_tools"):
        return "ขออภัย รุ่นโมเดลนี้ยังไม่รองรับ bind_tools()"

    tool_model = chat_model.bind_tools([retrieve_medical_context])
    for _ in range(3):
        ai: AIMessage = tool_model.invoke(msgs)
        msgs.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}
            call_id = call.get("id")
            if name == "retrieve_medical_context":
                tool_out = retrieve_medical_context.invoke(_sanitize_tool_args(args))
                msgs.append(ToolMessage(content=tool_out, tool_call_id=call_id))
            else:
                msgs.append(ToolMessage(content="{}", tool_call_id=call_id))

    final = ""
    if msgs and isinstance(msgs[-1], AIMessage):
        final = (msgs[-1].content or "").strip()
    return final or "ขออภัย ฉันไม่สามารถสร้างคำตอบได้ในขณะนี้"
