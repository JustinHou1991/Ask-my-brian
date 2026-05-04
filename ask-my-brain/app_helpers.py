"""
Ask My Brain - UI工具函数

共享的Streamlit渲染辅助函数，供各标签页模块使用。
"""
import streamlit as st
import config


def init_state():
    """初始化Streamlit会话状态"""
    defaults = {
        "conversation": [], "documents_loaded": [], "index_built": False,
        "eval_result_cache": None, "current_session_id": None,
        "use_hybrid": config.HYBRID_RETRIEVAL_ENABLED, "logged_in": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_index_count() -> int:
    try:
        from retriever import get_collection_count
        return get_collection_count()
    except Exception:
        return 0


def render_sources_html(sources: list) -> str:
    seen = set()
    tags = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            tags.append(f'<span class="source-tag">{s}</span>')
    return f'<div class="sources-row">{"".join(tags)}</div>' if tags else ""


def render_strategy_card(label: str, badge: str, answer: str, sources: list, tokens: int, time_s: float = None):
    hdr = "strategy-header-a" if "A" in label else "strategy-header-b"
    t = f" · {time_s}s" if time_s is not None else ""
    st.markdown(f'<div class="{hdr}"><span class="strategy-badge {badge}">{label}</span> <span style="color:#8b949e;font-size:0.8rem;">{tokens} tok{t}</span></div>', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown(render_sources_html(sources), unsafe_allow_html=True)


def _close_thinking(placeholder, thinking_text: str):
    if thinking_text:
        summary = thinking_text[:60].replace("\n", " ") + ("..." if len(thinking_text) > 60 else "")
        escaped = thinking_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        placeholder.markdown(
            f'<details class="think-block"><summary style="color:#8b949e;font-size:0.82rem;cursor:pointer;">💭 {summary}</summary>'
            f'<div style="color:#6e7681;font-size:0.82rem;padding:0.4rem 0;opacity:0.65;max-height:200px;overflow-y:auto;">{escaped}</div></details>',
            unsafe_allow_html=True,
        )
    else:
        placeholder.empty()


def _stream_with_thinking(gen):
    thinking = ""
    for kind, chunk in gen:
        if kind == "reasoning":
            thinking += chunk
        elif kind == "content":
            yield chunk
    _stream_with_thinking._last_thinking = thinking
