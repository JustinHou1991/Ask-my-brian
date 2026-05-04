"""
Ask My Brain - 评测报告标签页

展示批量评测结果: 总览指标、策略对比、分类筛选、逐题详情。
"""
import os
import json
import streamlit as st
import config
from app_helpers import render_strategy_card


def render_report_tab():
    ep = config.EVAL_RESULTS_PATH
    if not os.path.exists(ep):
        st.info("暂无评测结果。先在「双策略评测」运行批量评测。")
        return

    with open(ep, "r", encoding="utf-8") as f:
        rpt = json.load(f)
    stats = rpt.get("statistics", {})
    meta = rpt["metadata"]
    a_s, b_s = stats.get("strategy_a", {}), stats.get("strategy_b", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("总题数", meta.get("total_questions", meta.get("total", 0)))
    c2.metric("有效", stats.get("valid_questions", 0))
    c3.metric("失败", stats.get("failed_questions", 0))

    st.markdown("#### 对比")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="strategy-header-a"><span class="strategy-badge badge-a">策略A</span></div>', unsafe_allow_html=True)
        st.metric("平均响应", f"{a_s.get('avg_response_time', 0)}s")
        st.metric("平均Token", a_s.get('avg_context_tokens', 0))
    with col_b:
        st.markdown('<div class="strategy-header-b"><span class="strategy-badge badge-b">策略B</span></div>', unsafe_allow_html=True)
        st.metric("平均响应", f"{b_s.get('avg_response_time', 0)}s")
        st.metric("平均Token", b_s.get('avg_context_tokens', 0))

    at, bt = a_s.get('avg_response_time', 0), b_s.get('avg_response_time', 0)
    mx = max(at, bt, 0.01)
    st.markdown(f'<div class="compare-bar"><div class="bar-a" style="width:{at/mx*100}%;"></div></div><div class="compare-bar"><div class="bar-b" style="width:{bt/mx*100}%;"></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 详细")
    valid = [r for r in rpt.get("results", []) if "error" not in r]
    cats = sorted(set(r.get("category", "") for r in valid if r.get("category")))
    flt = st.multiselect("分类", cats, default=cats, label_visibility="collapsed") if cats else None
    cat_c = {"事实检索": "#3fb950", "理解分析": "#58a6ff", "应用场景": "#d29922", "技术细节": "#bc8cff", "综合应用": "#f778ba"}

    for r in valid:
        if flt and r.get("category", "") not in flt:
            continue
        cat = r.get("category", "")
        pill = f'<span class="category-pill" style="border:1px solid {cat_c.get(cat, "#8b949e")};color:{cat_c.get(cat, "#8b949e")};">{cat}</span>' if cat else ""
        with st.expander(f"{pill} {r['question'][:50]}"):
            st.markdown(f"**{r['question']}**")
            if r.get("reference_answer"):
                st.caption(f"参考: {r['reference_answer']}")
            c1, c2 = st.columns(2)
            with c1:
                render_strategy_card("策略A", "badge-a", r["strategy_a"]["answer"], r["strategy_a"]["sources"], r["strategy_a"]["context_tokens"], r["strategy_a"]["response_time"])
            with c2:
                render_strategy_card("策略B", "badge-b", r["strategy_b"]["answer"], r["strategy_b"]["sources"], r["strategy_b"]["context_tokens"], r["strategy_b"]["response_time"])
