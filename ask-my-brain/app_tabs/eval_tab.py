"""
Ask My Brain - 双策略评测标签页

单题评测、批量评测（50题）。
"""
import os
import json
import streamlit as st
import config
from eval_engine import evaluate_single, compute_stats
from utils import save_json
from model_registry import get_current_model
from app_helpers import render_strategy_card


def render_eval_tab(has_index: bool):
    st.markdown("#### 单题评测")
    eq_col1, eq_col2 = st.columns([4, 1])
    with eq_col1:
        eval_q = st.text_input("问题", key="eval_q", placeholder="什么是向量数据库？", label_visibility="collapsed")
    with eq_col2:
        if st.button("▶ 评测", use_container_width=True, type="primary", disabled=not eval_q):
            if not has_index:
                st.warning("请先构建索引")
            else:
                with st.spinner("评测中..."):
                    try:
                        st.session_state.eval_result_cache = evaluate_single(eval_q)
                    except RuntimeError as e:
                        st.error(f"失败: {e}")

    if st.session_state.eval_result_cache:
        r = st.session_state.eval_result_cache
        ra, rb = r["strategy_a"], r["strategy_b"]
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#58a6ff;">{ra["response_time"]}s</div><div class="metric-label">A 响应</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#58a6ff;">{ra["context_tokens"]}</div><div class="metric-label">A Token</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#bc8cff;">{rb["response_time"]}s</div><div class="metric-label">B 响应</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#bc8cff;">{rb["context_tokens"]}</div><div class="metric-label">B Token</div></div>', unsafe_allow_html=True)

        # 质量分展示
        qa, qb = ra.get("quality", {}), rb.get("quality", {})
        if qa.get("relevance", 0) > 0 or qb.get("relevance", 0) > 0:
            q1, q2, q3 = st.columns(3)
            q1.metric("相关性", f"A:{qa.get('relevance','-')} / B:{qb.get('relevance','-')}")
            q2.metric("完整性", f"A:{qa.get('completeness','-')} / B:{qb.get('completeness','-')}")
            q3.metric("忠实度", f"A:{qa.get('faithfulness','-')} / B:{qb.get('faithfulness','-')}")

        mt = max(ra["response_time"], rb["response_time"], 0.01)
        st.markdown(f'<div style="margin:0.8rem 0;"><div class="compare-bar"><div class="bar-a" style="width:{ra["response_time"]/mt*100}%;"></div></div><div class="compare-bar"><div class="bar-b" style="width:{rb["response_time"]/mt*100}%;"></div></div></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            render_strategy_card("策略A · 传统RAG", "badge-a", ra["answer"], ra["sources"], ra["context_tokens"], ra["response_time"])
        with c2:
            render_strategy_card("策略B · Long Context", "badge-b", rb["answer"], rb["sources"], rb["context_tokens"], rb["response_time"])

    st.markdown("---")
    st.markdown("#### 批量评测（50题）")

    if st.button("🚀 开始批量评测", use_container_width=True, type="primary"):
        if not has_index:
            st.warning("请先构建索引")
        else:
            tf = config.TEST_QUESTIONS_PATH
            if os.path.exists(tf):
                with open(tf, "r", encoding="utf-8") as f:
                    qs = json.load(f)
                prog = st.progress(0)
                status = st.empty()
                results = []
                for i, q in enumerate(qs):
                    status.caption(f"第 {i+1}/{len(qs)} 题: {q['question'][:35]}...")
                    try:
                        r = evaluate_single(q["question"], q.get("reference_answer"))
                        r["category"] = q.get("category", "")
                        results.append(r)
                    except Exception as e:
                        results.append({"question": q["question"], "category": q.get("category", ""), "error": str(e)})
                    prog.progress(int((i+1)/len(qs)*100))
                stats = compute_stats(results)
                save_json({"metadata": {"total": len(qs), "model": get_current_model()["id"]}, "statistics": stats, "results": results}, config.EVAL_RESULTS_PATH)
                prog.progress(100)
                status.empty()
                st.success(f"完成 — 有效 {stats.get('valid_questions', 0)} / 失败 {stats.get('failed_questions', 0)}")
            else:
                st.error(f"未找到: {tf}")
