"""
Ask My Brain - 智能问答标签页

会话管理、策略选择、对话历史、流式生成。
"""
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from embedder import embed_query
from retriever import retrieve_top_k, retrieve_full_documents
from generator import generate_answer_strategy_a_stream, generate_answer_strategy_b_stream
from hybrid_retriever import hybrid_retriever
import conversation_memory as conv_mem
from app_helpers import render_sources_html, _close_thinking, _stream_with_thinking


def render_chat_tab(has_index: bool):
    # 会话管理
    sess_col1, sess_col2, sess_col3 = st.columns([3, 1, 1])
    with sess_col1:
        sessions = conv_mem.list_sessions(limit=20)
        session_options = ["(新会话)"] + [f"{s['title']} ({s['message_count']}条)" for s in sessions]
        selected_sess = st.selectbox("会话", session_options, label_visibility="collapsed")
    with sess_col2:
        if st.button("🗑️", help="清空当前对话"):
            st.session_state.conversation = []
            st.rerun()
    with sess_col3:
        if selected_sess != "(新会话)":
            sess_idx = session_options.index(selected_sess) - 1
            if st.button("📂", help="加载会话"):
                sess = sessions[sess_idx]
                msgs = conv_mem.get_messages(sess["session_id"])
                st.session_state.conversation = [{"role": m["role"], "content": m["content"]} for m in msgs]
                st.session_state.current_session_id = sess["session_id"]
                st.rerun()

    # 策略选择
    top_col1, top_col2 = st.columns([5, 1])
    with top_col1:
        strategy = st.radio("策略", ["策略A: 传统RAG", "策略B: Long Context", "对比: A vs B"], horizontal=True, label_visibility="collapsed")

    # 对话历史
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # 输入框
    query = st.chat_input("输入你的问题...")
    if query:
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = conv_mem.create_session(title=query[:30])
        conv_mem.add_message(st.session_state.current_session_id, "user", query)

        if not has_index:
            with st.chat_message("assistant"):
                st.warning("⚠️ 尚未构建向量索引。请先在左侧上传文档并点击「构建索引」，然后再提问。")
        else:
            st.session_state.conversation.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

                try:
                    q_emb = embed_query(query)
                    hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation[:-1]]

                    ans_a, ans_b = None, None
                    is_cmp = "对比" in strategy
                    use_hybrid = st.session_state.use_hybrid and hybrid_retriever.is_ready

                    if is_cmp:
                        with st.spinner("检索中..."):
                            with ThreadPoolExecutor(max_workers=2) as ex:
                                if use_hybrid:
                                    fa = ex.submit(hybrid_retriever.hybrid_search, query, q_emb)
                                    fb = ex.submit(retrieve_full_documents, q_emb)
                                    ra, rb = fa.result(), fb.result()
                                else:
                                    fa = ex.submit(retrieve_top_k, q_emb)
                                    fb = ex.submit(retrieve_full_documents, q_emb)
                                    ra, rb = fa.result(), fb.result()

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f'<div class="strategy-header-a"><span class="strategy-badge badge-a">策略A · 传统RAG</span> <span style="color:#8b949e;font-size:0.8rem;">{ra["total_tokens"]} tok</span></div>', unsafe_allow_html=True)
                            think_a_ph = st.empty()
                            think_a_ph.markdown('<details class="think-block" open><summary style="color:#8b949e;font-size:0.85rem;cursor:pointer;">💭 思考中...</summary><div style="color:#6e7681;font-size:0.82rem;padding:0.3rem 0;opacity:0.7;"></div></details>', unsafe_allow_html=True)
                            gen_a = generate_answer_strategy_a_stream(query, ra, hist)
                            answer_a = st.write_stream(_stream_with_thinking(gen_a))
                            _close_thinking(think_a_ph, _stream_with_thinking._last_thinking)
                            st.markdown(render_sources_html(ra["sources"]), unsafe_allow_html=True)
                            ans_a = {"answer": answer_a, "sources": ra["sources"], "context_tokens": ra["total_tokens"]}
                        with c2:
                            st.markdown(f'<div class="strategy-header-b"><span class="strategy-badge badge-b">策略B · Long Context</span> <span style="color:#8b949e;font-size:0.8rem;">{rb["total_tokens"]} tok</span></div>', unsafe_allow_html=True)
                            think_b_ph = st.empty()
                            think_b_ph.markdown('<details class="think-block" open><summary style="color:#8b949e;font-size:0.85rem;cursor:pointer;">💭 思考中...</summary><div style="color:#6e7681;font-size:0.82rem;padding:0.3rem 0;opacity:0.7;"></div></details>', unsafe_allow_html=True)
                            gen_b = generate_answer_strategy_b_stream(query, rb, hist)
                            answer_b = st.write_stream(_stream_with_thinking(gen_b))
                            _close_thinking(think_b_ph, _stream_with_thinking._last_thinking)
                            st.markdown(render_sources_html(rb["sources"]), unsafe_allow_html=True)
                            ans_b = {"answer": answer_b, "sources": rb["sources"], "context_tokens": rb["total_tokens"]}

                    elif "A" in strategy:
                        with st.spinner("检索中..."):
                            if use_hybrid:
                                ra = hybrid_retriever.hybrid_search(query, q_emb)
                            else:
                                ra = retrieve_top_k(q_emb)
                        st.markdown(f'<div class="strategy-header-a"><span class="strategy-badge badge-a">策略A · 传统RAG</span> <span style="color:#8b949e;font-size:0.8rem;">{ra["total_tokens"]} tok</span></div>', unsafe_allow_html=True)
                        think_ph = st.empty()
                        think_ph.markdown('<details class="think-block" open><summary style="color:#8b949e;font-size:0.85rem;cursor:pointer;">💭 思考中...</summary><div style="color:#6e7681;font-size:0.82rem;padding:0.3rem 0;opacity:0.7;"></div></details>', unsafe_allow_html=True)
                        gen_a = generate_answer_strategy_a_stream(query, ra, hist)
                        answer_a = st.write_stream(_stream_with_thinking(gen_a))
                        _close_thinking(think_ph, _stream_with_thinking._last_thinking)
                        st.markdown(render_sources_html(ra["sources"]), unsafe_allow_html=True)
                        ans_a = {"answer": answer_a, "sources": ra["sources"], "context_tokens": ra["total_tokens"]}

                    else:
                        with st.spinner("检索中..."):
                            rb = retrieve_full_documents(q_emb)
                        st.markdown(f'<div class="strategy-header-b"><span class="strategy-badge badge-b">策略B · Long Context</span> <span style="color:#8b949e;font-size:0.8rem;">{rb["total_tokens"]} tok</span></div>', unsafe_allow_html=True)
                        think_ph = st.empty()
                        think_ph.markdown('<details class="think-block" open><summary style="color:#8b949e;font-size:0.85rem;cursor:pointer;">💭 思考中...</summary><div style="color:#6e7681;font-size:0.82rem;padding:0.3rem 0;opacity:0.7;"></div></details>', unsafe_allow_html=True)
                        gen_b = generate_answer_strategy_b_stream(query, rb, hist)
                        answer_b = st.write_stream(_stream_with_thinking(gen_b))
                        _close_thinking(think_ph, _stream_with_thinking._last_thinking)
                        st.markdown(render_sources_html(rb["sources"]), unsafe_allow_html=True)
                        ans_b = {"answer": answer_b, "sources": rb["sources"], "context_tokens": rb["total_tokens"]}

                    # 保存历史
                    fa = ""
                    if ans_a and ans_b:
                        fa = f"**策略A**: {ans_a['answer']}\n\n---\n\n**策略B**: {ans_b['answer']}"
                    elif ans_a:
                        fa = ans_a["answer"]
                    elif ans_b:
                        fa = ans_b["answer"]
                    if fa:
                        st.session_state.conversation.append({"role": "assistant", "content": fa})
                        conv_mem.add_message(st.session_state.current_session_id, "assistant", fa)

                except RuntimeError as e:
                    st.error(f"生成失败: {e}")
