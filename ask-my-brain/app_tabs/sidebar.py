"""
Ask My Brain - 侧边栏模块

包含: API状态、模型切换、用户认证、文档管理、检索模式、索引管理、Credits
"""
import os
import streamlit as st
import config
from utils import ensure_dir, credits, api_manager
from document_loader import load_document, load_url
from model_registry import get_available_models, switch_model, get_current_model
from hybrid_retriever import hybrid_retriever
from retriever import clear_index
from app_helpers import get_index_count


def _build_index_task(task, documents):
    from async_worker import async_build_index
    return async_build_index(task, documents)


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Ask My Brain")
        st.caption("个人知识库问答助手")

        if config.DEV_MODE:
            st.markdown('<div class="status-card warn">⚡ 开发模式 (max_tokens=500)</div>', unsafe_allow_html=True)

        errors = config.validate_config()
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.markdown('<div class="status-card ready"><span style="color:#3fb950;">●</span> API 已配置</div>', unsafe_allow_html=True)

        if api_manager.is_tripped:
            st.markdown(f'<div class="status-card error">⚡ API 熔断中 ({api_manager.remaining_cooldown():.0f}s)</div>', unsafe_allow_html=True)

        # ---- 模型切换 ----
        st.markdown("### 🤖 模型选择")
        models = get_available_models()
        current = get_current_model()
        model_names = [m["name"] for m in models]
        current_idx = next((i for i, m in enumerate(models) if m["id"] == current["id"]), 0)
        selected_model = st.selectbox("选择模型", model_names, index=current_idx, label_visibility="collapsed")
        selected_idx = model_names.index(selected_model)
        if models[selected_idx]["id"] != current["id"]:
            switch_model(models[selected_idx]["id"])
            st.toast(f"已切换: {models[selected_idx]['name']}", icon="✅")
        st.caption(f"当前: {current['name']}")

        # ---- 多用户 ----
        st.markdown("### 👤 用户")
        try:
            from user_manager import is_authenticated, get_current_user, login, logout, register_user, init_default_user
            init_default_user()
            if is_authenticated():
                user = get_current_user()
                st.markdown(f'<div class="status-card ready"><span style="color:#3fb950;">●</span> {user.get("display_name", "")}</div>', unsafe_allow_html=True)
                if st.button("登出"):
                    logout()
                    st.rerun()
            else:
                with st.expander("登录", expanded=True):
                    login_user = st.text_input("用户名", key="login_user", label_visibility="collapsed", placeholder="用户名")
                    login_pass = st.text_input("密码", key="login_pass", type="password", label_visibility="collapsed", placeholder="密码")
                    lc1, lc2 = st.columns(2)
                    with lc1:
                        if st.button("登录", use_container_width=True, type="primary"):
                            if login_user and login_pass:
                                result = login(login_user, login_pass)
                                if result["success"]:
                                    st.rerun()
                                else:
                                    st.error(result["error"])
                    with lc2:
                        if st.button("注册", use_container_width=True):
                            if login_user and login_pass:
                                result = register_user(login_user, login_pass)
                                if result["success"]:
                                    st.toast("注册成功，请登录", icon="✅")
                                else:
                                    st.error(result["error"])
        except Exception as e:
            st.caption(f"用户模块: {e}")

        st.markdown("---")

        # ---- 文档管理 ----
        st.markdown("### 📄 文档管理")
        uploaded_files = st.file_uploader(
            "上传文件", type=["pdf", "md", "txt", "html", "htm", "png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True, label_visibility="collapsed"
        )

        url_col1, url_col2 = st.columns([3, 1])
        with url_col1:
            url_input = st.text_input("网页URL", placeholder="https://...", label_visibility="collapsed")
        with url_col2:
            if st.button("抓取", use_container_width=True) and url_input:
                with st.spinner("抓取中..."):
                    try:
                        doc = load_url(url_input)
                        st.session_state.documents_loaded.append(doc)
                        st.toast(f"已抓取: {doc['filename']}", icon="✅")
                    except Exception as e:
                        st.toast(f"失败: {e}", icon="❌")

        if uploaded_files and st.button("📥 导入文件", use_container_width=True, type="primary"):
            ensure_dir(config.DOCUMENTS_DIR)
            ok = 0
            for f in uploaded_files:
                p = os.path.join(config.DOCUMENTS_DIR, f.name)
                with open(p, "wb") as out:
                    out.write(f.getvalue())
                try:
                    st.session_state.documents_loaded.append(load_document(p))
                    ok += 1
                except Exception as e:
                    st.toast(f"{f.name}: {e}", icon="❌")
            st.toast(f"导入 {ok}/{len(uploaded_files)}", icon="✅")

        if st.session_state.documents_loaded:
            with st.expander(f"📂 已导入 ({len(st.session_state.documents_loaded)})"):
                for d in st.session_state.documents_loaded:
                    ico = {"pdf": "📕", "md": "📗", "txt": "📄", "html": "🌐", "url": "🔗",
                           "png": "🖼️", "jpg": "🖼️", "jpeg": "🖼️", "bmp": "🖼️", "tiff": "🖼️"}.get(d["source_type"], "📄")
                    st.caption(f"{ico} {d['filename']} — {d['char_count']:,}字")

        st.markdown("---")

        # ---- 混合检索 ----
        st.markdown("### 🔍 检索模式")
        hybrid_toggle = st.toggle("混合检索 (BM25 + 向量)", value=st.session_state.use_hybrid)
        st.session_state.use_hybrid = hybrid_toggle
        if hybrid_toggle:
            st.caption("BM25关键词 + 向量语义融合")
        else:
            st.caption("仅向量语义检索")

        st.markdown("---")

        # ---- 索引 ----
        st.markdown("### ⚡ 索引")
        idx_count = get_index_count()
        if st.session_state.index_built or idx_count > 0:
            st.markdown(f'<div class="status-card ready"><span style="color:#3fb950;">●</span> 就绪 · {idx_count} 条</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card empty"><span style="color:#8b949e;">○</span> 未构建</div>', unsafe_allow_html=True)

        bi_col1, bi_col2 = st.columns(2)
        with bi_col1:
            if st.button("🔨 构建索引", use_container_width=True, type="primary", disabled=not st.session_state.documents_loaded):
                from async_worker import submit_task
                import uuid
                task_id = f"idx_{uuid.uuid4().hex[:6]}"
                submit_task(task_id, "构建索引", _build_index_task, st.session_state.documents_loaded)
                st.session_state["_build_task_id"] = task_id
                st.toast("索引构建已启动（后台进行）", icon="🚀")
        with bi_col2:
            if st.button("🗑️ 清空", use_container_width=True):
                clear_index()
                st.session_state.index_built = False
                st.session_state.documents_loaded = []
                st.session_state.conversation = []
                hybrid_retriever._bm25 = None
                st.rerun()

        # 索引内文档管理（增量删除）
        if idx_count > 0:
            from retriever import list_indexed_documents, remove_document
            indexed_docs = list_indexed_documents()
            if indexed_docs:
                with st.expander(f"📋 索引文档 ({len(indexed_docs)})"):
                    for doc in indexed_docs:
                        dc1, dc2 = st.columns([4, 1])
                        with dc1:
                            st.caption(f"📄 {doc['filename']} — {doc['chunk_count']} chunks")
                        with dc2:
                            if st.button("🗑️", key=f"del_{doc['filename']}", help=f"删除 {doc['filename']}"):
                                remove_document(doc["filename"])
                                hybrid_retriever.remove_document_chunks(doc["filename"])
                                st.session_state.documents_loaded = [
                                    d for d in st.session_state.documents_loaded if d["filename"] != doc["filename"]
                                ]
                                if not st.session_state.documents_loaded:
                                    st.session_state.index_built = False
                                st.rerun()

        # 异步任务进度
        if "_build_task_id" in st.session_state:
            from async_worker import get_task
            task = get_task(st.session_state["_build_task_id"])
            if task:
                if task.status == "running":
                    st.progress(task.progress, text=task.message or "构建中...")
                elif task.status == "completed":
                    st.success(f"构建完成 ({task.elapsed}s)")
                    st.session_state.index_built = True
                    chunk_data = (task.result or {}).get("chunk_data", [])
                    if chunk_data:
                        hybrid_retriever.build_bm25_index(chunk_data)
                    del st.session_state["_build_task_id"]
                    st.rerun()
                elif task.status == "failed":
                    st.error(f"构建失败: {task.error}")
                    del st.session_state["_build_task_id"]
                elif task.status == "cancelled":
                    st.warning("已取消")
                    del st.session_state["_build_task_id"]

        # Credits
        pct = credits.get_usage_percentage()
        if pct > 0:
            bar_color = "#3fb950" if pct < 0.7 else "#d29922" if pct < 0.9 else "#f85149"
            st.markdown(f'<div class="status-card"><div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8b949e;"><span>Credits</span><span>{credits.used:,}/{credits.budget:,} ({pct*100:.1f}%)</span></div><div style="background:#21262d;border-radius:3px;overflow:hidden;height:5px;margin-top:0.3rem;"><div style="background:{bar_color};height:100%;width:{min(pct*100,100)}%;border-radius:3px;"></div></div></div>', unsafe_allow_html=True)
