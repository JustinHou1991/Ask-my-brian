"""
Ask My Brain - Streamlit主入口

四个标签页: 对话问答 / 双策略评测 / 评测报告 / 评测可视化
支持: 模型热切换、混合检索、对话记忆、多用户、异步索引、多模态OCR
"""
import streamlit as st
from app_helpers import init_state, get_index_count
from app_tabs.sidebar import render_sidebar
from app_tabs.chat_tab import render_chat_tab
from app_tabs.eval_tab import render_eval_tab
from app_tabs.report_tab import render_report_tab
from app_tabs.viz_tab import render_viz_tab

# ========== 页面配置 ==========
st.set_page_config(page_title="Ask My Brain", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ========== CSS ==========
st.markdown("""
<style>
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1117 0%, #161b22 100%); }
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: #e6edf3; }
.status-card { background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%); border: 1px solid #30363d; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
.status-card.ready { border-left: 4px solid #3fb950; }
.status-card.empty { border-left: 4px solid #8b949e; }
.status-card.warn { border-left: 4px solid #d29922; }
.status-card.error { border-left: 4px solid #f85149; }
.strategy-header-a { background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%); border: 1px solid #1f6feb; border-radius: 10px; padding: 0.6rem 1rem; margin-bottom: 0.6rem; }
.strategy-header-b { background: linear-gradient(135deg, #1e1a2e 0%, #0d1117 100%); border: 1px solid #bc8cff; border-radius: 10px; padding: 0.6rem 1rem; margin-bottom: 0.6rem; }
.strategy-badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-a { background: rgba(31,111,235,0.2); color: #58a6ff; border: 1px solid rgba(31,111,235,0.3); }
.badge-b { background: rgba(188,140,255,0.2); color: #bc8cff; border: 1px solid rgba(188,140,255,0.3); }
.source-tag { display: inline-block; background: rgba(56,139,253,0.1); border: 1px solid rgba(56,139,253,0.2); border-radius: 6px; padding: 0.1rem 0.4rem; font-size: 0.75rem; color: #58a6ff; margin: 0.1rem; }
.sources-row { margin-top: 0.4rem; display: flex; flex-wrap: wrap; gap: 0.2rem; }
.metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 0.8rem; text-align: center; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #e6edf3; }
.metric-label { font-size: 0.75rem; color: #8b949e; margin-top: 0.15rem; }
.compare-bar { background: #21262d; border-radius: 4px; overflow: hidden; height: 6px; margin: 0.2rem 0; }
.bar-a { background: linear-gradient(90deg, #1f6feb, #58a6ff); height: 100%; border-radius: 4px; }
.bar-b { background: linear-gradient(90deg, #8957e5, #bc8cff); height: 100%; border-radius: 4px; }
.category-pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px; font-size: 0.7rem; font-weight: 500; margin-right: 0.2rem; }
div[data-testid="stChatMessage"] { border-radius: 10px; }
.think-block { background: rgba(110,118,129,0.06); border-radius: 8px; padding: 0.3rem 0.6rem; margin: 0.3rem 0; }
.think-block[open] { border-left: 2px solid #30363d; }
.think-block summary { padding: 0.2rem 0; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
button[data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; padding: 0.7rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ========== 初始化 ==========
init_state()
render_sidebar()

# ========== 主区域 ==========
tab_chat, tab_eval, tab_report, tab_viz = st.tabs(["💬 智能问答", "📊 双策略评测", "📈 评测报告", "📉 评测可视化"])

has_index = st.session_state.index_built or get_index_count() > 0

with tab_chat:
    render_chat_tab(has_index)

with tab_eval:
    render_eval_tab(has_index)

with tab_report:
    render_report_tab()

with tab_viz:
    render_viz_tab()
