"""
Ask My Brain - 评测可视化模块

使用Plotly生成交互式评测图表，支持双策略对比、分类统计、雷达图。

依赖: streamlit, json, plotly, config, utils
"""
import json
import os
import config
from utils import log

_plotly_available = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _plotly_available = True
except ImportError:
    log.warning("plotly未安装，评测可视化不可用。执行: pip install plotly")


def is_available() -> bool:
    return _plotly_available


def _load_results(filepath: str = None) -> dict:
    if filepath is None:
        filepath = config.EVAL_RESULTS_PATH
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def render_eval_dashboard(st, results: dict = None):
    """渲染完整评测仪表盘"""
    if not _plotly_available:
        st.warning("请先安装plotly: `pip install plotly`")
        return
    if results is None:
        results = _load_results()
    if not results:
        st.info("暂无评测结果，请先运行批量评测")
        return

    valid = [r for r in results.get("results", []) if "error" not in r]
    if not valid:
        st.warning("无有效评测结果")
        return

    stats = results.get("statistics", {})
    a_stats = stats.get("strategy_a", {})
    b_stats = stats.get("strategy_b", {})

    _render_overview_metrics(st, results, valid)
    _render_comparison_charts(st, a_stats, b_stats)
    _render_quality_comparison(st, a_stats, b_stats)
    _render_category_analysis(st, valid)
    _render_timeline_analysis(st, valid)


def _render_overview_metrics(st, results: dict, valid: list):
    """概览指标卡片"""
    meta = results.get("metadata", {})
    stats = results.get("statistics", {})
    st.markdown("#### 评测概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总题数", meta.get("total_questions", len(results.get("results", []))))
    c2.metric("有效题数", len(valid))
    c3.metric("失败题数", len(results.get("results", [])) - len(valid))
    c4.metric("模型", meta.get("model", "未知"))


def _render_comparison_charts(st, a_stats: dict, b_stats: dict):
    """双策略对比图表"""
    st.markdown("#### 双策略对比")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="策略A (传统RAG)",
            x=["平均响应(s)", "最大响应(s)", "最小响应(s)"],
            y=[a_stats.get("avg_response_time", 0), a_stats.get("max_response_time", 0), a_stats.get("min_response_time", 0)],
            marker_color="#1f6feb",
        ))
        fig.add_trace(go.Bar(
            name="策略B (Long Context)",
            x=["平均响应(s)", "最大响应(s)", "最小响应(s)"],
            y=[b_stats.get("avg_response_time", 0), b_stats.get("max_response_time", 0), b_stats.get("min_response_time", 0)],
            marker_color="#8957e5",
        ))
        fig.update_layout(title="响应时间对比", barmode="group", height=350,
                         template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="策略A",
            x=["平均Token"],
            y=[a_stats.get("avg_context_tokens", 0)],
            marker_color="#1f6feb",
        ))
        fig2.add_trace(go.Bar(
            name="策略B",
            x=["平均Token"],
            y=[b_stats.get("avg_context_tokens", 0)],
            marker_color="#8957e5",
        ))
        fig2.update_layout(title="上下文Token对比", barmode="group", height=350,
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)


def _render_quality_comparison(st, a_stats: dict, b_stats: dict):
    """LLM-as-Judge质量维度对比（仅当quality数据存在时渲染）"""
    dims = ["avg_relevance", "avg_completeness", "avg_faithfulness"]
    labels = ["相关性", "完整性", "忠实度"]
    a_vals = [a_stats.get(d, 0) for d in dims]
    b_vals = [b_stats.get(d, 0) for d in dims]

    if max(a_vals + b_vals) == 0:
        return

    st.markdown("#### 答案质量对比 (LLM-as-Judge)")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="策略A", x=labels, y=a_vals, marker_color="#1f6feb"))
    fig.add_trace(go.Bar(name="策略B", x=labels, y=b_vals, marker_color="#8957e5"))
    fig.update_layout(
        title="质量维度评分 (1-5分)", barmode="group", height=350,
        yaxis=dict(range=[0, 5]),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_category_analysis(st, valid: list):
    """按分类统计"""
    categories = sorted(set(r.get("category", "") for r in valid if r.get("category")))
    if not categories:
        return

    st.markdown("#### 分类统计")
    cat_stats = {}
    for cat in categories:
        cat_results = [r for r in valid if r.get("category") == cat]
        a_times = [r["strategy_a"]["response_time"] for r in cat_results]
        b_times = [r["strategy_b"]["response_time"] for r in cat_results]
        a_tokens = [r["strategy_a"]["context_tokens"] for r in cat_results]
        b_tokens = [r["strategy_b"]["context_tokens"] for r in cat_results]
        cat_stats[cat] = {
            "count": len(cat_results),
            "a_avg_time": sum(a_times) / len(a_times) if a_times else 0,
            "b_avg_time": sum(b_times) / len(b_times) if b_times else 0,
            "a_avg_tokens": sum(a_tokens) / len(a_tokens) if a_tokens else 0,
            "b_avg_tokens": sum(b_tokens) / len(b_tokens) if b_tokens else 0,
        }

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="策略A", x=categories,
            y=[cat_stats[c]["a_avg_time"] for c in categories],
            marker_color="#1f6feb",
        ))
        fig.add_trace(go.Bar(
            name="策略B", x=categories,
            y=[cat_stats[c]["b_avg_time"] for c in categories],
            marker_color="#8957e5",
        ))
        fig.update_layout(title="各分类平均响应时间", barmode="group", height=350,
                         template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[cat_stats[c]["a_avg_time"] for c in categories],
            y=[cat_stats[c]["a_avg_tokens"] for c in categories],
            mode="markers+text", text=categories, textposition="top center",
            marker=dict(size=[cat_stats[c]["count"] * 3 + 10 for c in categories], color="#1f6feb"),
            name="策略A",
        ))
        fig2.add_trace(go.Scatter(
            x=[cat_stats[c]["b_avg_time"] for c in categories],
            y=[cat_stats[c]["b_avg_tokens"] for c in categories],
            mode="markers+text", text=categories, textposition="top center",
            marker=dict(size=[cat_stats[c]["count"] * 3 + 10 for c in categories], color="#8957e5"),
            name="策略B",
        ))
        fig2.update_layout(title="响应时间 vs Token (气泡大小=题数)", height=350,
                          xaxis_title="平均响应时间(s)", yaxis_title="平均Token",
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("分类详情表"):
        for cat in categories:
            s = cat_stats[cat]
            st.markdown(
                f"**{cat}** ({s['count']}题) — "
                f"A: {s['a_avg_time']:.1f}s / {s['a_avg_tokens']:.0f}tok | "
                f"B: {s['b_avg_time']:.1f}s / {s['b_avg_tokens']:.0f}tok"
            )


def _render_timeline_analysis(st, valid: list):
    """逐题响应时间分布"""
    st.markdown("#### 逐题响应时间")
    questions = [r["question"][:25] + "..." for r in valid]
    a_times = [r["strategy_a"]["response_time"] for r in valid]
    b_times = [r["strategy_b"]["response_time"] for r in valid]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=a_times, mode="lines+markers", name="策略A",
        line=dict(color="#1f6feb", width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        y=b_times, mode="lines+markers", name="策略B",
        line=dict(color="#8957e5", width=2),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title="逐题响应时间", height=400, xaxis_title="题目序号", yaxis_title="响应时间(s)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(st, results: dict = None):
    """渲染雷达图（各维度对比）"""
    if not _plotly_available:
        return
    if results is None:
        results = _load_results()
    if not results:
        return

    valid = [r for r in results.get("results", []) if "error" not in r]
    if not valid:
        return

    categories = sorted(set(r.get("category", "") for r in valid if r.get("category")))
    if not categories:
        return

    a_scores, b_scores = [], []
    for cat in categories:
        cat_results = [r for r in valid if r.get("category") == cat]
        a_avg = sum(r["strategy_a"]["response_time"] for r in cat_results) / len(cat_results)
        b_avg = sum(r["strategy_b"]["response_time"] for r in cat_results) / len(cat_results)
        max_time = max(a_avg, b_avg, 0.01)
        a_scores.append(round((1 - a_avg / max_time) * 100, 1))
        b_scores.append(round((1 - b_avg / max_time) * 100, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=a_scores + [a_scores[0]], theta=categories + [categories[0]],
                                  fill="toself", name="策略A", line_color="#1f6feb"))
    fig.add_trace(go.Scatterpolar(r=b_scores + [b_scores[0]], theta=categories + [categories[0]],
                                  fill="toself", name="策略B", line_color="#8957e5"))
    fig.update_layout(
        title="策略表现雷达图", height=450, polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
