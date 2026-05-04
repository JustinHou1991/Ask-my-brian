"""
Ask My Brain - 评测可视化标签页

调用 eval_visualizer 渲染 Plotly 图表。
"""
import streamlit as st
from eval_visualizer import is_available, render_eval_dashboard, render_radar_chart


def render_viz_tab():
    if not is_available():
        st.warning("请先安装plotly: `pip install plotly`")
        return
    render_eval_dashboard(st)
    st.markdown("---")
    render_radar_chart(st)
