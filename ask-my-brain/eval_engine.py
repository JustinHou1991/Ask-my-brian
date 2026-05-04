"""
Ask My Brain - 评测引擎

提供单题评测和批量评测功能，自动对比策略A/B的响应时间和上下文Token。
含LLM-as-Judge质量评估（相关性/完整性/忠实度）。
评测结果持久化为JSON文件，含汇总统计。

输入: 问题列表 (list[dict])
输出: {"metadata": dict, "statistics": dict, "results": list}

依赖: time, datetime, json, config, generator, retriever, embedder, utils
"""
import json as _json
import re
import time
from datetime import datetime

import config
from generator import generate_answer_strategy_a, generate_answer_strategy_b, generate_with_mimo
from retriever import retrieve_top_k, retrieve_full_documents
from embedder import embed_query
from utils import estimate_tokens, save_json, log

# ========== LLM-as-Judge 质量评估 ==========

JUDGE_SYSTEM_PROMPT = """你是一个严格的答案质量评估专家。请对以下答案在三个维度上评分（每项1-5分）：
1. 相关性 (Relevance): 答案是否准确回答了问题？(5=完全切题, 1=完全无关)
2. 完整性 (Completeness): 答案是否涵盖了参考资料中的所有相关信息？(5=全面覆盖, 1=几乎没覆盖)
3. 忠实度 (Faithfulness): 答案是否忠实于参考资料，无编造内容？(5=完全忠实, 1=大量编造)

严格按JSON格式输出：{"relevance": int, "completeness": int, "faithfulness": int, "reasoning": "简要说明"}"""

JUDGE_USER_TEMPLATE = """## 用户问题
{question}

## 参考资料
{context}

## 待评估答案
{answer}

请评估答案质量。"""


def judge_answer(question: str, context: str, answer: str) -> dict:
    """
    LLM-as-Judge：用MiMo评估答案质量。

    Returns:
        dict: {"relevance": 1-5, "completeness": 1-5, "faithfulness": 1-5, "reasoning": str}
    """
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question, context=context[:4000], answer=answer[:3000]
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        result = generate_with_mimo(messages, max_tokens=256, temperature=0.1)
        text = result["answer"].strip()
        # 提取JSON块：支持```json...```或裸JSON
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if m:
                text = m.group(0)
        scores = _json.loads(text.strip())
        return {
            "relevance": max(1, min(5, int(scores.get("relevance", 3)))),
            "completeness": max(1, min(5, int(scores.get("completeness", 3)))),
            "faithfulness": max(1, min(5, int(scores.get("faithfulness", 3)))),
            "reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        log.warning(f"答案质量评估失败: {e}")
        return {"relevance": 0, "completeness": 0, "faithfulness": 0, "reasoning": f"评估失败: {e}"}


def evaluate_single(question: str, reference_answer: str = None) -> dict:
    """
    对单个问题执行双策略评测，测量响应时间和上下文Token。

    Args:
        question: 测试问题
        reference_answer: 可选的参考答案

    Returns:
        dict: 含策略A/B的回答、来源、Token数、响应时间
    """
    log.info(f"评测开始: {question[:50]}...")

    query_embedding = embed_query(question)

    # 策略A
    start_a = time.time()
    chunks_a = retrieve_top_k(query_embedding)
    result_a = generate_answer_strategy_a(question, chunks_a)
    time_a = round(time.time() - start_a, 2)

    # 策略B
    start_b = time.time()
    docs_b = retrieve_full_documents(query_embedding)
    result_b = generate_answer_strategy_b(question, docs_b)
    time_b = round(time.time() - start_b, 2)

    # LLM-as-Judge质量评估
    quality_a, quality_b = {}, {}
    if config.JUDGE_ENABLED:
        context_a = " ".join(c["text"] for c in chunks_a.get("chunks", []))
        quality_a = judge_answer(question, context_a, result_a["answer"])
        context_b = docs_b["chunks"][0]["text"] if docs_b.get("chunks") else ""
        quality_b = judge_answer(question, context_b, result_b["answer"])

    log.info(f"评测完成: A={time_a}s/{result_a['context_tokens']}tok, B={time_b}s/{result_b['context_tokens']}tok")
    out = {
        "question": question,
        "reference_answer": reference_answer,
        "strategy_a": {
            "answer": result_a["answer"],
            "sources": result_a["sources"],
            "context_tokens": result_a["context_tokens"],
            "response_time": time_a,
        },
        "strategy_b": {
            "answer": result_b["answer"],
            "sources": result_b["sources"],
            "context_tokens": result_b["context_tokens"],
            "response_time": time_b,
        },
        "timestamp": datetime.now().isoformat(),
    }
    if quality_a:
        out["strategy_a"]["quality"] = quality_a
    if quality_b:
        out["strategy_b"]["quality"] = quality_b
    return out


def run_evaluation(questions: list, output_file: str = None) -> dict:
    """
    批量运行评测，逐题执行双策略对比。

    Args:
        questions: 问题列表，每个元素需含 "question" 字段
        output_file: 输出文件路径，默认使用config.EVAL_RESULTS_PATH

    Returns:
        dict: {"metadata", "statistics", "results"}
    """
    if output_file is None:
        output_file = config.EVAL_RESULTS_PATH

    log.info(f"批量评测开始: {len(questions)} 道题")
    results = []
    for i, q in enumerate(questions):
        log.info(f"评测进度: {i+1}/{len(questions)} - {q['question'][:50]}...")
        try:
            result = evaluate_single(q["question"], q.get("reference_answer"))
            result["category"] = q.get("category", "")
            results.append(result)
        except Exception as e:
            log.error(f"评测失败 (Q{i+1}): {e}")
            results.append({
                "question": q["question"],
                "category": q.get("category", ""),
                "error": str(e),
            })

    stats = compute_stats(results)
    output = {
        "metadata": {
            "total_questions": len(questions),
            "model": config.MIMO_MODEL,
            "dev_mode": config.DEV_MODE,
            "timestamp": datetime.now().isoformat(),
        },
        "statistics": stats,
        "results": results,
    }
    save_json(output, output_file)
    log.info(f"批量评测完成，结果已保存: {output_file}")
    return output


def _avg_quality(valid: list, strategy_key: str) -> dict:
    """计算某策略的平均质量分（兼容无quality字段的旧结果）"""
    dims = ["relevance", "completeness", "faithfulness"]
    scores = {d: [] for d in dims}
    for r in valid:
        q = r.get(strategy_key, {}).get("quality", {})
        for d in dims:
            v = q.get(d, 0)
            if v > 0:
                scores[d].append(v)
    return {f"avg_{d}": round(sum(s) / len(s), 2) if s else 0.0 for d, s in scores.items()}


def compute_stats(results: list) -> dict:
    """
    计算评测统计数据（平均/最大/最小响应时间、平均Token数、平均质量分）。

    Args:
        results: 评测结果列表

    Returns:
        dict: 汇总统计
    """
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "无有效评测结果"}

    a_times = [r["strategy_a"]["response_time"] for r in valid]
    b_times = [r["strategy_b"]["response_time"] for r in valid]
    a_tokens = [r["strategy_a"]["context_tokens"] for r in valid]
    b_tokens = [r["strategy_b"]["context_tokens"] for r in valid]

    stats = {
        "strategy_a": {
            "avg_response_time": round(sum(a_times) / len(a_times), 2),
            "max_response_time": round(max(a_times), 2),
            "min_response_time": round(min(a_times), 2),
            "avg_context_tokens": round(sum(a_tokens) / len(a_tokens)),
        },
        "strategy_b": {
            "avg_response_time": round(sum(b_times) / len(b_times), 2),
            "max_response_time": round(max(b_times), 2),
            "min_response_time": round(min(b_times), 2),
            "avg_context_tokens": round(sum(b_tokens) / len(b_tokens)),
        },
        "valid_questions": len(valid),
        "failed_questions": len(results) - len(valid),
    }

    # 质量评估统计（兼容无quality字段的旧结果）
    stats["strategy_a"].update(_avg_quality(valid, "strategy_a"))
    stats["strategy_b"].update(_avg_quality(valid, "strategy_b"))

    return stats
