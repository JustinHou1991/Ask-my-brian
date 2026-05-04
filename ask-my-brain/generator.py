"""
Ask My Brain - LLM问答生成模块（核心）

通过MiMo API生成回答，含：
- 按规范设计的System Prompt（策略A精准/策略B深度）
- API错误分类处理（429/400/401/5xx）
- 对话历史滑动窗口管理
- 策略A/B差异化temperature
- Credits自动累计

输入: 查询 + 检索结果 + 可选对话历史
输出: {"answer", "strategy", "sources", "context_tokens", "model", "input_tokens", "output_tokens", "latency_ms"}

依赖: openai, config, utils
"""
import copy
import time
from typing import Generator
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError, BadRequestError, AuthenticationError

import config
from utils import log, api_manager, credits, count_tokens
from model_registry import get_model_config, supports_reasoning

def _parse_retry_after(e) -> int:
    """从RateLimitError中安全提取Retry-After秒数，默认5s"""
    try:
        resp = getattr(e, 'response', None)
        if resp:
            headers = getattr(resp, 'headers', {})
            val = headers.get('retry-after', '')
            if val and val.isdigit():
                return int(val)
    except Exception:
        pass
    return 5


_client = None
_client_base_url = None


def get_client() -> OpenAI:
    global _client, _client_base_url
    model_cfg = get_model_config()
    base_url = model_cfg["base_url"]
    if _client is None or _client_base_url != base_url:
        _client = OpenAI(api_key=model_cfg["api_key"], base_url=base_url)
        _client_base_url = base_url
        log.info(f"API客户端已初始化: {base_url} (模型: {model_cfg['model']})")
    return _client


# ========== Prompt 定义 ==========

SYSTEM_PROMPT_A = """你是一个个人知识库助手。你的任务是基于用户提供的文档片段，准确回答用户的问题。

## 规则
1. 只基于下面提供的【参考资料】回答，不要使用你自己的知识
2. 如果参考资料不足以回答问题，明确说"根据现有资料无法回答此问题"
3. 每个关键陈述必须标注来源，格式为 [来源: 文件名, 第X段]
4. 回答结构清晰：先给结论，再给论据，最后给引用列表
5. 如果用户问题涉及时间线或因果关系，请按逻辑顺序组织回答
6. 回答中使用Markdown格式增强可读性"""

SYSTEM_PROMPT_B = """你是一个个人知识库助手，拥有超长上下文理解能力。你的任务是基于用户提供的完整文档，准确回答用户的问题。

## 重要提示
你收到的全部文档中，可能只有部分与用户问题相关。请：
1. 首先快速扫描所有文档，识别与问题相关的段落
2. 只基于相关段落构建回答
3. 明确忽略与问题无关的文档内容
4. 在你的回答中标注使用了哪些文档

## 规则
1. 以下是你可用的全部文档。请通读后找出与用户问题相关的信息
2. 文档之间可能存在关联（比如一篇是另一篇的引用、补充或反驳），请综合考虑
3. 如果文档内存在互相矛盾的信息，请指出矛盾并分别说明
4. 每个关键陈述必须标注来源，格式为 [来源: 文件名, 原文段落摘录]
5. 回答结构：先给结论 → 详细论述 → 引用列表
6. 使用Markdown格式增强可读性"""

DEV_SYSTEM_PROMPT_A = "你是知识库问答助手。基于以下检索片段回答问题，标注[来源:文件名]。无相关信息则说明。"
DEV_SYSTEM_PROMPT_B = "你是知识库助手。基于完整文档回答问题，标注[来源:文件名]。无相关信息则说明。"


# ========== 对话历史管理 ==========

def trim_conversation_history(history: list, max_tokens: int = None) -> list:
    """
    滑动窗口对话历史管理。
    保留最近消息，总token不超过max_tokens。超出时对早期对话生成摘要。

    Args:
        history: 对话历史 [{"role": str, "content": str}]
        max_tokens: token上限

    Returns:
        list: 修剪后的对话历史
    """
    if not history:
        return []
    if max_tokens is None:
        max_tokens = config.MAX_HISTORY_TOKENS

    total = sum(count_tokens(m["content"]) for m in history)
    if total <= max_tokens:
        return history

    # 从前面开始丢弃，直到总token在限制内
    trimmed = list(history)
    while trimmed and sum(count_tokens(m["content"]) for m in trimmed) > max_tokens:
        trimmed.pop(0)

    log.debug(f"对话历史修剪: {len(history)} -> {len(trimmed)} 条")
    return trimmed


# ========== 核心调用 ==========

def generate_with_mimo(messages: list, max_tokens: int = None, temperature: float = None, strategy: str = None) -> dict:
    """
    封装MiMo API调用，含错误分类处理和Credits累计。

    Args:
        messages: OpenAI格式的消息列表
        max_tokens: 最大输出token
        temperature: 温度参数
        strategy: "A" 或 "B"，影响默认temperature

    Returns:
        dict: {"answer": str, "input_tokens": int, "output_tokens": int, "latency_ms": int}

    Raises:
        RuntimeError: API调用最终失败
    """
    if not api_manager.can_call():
        remaining = api_manager.remaining_cooldown()
        raise RuntimeError(f"API熔断中，请等待 {remaining:.0f}s 后重试（连续失败 {api_manager.failure_count} 次）")

    client = get_client()
    model_cfg = get_model_config(strategy or "A")
    if max_tokens is None:
        max_tokens = model_cfg["max_tokens"]
    if temperature is None:
        temperature = model_cfg["temperature"]
    model_name = model_cfg["model"]

    start_time = time.time()

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            answer = response.choices[0].message.content
            usage = response.usage
            in_tok = usage.prompt_tokens if usage else count_tokens(" ".join(m["content"] for m in messages))
            out_tok = usage.completion_tokens if usage else count_tokens(answer)

            api_manager.record_success()
            credits.add(in_tok, out_tok)

            log.info(f"API调用成功: {in_tok}+{out_tok} tokens, {latency_ms}ms")
            return {
                "answer": answer,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": latency_ms,
            }

        except RateLimitError as e:
            # 429: 等待Retry-After
            retry_after = _parse_retry_after(e)
            log.warning(f"429速率限制, 等待 {retry_after}s (attempt {attempt+1})")
            api_manager.record_failure()
            time.sleep(retry_after)

        except BadRequestError as e:
            # 400: prompt太长，截断后重试（deep copy避免污染原始messages）
            if "maximum context length" in str(e).lower() or "too long" in str(e).lower():
                max_tokens = int(max_tokens * 0.8)
                log.warning(f"400 prompt过长, 截断max_tokens到 {max_tokens}")
                messages = copy.deepcopy(messages)
                # 截断所有user消息（策略B的长上下文在user消息中）
                for msg in reversed(messages):
                    if msg["role"] == "user" and len(msg["content"]) > 500:
                        msg["content"] = msg["content"][:int(len(msg["content"]) * 0.8)]
                        break
                api_manager.record_failure()
                continue
            raise RuntimeError(f"API请求错误: {e}") from e

        except AuthenticationError as e:
            # 401/403: 立即停止
            api_manager.record_failure()
            raise RuntimeError(f"API认证失败，请检查 MIMO_API_KEY: {e}") from e

        except (APITimeoutError, APIConnectionError) as e:
            # 网络问题，指数退避
            wait = 2 ** attempt
            log.warning(f"网络错误, 等待 {wait}s (attempt {attempt+1}): {e}")
            api_manager.record_failure()
            time.sleep(wait)

        except APIError as e:
            # 5xx: 指数退避
            if hasattr(e, 'status_code') and e.status_code >= 500:
                wait = 2 ** attempt
                log.warning(f"服务端错误 {e.status_code}, 等待 {wait}s")
                api_manager.record_failure()
                time.sleep(wait)
            else:
                raise RuntimeError(f"API错误: {e}") from e

        except Exception as e:
            api_manager.record_failure()
            raise RuntimeError(f"LLM调用异常: {e}") from e

    api_manager.record_failure()
    raise RuntimeError("API调用失败，已重试3次")


# ========== 策略A生成 ==========

def generate_answer_strategy_a(query: str, retrieved_chunks: dict, conversation_history: list = None) -> dict:
    """
    策略A：基于Top-K检索片段生成回答。

    Returns:
        dict: {"answer", "strategy", "sources", "context_tokens", "model", "input_tokens", "output_tokens", "latency_ms"}
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks["chunks"], 1):
        source = chunk["metadata"].get("source", "未知")
        context_parts.append(f"【参考资料 - {source}, 第{i}段】\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    if config.DEV_MODE:
        system_prompt = DEV_SYSTEM_PROMPT_A
        user_message = f"参考资料：\n{context}\n\n问题：{query}"
    else:
        system_prompt = SYSTEM_PROMPT_A
        user_message = f"## 参考资料\n{context}\n\n## 用户问题\n{query}"

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(trim_conversation_history(conversation_history))
    messages.append({"role": "user", "content": user_message})

    result = generate_with_mimo(messages, strategy="A")
    result.update({
        "strategy": "A (Traditional RAG)",
        "sources": retrieved_chunks["sources"],
        "context_tokens": retrieved_chunks["total_tokens"],
        "model": get_model_config()["model"],
    })
    return result


# ========== 策略B生成 ==========

def generate_answer_strategy_b(query: str, full_docs: dict, conversation_history: list = None) -> dict:
    """
    策略B：基于完整源文档生成回答（Long Context）。

    Returns:
        dict: {"answer", "strategy", "sources", "context_tokens", "model", "input_tokens", "output_tokens", "latency_ms"}
    """
    context = full_docs["chunks"][0]["text"]
    sources = full_docs["sources"]

    if config.DEV_MODE:
        system_prompt = DEV_SYSTEM_PROMPT_B
        user_message = f"文档：\n{context}\n\n问题：{query}"
    else:
        system_prompt = SYSTEM_PROMPT_B
        user_message = f"## 全部文档\n{context}\n\n## 用户问题\n{query}"

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(trim_conversation_history(conversation_history))
    messages.append({"role": "user", "content": user_message})

    result = generate_with_mimo(messages, strategy="B")
    result.update({
        "strategy": "B (Long Context)",
        "sources": sources,
        "context_tokens": full_docs["total_tokens"],
        "model": get_model_config()["model"],
    })
    return result


# ========== 流式生成 ==========

def generate_streaming(messages: list, max_tokens: int = None, temperature: float = None, strategy: str = None) -> Generator:
    """
    流式调用MiMo API，逐token yield。

    Yields:
        ("reasoning", str): 推理过程片段
        ("content", str): 最终回答片段
        ("done", dict): 完成统计 {"input_tokens", "output_tokens", "latency_ms"}
    """
    if not api_manager.can_call():
        remaining = api_manager.remaining_cooldown()
        raise RuntimeError(f"API熔断中，请等待 {remaining:.0f}s 后重试")

    client = get_client()
    model_cfg = get_model_config(strategy or "A")
    if max_tokens is None:
        max_tokens = model_cfg["max_tokens"]
    if temperature is None:
        temperature = model_cfg["temperature"]
    model_name = model_cfg["model"]

    start_time = time.time()
    in_tok = count_tokens(" ".join(m["content"] for m in messages))

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        out_tok = 0
        for chunk in stream:
            if not chunk.choices:
                # usage统计在最后一个chunk
                if chunk.usage:
                    in_tok = chunk.usage.prompt_tokens
                    out_tok = chunk.usage.completion_tokens
                continue

            delta = chunk.choices[0].delta
            # 推理模型：先输出reasoning_content，再输出content
            if supports_reasoning() and hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield ("reasoning", delta.reasoning_content)
            if delta.content:
                yield ("content", delta.content)

        latency_ms = int((time.time() - start_time) * 1000)
        api_manager.record_success()
        credits.add(in_tok, out_tok)
        log.info(f"流式API完成: {in_tok}+{out_tok} tokens, {latency_ms}ms")
        yield ("done", {"input_tokens": in_tok, "output_tokens": out_tok, "latency_ms": latency_ms})

    except RateLimitError as e:
        retry_after = _parse_retry_after(e)
        api_manager.record_failure()
        raise RuntimeError(f"429速率限制，请等待 {retry_after}s") from e

    except AuthenticationError as e:
        api_manager.record_failure()
        raise RuntimeError(f"API认证失败，请检查 MIMO_API_KEY") from e

    except BadRequestError as e:
        api_manager.record_failure()
        raise RuntimeError(f"API请求错误: {e}") from e

    except (APITimeoutError, APIConnectionError) as e:
        api_manager.record_failure()
        raise RuntimeError(f"网络错误: {e}") from e

    except APIError as e:
        api_manager.record_failure()
        raise RuntimeError(f"API错误: {e}") from e

    except Exception as e:
        api_manager.record_failure()
        raise RuntimeError(f"LLM调用异常: {e}") from e


def generate_answer_strategy_a_stream(query: str, retrieved_chunks: dict, conversation_history: list = None) -> Generator:
    """策略A流式版本：基于Top-K检索片段生成回答。"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks["chunks"], 1):
        source = chunk["metadata"].get("source", "未知")
        context_parts.append(f"【参考资料 - {source}, 第{i}段】\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    if config.DEV_MODE:
        system_prompt = DEV_SYSTEM_PROMPT_A
        user_message = f"参考资料：\n{context}\n\n问题：{query}"
    else:
        system_prompt = SYSTEM_PROMPT_A
        user_message = f"## 参考资料\n{context}\n\n## 用户问题\n{query}"

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(trim_conversation_history(conversation_history))
    messages.append({"role": "user", "content": user_message})

    yield from generate_streaming(messages, strategy="A")


def generate_answer_strategy_b_stream(query: str, full_docs: dict, conversation_history: list = None) -> Generator:
    """策略B流式版本：基于完整文档生成回答。"""
    context = full_docs["chunks"][0]["text"]
    sources = full_docs["sources"]

    if config.DEV_MODE:
        system_prompt = DEV_SYSTEM_PROMPT_B
        user_message = f"文档：\n{context}\n\n问题：{query}"
    else:
        system_prompt = SYSTEM_PROMPT_B
        user_message = f"## 全部文档\n{context}\n\n## 用户问题\n{query}"

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(trim_conversation_history(conversation_history))
    messages.append({"role": "user", "content": user_message})

    yield from generate_streaming(messages, strategy="B")
