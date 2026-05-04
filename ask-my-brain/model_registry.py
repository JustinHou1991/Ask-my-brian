"""
Ask My Brain - 模型热切换模块

支持多个LLM模型动态切换，预设常用模型配置。
运行时切换模型无需重启应用。

依赖: os, config
"""
import os
import config
from utils import log

# ========== 模型预设 ==========
MODEL_PRESETS = [
    {
        "id": "mimo-v2.5-pro",
        "name": "MiMo v2.5 Pro (小米)",
        "provider": "xiaomi",
        "base_url": "https://api.xiaomimimo.com/v1",
        "max_tokens": 4096,
        "temperature_a": 0.3,
        "temperature_b": 0.15,
        "supports_streaming": True,
        "supports_reasoning": True,
    },
    {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 4096,
        "temperature_a": 0.3,
        "temperature_b": 0.15,
        "supports_streaming": True,
        "supports_reasoning": False,
    },
    {
        "id": "deepseek-reasoner",
        "name": "DeepSeek Reasoner",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 4096,
        "temperature_a": 0.3,
        "temperature_b": 0.15,
        "supports_streaming": True,
        "supports_reasoning": True,
    },
    {
        "id": "qwen-plus",
        "name": "通义千问 Plus",
        "provider": "alibaba",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens": 4096,
        "temperature_a": 0.3,
        "temperature_b": 0.15,
        "supports_streaming": True,
        "supports_reasoning": False,
    },
    {
        "id": "glm-4-flash",
        "name": "智谱 GLM-4-Flash",
        "provider": "zhipu",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "max_tokens": 4096,
        "temperature_a": 0.3,
        "temperature_b": 0.15,
        "supports_streaming": True,
        "supports_reasoning": False,
    },
]

# 当前活跃模型配置（模块级全局状态）
_current_model = None


def get_current_model() -> dict:
    """获取当前模型配置"""
    global _current_model
    if _current_model is None:
        _current_model = {
            "id": config.MIMO_MODEL,
            "name": config.MIMO_MODEL,
            "provider": "custom",
            "base_url": config.MIMO_BASE_URL,
            "max_tokens": config.MAX_TOKENS,
            "temperature_a": config.TEMPERATURE_A,
            "temperature_b": config.TEMPERATURE_B,
            "supports_streaming": True,
            "supports_reasoning": True,
        }
    return _current_model


def switch_model(model_id: str) -> bool:
    """
    切换当前使用的模型。

    Args:
        model_id: 模型ID（预设ID或自定义模型名）

    Returns:
        bool: 切换是否成功
    """
    global _current_model

    preset = next((m for m in MODEL_PRESETS if m["id"] == model_id), None)
    if preset:
        _current_model = preset.copy()
    else:
        _current_model = {
            "id": model_id,
            "name": model_id,
            "provider": "custom",
            "base_url": config.MIMO_BASE_URL,
            "max_tokens": config.MAX_TOKENS,
            "temperature_a": config.TEMPERATURE_A,
            "temperature_b": config.TEMPERATURE_B,
            "supports_streaming": True,
            "supports_reasoning": True,
        }

    log.info(f"模型已切换: {_current_model['name']} ({_current_model['id']})")
    return True


def get_model_config(strategy: str = "A") -> dict:
    """
    获取当前模型的完整调用配置。

    Args:
        strategy: "A" 或 "B"

    Returns:
        dict: {"model", "base_url", "api_key", "max_tokens", "temperature"}
    """
    m = get_current_model()
    temp = m["temperature_a"] if strategy == "A" else m["temperature_b"]
    if config.DEV_MODE:
        temp = config.DEV_TEMPERATURE

    return {
        "model": m["id"],
        "base_url": m["base_url"],
        "api_key": config.MIMO_API_KEY,
        "max_tokens": config.DEV_MAX_TOKENS if config.DEV_MODE else m["max_tokens"],
        "temperature": temp,
    }


def get_available_models() -> list:
    """获取所有可选模型列表"""
    return MODEL_PRESETS


def supports_reasoning() -> bool:
    """当前模型是否支持推理过程输出"""
    return get_current_model().get("supports_reasoning", False)


def supports_streaming() -> bool:
    """当前模型是否支持流式输出"""
    return get_current_model().get("supports_streaming", True)
