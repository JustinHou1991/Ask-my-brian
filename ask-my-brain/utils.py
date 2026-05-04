"""
Ask My Brain - 工具函数模块

提供: 日志、APICallManager熔断器、Credits计数器、Token估算、JSON读写。

依赖: os, json, hashlib, logging, time, threading, tiktoken
"""
import os
import json
import hashlib
import logging
import time
import threading
from datetime import datetime

import tiktoken

import config

# ========== 目录 / 哈希 ==========

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ========== 日志 ==========

def setup_logger(name: str = "ask_my_brain") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    ensure_dir(config.LOG_DIR)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = setup_logger()


def file_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def chunk_hash(text: str, source: str, index: int) -> str:
    return hashlib.md5(f"{source}:{index}:{text[:100]}".encode()).hexdigest()[:12]


# ========== Token 计算 ==========

_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """用tiktoken精确计算token数"""
    enc = _get_tokenizer()
    if enc:
        return len(enc.encode(text))
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """粗略估算（tiktoken不可用时的fallback）"""
    if not text:
        return 0
    cn = sum(1 for c in text if "一" <= c <= "鿿")
    return int(cn / 1.5 + (len(text) - cn) / 4)


def format_sources(sources: list) -> str:
    if not sources:
        return "无来源"
    seen, result = set(), []
    for s in sources:
        name = s if isinstance(s, str) else s.get("source", "未知")
        if name not in seen:
            seen.add(name)
            result.append(f"- {name}")
    return "\n".join(result)


# ========== JSON ==========

def save_json(data, filepath: str):
    ensure_dir(os.path.dirname(filepath) or ".")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ========== APICallManager（全局熔断器） ==========

class APICallManager:
    """
    API调用熔断器。
    连续失败N次 → 进入冷却期 → 冷却期间拦截所有调用。
    """

    def __init__(self, threshold: int = None, cooldown: int = None):
        self._threshold = threshold or config.API_CIRCUIT_BREAKER_THRESHOLD
        self._cooldown = cooldown or config.API_CIRCUIT_BREAKER_COOLDOWN
        self._failures = 0
        self._tripped_at = 0.0
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self._failures = 0

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._tripped_at = time.time()
                log.warning(f"APICallManager 熔断! 连续失败 {self._failures} 次, 冷却 {self._cooldown}s")

    def can_call(self) -> bool:
        with self._lock:
            if self._failures < self._threshold:
                return True
            elapsed = time.time() - self._tripped_at
            if elapsed >= self._cooldown:
                self._failures = 0
                log.info("APICallManager 冷却结束，恢复调用")
                return True
            return False

    def remaining_cooldown(self) -> float:
        with self._lock:
            if self._failures < self._threshold:
                return 0.0
            elapsed = time.time() - self._tripped_at
            return max(0.0, self._cooldown - elapsed)

    @property
    def is_tripped(self) -> bool:
        return not self.can_call()

    @property
    def failure_count(self) -> int:
        return self._failures


# 全局实例
api_manager = APICallManager()


# ========== Credits 计数器 ==========

class CreditsCounter:
    """全局Credits消耗计数器（线程安全）"""

    def __init__(self):
        self._used = 0
        self._lock = threading.Lock()

    def add(self, input_tokens: int, output_tokens: int):
        cost = input_tokens * config.CREDITS_INPUT_RATE + output_tokens * config.CREDITS_OUTPUT_RATE
        with self._lock:
            self._used += cost

    @property
    def used(self) -> int:
        return self._used

    @property
    def budget(self) -> int:
        return config.CREDITS_BUDGET

    def get_usage_percentage(self) -> float:
        return self._used / config.CREDITS_BUDGET if config.CREDITS_BUDGET > 0 else 0.0

    def is_budget_warning(self) -> bool:
        return self.get_usage_percentage() >= config.CREDITS_WARNING_THRESHOLD

    def is_budget_critical(self) -> bool:
        return self.get_usage_percentage() >= config.CREDITS_CRITICAL_THRESHOLD


# 全局实例
credits = CreditsCounter()
