"""
Ask My Brain - 全局配置模块

定义所有可配置参数，包括API、Embedding、分块、检索、Credits预算、路径等。
启动时通过 validate_config() 校验关键配置。

依赖: os, dotenv
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ========== 项目根目录 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ========== MiMo API 配置 ==========
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")

# ========== Embedding 模型 ==========
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = None  # 运行时自动检测

# ========== 文档分块参数 ==========
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ========== 检索参数 ==========
TOP_K_RETRIEVAL = 5
LONG_CONTEXT_MAX_CHARS = 600000
MAX_TOKENS_LONG_CONTEXT = 900000  # 1M窗口留10%给prompt模板

# ========== ChromaDB ==========
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "ask_my_brain"

# ========== 文档存储 ==========
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "documents")

# ========== 评测结果 ==========
EVAL_RESULTS_PATH = os.path.join(PROJECT_ROOT, "eval_results.json")
TEST_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "test_questions.json")

# ========== 日志 ==========
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "ask_my_brain.log")

# ========== LLM 参数 ==========
MAX_TOKENS = 4096
TEMPERATURE_A = 0.3   # 策略A
TEMPERATURE_B = 0.15  # 策略B
MAX_HISTORY_TOKENS = 4000

# ========== 开发模式 ==========
DEV_MODE = os.environ.get("ASK_MY_BRAIN_DEV", "0") == "1"
DEV_MAX_TOKENS = 500
DEV_TEMPERATURE = 0.1

# ========== 多用户支持 ==========
MULTI_USER_ENABLED = os.environ.get("MULTI_USER_ENABLED", "0") == "1"
USER_DATA_ROOT = os.path.join(PROJECT_ROOT, "user_data")

# ========== 混合检索 ==========
HYBRID_RETRIEVAL_ENABLED = os.environ.get("HYBRID_RETRIEVAL", "1") == "1"
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "1.0"))
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", "1.0"))
RRF_K = int(os.environ.get("RRF_K", "60"))

# ========== API 调用保护（APICallManager使用） ==========
API_CIRCUIT_BREAKER_THRESHOLD = 5  # 连续失败N次触发熔断
API_CIRCUIT_BREAKER_COOLDOWN = 60  # 熔断冷却秒数

# ========== Credits 预算 ==========
CREDITS_BUDGET = 200_000_000  # 2亿 Credits 总预算
CREDITS_WARNING_THRESHOLD = 0.7   # 70%时Streamlit告警
CREDITS_CRITICAL_THRESHOLD = 0.9  # 90%时停止自动API调用

# ========== 费率（per token） ==========
CREDITS_INPUT_RATE = 1
CREDITS_OUTPUT_RATE = 3

# ========== LLM-as-Judge 质量评估 ==========
JUDGE_ENABLED = os.environ.get("JUDGE_ENABLED", "1") == "1"


def get_max_tokens() -> int:
    return DEV_MAX_TOKENS if DEV_MODE else MAX_TOKENS


def get_temperature(strategy: str = "A") -> float:
    if DEV_MODE:
        return DEV_TEMPERATURE
    return TEMPERATURE_A if strategy == "A" else TEMPERATURE_B


def validate_config() -> list:
    """
    启动时校验关键配置，返回友好错误信息列表。
    """
    errors = []
    if not MIMO_API_KEY or MIMO_API_KEY == "your_api_key_here":
        errors.append(
            "MIMO_API_KEY 未设置\n"
            "请执行: export MIMO_API_KEY='你的密钥'\n"
            "获取密钥: https://platform.xiaomimimo.com → 控制台 → API Key"
        )
    return errors
