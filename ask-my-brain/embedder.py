"""
Ask My Brain - 向量化模块

使用 sentence-transformers 加载本地Embedding模型。
国内环境自动使用 HuggingFace 镜像加速下载。
模型已缓存时自动切换离线模式，避免联网检查卡顿。

依赖: sentence_transformers, os, config, utils
"""
import os
from pathlib import Path
from functools import lru_cache
import config
from utils import log

_model = None
_sentence_transformer_cls = None

# ========== 国内镜像加速 + 离线优化 ==========
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_CACHE_NAME = "models--sentence-transformers--all-MiniLM-L6-v2"
_model_cached = (HF_CACHE_DIR / MODEL_CACHE_NAME).exists()

if _model_cached:
    # 模型已缓存，强制离线模式，跳过联网检查（秒加载）
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    log.info("Embedding模型已缓存，启用离线模式")
elif not os.environ.get("HF_ENDPOINT"):
    # 模型未缓存，使用镜像加速下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    log.info(f"HuggingFace镜像: {os.environ['HF_ENDPOINT']}")

# 懒加载 sentence_transformers（模块导入耗时10秒，延迟到首次使用）
_sentence_transformer_cls = None


def get_model():
    """
    懒加载Embedding模型（单例）。
    已缓存时离线加载（<1秒），未缓存时通过hf-mirror.com下载。
    """
    global _model, _sentence_transformer_cls
    if _model is None:
        log.info(f"正在加载Embedding模型: {config.EMBEDDING_MODEL}")
        if not _model_cached:
            log.info("首次使用需下载模型（约90MB），请稍候...")
        try:
            if _sentence_transformer_cls is None:
                from sentence_transformers import SentenceTransformer
                _sentence_transformer_cls = SentenceTransformer
            _model = _sentence_transformer_cls(config.EMBEDDING_MODEL, device="cpu")
            dim = _model.get_sentence_embedding_dimension()
            config.EMBEDDING_DIMENSION = dim
            log.info(f"Embedding模型加载完成，维度: {dim}")
        except Exception as e:
            log.error(f"Embedding模型加载失败: {e}")
            log.error(
                "如网络问题导致下载失败，请尝试:\n"
                "  1. 设置代理: set HTTPS_PROXY=http://127.0.0.1:7890\n"
                "  2. 手动下载: pip install -U sentence-transformers\n"
                "  3. 使用国内源: set HF_ENDPOINT=https://hf-mirror.com"
            )
            raise RuntimeError(f"Embedding模型加载失败: {e}") from e
    return _model


def embed_chunks(chunks: list, batch_size: int = 64) -> list:
    """
    将chunk列表向量化。批处理避免内存溢出。

    Args:
        chunks: chunk列表，每个含 "text" 字段
        batch_size: 每批大小，默认64（比32快一倍）

    Returns:
        list[list[float]]: 向量列表
    """
    model = get_model()
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        try:
            embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.extend(embs.tolist())
        except Exception as e:
            log.error(f"Embedding失败 (batch {i//batch_size+1}): {e}")
            raise RuntimeError(f"Embedding计算失败: {e}") from e

    log.info(f"向量化完成: {total} chunk -> {len(all_embeddings)} 向量")
    return all_embeddings


@lru_cache(maxsize=128)
def _embed_query_cached(query: str) -> tuple:
    """内部缓存函数，返回不可变tuple以支持LRU缓存"""
    model = get_model()
    try:
        return tuple(model.encode([query], normalize_embeddings=True)[0].tolist())
    except Exception as e:
        raise RuntimeError(f"查询向量化失败: {e}") from e


def embed_query(query: str) -> list:
    """单条查询向量化（带LRU缓存，128条上限约393KB）"""
    return list(_embed_query_cached(query))


def get_embed_cache_stats() -> dict:
    """获取embedding缓存统计"""
    info = _embed_query_cached.cache_info()
    return {"hits": info.hits, "misses": info.misses, "size": info.currsize, "maxsize": info.maxsize}


def clear_embed_cache():
    """清空embedding缓存"""
    _embed_query_cached.cache_clear()
