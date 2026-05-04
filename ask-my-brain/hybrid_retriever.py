"""
Ask My Brain - 混合检索模块

BM25关键词检索 + 向量语义检索融合，使用Reciprocal Rank Fusion (RRF)提升召回率。

依赖: rank_bm25, jieba, retriever, embedder, config, utils
"""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from utils import log
import config

# 懒加载jieba
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba
        _jieba = jieba
        _jieba.setLogLevel(20)
    return _jieba


def _tokenize_chinese(text: str) -> list:
    """中文分词（jieba）"""
    jieba = _get_jieba()
    text = re.sub(r'[^\w一-鿿]+', ' ', text)
    return [w for w in jieba.cut(text) if w.strip()]


def _tokenize_english(text: str) -> list:
    """英文分词"""
    text = re.sub(r'[^\w]+', ' ', text.lower())
    return text.split()


def tokenize(text: str, language: str = "auto") -> list:
    """自动选择分词策略"""
    if language == "auto":
        cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
        language = "chinese" if cn_chars / max(len(text), 1) > 0.3 else "english"
    return _tokenize_chinese(text) if language == "chinese" else _tokenize_english(text)


class HybridRetriever:
    """混合检索器：BM25 + 向量检索 + RRF融合"""

    def __init__(self):
        self._bm25 = None
        self._bm25_docs = []
        self._bm25_tokens = []
        self._bm25_language = "auto"

    def build_bm25_index(self, chunks: list, language: str = "auto"):
        """从chunk列表构建BM25索引"""
        self._bm25_docs = chunks
        self._bm25_language = language
        self._bm25_tokens = [tokenize(c["text"], language) for c in chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)
        log.info(f"BM25索引构建完成: {len(chunks)} 文档, 语言={language}")

    def remove_document_chunks(self, filename: str):
        """从BM25索引中移除指定文档的chunk"""
        if self._bm25 is None:
            return
        new_docs = [c for c in self._bm25_docs if c.get("metadata", {}).get("source") != filename]
        if len(new_docs) < len(self._bm25_docs):
            removed = len(self._bm25_docs) - len(new_docs)
            self.build_bm25_index(new_docs, self._bm25_language)
            log.info(f"BM25索引已更新: 移除 {filename} ({removed} chunks)")

    def bm25_search(self, query: str, top_k: int = None) -> list:
        """BM25关键词检索"""
        if self._bm25 is None:
            return []
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL * 2
        query_tokens = tokenize(query, self._bm25_language)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"index": int(idx), "score": float(scores[idx]), "chunk": self._bm25_docs[idx]}
            for idx in top_indices if scores[idx] > 0
        ]

    def vector_search(self, query_embedding: list, top_k: int = None) -> list:
        """向量语义检索"""
        from retriever import retrieve_top_k
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL * 2
        results = retrieve_top_k(query_embedding, k=top_k)
        return [
            {"index": i, "score": c.get("score", 0), "chunk": c}
            for i, c in enumerate(results["chunks"])
        ]

    def hybrid_search(self, query: str, query_embedding: list, top_k: int = None,
                      bm25_weight: float = 1.0, vector_weight: float = 1.0,
                      rrf_k: int = 60) -> dict:
        """
        混合检索：RRF融合BM25和向量检索结果。

        Args:
            query: 查询文本
            query_embedding: 查询向量
            top_k: 返回结果数
            bm25_weight: BM25权重
            vector_weight: 向量检索权重
            rrf_k: RRF参数（默认60）

        Returns:
            dict: {"chunks": list, "sources": list, "total_tokens": int, "fusion_method": str}
        """
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        from utils import estimate_tokens

        bm25_results = self.bm25_search(query)
        vector_results = self.vector_search(query_embedding)

        def _rrf_key(chunk: dict) -> str:
            """用source+chunk_index作为唯一键，避免text[:100]碰撞"""
            meta = chunk.get("metadata", {})
            return f"{meta.get('source', '')}:{meta.get('chunk_index', 0)}"

        rrf_scores = {}
        for rank, r in enumerate(bm25_results):
            key = _rrf_key(r["chunk"])
            rrf_scores[key] = rrf_scores.get(key, 0) + bm25_weight / (rrf_k + rank + 1)
        for rank, r in enumerate(vector_results):
            key = _rrf_key(r["chunk"])
            rrf_scores[key] = rrf_scores.get(key, 0) + vector_weight / (rrf_k + rank + 1)

        all_results = {}
        for r in bm25_results:
            key = _rrf_key(r["chunk"])
            all_results[key] = r["chunk"]
        for r in vector_results:
            key = _rrf_key(r["chunk"])
            all_results[key] = r["chunk"]

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]
        chunks = [all_results[k] for k in sorted_keys]
        sources = list(dict.fromkeys(c.get("metadata", {}).get("source", "") for c in chunks if c.get("metadata", {}).get("source")))
        total_text = " ".join(c["text"] for c in chunks)

        log.info(f"混合检索: BM25={len(bm25_results)}, Vector={len(vector_results)}, 融合后={len(chunks)}")
        return {
            "chunks": chunks,
            "sources": sources,
            "total_tokens": estimate_tokens(total_text),
            "fusion_method": "RRF",
            "bm25_count": len(bm25_results),
            "vector_count": len(vector_results),
        }

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None


# 全局实例
hybrid_retriever = HybridRetriever()
