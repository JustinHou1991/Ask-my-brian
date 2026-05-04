"""
Ask My Brain - 检索模块（含双策略 + 智能文档选择）

策略A: retrieve_top_k — Top-K相似chunk
策略B: retrieve_full_documents — 智能选择文档内容（<5k全文/5k-50k摘要+chunk/>50k窗口）
评估: evaluate_retrieval — 独立于LLM的检索质量评估（Recall@5, MRR）

依赖: chromadb, hashlib, config, utils
"""
import os
import hashlib
import threading
import chromadb

import config
from utils import estimate_tokens, count_tokens, log

_client = None
_collection = None
_documents_store = {}
_store_lock = threading.Lock()


def get_collection():
    global _client, _collection
    if _collection is None:
        try:
            _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            _collection = _client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"ChromaDB collection已加载: {config.COLLECTION_NAME}")
        except Exception as e:
            log.error(f"ChromaDB初始化失败: {e}")
            raise RuntimeError(f"ChromaDB初始化失败: {e}") from e
    return _collection


def store_full_document(filename: str, content: str):
    with _store_lock:
        _documents_store[filename] = content
    log.debug(f"完整文档已缓存: {filename} ({len(content)} 字符)")


def _chunk_id(text: str, source: str, index: int) -> str:
    h = hashlib.md5(f"{source}:{index}:{text[:100]}".encode()).hexdigest()[:12]
    return f"{source}_{index}_{h}"


def build_index(chunks: list, embeddings: list):
    collection = get_collection()
    ids = [_chunk_id(c["text"], c["metadata"].get("source", ""), c["metadata"].get("chunk_index", i)) for i, c in enumerate(chunks)]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        try:
            collection.upsert(ids=ids[i:end], embeddings=embeddings[i:end], documents=documents[i:end], metadatas=metadatas[i:end])
        except Exception as e:
            log.error(f"ChromaDB写入失败: {e}")
            raise RuntimeError(f"索引写入失败: {e}") from e
    log.info(f"索引构建完成: {len(chunks)} 个chunk写入ChromaDB")


def remove_document(filename: str) -> int:
    """从索引中移除指定文档的所有chunk，返回移除数量"""
    collection = get_collection()
    if collection.count() == 0:
        return 0
    try:
        before = collection.count()
        collection.delete(where={"source": filename})
        after = collection.count()
        removed = before - after
    except Exception as e:
        log.error(f"删除文档失败 [{filename}]: {e}")
        raise RuntimeError(f"删除文档失败: {e}") from e
    with _store_lock:
        _documents_store.pop(filename, None)
    log.info(f"文档已从索引移除: {filename} ({removed} chunks)")
    return removed


def list_indexed_documents() -> list:
    """列出索引中的所有文档及其chunk数量"""
    collection = get_collection()
    if collection.count() == 0:
        return []
    try:
        all_meta = collection.get(include=["metadatas"])
        doc_counts = {}
        for meta in all_meta["metadatas"]:
            src = meta.get("source", "unknown")
            doc_counts[src] = doc_counts.get(src, 0) + 1
        return [{"filename": f, "chunk_count": c} for f, c in sorted(doc_counts.items())]
    except Exception as e:
        log.error(f"列出文档失败: {e}")
        return []


def retrieve_top_k(query_embedding: list, k: int = None) -> dict:
    if k is None:
        k = config.TOP_K_RETRIEVAL
    collection = get_collection()
    if collection.count() == 0:
        return {"chunks": [], "sources": [], "total_tokens": 0}

    k = min(k, collection.count())
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"])
    except Exception as e:
        raise RuntimeError(f"检索失败: {e}") from e

    chunks, sources = [], []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
        })
        sources.append(results["metadatas"][0][i].get("source", "未知"))

    total_text = " ".join(c["text"] for c in chunks)
    log.info(f"策略A检索: {k} chunk, {estimate_tokens(total_text)} tokens")
    return {"chunks": chunks, "sources": sources, "total_tokens": estimate_tokens(total_text)}


def _smart_select_document(filename: str, doc_text: str, hit_chunk_texts: list) -> str:
    """
    智能文档选择：
    - < 5000字符 → 全文
    - 5000-50000字符 → 命中chunk + 前后各2000字符窗口
    - > 50000字符 → 只取命中chunk及其前后各2000字符窗口
    """
    doc_len = len(doc_text)

    if doc_len < 5000:
        return doc_text

    # 对于较长文档，提取命中chunk附近的上下文
    window = 2000
    selected_regions = []
    for chunk_text in hit_chunk_texts:
        pos = doc_text.find(chunk_text[:80])
        if pos == -1:
            pos = doc_text.find(chunk_text[:40])
        if pos >= 0:
            start = max(0, pos - window)
            end = min(doc_len, pos + len(chunk_text) + window)
            selected_regions.append((start, end))

    if not selected_regions:
        # 未找到位置，取前5000字符
        return doc_text[:5000]

    # 合并重叠区域
    selected_regions.sort()
    merged = [selected_regions[0]]
    for start, end in selected_regions[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts = [doc_text[s:e] for s, e in merged]
    result = "\n[...]\n".join(parts)

    # 如果合并后仍然太长，截断
    if len(result) > config.LONG_CONTEXT_MAX_CHARS:
        result = result[:config.LONG_CONTEXT_MAX_CHARS]

    return result


def retrieve_full_documents(query_embedding: list, threshold: float = 0.3) -> dict:
    """策略B：智能文档选择"""
    collection = get_collection()
    if collection.count() == 0:
        return {"chunks": [{"text": "", "metadata": {}}], "sources": [], "total_tokens": 0}

    n_results = min(collection.count(), 50)
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["metadatas", "distances", "documents"])
    except Exception as e:
        raise RuntimeError(f"检索失败: {e}") from e

    # 按文档分组命中的chunk
    hit_docs = {}  # filename -> [chunk_text, ...]
    for i in range(len(results["ids"][0])):
        score = 1 - results["distances"][0][i]
        if score >= threshold:
            source = results["metadatas"][0][i].get("source", "")
            if source:
                if source not in hit_docs:
                    hit_docs[source] = []
                hit_docs[source].append(results["documents"][0][i])

    if not hit_docs:
        return {"chunks": [{"text": "", "metadata": {}}], "sources": [], "total_tokens": 0}

    # 智能选择文档内容
    full_texts = []
    sources = list(hit_docs.keys())
    total_chars = 0

    for src in sources:
        with _store_lock:
            doc_text = _documents_store.get(src)
        if doc_text:
            selected = _smart_select_document(src, doc_text, hit_docs.get(src, []))

            if total_chars + len(selected) > config.LONG_CONTEXT_MAX_CHARS:
                remaining = config.LONG_CONTEXT_MAX_CHARS - total_chars
                if remaining > 0:
                    selected = selected[:remaining]
                else:
                    break

            full_texts.append(f"【文档: {src}】\n{selected}")
            total_chars += len(selected)

    combined_text = "\n\n---\n\n".join(full_texts)
    total_tok = estimate_tokens(combined_text)
    log.info(f"策略B检索: {len(sources)} 篇文档, {total_tok} tokens (智能选择)")
    return {
        "chunks": [{"text": combined_text, "metadata": {"source": ", ".join(sources)}}],
        "sources": sources,
        "total_tokens": total_tok,
    }


def evaluate_retrieval(test_queries: list) -> dict:
    """
    独立于LLM的检索质量评估。

    Args:
        test_queries: [{"query": str, "expected_sources": [str]}]

    Returns:
        dict: {"recall_at_5": float, "mrr": float, "details": list}
    """
    collection = get_collection()
    if collection.count() == 0:
        return {"recall_at_5": 0.0, "mrr": 0.0, "details": []}

    from embedder import embed_query

    details = []
    recall_hits = 0
    mrr_sum = 0.0

    for tq in test_queries:
        query = tq["query"]
        expected = set(tq.get("expected_sources", []))

        q_emb = embed_query(query)
        results = retrieve_top_k(q_emb, k=5)
        retrieved_sources = [c["metadata"].get("source", "") for c in results["chunks"]]

        # Recall@5
        hit = bool(expected & set(retrieved_sources))
        if hit:
            recall_hits += 1

        # MRR
        rank = 0
        for j, src in enumerate(retrieved_sources):
            if src in expected:
                rank = 1.0 / (j + 1)
                break
        mrr_sum += rank

        details.append({
            "query": query,
            "retrieved": retrieved_sources,
            "expected": list(expected),
            "recall_hit": hit,
            "reciprocal_rank": rank,
        })

    n = len(test_queries)
    log.info(f"检索评估完成: Recall@5={recall_hits/n:.2f}, MRR={mrr_sum/n:.2f}")
    return {
        "recall_at_5": round(recall_hits / n, 4) if n else 0.0,
        "mrr": round(mrr_sum / n, 4) if n else 0.0,
        "details": details,
    }


def get_collection_count() -> int:
    """获取索引数量。快速路径：DB目录不存在直接返回0，不初始化ChromaDB。"""
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        return 0
    try:
        return get_collection().count()
    except Exception:
        return 0


def clear_index():
    global _collection, _documents_store
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    with _store_lock:
        _documents_store.clear()
    log.info("索引已清空")
