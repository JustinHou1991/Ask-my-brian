"""
Ask My Brain - 文档分块模块

功能:
- 语言检测（中/英文），自动选择分隔符
- 常规分块（策略A）+ 粗粒度长上下文分块（策略B，用tiktoken精确控制token）
- 每个chunk保留完整元数据

输入: document dict
输出: list[{"text": str, "metadata": dict}]

依赖: langchain_text_splitters, tiktoken, config, utils
"""
import config
from utils import count_tokens, log

# 懒加载，避免启动时导入langchain（耗时8秒）
_Splitter = None

def _get_splitter():
    global _Splitter
    if _Splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _Splitter = RecursiveCharacterTextSplitter
    return _Splitter

# 中文专用分隔符
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]
# 英文默认分隔符
ENGLISH_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def detect_language(text: str) -> str:
    """
    通过Unicode范围检测文本主要语言。
    中文字符占比超过30%视为中文文档。

    Returns:
        str: "chinese" 或 "english"
    """
    if not text:
        return "english"
    sample = text[:2000]
    cn = sum(1 for c in sample if "一" <= c <= "鿿")
    return "chinese" if cn / max(len(sample), 1) > 0.3 else "english"


def chunk_document(doc: dict) -> list:
    """
    将单个文档分块，根据语言自动选择分隔符。

    Args:
        doc: 文档字典，需含 "content" 和 "filename"

    Returns:
        list: chunk列表
    """
    text = doc["content"]
    if not text.strip():
        log.warning(f"文档内容为空: {doc.get('filename')}")
        return []

    lang = detect_language(text)
    separators = CHINESE_SEPARATORS if lang == "chinese" else ENGLISH_SEPARATORS

    splitter = _get_splitter()(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=separators,
    )

    split_texts = splitter.split_text(text)
    chunks = []
    search_pos = 0

    for i, chunk_text in enumerate(split_texts):
        anchor = chunk_text[:80]
        start = text.find(anchor, search_pos)
        if start == -1:
            start = text.find(anchor)
        if start == -1:
            start = search_pos
        end = start + len(chunk_text)
        search_pos = max(start + 1, end - config.CHUNK_OVERLAP)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": doc["filename"],
                "chunk_index": i,
                "char_start": start,
                "char_end": end,
                "language": lang,
            },
        })

    log.debug(f"分块完成: {doc.get('filename')}, {len(chunks)} chunk, 语言={lang}")
    return chunks


def chunk_for_long_context(doc: dict) -> list:
    """
    粗粒度切分，用于策略B。
    用tiktoken精确控制token数，上限为MAX_TOKENS_LONG_CONTEXT。

    Args:
        doc: 文档字典

    Returns:
        list: 大段chunk列表
    """
    text = doc["content"]
    token_count = count_tokens(text)
    max_tok = config.MAX_TOKENS_LONG_CONTEXT

    if token_count <= max_tok:
        return [{
            "text": text,
            "metadata": {
                "source": doc["filename"],
                "chunk_index": 0,
                "char_start": 0,
                "char_end": len(text),
                "is_long_context": True,
                "token_count": token_count,
            },
        }]

    # 按token切分（复用utils中的tokenizer）
    from utils import _get_tokenizer
    enc = _get_tokenizer()
    if enc is None:
        raise RuntimeError("tiktoken不可用，无法进行长上下文分块")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + max_tok, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        char_start = len(enc.decode(tokens[:start]))
        char_end = len(enc.decode(tokens[:end]))

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": doc["filename"],
                "chunk_index": idx,
                "char_start": char_start,
                "char_end": char_end,
                "is_long_context": True,
                "token_count": end - start,
            },
        })
        start = end
        idx += 1

    log.debug(f"长上下文分块: {doc.get('filename')}, {len(chunks)} 段, {token_count} tokens")
    return chunks


def chunk_documents(documents: list) -> list:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    log.info(f"批量分块完成: {len(documents)} 篇 -> {len(all_chunks)} chunk")
    return all_chunks
