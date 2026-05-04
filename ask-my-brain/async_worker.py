"""
Ask My Brain - 异步任务执行模块

后台异步构建索引，不阻塞UI。提供进度追踪和取消功能。

依赖: threading, concurrent.futures, utils
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from utils import log

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="async_worker")
_tasks = {}
_lock = threading.Lock()


class AsyncTask:
    """异步任务封装"""

    def __init__(self, task_id: str, name: str):
        self.task_id = task_id
        self.name = name
        self.status = "pending"  # pending, running, completed, failed, cancelled
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self._cancel_flag = threading.Event()

    @property
    def elapsed(self) -> float:
        if self.start_time:
            end = self.end_time or time.time()
            return round(end - self.start_time, 1)
        return 0.0

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag.is_set()

    def cancel(self):
        self._cancel_flag.set()
        self.status = "cancelled"


def submit_task(task_id: str, name: str, func, *args, **kwargs) -> str:
    """提交异步任务"""
    task = AsyncTask(task_id, name)
    with _lock:
        _tasks[task_id] = task

    def _run():
        task.status = "running"
        task.start_time = time.time()
        try:
            result = func(task, *args, **kwargs)
            if not task.is_cancelled:
                task.result = result
                task.status = "completed"
                task.progress = 1.0
        except Exception as e:
            if not task.is_cancelled:
                task.error = str(e)
                task.status = "failed"
                log.error(f"异步任务失败 [{name}]: {e}")
        finally:
            task.end_time = time.time()

    future = _executor.submit(_run)
    with _lock:
        _tasks[task_id]._future = future
    log.info(f"异步任务已提交: {name} ({task_id})")
    return task_id


def get_task(task_id: str) -> AsyncTask:
    """获取任务状态"""
    with _lock:
        return _tasks.get(task_id)


def cancel_task(task_id: str) -> bool:
    """取消任务"""
    with _lock:
        task = _tasks.get(task_id)
    if task and task.status in ("pending", "running"):
        task.cancel()
        log.info(f"任务已取消: {task.name}")
        return True
    return False


def list_tasks() -> list:
    """列出所有任务"""
    with _lock:
        return [
            {"task_id": t.task_id, "name": t.name, "status": t.status,
             "progress": t.progress, "message": t.message, "elapsed": t.elapsed}
            for t in _tasks.values()
        ]


def cleanup_tasks(keep_recent: int = 10):
    """清理已完成的任务，保留最近N个"""
    with _lock:
        completed = [(tid, t) for tid, t in _tasks.items()
                     if t.status in ("completed", "failed", "cancelled")]
        if len(completed) > keep_recent:
            completed.sort(key=lambda x: x[1].end_time or 0, reverse=True)
            for tid, _ in completed[keep_recent:]:
                del _tasks[tid]


def update_progress(task_id: str, progress: float, message: str = ""):
    """更新任务进度"""
    with _lock:
        task = _tasks.get(task_id)
    if task:
        task.progress = min(max(progress, 0.0), 1.0)
        if message:
            task.message = message


# ========== 便捷函数：异步索引构建 ==========

def async_build_index(task: AsyncTask, documents: list):
    """
    异步构建向量索引（供submit_task调用）。

    Args:
        task: AsyncTask实例（自动传入）
        documents: 文档列表

    Returns:
        dict: {"chunks": int, "documents": int}
    """
    from chunker import chunk_document
    from embedder import embed_chunks
    from retriever import build_index, store_full_document

    if task.is_cancelled:
        return None

    total_docs = len(documents)
    all_chunks = []
    for i, doc in enumerate(documents):
        if task.is_cancelled:
            return None
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        task.progress = (i + 1) / total_docs * 0.3
        task.message = f"分块中 [{i+1}/{total_docs}]: {doc.get('filename', '')}"
        update_progress(task.task_id, task.progress, task.message)

    for doc in documents:
        store_full_document(doc["filename"], doc["content"])

    if task.is_cancelled:
        return None
    task.message = "向量化中..."
    update_progress(task.task_id, 0.4, task.message)
    embeddings = embed_chunks(all_chunks)

    if task.is_cancelled:
        return None
    task.message = "写入数据库..."
    update_progress(task.task_id, 0.8, task.message)
    build_index(all_chunks, embeddings)

    task.progress = 1.0
    task.message = f"完成: {len(all_chunks)} chunks, {total_docs} 文档"
    log.info(f"异步索引构建完成: {len(all_chunks)} chunks")
    return {"chunks": len(all_chunks), "documents": total_docs, "chunk_data": all_chunks}
