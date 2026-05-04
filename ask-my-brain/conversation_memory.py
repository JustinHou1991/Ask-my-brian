"""
Ask My Brain - 对话记忆模块

SQLite长期存储对话历史，支持跨会话上下文管理。
- 按session_id隔离对话
- 自动摘要旧消息（滑动窗口）
- 持久化到SQLite数据库

依赖: sqlite3, json, config, utils
"""
import os
import json
import sqlite3
from datetime import datetime
from utils import log, ensure_dir, count_tokens
import config

DB_PATH = os.path.join(config.PROJECT_ROOT, "conversation_memory.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库表"""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
        """)
        conn.commit()
        log.info("对话记忆数据库已初始化")
    finally:
        conn.close()


def create_session(title: str = "", metadata: dict = None) -> str:
    """创建新会话，返回session_id"""
    import uuid
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, title or f"会话 {now[:10]}", now, now, json.dumps(metadata or {}))
        )
        conn.commit()
        log.info(f"会话已创建: {session_id}")
        return session_id
    finally:
        conn.close()


def add_message(session_id: str, role: str, content: str, metadata: dict = None):
    """添加消息到会话"""
    now = datetime.now().isoformat()
    tokens = count_tokens(content)
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tokens, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, tokens, now, json.dumps(metadata or {}))
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        conn.commit()
    finally:
        conn.close()


def get_messages(session_id: str, limit: int = None) -> list:
    """获取会话消息"""
    conn = _get_conn()
    try:
        if limit:
            rows = conn.execute(
                "SELECT role, content, tokens, created_at, metadata FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                "SELECT role, content, tokens, created_at, metadata FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,)
            ).fetchall()
        return [{"role": r["role"], "content": r["content"], "tokens": r["tokens"],
                 "created_at": r["created_at"], "metadata": json.loads(r["metadata"])} for r in rows]
    finally:
        conn.close()


def get_conversation_context(session_id: str, max_tokens: int = None) -> list:
    """
    获取适合作为LLM上下文的消息列表。
    自动截断早期消息以满足token限制。
    """
    if max_tokens is None:
        max_tokens = config.MAX_HISTORY_TOKENS
    messages = get_messages(session_id)
    result = []
    total_tokens = 0
    for msg in reversed(messages):
        if total_tokens + msg["tokens"] > max_tokens:
            break
        result.insert(0, {"role": msg["role"], "content": msg["content"]})
        total_tokens += msg["tokens"]
    return result


def list_sessions(limit: int = 50, offset: int = 0) -> list:
    """列出所有会话"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT s.session_id, s.title, s.created_at, s.updated_at,
                      COUNT(m.id) as message_count, SUM(m.tokens) as total_tokens
               FROM sessions s LEFT JOIN messages m ON s.session_id = m.session_id
               GROUP BY s.session_id ORDER BY s.updated_at DESC LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
        return [{"session_id": r["session_id"], "title": r["title"],
                 "created_at": r["created_at"], "updated_at": r["updated_at"],
                 "message_count": r["message_count"], "total_tokens": r["total_tokens"] or 0} for r in rows]
    finally:
        conn.close()


def delete_session(session_id: str):
    """删除会话及其所有消息"""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        log.info(f"会话已删除: {session_id}")
    finally:
        conn.close()


def search_messages(keyword: str, limit: int = 20) -> list:
    """跨会话搜索消息内容"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT m.session_id, m.role, m.content, m.created_at, s.title
               FROM messages m JOIN sessions s ON m.session_id = s.session_id
               WHERE m.content LIKE ? ORDER BY m.created_at DESC LIMIT ?""",
            (f"%{keyword}%", limit)
        ).fetchall()
        return [{"session_id": r["session_id"], "role": r["role"], "content": r["content"],
                 "created_at": r["created_at"], "session_title": r["title"]} for r in rows]
    finally:
        conn.close()


# 初始化
init_db()
