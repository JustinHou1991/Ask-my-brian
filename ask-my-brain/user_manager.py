"""
Ask My Brain - 多用户支持模块

用户登录、ChromaDB按用户隔离、独立文档空间。

依赖: os, json, hashlib, uuid, chromadb, config, utils
"""
import os
import hmac
import json
import hashlib
import uuid
from datetime import datetime
import chromadb
from config import PROJECT_ROOT, COLLECTION_NAME
from utils import log, ensure_dir

USERS_FILE = os.path.join(PROJECT_ROOT, "users.json")
USER_DATA_ROOT = os.path.join(PROJECT_ROOT, "user_data")
_current_user = None


def _hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 with random salt (OWASP 2023: 260k iterations)"""
    pepper = os.environ.get("PASSWORD_PEPPER", "")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", (password + pepper).encode(), salt, 260_000)
    return f"pbkdf2:260000:{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """验证密码，兼容旧SHA-256格式自动升级"""
    pepper = os.environ.get("PASSWORD_PEPPER", "")
    if stored_hash.startswith("pbkdf2:"):
        parts = stored_hash.split(":")
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
        dk = hashlib.pbkdf2_hmac("sha256", (password + pepper).encode(), salt, iterations)
        return hmac.compare_digest(dk, expected)
    else:
        # 兼容旧版无盐SHA-256
        return hmac.compare_digest(
            hashlib.sha256(password.encode()).hexdigest().encode(),
            stored_hash.encode()
        )


def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def register_user(username: str, password: str, display_name: str = "") -> dict:
    """注册新用户"""
    users = _load_users()
    if username in users:
        return {"success": False, "error": "用户名已存在"}
    if len(username) < 2 or len(password) < 8:
        return {"success": False, "error": "用户名至少2字符，密码至少8字符"}

    user_id = str(uuid.uuid4())[:8]
    users[username] = {
        "user_id": user_id,
        "password_hash": _hash_password(password),
        "display_name": display_name or username,
        "created_at": datetime.now().isoformat(),
    }
    _save_users(users)
    ensure_dir(os.path.join(USER_DATA_ROOT, user_id, "documents"))
    log.info(f"用户已注册: {username} ({user_id})")
    return {"success": True, "user_id": user_id}


def login(username: str, password: str) -> dict:
    """用户登录"""
    global _current_user
    users = _load_users()
    if username not in users:
        return {"success": False, "error": "用户不存在"}
    user = users[username]
    if not _verify_password(password, user["password_hash"]):
        return {"success": False, "error": "密码错误"}

    # 自动升级旧SHA-256哈希到PBKDF2
    if not user["password_hash"].startswith("pbkdf2:"):
        user["password_hash"] = _hash_password(password)
        _save_users(users)
        log.info(f"密码哈希已升级: {username}")

    _current_user = {"username": username, **user}
    log.info(f"用户已登录: {username}")
    return {"success": True, "user": _current_user}


def logout():
    """用户登出"""
    global _current_user
    _current_user = None


def get_current_user() -> dict:
    """获取当前用户"""
    return _current_user


def is_authenticated() -> bool:
    return _current_user is not None


def get_user_collection_name() -> str:
    """获取当前用户的ChromaDB集合名"""
    if _current_user:
        return f"{COLLECTION_NAME}_{_current_user['user_id']}"
    return COLLECTION_NAME


def get_user_doc_dir() -> str:
    """获取当前用户的文档目录"""
    if _current_user:
        path = os.path.join(USER_DATA_ROOT, _current_user["user_id"], "documents")
    else:
        from config import DOCUMENTS_DIR
        path = DOCUMENTS_DIR
    ensure_dir(path)
    return path


def get_user_chroma_dir() -> str:
    """获取当前用户的ChromaDB目录"""
    if _current_user:
        path = os.path.join(USER_DATA_ROOT, _current_user["user_id"], "chroma_db")
    else:
        from config import CHROMA_PERSIST_DIR
        path = CHROMA_PERSIST_DIR
    return path


def get_user_collection():
    """获取当前用户的ChromaDB集合"""
    chroma_dir = get_user_chroma_dir()
    collection_name = get_user_collection_name()
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def list_users() -> list:
    """列出所有用户（不含密码）"""
    users = _load_users()
    return [{"username": u, "display_name": d.get("display_name", u),
             "user_id": d.get("user_id", ""), "created_at": d.get("created_at", "")}
            for u, d in users.items()]


def delete_user(username: str) -> bool:
    """删除用户及其数据"""
    users = _load_users()
    if username not in users:
        return False
    user_id = users[username]["user_id"]
    import shutil
    user_data_dir = os.path.join(USER_DATA_ROOT, user_id)
    if os.path.exists(user_data_dir):
        shutil.rmtree(user_data_dir, ignore_errors=True)
    del users[username]
    _save_users(users)
    log.info(f"用户已删除: {username}")
    return True


def init_default_user():
    """如果没有用户，创建默认用户"""
    users = _load_users()
    if not users:
        register_user("default", "default123", "默认用户")
        log.info("已创建默认用户: default/default123")
