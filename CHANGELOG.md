# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-05-04 - Code Review & Optimization

### Fixed

#### generator.py - Bug Fixes
- **API 400 重试时污染调用方数据**: `generate_with_mimo` 在收到 400 错误（prompt 过长）重试时，直接截断了调用方传入的 `messages` 列表。后续调用会使用被截断的内容，导致不可预期的回答质量下降。修复：使用 `copy.deepcopy` 创建副本后再截断。
- **RateLimitError retry-after 解析崩溃**: 原逻辑 `int(getattr(e, 'response', None) and ...)` 在 response 为 `None` 时执行 `int(None)` 抛出 `TypeError`，`or 5` 的 fallback 永远不会生效。修复：抽取 `_parse_retry_after()` 函数，安全解析 header 值，异常时返回默认 5s。

#### chunker.py - Bug Fixes
- **中文检测 Unicode 范围冗余**: `"一" <= c <= "鿿" or "一" <= c <= "龥"` — 两个范围完全重叠（`龥`=龟 < `鿿`=鿿），`or` 后半部分永远为 `True`，无实际效果。修复：移除冗余条件。
- **tiktoken 编码器重复创建**: `chunk_for_long_context` 每次调用都执行 `tiktoken.get_encoding("cl100k_base")`，编码器初始化开销约 200ms。修复：复用 `utils._get_tokenizer()` 单例。

#### hybrid_retriever.py - Bug Fixes
- **RRF 融合键碰撞**: 使用 `chunk["text"][:100]` 作为 RRF 融合的字典键，当前 100 字符相同时不同 chunk 会被错误合并，丢失检索结果。修复：改用 `source:chunk_index` 作为唯一键。

#### retriever.py - Thread Safety
- **`_documents_store` 竞态条件**: 全局字典 `_documents_store` 被异步 worker（写入）和主线程（读取）同时访问，无同步保护。修复：添加 `threading.Lock` 保护所有读写路径。

#### user_manager.py - Code Quality
- **运行时动态导入**: `__import__("datetime").datetime.now()` 在函数体内动态导入，IDE 无法静态分析，代码可读性差。修复：改为顶部 `from datetime import datetime`。

#### conversation_memory.py - Thread Safety
- **SQLite 并发读写**: 默认 journal mode 在多线程环境下可能导致写入阻塞或死锁。修复：启用 WAL mode（`PRAGMA journal_mode=WAL`）提升并发读性能，设置 `timeout=10` 避免立即报锁冲突。

#### eval_engine.py - Code Quality
- **无用导入**: 导入了 `retrieve_top_k` 和 `retrieve_full_documents` 但从未使用，增加不必要的模块加载开销。修复：移除。

#### requirements.txt - Dependencies
- **冗余 langchain 依赖**: 代码只使用 `langchain-text-splitters`，但 `requirements.txt` 同时列出了完整的 `langchain` 包，拉入大量无用传递依赖。修复：移除 `langchain>=0.1.0`，保留 `langchain-text-splitters>=0.0.1`。

### Architecture Notes (Not Changed - Documented for Reference)

以下问题在本次审查中识别但未修改，记录供后续参考：

- **app.py `_stream_with_thinking` 状态管理**: 用函数属性 `_last_thinking` 存储思考文本，在 Streamlit 单线程模型下可工作，但非理想模式。可改为返回元组 `(content_stream, thinking_text)`。
- **conversation_memory.py 模块级初始化**: `init_db()` 在 import 时执行，每次导入都打开 SQLite 连接。当前可接受（模块总是在 app.py 中被使用），但用于测试时会增加不必要开销。

### 已在后续优化中解决

- **user_manager.py 密码存储** → 已升级为 PBKDF2-HMAC-SHA256 + 随机盐（见上方优化 #1）
- **app.py 代码长度** → 已拆分为 7 个文件（见上方优化 #5）

## [Unreleased] - 2026-05-04 - 5项优化实施

### 1. 密码安全升级 (user_manager.py)

- **PBKDF2-HMAC-SHA256**: 260,000次迭代，16字节随机盐，支持`PASSWORD_PEPPER`环境变量
- **向后兼容**: `login()`自动识别旧SHA-256哈希并验证，登录成功后自动升级为新格式
- **常量时间比较**: 使用`hmac.compare_digest`防止时序攻击
- **密码策略**: 最小长度从4提升至8，默认用户密码`default123`

### 2. 向量检索缓存 (embedder.py)

- **LRU缓存**: `@lru_cache(maxsize=128)`缓存查询embedding结果
- **哈希兼容**: 内部`_embed_query_cached`返回tuple(可哈希)，公共`embed_query`转list
- **监控接口**: `get_embed_cache_stats()`返回hits/misses/size，`clear_embed_cache()`清空缓存

### 3. 文档增量更新 (retriever.py, hybrid_retriever.py, sidebar.py)

- **`remove_document(filename)`**: 使用`collection.delete(where={"source": filename})`按文件名删除索引
- **`list_indexed_documents()`**: 查询所有metadata，聚合每个文件的chunk数量
- **`HybridRetriever.remove_document_chunks(filename)`**: 同步清理BM25索引中的对应chunks
- **侧边栏UI**: 索引文档列表，每个文件显示chunk数和删除按钮

### 4. LLM-as-Judge质量评估 (eval_engine.py, eval_visualizer.py, config.py)

- **`JUDGE_ENABLED`配置**: 环境变量开关（默认开启）
- **`judge_answer()`**: 三维评分 — 相关性(relevance)、完整性(completeness)、忠实度(faithfulness)，1-5分
- **`evaluate_single()`更新**: 当JUDGE_ENABLED时自动调用judge评分
- **`compute_stats()`更新**: 增加avg_relevance/avg_completeness/avg_faithfulness维度
- **`_render_quality_comparison()`**: Plotly分组柱状图展示质量维度对比

### 5. App.py重构 (app.py → 7个文件)

- **`app_helpers.py`**: `init_state()`, `get_index_count()`, `render_sources_html()`, `render_strategy_card()`, `_close_thinking()`, `_stream_with_thinking()`
- **`app_tabs/__init__.py`**: 空包文件
- **`app_tabs/sidebar.py`**: `render_sidebar()` — 模型切换、用户认证、文档管理、混合检索、索引构建/删除、异步进度、Credits
- **`app_tabs/chat_tab.py`**: `render_chat_tab(has_index)` — 会话管理、策略选择、对话历史、流式生成
- **`app_tabs/eval_tab.py`**: `render_eval_tab(has_index)` — 单题评测、质量分展示、批量评测(50题)
- **`app_tabs/report_tab.py`**: `render_report_tab()` — 评测结果总览、策略对比、分类筛选、逐题详情
- **`app_tabs/viz_tab.py`**: `render_viz_tab()` — Plotly图表
- **`app.py`**: ~80行，仅保留页面配置、CSS、`init_state()`、`render_sidebar()`、标签页创建与调度

## [Unreleased] - 2026-05-04 - Code Review Round 2

### Fixed

#### eval_engine.py - Bugs
- **`chunks_b` 未定义变量**: `evaluate_single()` 第109行使用 `chunks_b` 但该变量不存在（应为 `docs_b`）。当 `JUDGE_ENABLED=True` 时触发 `NameError`，导致批量评测全部失败。修复：改为 `docs_b`。
- **JSON提取逻辑脆弱**: `judge_answer()` 使用 `text.split("```")[1]` 提取LLM返回的JSON，当LLM返回非标准格式（如无换行、额外文本）时解析失败。修复：改用正则表达式 `re.search` 提取 ```json...``` 块或裸JSON对象。

#### conversation_memory.py - Bug
- **WAL模式仅首次连接生效**: `_get_conn()` 通过 `_db_initialized` 标志控制 `PRAGMA journal_mode=WAL`，但每次调用都创建新连接，只有第一条连接设置了WAL。后续连接回退到默认 journal mode，并发读写时可能死锁。修复：移除 `_db_initialized` 标志，每次连接都设置 WAL + foreign_keys。

#### generator.py - Code Quality
- **400错误重试截断逻辑不精确**: `generate_with_mimo` 收到 400（prompt过长）时只截断最后一条user消息，但策略B的长上下文在user消息中且可能很长。原逻辑不检查消息长度，可能截断短消息而放过长消息。修复：添加 `len(msg["content"]) > 500` 条件，只截断长消息。
