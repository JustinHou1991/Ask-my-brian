# Ask My Brain - 从零到上线：全流程总结报告

## 一、项目概述

| 项目 | 说明 |
|------|------|
| 名称 | Ask My Brain - 个人知识库问答助手 |
| 核心技术 | RAG (检索增强生成) + 双策略对比 |
| LLM | MiMo-V2.5-Pro (小米大模型) |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 (384维) |
| 向量数据库 | ChromaDB (本地持久化) |
| 前端框架 | Streamlit |
| 测试文档 | 14份PDF产品管理课程讲义 (共约72MB) |

**双策略设计：**
- **策略A (传统RAG)**：Top-5相似chunk检索 → 精准回答 (temperature=0.3)
- **策略B (Long Context)**：智能文档选择 → 深度全文分析 (temperature=0.15)

---

## 二、架构设计

```
用户提问
  │
  ├─→ embed_query() ──→ 查询向量化
  │
  ├─→ 策略A: retrieve_top_k() / hybrid_search() ──→ Top-5 chunk
  │     └─ hybrid: BM25关键词 + 向量语义 → RRF融合
  ├─→ 策略B: retrieve_full_documents() ──→ 智能文档选择
  │     ├─ <5k字符 → 全文
  │     ├─ 5k-50k → 命中chunk ±2000字符窗口
  │     └─ >50k   → 窗口拼接，上限600k字符
  │
  ├─→ generate_streaming() ──→ MiMo API (流式)
  │     ├─ reasoning_content → 可收起的思考过程
  │     └─ content → 逐token流式回答
  │
  └─→ Streamlit UI 渲染（四标签页）
        ├─ 智能问答（会话管理、策略选择）
        ├─ 双策略评测（单题/批量50题）
        ├─ 评测报告（分类筛选、详细对比）
        └─ 评测可视化（Plotly交互图表）
```

**核心模块：**

| 模块 | 文件 | 职责 |
|------|------|------|
| 配置 | `config.py` | 环境变量、阈值、模型参数 |
| 工具 | `utils.py` | 熔断器、Credits计数、日志、token计算 |
| 文档加载 | `document_loader.py` | PDF/MD/HTML/TXT/URL解析，MIME白名单 |
| 多模态 | `multimodal_loader.py` | 图片OCR、扫描件PDF检测、PDF图片提取 |
| 分块 | `chunker.py` | 语言检测、中英文分隔符、tiktoken精确切分 |
| 向量化 | `embedder.py` | sentence-transformers懒加载、批量编码、离线/镜像自动切换 |
| 检索 | `retriever.py` | ChromaDB查询、智能文档选择、检索评估、线程安全文档缓存 |
| 混合检索 | `hybrid_retriever.py` | BM25(jieba) + 向量检索RRF融合 |
| 模型管理 | `model_registry.py` | 多模型预设、热切换、per-model配置 |
| 生成 | `generator.py` | MiMo API调用、流式输出(reasoning+content)、错误分类处理 |
| 评测 | `eval_engine.py` | 单题/批量评测、Recall@5/MRR指标 |
| 评测可视化 | `eval_visualizer.py` | Plotly交互式图表（对比柱状图、分类分析、雷达图） |
| 对话记忆 | `conversation_memory.py` | SQLite长期存储、会话管理、跨会话搜索 |
| 多用户 | `user_manager.py` | 用户登录注册、per-user ChromaDB隔离 |
| 异步任务 | `async_worker.py` | 后台索引构建、进度追踪、取消支持 |
| UI工具 | `app_helpers.py` | 共享渲染函数、会话状态初始化 |
| 侧边栏 | `app_tabs/sidebar.py` | 模型切换、用户认证、文档管理、索引 |
| 对话标签页 | `app_tabs/chat_tab.py` | 会话管理、策略选择、流式生成 |
| 评测标签页 | `app_tabs/eval_tab.py` | 单题/批量评测、质量分展示 |
| 报告标签页 | `app_tabs/report_tab.py` | 评测结果详情、分类筛选 |
| 可视化标签页 | `app_tabs/viz_tab.py` | Plotly交互式图表 |
| 主入口 | `app.py` | Streamlit UI入口(~60行)、CSS、标签页调度 |

---

## 三、从零搭建过程

### 3.1 项目初始化
- 创建项目目录结构
- 编写 `requirements.txt` 定义依赖
- 创建 `config.py` 集中管理配置
- 创建 `.env.example` 和 `.gitignore`

### 3.2 核心模块开发（按依赖顺序）
1. **utils.py** — 基础工具（日志、目录、token计算）
2. **document_loader.py** — 文档解析（PDF三级降级、MIME检查）
3. **chunker.py** — 文档分块（语言检测、RecursiveCharacterTextSplitter）
4. **embedder.py** — 向量化（sentence-transformers）
5. **retriever.py** — 向量检索（ChromaDB + 智能文档选择）
6. **generator.py** — LLM生成（MiMo API + 流式输出）
7. **eval_engine.py** — 评测引擎（Recall@5、MRR）
8. **app.py** — Streamlit前端

### 3.3 前端UI设计
- 三标签页：智能问答 / 双策略评测 / 评测报告
- 暗色主题CSS（渐变卡片、策略徽章、来源标签）
- 侧边栏：API状态、熔断器、Credits进度条、文档管理、索引构建

---

## 四、遇到的问题与解决方案

### 4.1 白屏/加载卡住

**现象：** 打开页面白屏，长时间无响应

**根因：** `get_collection_count()` 在首次调用时会初始化ChromaDB，如果 `chroma_db/` 目录不存在会阻塞

**修复：**
```python
def get_collection_count() -> int:
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        return 0  # 快速路径，不初始化ChromaDB
    try:
        return get_collection().count()
    except Exception:
        return 0
```

### 4.2 索引构建极慢（20+分钟无响应）

**现象：** 点击「构建索引」后20分钟仍未完成

**根因（三层）：**
1. `sentence-transformers v5.4.1` 有 `Pooling.__init__()` 兼容性bug
2. HuggingFace模型缓存为空，首次需下载90MB模型
3. 国内网络无法直接访问 huggingface.co

**修复：**
1. 降级 `sentence-transformers==3.4.1`
2. `embedder.py` 自动检测并设置 `HF_ENDPOINT=https://hf-mirror.com`
3. 预下载模型到本地缓存
4. 验证：100条文本Embedding仅需1.64秒

### 4.3 启动极慢（18秒+）

**现象：** 页面打开后长时间loading

**根因：** `langchain_text_splitters`（8秒）和 `sentence_transformers`（10秒）在模块导入时就加载

**修复 — 懒加载：**
```python
# chunker.py - 懒加载langchain
_Splitter = None
def _get_splitter():
    global _Splitter
    if _Splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _Splitter = RecursiveCharacterTextSplitter
    return _Splitter

# embedder.py - 懒加载sentence_transformers
_sentence_transformer_cls = None
# 在get_model()中才真正导入
```

**效果：** 启动时间从18秒降至2.4秒

### 4.4 API调用404错误

**现象：** 生成回答时报 `404 Not Found` (openresty)

**根因：** `.env` 中 `MIMO_BASE_URL` 指向 `/anthropic` 端点（Anthropic格式），但代码使用OpenAI SDK，会拼接 `/chat/completions`

**修复：**
```env
# 错误
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic
# 正确
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
```

**补充发现：** 模型名必须小写 `mimo-v2.5-pro`（API区分大小写）

### 4.5 MiMo API响应解析

**发现：** MiMo是推理模型，响应结构特殊：
```json
{
  "choices": [{
    "message": {
      "content": "最终回答",
      "reasoning_content": "推理思考过程..."
    }
  }],
  "usage": {
    "completion_tokens_details": {"reasoning_tokens": 84}
  }
}
```

**处理：** 流式输出时分别处理 `reasoning_content`（思考过程）和 `content`（最终回答）

### 4.6 langchain导入路径变更

**现象：** `from langchain.text_splitter import ...` 报错

**修复：**
```python
# 旧
from langchain.text_splitters import RecursiveCharacterTextSplitter
# 新
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### 4.7 其他修复

| 问题 | 修复 |
|------|------|
| PDF `load_pdf` 返回类型错误 | `-> dict` 改为 `-> str` |
| URL保存崩溃 | 将文件保存逻辑移入 `load_url()` |
| 重复索引构建 | 使用内容哈希作为chunk ID |
| OpenAI客户端重复创建 | 全局单例 `_client` |
| `ensure_dir` NameError | 移动定义到 `setup_logger` 之前 |
| 对话历史在对比模式丢失 | 同时保存A和B的回答 |

---

## 五、性能优化

### 5.1 启动优化

| 阶段 | 优化前 | 优化后 |
|------|--------|--------|
| 模块导入 | 18s | 2.4s |
| 页面加载 | 白屏卡住 | 3ms |

**关键手段：** `langchain_text_splitters` 和 `sentence_transformers` 懒加载

### 5.2 索引构建优化

| 阶段 | 优化前 | 优化后 |
|------|--------|--------|
| 模型下载 | 超时失败 | hf-mirror镜像 |
| 模型加载 | 10min+ | 0.6s（缓存后） |
| 14份PDF索引 | 20min+未完成 | ~2min |
| 100条Embedding | - | 1.64s |

**关键手段：**
- 离线模式（`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`）
- 国内镜像 `hf-mirror.com`
- 批量编码 `batch_size=64`
- `normalize_embeddings=True`

### 5.3 流式输出优化

| 阶段 | 优化前 | 优化后 |
|------|--------|--------|
| 回答呈现 | 全部生成后一次性显示 | 逐token流式输出 |
| 首token延迟 | ~7s（等待全部生成） | ~2-3s |
| 思考过程 | 不可见/混入回答 | 可收起的灰色折叠块 |
| 对比模式检索 | 串行（A等B） | 并行（ThreadPoolExecutor） |

**关键手段：**
```python
# 流式API调用
stream = client.chat.completions.create(..., stream=True, stream_options={"include_usage": True})

# 检索并行化
with ThreadPoolExecutor(max_workers=2) as ex:
    fa = ex.submit(retrieve_top_k, q_emb)
    fb = ex.submit(retrieve_full_documents, q_emb)
    ra, rb = fa.result(), fb.result()
```

### 5.4 离线模式检测

```python
# embedder.py - 自动检测缓存，决定离线/镜像模式
_model_cached = (HF_CACHE_DIR / MODEL_CACHE_NAME).exists()
if _model_cached:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
elif not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

## 六、安全与防护机制

### 6.1 API熔断器 (APICallManager)
- 连续5次失败 → 熔断60秒
- 线程安全（`threading.Lock`）
- 状态：`can_call()` / `record_success()` / `record_failure()`

### 6.2 Credits预算管理 (CreditsCounter)
- 总预算：200M credits
- 输入token权重：1x，输出token权重：3x
- 70%警告，90%严重
- 侧边栏实时显示用量进度条

### 6.3 API错误分类处理

| 错误码 | 处理策略 |
|--------|----------|
| 429 (RateLimit) | 读取 Retry-After header，等待后重试 |
| 400 (prompt过长) | 截断80%后重试 |
| 401/403 (认证失败) | 立即停止，提示检查API Key |
| 5xx (服务端错误) | 指数退避 (1s, 2s, 4s) |
| 网络超时 | 指数退避重试 |

### 6.4 文档安全
- MIME白名单检查（PDF/TXT/MD/HTML/CSV）
- PDF三级降级（pdfplumber → PyPDF2 → 友好提示）
- 大文件超时保护

### 6.5 对话历史管理
- 滑动窗口：最大4000 tokens
- 超出时从前面丢弃旧消息
- 使用tiktoken精确计数

---

## 七、流式输出实现细节

### 7.1 生成器链路

```
generate_streaming(messages)
  │
  ├─ stream=True 调用MiMo API
  ├─ 每个chunk判断类型：
  │   ├─ delta.reasoning_content → yield ("reasoning", text)
  │   └─ delta.content → yield ("content", text)
  └─ 最后一个chunk → yield ("done", stats)

_stream_with_thinking(gen)
  │
  ├─ 收集所有 ("reasoning", text) 到 thinking 变量
  ├─ yield 所有 ("content", text) 给 st.write_stream
  └─ 设置 _last_thinking 属性

app.py 调用链：
  think_ph = st.empty()           # 思考块占位
  think_ph.markdown(展开的思考中...)  # 显示思考状态
  answer = st.write_stream(_stream_with_thinking(gen))  # 流式回答
  _close_thinking(think_ph, thinking)  # 收起思考块
```

### 7.2 思考过程UI

- **思考中：** 展开的灰色折叠块，显示 `💭 思考中...`
- **回答后：** 自动收起，显示摘要（前60字），可点击展开查看完整思考
- **CSS样式：** 半透明背景 + 左侧边框 + 滚动限制（max-height: 200px）

---

## 八、关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `CHUNK_SIZE` | 800 | 每个chunk的字符数 |
| `CHUNK_OVERLAP` | 150 | chunk之间的重叠字符数 |
| `TOP_K_RETRIEVAL` | 5 | 策略A检索的chunk数量 |
| `LONG_CONTEXT_MAX_CHARS` | 600,000 | 策略B最大文档字符数 |
| `MAX_TOKENS_LONG_CONTEXT` | 900,000 | 长上下文token上限 |
| `MAX_TOKENS` | 4,096 | 生成最大token数 |
| `DEV_MAX_TOKENS` | 500 | 开发模式token数 |
| `TEMPERATURE_A` | 0.3 | 策略A温度 |
| `TEMPERATURE_B` | 0.15 | 策略B温度 |
| `MAX_HISTORY_TOKENS` | 4,000 | 对话历史token上限 |
| `API_CIRCUIT_BREAKER_THRESHOLD` | 5 | 熔断触发次数 |
| `API_CIRCUIT_BREAKER_COOLDOWN` | 60s | 熔断冷却时间 |
| `CREDITS_BUDGET` | 200,000,000 | Credits总预算 |

---

## 九、测试验证

### 9.1 测试数据
- `test_questions.json`：50道测试题，5个类别
  - 事实检索、理解分析、应用场景、技术细节、综合应用

### 9.2 检索评估指标
- **Recall@5**：Top-5检索中命中期望来源的比例
- **MRR (Mean Reciprocal Rank)**：期望来源的平均倒数排名
- 独立于LLM的纯检索质量评估

### 9.3 端到端验证结果

| 测试项 | 结果 |
|--------|------|
| PDF解析 (14份) | 全部成功，~2.2s/5MB |
| 模型加载 (缓存后) | 0.6s |
| 100条Embedding | 1.64s |
| 流式API调用 | 27个content chunk, ~7s |
| 页面加载 | 3ms |
| 模块导入 | 2.4s |

---

## 十、文件清单

```
D:\ask-my-brain\
├── app.py                 Streamlit主入口（~80行，CSS+标签页调度）
├── app_helpers.py         UI工具函数（init_state, 渲染辅助）
├── app_tabs/              标签页模块
│   ├── __init__.py
│   ├── sidebar.py         侧边栏（模型、用户、文档、索引、Credits）
│   ├── chat_tab.py        智能问答标签页
│   ├── eval_tab.py        双策略评测标签页
│   ├── report_tab.py      评测报告标签页
│   └── viz_tab.py         评测可视化标签页
├── config.py              配置中心（.env加载、阈值管理）
├── utils.py               工具集（熔断器、Credits、日志、token计算）
├── document_loader.py     文档解析（PDF四级降级、MIME检查、图片OCR）
├── multimodal_loader.py   多模态（图片OCR、扫描件PDF、PDF图片提取）
├── chunker.py             分块引擎（语言检测、tiktoken切分）
├── embedder.py            向量化（懒加载、离线模式、LRU缓存）
├── retriever.py           检索引擎（ChromaDB、智能文档选择、增量删除、线程安全）
├── hybrid_retriever.py    混合检索（BM25 + 向量RRF融合）
├── model_registry.py      模型管理（多模型预设、热切换）
├── generator.py           生成引擎（流式API、reasoning输出、错误分类）
├── eval_engine.py         评测引擎（单题/批量、LLM-as-Judge质量评分）
├── eval_visualizer.py     评测可视化（Plotly交互式图表、质量维度对比）
├── conversation_memory.py 对话记忆（SQLite、会话管理、跨会话搜索）
├── user_manager.py        多用户（PBKDF2密码哈希、per-user隔离）
├── async_worker.py        异步任务（后台索引构建、进度追踪）
├── test_questions.json    50道测试题
├── requirements.txt       依赖清单
├── CHANGELOG.md           变更日志（含代码审查记录）
├── CLAUDE.md              Claude Code项目指引
├── Makefile               构建脚本
├── README.md              项目文档
├── demo_script.md         演示脚本
├── eval_report.md         评测报告模板
└── PROJECT_REPORT.md      本文件
```

---

## 十一、启动方式

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API Key
cp .env.example .env
# 编辑 .env 填入真实的 MIMO_API_KEY

# 启动
streamlit run app.py

# 开发模式（低token消耗）
make dev
```

---

## 十二、后续可优化方向

### 已实现

- [x] **多模态支持** — `multimodal_loader.py`：图片OCR（tesseract/PaddleOCR），扫描件PDF检测与整页OCR
- [x] **混合检索** — `hybrid_retriever.py`：BM25(jieba) + 向量检索RRF融合
- [x] **对话记忆** — `conversation_memory.py`：SQLite长期存储，会话管理，跨会话搜索
- [x] **多用户支持** — `user_manager.py`：用户登录注册，per-user ChromaDB隔离
- [x] **异步索引** — `async_worker.py`：后台ThreadPoolExecutor，进度追踪，取消支持
- [x] **模型热切换** — `model_registry.py`：MiMo/DeepSeek/Qwen/GLM预设，sidebar切换
- [x] **评测可视化** — `eval_visualizer.py`：Plotly交互式图表（对比柱状图、分类分析、雷达图）

### 已完成优化 (2026-05-04)

- [x] **密码安全升级** — `user_manager.py`：PBKDF2-HMAC-SHA256 (260k迭代, 16字节随机盐)，向后兼容旧SHA-256哈希，最小密码长度8
- [x] **向量检索缓存** — `embedder.py`：`@lru_cache(maxsize=128)`缓存查询embedding，重复查询零开销
- [x] **文档增量更新** — `retriever.py`：`remove_document()`按文件名删除、`list_indexed_documents()`列出索引文档；`hybrid_retriever.py`同步清理BM25索引；侧边栏逐文档删除UI
- [x] **回答质量评估** — `eval_engine.py`：LLM-as-Judge三维评分(相关性/完整性/忠实度, 1-5分)；`eval_visualizer.py`：质量维度对比柱状图
- [x] **app.py拆分** — 599行→~80行。拆分为`app_helpers.py` + `app_tabs/`包(5个标签页模块)，主入口仅保留CSS和标签页调度

---

## 十三、代码审查记录 (2026-05-04)

对全部源码进行了系统性审查，修复10项问题，详见 [CHANGELOG.md](../CHANGELOG.md)。

### 审查范围

覆盖全部14个Python源文件 + requirements.txt，共计约2000行代码。

### 修复摘要

| 严重度 | 数量 | 典型问题 |
|--------|------|----------|
| Bug | 4 | API重试污染调用方数据、retry-after解析崩溃、Unicode范围冗余、RRF键碰撞 |
| 线程安全 | 3 | 文档缓存竞态、SQLite并发、消息存储锁 |
| 代码质量 | 3 | 动态导入、无用导入、冗余依赖 |

### 关键修复详情

1. **generator.py** — `generate_with_mimo` 400重试时用`copy.deepcopy`避免修改调用方messages
2. **generator.py** — 抽取`_parse_retry_after()`安全解析Retry-After header
3. **chunker.py** — 移除冗余Unicode范围条件，复用tiktoken编码器单例
4. **hybrid_retriever.py** — RRF融合键从`text[:100]`改为`source:chunk_index`
5. **retriever.py** — `_documents_store`添加`threading.Lock`保护
6. **conversation_memory.py** — 启用SQLite WAL mode + timeout=10
7. **user_manager.py** — `__import__("datetime")`改为顶部import
8. **requirements.txt** — 移除冗余`langchain`依赖
9. **eval_engine.py** — 移除无用的retriever导入
10. **CLAUDE.md** — 更新模块架构文档，反映16个模块的完整依赖关系
