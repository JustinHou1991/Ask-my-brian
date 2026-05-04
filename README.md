# Ask My Brain

个人知识库问答助手 — 基于 RAG（检索增强生成）的双策略对比系统。

上传文档，构建向量索引，用自然语言提问，获得基于你自己的资料的精准回答。

## 核心特性

**双策略对比**
- **策略A (传统RAG)** — Top-5 相似 chunk 检索，精准快速
- **策略B (Long Context)** — 智能文档选择，深度全文分析
- 支持单策略或 A/B 并排对比

**检索增强**
- ChromaDB 向量检索 + BM25 关键词检索（jieba 分词）
- Reciprocal Rank Fusion (RRF) 融合两种检索结果
- 智能文档选择：<5k 全文 / 5k-50k 命中 chunk 窗口 / >50k 窗口拼接

**文档支持**
- PDF（四级降级：pdfplumber → PyPDF2 → 扫描件 OCR → 图片 OCR）
- Markdown、TXT、HTML
- 图片 OCR（tesseract / PaddleOCR）
- 网页抓取

**智能问答**
- 流式输出，逐 token 实时显示
- 推理模型思考过程可收起展示
- SQLite 长期对话记忆，跨会话搜索
- 滑动窗口对话历史管理

**评测体系**
- 50 道测试题，5 个类别
- Recall@5 / MRR 检索质量指标
- LLM-as-Judge 三维质量评分（相关性 / 完整性 / 忠实度）
- Plotly 交互式可视化图表

**安全与可靠性**
- API 熔断器（连续 5 次失败 → 60s 冷却）
- Credits 预算管理（200M 总预算，输入 1x / 输出 3x）
- PBKDF2-HMAC-SHA256 密码哈希（260k 迭代 + 随机盐）
- 多用户支持，per-user ChromaDB 隔离

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 MIMO_API_KEY

# 3. 启动
streamlit run app.py

# 开发模式（低 token 消耗）
make dev
```

首次启动会自动下载 Embedding 模型（约 90MB）。国内环境自动使用 hf-mirror.com 镜像加速。

## 配置

所有可配置参数在 `config.py` 中集中管理。关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 800 | 分块字符数 |
| `TOP_K_RETRIEVAL` | 5 | 策略A检索数量 |
| `LONG_CONTEXT_MAX_CHARS` | 600,000 | 策略B最大文档字符 |
| `MAX_TOKENS` | 4,096 | 生成最大 token |
| `TEMPERATURE_A` / `B` | 0.3 / 0.15 | 策略温度 |
| `HYBRID_RETRIEVAL_ENABLED` | 1 | 混合检索开关 |
| `JUDGE_ENABLED` | 1 | LLM-as-Judge 开关 |

环境变量（`.env`）：

| 变量 | 必需 | 说明 |
|------|------|------|
| `MIMO_API_KEY` | Yes | API 密钥 |
| `MIMO_BASE_URL` | No | API 端点（必须以 `/v1` 结尾） |
| `MIMO_MODEL` | No | 模型名（小写 `mimo-v2.5-pro`） |
| `ASK_MY_BRAIN_DEV` | No | `1` 启用开发模式 |
| `MULTI_USER_ENABLED` | No | `1` 启用多用户 |
| `HYBRID_RETRIEVAL` | No | `0` 关闭混合检索 |
| `JUDGE_ENABLED` | No | `0` 关闭质量评估 |
| `PASSWORD_PEPPER` | No | 密码哈希 pepper |

## 架构

```
用户提问
  │
  ├─→ embed_query() ──→ 查询向量化
  │
  ├─→ 策略A: retrieve_top_k() / hybrid_search() ──→ Top-5 chunk
  │     └─ hybrid: BM25 + 向量 → RRF 融合
  ├─→ 策略B: retrieve_full_documents() ──→ 智能文档选择
  │     ├─ <5k → 全文
  │     ├─ 5k-50k → 命中 chunk ±2000 字符窗口
  │     └─ >50k → 窗口拼接，上限 600k
  │
  ├─→ generate_streaming() ──→ MiMo API (流式)
  │     ├─ reasoning_content → 可收起的思考过程
  │     └─ content → 逐 token 流式回答
  │
  └─→ Streamlit UI（四标签页）
        ├─ 智能问答（会话管理、策略选择、流式生成）
        ├─ 双策略评测（单题 / 批量 50 题）
        ├─ 评测报告（分类筛选、详细对比）
        └─ 评测可视化（Plotly 交互图表）
```

## 模块结构

```
app.py                 Streamlit 入口（~80 行，CSS + 标签页调度）
app_helpers.py         共享 UI 工具函数
app_tabs/              标签页模块
  sidebar.py           侧边栏（模型、用户、文档、索引、Credits）
  chat_tab.py          智能问答
  eval_tab.py          双策略评测
  report_tab.py        评测报告
  viz_tab.py           评测可视化
config.py              配置中心
utils.py               工具集（熔断器、Credits、日志、token 计算）
document_loader.py     文档解析（PDF 四级降级、MIME 白名单）
multimodal_loader.py   多模态（图片 OCR、扫描件 PDF）
chunker.py             分块引擎（语言检测、tiktoken 切分）
embedder.py            向量化（懒加载、离线模式、LRU 缓存）
retriever.py           检索引擎（ChromaDB、智能文档选择、增量删除）
hybrid_retriever.py    混合检索（BM25 + 向量 RRF 融合）
model_registry.py      模型管理（多模型预设、热切换）
generator.py           生成引擎（流式 API、reasoning 输出、错误分类）
eval_engine.py         评测引擎（单题/批量、LLM-as-Judge 质量评分）
eval_visualizer.py     评测可视化（Plotly 交互图表）
conversation_memory.py 对话记忆（SQLite、会话管理、跨会话搜索）
user_manager.py        多用户（PBKDF2 密码哈希、per-user 隔离）
async_worker.py        异步任务（后台索引构建、进度追踪）
```

依赖顺序（自底向上）：`config` → `utils` → `document_loader` / `multimodal_loader` → `chunker` → `embedder` → `retriever` → `hybrid_retriever` → `model_registry` → `generator` → `eval_engine` / `eval_visualizer` → `conversation_memory` → `user_manager` → `async_worker` → `app_helpers` → `app_tabs/*` → `app.py`

## 测试

`test_questions.json` 包含 50 道测试题，覆盖 5 个类别：事实检索、理解分析、应用场景、技术细节、综合应用。

运行方式：启动应用后进入「双策略评测」标签页，点击「开始批量评测」。

评测指标：
- **Recall@5** — Top-5 检索命中期望来源的比例
- **MRR** — 期望来源的平均倒数排名
- **质量评分** — LLM-as-Judge 三维评分（1-5 分）

## 文档

- [CHANGELOG.md](CHANGELOG.md) — 变更日志（含两轮代码审查记录）
- [PROJECT_REPORT.md](PROJECT_REPORT.md) — 从零到上线全流程报告
- [CLAUDE.md](CLAUDE.md) — Claude Code 项目指引

## 许可

个人学习项目。
