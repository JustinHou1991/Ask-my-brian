# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ask My Brain** — a personal knowledge base Q&A assistant built on RAG (Retrieval-Augmented Generation) with dual-strategy comparison. Uses MiMo-V2.5-Pro (Xiaomi LLM) via OpenAI-compatible API, ChromaDB for vector storage, and Streamlit for the UI.

**Dual strategies:**
- **Strategy A (Traditional RAG)**: Top-5 similar chunk retrieval → precise answers (temperature=0.3)
- **Strategy B (Long Context)**: Smart document selection → deep full-document analysis (temperature=0.15)

## Commands

```bash
# Setup
make setup                    # Install deps + pre-download embedding model
cp .env.example .env          # Then fill in MIMO_API_KEY

# Run
make run                      # streamlit run app.py
make dev                      # Dev mode (max_tokens=500, low cost)

# Database
make init-db                  # Initialize ChromaDB
make clean                    # Remove chroma_db/, documents/, logs/
```

## Architecture

**Pipeline flow:**
```
User query → embed_query() → vectorize
  ├→ Strategy A: retrieve_top_k() / hybrid_search() → Top-5 chunks
  ├→ Strategy B: retrieve_full_documents() → smart doc selection
  │     (<5k chars → full text, 5k-50k → hit chunks ±2000 char window, >50k → window concat up to 600k)
  └→ generate_streaming() → MiMo API (streaming) → Streamlit UI
```

**Module dependency order** (bottom-up):
1. `config.py` — all configuration, env vars, thresholds
2. `utils.py` — logging, APICallManager (circuit breaker), CreditsCounter, tiktoken counting
3. `document_loader.py` — PDF/MD/HTML/TXT/URL parsing with MIME whitelist, image OCR
4. `multimodal_loader.py` — image OCR (tesseract/PaddleOCR), scanned PDF detection, PDF image extraction
5. `chunker.py` — language detection (CN/EN), RecursiveCharacterTextSplitter, long-context tiktoken splitting
6. `embedder.py` — sentence-transformers lazy loading, batch encoding, offline/mirror auto-detection, LRU query cache
7. `retriever.py` — ChromaDB queries, smart document selection, retrieval evaluation (Recall@5, MRR), incremental document delete, thread-safe document store
8. `hybrid_retriever.py` — BM25 (jieba tokenization) + vector search fusion via RRF, per-document chunk cleanup
9. `model_registry.py` — multi-model presets (MiMo/DeepSeek/Qwen/GLM), hot-switching, per-model config
10. `generator.py` — MiMo API calls, streaming (reasoning_content + content), error classification (429/400/401/5xx), circuit breaker integration
11. `eval_engine.py` — single/batch evaluation, LLM-as-Judge quality scoring (relevance/completeness/faithfulness)
12. `eval_visualizer.py` — Plotly interactive charts (comparison bars, quality dimensions, category analysis, radar chart)
13. `conversation_memory.py` — SQLite long-term conversation storage, session management, cross-session search
14. `user_manager.py` — multi-user support, PBKDF2 password hashing, per-user ChromaDB isolation, login/register
15. `async_worker.py` — background task execution (index building), progress tracking, cancellation
16. `app_helpers.py` — shared UI helpers (init_state, render functions, streaming with thinking)
17. `app_tabs/` — tab modules (sidebar, chat, eval, report, viz)
18. `app.py` — Streamlit entry point (~80 lines): page config, CSS, tab orchestration

## Key Design Decisions

**Lazy loading is critical.** `langchain_text_splitters` (8s) and `sentence_transformers` (10s) are lazily imported to keep startup under 3s. Never import these at module level.

**Embedding model caching.** `embedder.py` auto-detects if the model is cached in `~/.cache/huggingface/hub/`. If cached, sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` for instant load. If not cached, sets `HF_ENDPOINT=https://hf-mirror.com` for China network.

**MiMo is a reasoning model.** Streaming responses have two fields: `reasoning_content` (thinking process) and `content` (final answer). The UI shows thinking in a collapsible block via `_stream_with_thinking()`.

**API circuit breaker** (`APICallManager` in utils.py): 5 consecutive failures → 60s cooldown. All API calls go through this.

**Credits budget**: 200M total, input tokens ×1, output tokens ×3. Warning at 70%, critical at 90%.

**Hybrid retrieval** (`hybrid_retriever.py`): BM25 keyword search (jieba tokenization) fused with vector semantic search via Reciprocal Rank Fusion (RRF). Uses `source:chunk_index` as unique key to avoid collision. Toggle in sidebar.

**Model hot-switching** (`model_registry.py`): Preset models (MiMo, DeepSeek, Qwen, GLM) with per-model base_url, temperature, max_tokens. Switching via sidebar selectbox, no restart needed. Custom models fall back to config defaults.

**Conversation memory** (`conversation_memory.py`): SQLite with WAL mode for concurrent reads. Per-session isolation, auto-truncation by token limit. Thread-safe with `timeout=10`.

**Async index building** (`async_worker.py`): `ThreadPoolExecutor(max_workers=2)`. Chunking → vectorization → DB write pipeline with progress tracking. Cancellable via `threading.Event`.

**Thread-safe document store** (`retriever.py`): `_documents_store` dict protected by `threading.Lock` for concurrent read (main thread) / write (async worker) access.

**Password security** (`user_manager.py`): PBKDF2-HMAC-SHA256 with 260k iterations, 16-byte random salt, optional `PASSWORD_PEPPER` env var. `login()` auto-upgrades old SHA-256 hashes on successful verification. Uses `hmac.compare_digest` for constant-time comparison.

**Embedding query cache** (`embedder.py`): `@lru_cache(maxsize=128)` on internal `_embed_query_cached()` (returns tuple for hashability). Public `embed_query()` converts to list. Cache stats via `get_embed_cache_stats()`.

**Incremental document updates** (`retriever.py`, `hybrid_retriever.py`): `remove_document(filename)` deletes from ChromaDB by metadata filter. `list_indexed_documents()` aggregates chunk counts per source. `HybridRetriever.remove_document_chunks()` syncs BM25 index. Sidebar UI provides per-document delete buttons.

**LLM-as-Judge** (`eval_engine.py`): 3-dimension scoring (relevance/completeness/faithfulness, 1-5 scale) via structured JSON prompt. Enabled by default (`JUDGE_ENABLED=1`). Quality stats aggregated in `compute_stats()` and visualized in `eval_visualizer.py`.

**App.py refactoring**: 599 lines → ~80 lines. Shared helpers in `app_helpers.py`, tab logic in `app_tabs/*.py`. Main entry point only does page config, CSS, sidebar, and tab orchestration.

## Configuration

All tunable parameters are in `config.py`. Key ones:
- `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`
- `TOP_K_RETRIEVAL=5`
- `LONG_CONTEXT_MAX_CHARS=600000`
- `MAX_TOKENS=4096` (dev: 500)
- `TEMPERATURE_A=0.3`, `TEMPERATURE_B=0.15`
- `API_CIRCUIT_BREAKER_THRESHOLD=5`, `API_CIRCUIT_BREAKER_COOLDOWN=60`
- `HYBRID_RETRIEVAL_ENABLED=1`, `BM25_WEIGHT=1.0`, `VECTOR_WEIGHT=1.0`, `RRF_K=60`
- `JUDGE_ENABLED=1` — LLM-as-Judge quality evaluation

Environment variables (`.env`):
- `MIMO_API_KEY` — required
- `MIMO_BASE_URL` — must end with `/v1` (NOT `/anthropic`)
- `MIMO_MODEL` — must be lowercase `mimo-v2.5-pro` (API is case-sensitive)
- `ASK_MY_BRAIN_DEV=1` — enables dev mode
- `MULTI_USER_ENABLED=1` — enables multi-user mode
- `HYBRID_RETRIEVAL=1` — enables BM25+vector hybrid retrieval (default on)
- `JUDGE_ENABLED=0` — disable LLM-as-Judge (default on)
- `PASSWORD_PEPPER` — optional pepper for password hashing

## Gotchas

- `MIMO_BASE_URL` must point to `/v1` endpoint, not `/anthropic`. The OpenAI SDK appends `/chat/completions` automatically.
- Model name must be lowercase (`mimo-v2.5-pro`), API is case-sensitive.
- `get_collection_count()` has a fast path: returns 0 if `chroma_db/` doesn't exist, avoiding ChromaDB initialization blocking.
- `ensure_dir` is defined before `setup_logger` in utils.py — order matters.
- langchain import path: `from langchain_text_splitters import ...` (not `from langchain.text_splitters`).
- `conversation_memory.py` calls `init_db()` at import time — creates SQLite DB on first import.
- `hybrid_retriever.py` RRF fusion uses `source:chunk_index` as key, not `text[:100]` (fixed in code review).
- `generator.py` API 400 retry uses `copy.deepcopy` to avoid mutating caller's messages (fixed in code review).
- `retriever.py` `_documents_store` is protected by `threading.Lock` — always use the lock when accessing it.
- `embedder.py` `_embed_query_cached` uses `@lru_cache` — returns tuple (hashable), not list. Use public `embed_query()` for list.
- `user_manager.py` `login()` auto-upgrades old SHA-256 hashes to PBKDF2 on successful verification — no manual migration needed.
- `app_tabs/sidebar.py` imports `_build_index_task` locally to avoid circular imports with `async_worker`.
- `eval_engine.py` `judge_answer()` returns `None` on any error — check before accessing quality scores.

## Testing

`test_questions.json` contains 50 test questions across 5 categories (事实检索, 理解分析, 应用场景, 技术细节, 综合应用). Run batch evaluation via the "双策略评测" tab or programmatically with `eval_engine.run_evaluation()`.

Retrieval quality metrics: Recall@5 and MRR (independent of LLM).

Evaluation results are visualized in the "评测可视化" tab via `eval_visualizer.py` (Plotly): comparison bars, category analysis, timeline, and radar chart.

## Code Review Log

See [CHANGELOG.md](CHANGELOG.md) for detailed code review findings and fixes (2026-05-04).
