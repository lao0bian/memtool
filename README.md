# memtool

一个基于 SQLite 的"记忆管理工具"，面向 Claude Code / Codex 等 AI Agent：默认输出 JSON，方便读取、更新、检索。

**当前版本**：v0.7.0  
**最后更新**：2026-03-11

## 核心特性

- **三层记忆**：project / feature / run，匹配不同生命周期
- **FTS5 全文检索**：中文 jieba 分词，自动降级 LIKE
- **向量语义搜索**：BAAI/bge-small-zh-v1.5 本地 embedding，支持 OpenAI/Ollama
- **混合搜索 (Hybrid)**：FTS + 向量加权融合，兼顾精确匹配与语义理解
- **异步降级**：向量模型冷启动时自动降级为纯 FTS（2ms 秒回），后台加载完成后自动升级
- **元认知评估**：4 维度 breakdown + bottleneck 识别，AI 知道自己知道什么
- **记忆活化**：访问追踪、巩固分数、自动衰减、主动推荐
- **生命周期管理**：基于遗忘曲线的自动清理，动态半衰期

## 1. 安装

基础 CLI 仅依赖 Python 3.9+；中文检索使用 jieba 分词（已作为依赖）。

```bash
pip install -e .
```

### MCP Server

```bash
memtool-mcp
```

环境变量：
- `MEMTOOL_DB`：数据库路径（默认 `./memtool.db`）
- `LOG_LEVEL`：日志级别（INFO/DEBUG）
- `MEMTOOL_EMBEDDING_PROVIDER`：embedding 提供商（local/openai/ollama，默认 local）
- `MEMTOOL_EMBEDDING_MODEL`：自定义 embedding 模型名
- `MEMTOOL_VECTOR_ENABLED`：向量搜索开关（on/off/auto，默认 auto）

### 向量搜索依赖（可选）

```bash
pip install -e ".[vector]"
```

### Web Dashboard（可选）

```bash
pip install -e ".[web]"
memtool-web
```

访问：`http://127.0.0.1:8765`

### Claude Code Hook（可选）

UserPromptSubmit 事件自动注入相关记忆上下文：
```bash
hooks/memory_inject.py
```

环境变量：
- `MEMTOOL_HOOK_MAX_ITEMS`：最多注入条数（默认 3）
- `MEMTOOL_HOOK_MAX_TOKENS`：检索 token 上限（默认 6）
- `MEMTOOL_HOOK_MIN_TOKEN_LEN`：最小 token 长度（默认 2）
- `MEMTOOL_HOOK_MIN_RELEVANCE`：相关度阈值（默认 0.25）

## 2. MCP 工具一览

| 工具名 | 功能 | 阶段 |
|--------|------|------|
| `memory_store` | 写入/更新记忆 (upsert) | Phase 1 |
| `memory_recall` | 按 ID 或逻辑键读取 | Phase 1 |
| `memory_search` | FTS5 全文搜索 | Phase 1 |
| `memory_list` | 列表+过滤 | Phase 1 |
| `memory_recommend` | 上下文推荐 | Phase 2 |
| `memory_cleanup` | 清理过期记忆 | Phase 2 |
| `memory_delete` | 删除记忆 | Phase 2 |
| `memory_export` | 导出记忆 | Phase 2 |
| `memory_semantic_search` | 向量语义搜索 | Phase 2.1 |
| `memory_hybrid_search` | 混合搜索 (FTS+向量) | Phase 2.1 |
| `memory_vector_sync` | 向量索引同步 | Phase 2.1 |
| `memory_vector_status` | 向量搜索状态 | Phase 2.1 |
| `memory_stats` | 统计信息 | Phase 2.1 |
| `memory_health_check` | 健康检查 | Phase 2.1 |
| `memory_contextual_search` | 情境检索 | Phase 2.2 |
| `memory_parse_context` | 自然语言→情境条件 | Phase 2.2 |
| `memory_history` | 变更历史 | Phase 2.6 |
| `memory_suggest_merge` | 记忆合并建议 | Phase 2.6 |
| `memory_assess_knowledge` | 元认知评估 (4维度) | Phase 2.7 |
| `memory_touch` | 强化记忆 | Phase 3 |
| `memory_auto_decay` | 类型化自动衰减 | Phase 3 |
| `memory_proactive_recommend` | 主动推荐 (24h冷却) | Phase 3 |

## 3. 基础用法

### 写入 / 更新（put）

```bash
./memtool.py put --type project --key glossary --file glossary.md
./memtool.py put --type feature --key contract --content '{"api":"/v1/login"}' --tag auth,login
```

### 读取（get）

```bash
./memtool.py get --id <id>
./memtool.py get --type feature --key contract
```

### 列表（list）

```bash
./memtool.py list --type feature --limit 50
./memtool.py list --tag auth --sort-by mixed --exclude-stale
```

### 检索（search）

```bash
./memtool.py search --query "jwt refresh" --type feature
./memtool.py search --query "画布 核心 运行机制" --type project
```

## 4. 高级搜索

### 混合搜索（推荐）

FTS 关键词匹配 + 向量语义理解，效果最佳：

```python
result = store.hybrid_search(
    query="数据库超时",
    fts_weight=0.3,
    vector_weight=0.7,
    limit=10
)
# 能找到 "DB timeout", "connection slow" 等语义相关记录
```

**异步降级机制**：
- 向量模型未加载时 → 自动返回纯 FTS 结果（`mode: fts_degraded`），2ms 秒回
- 模型加载完成后 → 自动切换为混合模式（`mode: hybrid`）
- MCP 启动时后台预热线程自动加载模型

### 元认知评估

```python
result = store.assess_knowledge(topic="React 性能优化")
# 返回 4 维度 breakdown + bottleneck 识别 + 针对性建议
```

## 5. 导入 / 导出 / 清理

```bash
./memtool.py export --output backup.jsonl
./memtool.py import --input backup.jsonl
./memtool.py cleanup --type run --older-than-days 14        # 预览
./memtool.py cleanup --type run --older-than-days 14 --apply # 执行
```

## 6. 给 Agent 的调用建议

- 默认输出 JSON，适合直接解析
- 推荐使用 `hybrid_search`，兼顾精确匹配与语义理解
- 典型流程：search → get → 修复 → put
- 使用 `assess_knowledge` 评估知识边界，避免幻觉
- run 层记忆按 step_id 分桶，避免互相覆盖
