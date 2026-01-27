# memtool（MVP）

一个基于 SQLite 的“记忆管理工具”，面向 Claude Code / Codex 等 agent：默认输出 JSON，方便读取、更新、检索。

## 1. 安装
基础 CLI 仅依赖 Python 3.9+；中文检索会使用 jieba 分词（已作为依赖）。

```bash
chmod +x memtool.py
./memtool.py --db ./mem.db init
```

也可以把 `MEMTOOL_DB` 设成默认路径：

```bash
export MEMTOOL_DB=./mem.db
./memtool.py init
```

### MCP Server（可选）
MCP 需要额外依赖：
```bash
pip install -e .
```
启动 MCP Server：
```bash
memtool-mcp
```
环境变量：
- `MEMTOOL_DB`：共享数据库路径
- `LOG_LEVEL`：日志级别（INFO/DEBUG）

### Web Dashboard（可选）
Web 仪表板需要额外依赖：
```bash
pip install -e ".[web]"
```
启动 Web 仪表板：
```bash
memtool-web
```
访问：`http://127.0.0.1:8765` (默认)
环境变量：
- `MEMTOOL_DB`：共享数据库路径
- `LOG_LEVEL`：日志级别（INFO/DEBUG）

## 2. 写入 / 更新（put）
- `--id` 不填会自动生成
- 同一个 `--id` 再 put 会更新并 version+1
- content 可以来自 `--content`、`--file` 或 stdin

```bash
./memtool.py put --type project --key glossary --file glossary.md
./memtool.py put --type feature --task-id T123 --key contract --content '{"api":"/v1/login"}' --tag auth,login
echo "stacktrace..." | ./memtool.py put --type run --task-id T123 --step-id S9 --key stacktrace --source log
```

## 3. 读取（get）
按 id：
```bash
./memtool.py get --id <id>
```

按 (type,key,task_id,step_id) 获取最新一条：
```bash
./memtool.py get --type feature --task-id T123 --key contract
```

## 4. 列表（list）
```bash
./memtool.py list --type feature --task-id T123 --limit 50
./memtool.py list --tag auth
./memtool.py list --sort-by mixed --exclude-stale
```

## 5. 检索（search）
优先使用 SQLite FTS5（如环境不支持会自动降级为 LIKE）。中文内容会做 jieba 分词，过滤长度 < 2 的 token：

```bash
./memtool.py search --query "jwt refresh" --type feature --task-id T123
./memtool.py search --query "画布 核心 运行机制" --type project
./memtool.py search --query "timeout" --sort-by confidence
```

## 6. 导入 / 导出
```bash
./memtool.py export --output backup.jsonl
./memtool.py import --input backup.jsonl
```

## 6.1 删除（delete）
```bash
./memtool.py delete --id <id>
```

## 6.2 推荐（recommend）
```bash
./memtool.py recommend --context "redis timeout" --limit 5
./memtool.py recommend --task-id T123 --tag auth
```

## 6.3 清理（cleanup）
```bash
# 预览（dry-run）
./memtool.py cleanup --type run --older-than-days 14
# 执行删除
./memtool.py cleanup --type run --older-than-days 14 --apply
```

## 7. 给 agent 的调用建议
- 默认输出 JSON，适合直接解析
- 典型用法：先 list / search 拿到 id，再 get 取 full content
- 写入 run memory 时尽量按 step_id 分桶，避免互相覆盖

示例（伪代码）：
1) `memtool search --query "...error..." --task-id T123`
2) `memtool get --id <top_hit_id>`
3) 修复后：`memtool put --type run --task-id T123 --step-id S9 --key diff_summary --content "..."`
