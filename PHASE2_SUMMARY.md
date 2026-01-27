# Phase 2 Summary

## 已实现功能

1. 混合排序（confidence + recency）
   - list/search 新增 `--sort-by {updated,confidence,recency,mixed}`
   - 基于遗忘曲线的 recency_score，混合评分公式：
     `mixed = 0.6 * confidence_score + 0.4 * recency_score`

2. 记忆推荐引擎
   - 新增 CLI `recommend` 与 MCP `memory_recommend`
   - 评分维度：关键词相关性 + 置信度 + 新鲜度 + 权重
   - 输出推荐理由（context_match / high_confidence / recent / high_weight）

3. 生命周期管理（遗忘曲线）
   - 输出 `decay_score`、`age_days`、`is_stale`
   - 新增 CLI `cleanup`（默认 dry-run，显式 `--apply` 才删除）
   - 半衰期：run=14d，feature=180d，project=365d

## 文件变更

- ✅ memtool_lifecycle.py - 生命周期与衰减评分
- ✅ memtool_rank.py - 混合排序评分
- ✅ memtool_recommend.py - 推荐引擎评分逻辑
- ✅ memtool_core.py - list/search/recommend/cleanup 接口扩展
- ✅ memtool.py - CLI 新增 sort-by / recommend / cleanup
- ✅ mcp_server.py - MCP 新增 memory_recommend / memory_cleanup
- ✅ test_phase2.py - Phase 2 测试
- ✅ README.md / 技术设计文档.md - 文档更新

## 快速体验

### 1) 混合排序
```bash
./memtool.py list --sort-by mixed --exclude-stale
./memtool.py search --query "timeout" --sort-by confidence
```

### 2) 推荐引擎
```bash
./memtool.py recommend --context "redis timeout" --limit 5
```

### 3) 生命周期清理
```bash
# 预览（dry-run）
./memtool.py cleanup --type run --older-than-days 14
# 执行删除
./memtool.py cleanup --type run --older-than-days 14 --apply
```

