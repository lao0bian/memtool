# Memtool MCP Refactoring Task

## Goal
Execute memtool MCP refactoring to slim down the codebase by ignoring deprecated fields at the code level. No data migration needed.

## Technical Spec Reference
Read the full specification at: /Users/bianmengkai/.openclaw/workspace/coder/memtool_refactor_spec.md

## Step-by-Step Tasks

### Step 1: Backup
```bash
git add -A && git commit -m "Pre-refactor backup"
```

### Step 2: Modify memtool_core.py
- Add CORE_FIELDS and DEPRECATED_FIELDS constants at the top
- Add _filter_core_fields helper function
- Modify put_memory to set default values for deprecated fields, ignore user input for deprecated fields
- Modify get_memory to filter output to core fields only
- Modify list_memories to filter output to core fields only
- Remove or comment out _calculate_consolidation_score, _calculate_decay_score functions if they exist

### Step 3: Modify mcp_server.py
- Simplify memory_store tool parameters: remove deprecated fields like context_tags, emotional_valence, urgency_level, task_id, step_id, source, weight, related, session_id
- Simplify memory_assess_knowledge to not rely on deprecated fields

### Step 4: Modify memtool.py CLI
- Remove deprecated field arguments from put command

### Step 5: Update Tests
- Update tests/test_memtool_core.py to remove assertions on deprecated fields
- Update tests/test_mcp_server.py similarly
- Add test_deprecated_fields_ignored test

### Step 6: Run Tests
```bash
source .venv/bin/activate && pytest tests/ -v
```

### Step 7: Integration Test
```bash
mcporter call memtool.memory_store type:project key:"test:refactor" content:"测试瘦身后的存储"
mcporter call memtool.memory_recall type:project key:"test:refactor"
mcporter call memtool.memory_list limit:5
```

### Step 8: Final Commit
```bash
git add -A && git commit -m "refactor: 瘦身重构 - 移除低价值特性，聚焦核心能力"
```

## Important Constraints
- Maintain backward compatibility with existing 56 memories
- No data migration - deprecated fields ignored at code level only
- Keep core capabilities: store/retrieve/vector search/access tracking
- All tests must pass before final commit

## Project Info
- Project path: /Users/bianmengkai/Downloads/memtool_mvp
- Virtual env: .venv
- Database: /Users/bianmengkai/.memtool/shared.db
