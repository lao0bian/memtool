#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memtool: 基于 SQLite 的记忆管理工具（MVP）
- 面向 Claude Code / Codex 等 agent：输出默认 JSON，便于解析与调用
- 支持：初始化、写入/更新、读取、列表、检索（FTS5 优先）、删除、导入/导出

依赖：Python 3.9+（含 jieba 分词用于中文检索）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional
import yaml
from string import Formatter

from memtool_core import DEFAULT_DB, MemtoolError, MemoryStore, init_db, parse_tags


def read_stdin() -> str:
    return sys.stdin.read()


def load_content(args: argparse.Namespace) -> str:
    if getattr(args, "content", None) is not None:
        return args.content
    if getattr(args, "file", None) is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    return read_stdin()


def emit(obj: Any, fmt: str) -> None:
    if fmt == "json":
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        # text：尽量给人看
        if isinstance(obj, dict) and "content" in obj and len(obj) == 1:
            sys.stdout.write(str(obj["content"]))
        else:
            sys.stdout.write(str(obj) + "\n")


def _exit_with_error(e: MemtoolError, fmt: str) -> None:
    emit(e.payload, fmt)
    sys.exit(e.exit_code)


def cmd_put(args: argparse.Namespace) -> None:
    try:
        content = load_content(args)
        tags = parse_tags(args.tag)
        store = MemoryStore(args.db)
        result = store.put(
            item_id=args.id,
            type=args.type,
            key=args.key,
            content=content,
            task_id=args.task_id,
            step_id=args.step_id,
            tags=tags,
            source=args.source,
            weight=args.weight,
            confidence_level=getattr(args, "confidence", "medium"),
            verified_by=getattr(args, "verified_by", None),
        )
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def cmd_get(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        result = store.get(
            item_id=args.id,
            type=args.type,
            key=args.key,
            task_id=args.task_id,
            step_id=args.step_id,
        )
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def cmd_list(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        tags = parse_tags(args.tag)
        result = store.list(
            type=args.type,
            task_id=args.task_id,
            step_id=args.step_id,
            key=args.key,
            tags=tags,
            limit=args.limit,
            sort_by=getattr(args, "sort_by", "updated"),
            include_stale=not getattr(args, "exclude_stale", False),
        )
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def cmd_search(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        result = store.search(
            query=args.query,
            type=args.type,
            task_id=args.task_id,
            step_id=args.step_id,
            key=args.key,
            limit=args.limit,
            sort_by=getattr(args, "sort_by", "updated"),
            include_stale=not getattr(args, "exclude_stale", False),
        )
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def cmd_delete(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        result = store.delete(item_id=args.id)
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def cmd_export(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        rows = store.export_items()
        out_path = args.output
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        emit({"ok": True, "output": os.path.abspath(out_path), "count": len(rows)}, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def cmd_import(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        items = _iter_jsonl(args.input)
        try:
            result = store.import_items(items)
        finally:
            if hasattr(items, "close"):
                items.close()
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def cmd_recommend(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        tags = parse_tags(args.tag)
        result = store.recommend(
            context=args.context,
            type=args.type,
            task_id=args.task_id,
            tags=tags,
            key_prefix=args.key_prefix,
            limit=args.limit,
            include_stale=not getattr(args, "exclude_stale", False),
        )
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def cmd_cleanup(args: argparse.Namespace) -> None:
    try:
        store = MemoryStore(args.db)
        result = store.cleanup(
            type=args.type,
            older_than_days=args.older_than_days,
            stale_threshold=args.stale_threshold,
            limit=args.limit,
            apply=args.apply,
        )
        emit(result, args.format)
    except MemtoolError as e:
        _exit_with_error(e, args.format)


def _load_templates() -> dict:
    """从 memtool_templates.yaml 加载模板定义"""
    template_file = os.path.join(os.path.dirname(__file__), "memtool_templates.yaml")
    if not os.path.exists(template_file):
        raise MemtoolError({
            "ok": False,
            "error": "NOT_FOUND",
            "message": f"Template file not found: {template_file}"
        }, 3)
    with open(template_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("templates", {})


def cmd_template(args: argparse.Namespace) -> None:
    """处理 template 子命令"""
    try:
        templates = _load_templates()
        template_name = args.template_name

        if template_name == "list":
            # 列出所有可用的模板
            result = {
                "ok": True,
                "templates": [
                    {
                        "name": name,
                        "description": tmpl.get("description", ""),
                        "type": tmpl.get("type", ""),
                        "fields": [f["name"] for f in tmpl.get("fields", [])],
                    }
                    for name, tmpl in templates.items()
                ]
            }
            emit(result, args.format)
            return

        if template_name not in templates:
            raise MemtoolError({
                "ok": False,
                "error": "NOT_FOUND",
                "message": f"Template not found: {template_name}",
                "available": list(templates.keys())
            }, 3)

        tmpl = templates[template_name]

        # 检查必填字段
        required_fields = {
            f["name"] for f in tmpl.get("fields", []) if f.get("required", True)
        }
        provided_fields = {k: v for k, v in vars(args).items()
                          if k not in ["db", "format", "func", "template_name", "cmd", "tag", "source", "weight"]
                          and v is not None}

        missing = required_fields - set(provided_fields.keys())
        if missing:
            raise MemtoolError({
                "ok": False,
                "error": "PARAM_ERROR",
                "message": f"Missing required fields: {', '.join(missing)}",
                "required_fields": list(required_fields),
                "provided_fields": list(provided_fields.keys())
            }, 2)

        # 生成自动 key
        auto_key_pattern = tmpl.get("auto_key", "")
        if auto_key_pattern:
            key = auto_key_pattern
            for field_name in Formatter().parse(auto_key_pattern):
                if field_name[1]:  # {name} 的占位符
                    key = key.replace("{" + field_name[1] + "}", str(provided_fields.get(field_name[1], "")))
        else:
            key = template_name

        # 生成内容
        content_template = tmpl.get("content_template", "")
        if content_template:
            content = content_template
            for field_name in Formatter().parse(content_template):
                if field_name[1]:  # {name} 的占位符
                    content = content.replace("{" + field_name[1] + "}", str(provided_fields.get(field_name[1], "")))
        else:
            content = json.dumps(provided_fields, ensure_ascii=False, indent=2)

        # 调用 put 存储
        store = MemoryStore(args.db)
        result = store.put(
            item_id=None,
            type=tmpl.get("type", "run"),
            key=key,
            content=content,
            task_id=args.task_id,
            step_id=args.step_id,
            tags=parse_tags(args.tag),
            source=args.source,
            weight=args.weight,
            confidence_level=getattr(args, "confidence", "medium"),
            verified_by=getattr(args, "verified_by", None),
        )
        emit(result, args.format)

    except MemtoolError as e:
        _exit_with_error(e, args.format)
    except yaml.YAMLError as e:
        _exit_with_error(MemtoolError({
            "ok": False,
            "error": "CONFIG_ERROR",
            "message": f"Failed to parse template file: {e}"
        }, 4), args.format)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memtool", description="SQLite 记忆管理工具（便于 agent 读取/更新/检索）")
    p.add_argument("--db", default=DEFAULT_DB, help="SQLite 数据库路径（或设置环境变量 MEMTOOL_DB）")
    p.add_argument("--format", default="json", choices=["json","text"], help="输出格式，默认 json 便于 agent 解析")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="初始化数据库")
    s.set_defaults(func=lambda a: emit(init_db(a.db), a.format))

    s = sub.add_parser("put", help="写入或更新一条记忆（id 存在则更新）")
    s.add_argument("--id", help="可选：指定 id；不填则自动生成")
    s.add_argument("--type", required=True, choices=["project","feature","run"])
    s.add_argument("--task-id")
    s.add_argument("--step-id")
    s.add_argument("--key", required=True)
    s.add_argument("--tag", action="append", help="可重复；或用逗号分隔，例如 --tag auth,login")
    s.add_argument("--source", help="来源，例如 PRD/stacktrace/manual")
    s.add_argument("--weight", default=1.0, help="权重（Router 可用），默认 1.0")
    s.add_argument("--confidence", choices=["high","medium","low"], default="medium", help="置信度：high/medium/low，默认 medium")
    s.add_argument("--verified-by", help="可选：验证来源，例如 code_review#123")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--content", help="直接写 content")
    g.add_argument("--file", help="从文件读取 content；不填则从 stdin 读取")
    s.set_defaults(func=cmd_put)

    s = sub.add_parser("get", help="按 id 获取，或按 (type,key,task_id,step_id) 获取最新一条")
    s.add_argument("--id")
    s.add_argument("--type", choices=["project","feature","run"])
    s.add_argument("--key")
    s.add_argument("--task-id")
    s.add_argument("--step-id")
    s.set_defaults(func=cmd_get)

    s = sub.add_parser("list", help="列表查询（默认按 updated_at 倒序）")
    s.add_argument("--type", choices=["project","feature","run"])
    s.add_argument("--task-id")
    s.add_argument("--step-id")
    s.add_argument("--key")
    s.add_argument("--tag", action="append")
    s.add_argument("--limit", default=50)
    s.add_argument("--sort-by", choices=["updated","confidence","recency","mixed"], default="updated")
    s.add_argument("--exclude-stale", action="store_true", help="过滤掉衰减后的旧记忆")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("search", help="全文检索（FTS5 优先，自动降级 LIKE）")
    s.add_argument("--query", required=True)
    s.add_argument("--type", choices=["project","feature","run"])
    s.add_argument("--task-id")
    s.add_argument("--step-id")
    s.add_argument("--key")
    s.add_argument("--limit", default=20)
    s.add_argument("--sort-by", choices=["updated","confidence","recency","mixed"], default="updated")
    s.add_argument("--exclude-stale", action="store_true", help="过滤掉衰减后的旧记忆")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("delete", help="按 id 删除")
    s.add_argument("--id", required=True)
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser("export", help="导出为 JSONL（每行一条）")
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("import", help="从 JSONL 导入（按 id upsert）")
    s.add_argument("--input", required=True)
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("recommend", help="基于上下文推荐相关记忆")
    s.add_argument("--context", help="当前上下文或问题描述")
    s.add_argument("--type", choices=["project","feature","run"])
    s.add_argument("--task-id")
    s.add_argument("--tag", action="append")
    s.add_argument("--key-prefix")
    s.add_argument("--limit", default=10)
    s.add_argument("--exclude-stale", action="store_true", help="过滤掉衰减后的旧记忆")
    s.set_defaults(func=cmd_recommend)

    s = sub.add_parser("cleanup", help="清理过期/衰减记忆（默认 dry-run）")
    s.add_argument("--type", choices=["project","feature","run"])
    s.add_argument("--older-than-days", type=float, help="按更新时间过滤（天）")
    s.add_argument("--stale-threshold", type=float, help="衰减阈值（0-1），低于则清理")
    s.add_argument("--limit", default=1000)
    s.add_argument("--apply", action="store_true", help="执行删除（默认仅预览）")
    s.set_defaults(func=cmd_cleanup)

    # template 子命令
    s = sub.add_parser("template", help="使用模板快速记录信息（template list 查看可用模板）")
    s.add_argument("template_name", help="模板名称（如 error_analysis, design_decision）或 'list' 查看所有模板")
    s.add_argument("--task-id", help="可选：关联的 task_id")
    s.add_argument("--step-id", help="可选：关联的 step_id")
    s.add_argument("--tag", action="append", help="可重复标签")
    s.add_argument("--source", help="可选：来源")
    s.add_argument("--weight", default=1.0, help="可选：权重，默认 1.0")
    s.add_argument("--confidence", choices=["high","medium","low"], default="medium", help="可选：置信度，默认 medium")
    s.add_argument("--verified-by", help="可选：验证来源")

    s.set_defaults(func=cmd_template)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()

    # 特殊处理 template 子命令：使用 parse_known_args 以支持动态参数
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "template":
        # 自定义处理 template 子命令的参数
        from memtool_core import parse_tags
        # 提取已知参数和未知参数
        known_args, unknown_args = parser.parse_known_args(argv)

        # 处理未知参数（动态字段）
        template_fields = {}
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if arg.startswith("--"):
                field_name = arg[2:].replace("-", "_")
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    template_fields[field_name] = unknown_args[i + 1]
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # 合并到 args 中
        for key, value in template_fields.items():
            setattr(known_args, key, value)

        args = known_args
    else:
        args = parser.parse_args(argv)

    # get 的参数校验：非 id 查询需要 type + key
    if args.cmd == "get" and not args.id:
        if not args.type or not args.key:
            emit({"ok": False, "error": "PARAM_ERROR", "message": "get command requires --type and --key when --id is not provided", "hint": "Use --id <id> or --type <type> --key <key>"}, args.format)
            sys.exit(2)

    # 需要初始化就能用：如果 db 不存在且不是 init 和 template list，则提示
    if args.cmd != "init" and not (args.cmd == "template" and args.template_name == "list") and not os.path.exists(args.db):
        emit({"ok": False, "error": "PARAM_ERROR", "message": f"Database not found: {args.db}", "hint": f"Run: memtool --db {args.db} init"}, args.format)
        sys.exit(2)

    args.func(args)


if __name__ == "__main__":
    main()
