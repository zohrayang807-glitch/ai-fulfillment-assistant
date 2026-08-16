#!/usr/bin/env python3
"""
migrate_to_db.py — 将本地 JSONL 数据迁移到 Supabase
用法：python3 migrate_to_db.py
"""
import json
import sys
from pathlib import Path

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from db import _get_client, _load_jsonl

LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "eval"


def migrate_conversations(client):
    """迁移对话日志"""
    items = _load_jsonl(LOGS_DIR / "conversations.jsonl")
    if not items:
        print("  conversations.jsonl 为空，跳过")
        return 0

    rows = []
    for item in items:
        rows.append({
            "ts": item.get("ts"),
            "user": item.get("user", "我"),
            "question": item.get("question", ""),
            "intent": item.get("intent", ""),
            "answer": item.get("answer", ""),
        })

    client.table("conversations").insert(rows).execute()
    print(f"  ✅ conversations: 迁移 {len(rows)} 条")
    return len(rows)


def migrate_evaluations(client):
    """迁移评审记录"""
    items = _load_jsonl(LOGS_DIR / "evaluations.jsonl")
    if not items:
        print("  evaluations.jsonl 为空，跳过")
        return 0

    rows = []
    for item in items:
        rows.append({
            "ts": item.get("ts"),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "framework": item.get("framework", {}),
            "model_judge": item.get("model_judge", {}),
        })

    # 分批插入（Supabase 限制）
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        client.table("evaluations").insert(batch).execute()

    print(f"  ✅ evaluations: 迁移 {len(rows)} 条")
    return len(rows)


def migrate_cases(client):
    """迁移 Eval 用例"""
    items = _load_jsonl(EVAL_DIR / "cases.jsonl")
    if not items:
        print("  cases.jsonl 为空，跳过")
        return 0

    rows = []
    for item in items:
        row = {
            "section": item.get("section", "防复发"),
            "q": item.get("q"),
            "expected_intents": item.get("expected_intents"),
            "banned": item.get("banned"),
            "required": item.get("required"),
            "required_any": item.get("required_any"),
            "user": item.get("user", "我"),
            "reason": item.get("reason"),
            "note": item.get("note"),
            "turns": item.get("turns"),
            # banned_answer_contains_intent 历史上有时是数组、有时是 bool，
            # 统一转 bool（有值 → True，空 → False）
            "banned_answer_contains_intent": bool(item.get("banned_answer_contains_intent")),
            "legacy_id": item.get("id"),
        }
        rows.append(row)

    client.table("cases").insert(rows).execute()
    print(f"  ✅ cases: 迁移 {len(rows)} 条")
    return len(rows)


def migrate_bug_feedback(client):
    """迁移 BUG 反馈"""
    items = _load_jsonl(LOGS_DIR / "bug_feedback.jsonl")
    if not items:
        print("  bug_feedback.jsonl 为空，跳过")
        return 0

    rows = []
    for item in items:
        rows.append({
            "ts": item.get("ts"),
            "user": item.get("user", "admin"),
            "question": item.get("question", ""),
            "intent": item.get("intent", ""),
            "answer": item.get("answer", ""),
            "status": item.get("status", "待修复"),
            "reason": item.get("reason", ""),
            "note": item.get("note", ""),
            "fixed_at": item.get("fixed_at"),
        })

    client.table("bug_feedback").insert(rows).execute()
    print(f"  ✅ bug_feedback: 迁移 {len(rows)} 条")
    return len(rows)


def migrate_prompt_versions(client):
    """迁移提示词版本"""
    items = _load_jsonl(LOGS_DIR / "prompt_versions.jsonl")
    if not items:
        print("  prompt_versions.jsonl 为空，跳过")
        return 0

    rows = []
    for item in items:
        rows.append({
            "ts": item.get("ts"),
            "key": item.get("key"),
            "content": item.get("content"),
            "version": item.get("version"),
            "user": item.get("user", "admin"),
        })

    client.table("prompt_versions").insert(rows).execute()
    print(f"  ✅ prompt_versions: 迁移 {len(rows)} 条")
    return len(rows)


def main():
    print("═══ Supabase 数据迁移 ═══\n")

    client = _get_client()
    if not client:
        print("❌ Supabase 未配置。请检查 .env 中的 SUPABASE_URL 和 SUPABASE_KEY")
        sys.exit(1)

    print("✅ Supabase 连接成功\n")

    total = 0
    print("开始迁移...")
    total += migrate_conversations(client)
    total += migrate_evaluations(client)
    total += migrate_cases(client)
    total += migrate_bug_feedback(client)
    total += migrate_prompt_versions(client)

    print(f"\n═══ 迁移完成：共 {total} 条 ═══")
    print("本地 JSONL 文件已保留（作为备份）")


if __name__ == "__main__":
    main()
