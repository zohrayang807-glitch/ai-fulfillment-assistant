"""
db.py — Supabase 数据访问层
所有读写操作封装在此，返回结构与原 jsonl 读写一致（dict/list）。
如 Supabase 不可用则自动 fallback 到本地 jsonl。
"""
import json
import os
from datetime import datetime
from pathlib import Path

# ── 环境变量 ──
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── 本地路径（fallback）──
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "eval"

CONV_FILE = LOGS_DIR / "conversations.jsonl"
EVAL_LOG = LOGS_DIR / "evaluations.jsonl"
EVALUATIONS_FILE = EVAL_LOG
CASES_FILE = EVAL_DIR / "cases.jsonl"
BUG_FILE = LOGS_DIR / "bug_feedback.jsonl"
PV_FILE = LOGS_DIR / "prompt_versions.jsonl"
TOKEN_FILE = LOGS_DIR / "token_usage.jsonl"


def _load_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]"""
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def _save_jsonl(path, items):
    """写入 jsonl 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _append_jsonl(path, item):
    """追加一条到 jsonl 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Supabase 客户端（懒加载）──
_client = None


def _get_client():
    """获取 Supabase 客户端，不可用时返回 None"""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _client
    except Exception:
        return None


# ═══════════════════════════════════════════════
# 1. conversations（对话日志）
# ═══════════════════════════════════════════════

def insert_conversation(ts=None, user="我", question="", intent="", answer=""):
    """插入一条对话记录"""
    row = {
        "ts": ts or datetime.now().isoformat(),
        "user": user,
        "question": question,
        "intent": intent,
        "answer": answer,
    }
    client = _get_client()
    if client:
        try:
            client.table("conversations").insert(row).execute()
            return True
        except Exception:
            pass
    _append_jsonl(CONV_FILE, row)
    return True


def get_conversations(limit=200):
    """获取对话列表，按时间倒序"""
    client = _get_client()
    if client:
        try:
            resp = client.table("conversations").select("*").order("ts", desc=True).limit(limit).execute()
            return resp.data
        except Exception:
            pass
    items = _load_jsonl(CONV_FILE)
    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items[:limit]


def delete_conversation(row_id):
    """删除一条对话记录"""
    client = _get_client()
    if client:
        try:
            client.table("conversations").delete().eq("id", row_id).execute()
            return True
        except Exception:
            pass
    # fallback: jsonl 不支持按 id 删，返回 False
    return False


# ═══════════════════════════════════════════════
# 2. evaluations（评审记录）
# ═══════════════════════════════════════════════

def insert_evaluation(ts=None, question="", answer="", scores=None, overall=0,
                       comment="", user="", framework=None, model_judge=None):
    """插入一条评审记录"""
    row = {
        "ts": ts or datetime.now().isoformat(),
        "user": user,
        "question": question,
        "answer": answer,
        "scores": scores or {},
        "overall": overall,
        "comment": comment,
        "framework": framework,
        "model_judge": model_judge,
    }
    client = _get_client()
    if client:
        try:
            client.table("evaluations").insert(row).execute()
            return True
        except Exception as e:
            print(f"[db] insert_evaluation 失败，已回退本地: {e}")
    _append_jsonl(EVAL_LOG, row)
    return True


def get_evaluations(limit=200):
    """获取评审记录，按时间倒序"""
    client = _get_client()
    if client:
        try:
            resp = client.table("evaluations").select("*").order("ts", desc=True).limit(limit).execute()
            return resp.data
        except Exception:
            pass
    items = _load_jsonl(EVAL_LOG)
    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items[:limit]


# ═══════════════════════════════════════════════
# 3. cases（Eval 用例）
# ═══════════════════════════════════════════════

def normalize_case(case_dict: dict) -> dict:
    """规范化用例字段，保证写入格式统一。
    - expected_intents: 数组（兼容 expect_intent 字符串）
    - banned / required / required_any: 数组（不是 null 或字符串）
    - banned_answer_contains_intent: bool
    - turns: 数组或 None（单轮为 None）
    - 不带 id（Supabase 自动生成）
    """
    row = dict(case_dict)
    row.pop("id", None)  # Supabase 自增 id

    # ── 字段名统一 ──
    # expect_intent (legacy) → expected_intents (数组)
    if "expect_intent" in row:
        val = row.pop("expect_intent")
        if isinstance(val, str):
            row["expected_intents"] = [v.strip() for v in val.split(",") if v.strip()]
        elif isinstance(val, list):
            row["expected_intents"] = val
    # banned_words (legacy) → banned
    if "banned_words" in row:
        row["banned"] = row.pop("banned_words")
    # required_words (legacy) → required
    if "required_words" in row:
        row["required"] = row.pop("required_words")

    # ── 数组字段强制为 list ──
    for key in ("expected_intents", "banned", "required", "required_any"):
        val = row.get(key)
        if val is None:
            row[key] = []
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                row[key] = parsed if isinstance(parsed, list) else [val]
            except Exception:
                row[key] = [v.strip() for v in val.split(",") if v.strip()]
        elif not isinstance(val, list):
            row[key] = [val]

    # ── banned_answer_contains_intent: 必须是 bool ──
    val = row.get("banned_answer_contains_intent")
    if val is None:
        row["banned_answer_contains_intent"] = False
    elif isinstance(val, list):
        # 数组（历史遗留）有值即 True，转为 bool（否则写 Supabase boolean 字段会报错）
        row["banned_answer_contains_intent"] = bool(val)
    elif not isinstance(val, bool):
        row["banned_answer_contains_intent"] = bool(val)

    # ── turns: 多轮才有，单轮为 None ──
    turns = row.get("turns")
    if turns is not None:
        if isinstance(turns, str):
            try:
                row["turns"] = json.loads(turns)
            except Exception:
                row["turns"] = [turns]
        elif not isinstance(turns, list):
            row["turns"] = None
    # 单轮用例不带 turns 字段
    if row.get("turns") is None:
        row.pop("turns", None)

    # ── 字符串字段兜底 ──
    for key in ("q", "section", "user", "reason", "note"):
        if key not in row:
            row[key] = ""

    # ── expected_intents 规范化 ──
    # 只保留合法 operation 的期望意图；非法格式置空（不崩后台，Eval 里显示未通过即可）
    _VALID_OPS = {"query", "compare", "aggregate", "recommend", "capability", "引导", "other", "unsupported"}
    _expected = row.get("expected_intents") or []
    if isinstance(_expected, list):
        _valid = []
        for _exp in _expected:
            if not isinstance(_exp, str) or not _exp:
                continue
            _op = _exp.split("×")[0].strip()
            if _op in _VALID_OPS:
                _valid.append(_exp)
        row["expected_intents"] = _valid

    return row


def get_cases():
    """获取全部 Eval 用例（数组字段 null → [] 规范化）"""
    client = _get_client()
    data = []
    if client:
        try:
            resp = client.table("cases").select("*").order("id").execute()
            data = resp.data
        except Exception:
            pass
    if not data:
        data = _load_jsonl(CASES_FILE)
    # 数组字段 null → []，避免下游 join 崩
    _LIST_FIELDS = ("banned", "required", "required_any", "expected_intents", "turns")
    for c in data:
        for k in _LIST_FIELDS:
            v = c.get(k)
            if v is None:
                c[k] = []
            elif not isinstance(v, list):
                c[k] = [v]
    return data


def add_case(case_dict):
    """新增一条 Eval 用例（自动规范化）"""
    row = normalize_case(case_dict)

    client = _get_client()
    if client:
        try:
            client.table("cases").insert(row).execute()
            return True
        except Exception:
            pass
    _append_jsonl(CASES_FILE, row)
    return True


def update_case(case_id, updates):
    """更新一条 Eval 用例"""
    client = _get_client()
    if client:
        try:
            # jsonb 字段序列化
            for key in ("expected_intents", "banned", "required", "required_any", "turns"):
                if key in updates and not isinstance(updates[key], (dict, list)):
                    try:
                        updates[key] = json.loads(updates[key])
                    except Exception:
                        pass
            client.table("cases").update(updates).eq("id", case_id).execute()
            return True
        except Exception:
            pass
    # fallback: 按 index 更新
    cases = _load_jsonl(CASES_FILE)
    for i, c in enumerate(cases):
        if c.get("id") == case_id or i == case_id:
            cases[i].update(updates)
            _save_jsonl(CASES_FILE, cases)
            return True
    return False


def delete_case(case_id):
    """删除一条 Eval 用例"""
    client = _get_client()
    if client:
        try:
            client.table("cases").delete().eq("id", case_id).execute()
            return True
        except Exception:
            pass
    # fallback
    cases = _load_jsonl(CASES_FILE)
    new_cases = [c for c in cases if c.get("id") != case_id]
    if len(new_cases) < len(cases):
        _save_jsonl(CASES_FILE, new_cases)
        return True
    return False


# ═══════════════════════════════════════════════
# 4. bug_feedback（BUG 反馈）
# ═══════════════════════════════════════════════

def insert_bug(ts=None, user="admin", question="", intent="", answer="",
               status="待修复", reason="", note=""):
    """插入一条 BUG 反馈"""
    row = {
        "ts": ts or datetime.now().isoformat(),
        "user": user,
        "question": question,
        "intent": intent,
        "answer": answer,
        "status": status,
        "reason": reason,
        "note": note,
    }
    client = _get_client()
    if client:
        try:
            client.table("bug_feedback").insert(row).execute()
            return True
        except Exception:
            pass
    _append_jsonl(BUG_FILE, row)
    return True


def get_bugs(status_filter=None, limit=200):
    """获取 BUG 反馈列表"""
    client = _get_client()
    if client:
        try:
            q = client.table("bug_feedback").select("*").order("ts", desc=True).limit(limit)
            if status_filter and status_filter != "全部":
                q = q.eq("status", status_filter)
            resp = q.execute()
            return resp.data
        except Exception:
            pass
    items = _load_jsonl(BUG_FILE)
    if status_filter and status_filter != "全部":
        items = [b for b in items if b.get("status") == status_filter]
    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items[:limit]


def update_bug_status(bug_id, new_status):
    """更新 BUG 状态（待修复 → 已修复）"""
    client = _get_client()
    if client:
        try:
            client.table("bug_feedback").update({
                "status": new_status,
                "fixed_at": datetime.now().isoformat() if new_status == "已修复" else None,
            }).eq("id", bug_id).execute()
            return True
        except Exception:
            pass
    # fallback
    bugs = _load_jsonl(BUG_FILE)
    for b in bugs:
        if b.get("id") == bug_id or b.get("question") == bug_id:
            b["status"] = new_status
            if new_status == "已修复":
                b["fixed_at"] = datetime.now().isoformat()
            break
    _save_jsonl(BUG_FILE, bugs)
    return True


# ═══════════════════════════════════════════════
# 5. prompt_versions（提示词版本）
# ═══════════════════════════════════════════════

def save_prompt_version(key, content, version, user="admin", ts=None):
    """保存一个提示词版本快照"""
    row = {
        "ts": ts or datetime.now().isoformat(),
        "key": key,
        "content": content,
        "version": version,
        "user": user,
    }
    client = _get_client()
    if client:
        try:
            client.table("prompt_versions").insert(row).execute()
            return True
        except Exception:
            pass
    _append_jsonl(PV_FILE, row)
    return True


def get_prompt_versions(key=None):
    """获取提示词版本列表"""
    client = _get_client()
    if client:
        try:
            q = client.table("prompt_versions").select("*").order("ts")
            if key:
                q = q.eq("key", key)
            resp = q.execute()
            return resp.data
        except Exception:
            pass
    items = _load_jsonl(PV_FILE)
    if key:
        items = [v for v in items if v.get("key") == key]
    return items


def delete_prompt_version(version_id):
    """删除一个提示词版本"""
    client = _get_client()
    if client:
        try:
            client.table("prompt_versions").delete().eq("id", version_id).execute()
            return True
        except Exception:
            pass
    # fallback
    versions = _load_jsonl(PV_FILE)
    new_versions = [v for v in versions if v.get("id") != version_id]
    if len(new_versions) < len(versions):
        _save_jsonl(PV_FILE, new_versions)
        return True
    return False


# ═══════════════════════════════════════════════
# 6. token_usage（Token 用量）
# ═══════════════════════════════════════════════

def insert_token_usage(caller="unknown", model="", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, ts=None):
    """记录一次 API 调用的 token 用量"""
    row = {
        "ts": ts or datetime.now().isoformat(),
        "caller": caller,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
    }
    client = _get_client()
    if client:
        try:
            client.table("token_usage").insert(row).execute()
            return True
        except Exception as e:
            print(f"[db] insert_token_usage 失败，已回退本地: {e}")
    _append_jsonl(TOKEN_FILE, row)
    return True


def get_token_usage(limit=1000):
    """获取 Token 用量记录，按时间倒序"""
    client = _get_client()
    if client:
        try:
            resp = client.table("token_usage").select("*").order("ts", desc=True).limit(limit).execute()
            return resp.data
        except Exception:
            pass
    items = _load_jsonl(TOKEN_FILE)
    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items[:limit]


# ═══════════════════════════════════════════════
# 诊断工具
# ═══════════════════════════════════════════════

def test_connection():
    """测试 Supabase 连接，返回 (success, message)"""
    client = _get_client()
    if not client:
        return False, "Supabase 未配置（SUPABASE_URL 或 SUPABASE_KEY 缺失）"
    try:
        resp = client.table("conversations").select("id", count="exact").limit(1).execute()
        return True, f"✅ Supabase 连接成功（conversations 表有 {resp.count or 0} 条）"
    except Exception as e:
        return False, f"❌ Supabase 连接失败：{e}"


def is_using_supabase():
    """当前是否走 Supabase（而非 fallback jsonl）"""
    return _get_client() is not None
