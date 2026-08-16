-- ========================================================
-- 履约 AI 助手 V2.0 — Supabase 建表脚本
-- 复制此文件内容到 Supabase Dashboard > SQL Editor 执行
-- ========================================================

-- 1. 对话日志
CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "user" TEXT NOT NULL DEFAULT 'anonymous',
    question TEXT NOT NULL,
    intent TEXT,
    answer TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_ts ON conversations (ts DESC);

-- 2. 评审记录
CREATE TABLE IF NOT EXISTS evaluations (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "user" TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    scores JSONB DEFAULT '{}',
    overall NUMERIC DEFAULT 0,
    comment TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_evaluations_ts ON evaluations (ts DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_user ON evaluations ("user");

-- 3. 坏例库
CREATE TABLE IF NOT EXISTS cases (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    intent TEXT,
    ban TEXT DEFAULT '',
    why TEXT DEFAULT '',
    fix TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status);

-- 4. Bug 反馈
CREATE TABLE IF NOT EXISTS bug_feedback (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "user" TEXT NOT NULL DEFAULT '',
    bug TEXT NOT NULL,
    expect TEXT DEFAULT '',
    status TEXT DEFAULT '待处理',
    reply TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bug_feedback_ts ON bug_feedback (ts DESC);

-- 5. 提示词版本
CREATE TABLE IF NOT EXISTS prompt_versions (
    id BIGSERIAL PRIMARY KEY,
    section TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0',
    content TEXT NOT NULL,
    user TEXT NOT NULL DEFAULT 'system',
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    note TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_prompt_versions_section ON prompt_versions (section, version);
