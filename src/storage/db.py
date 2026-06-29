"""
src/storage/db.py

SQLite storage for:
1. Resume keyword cache  — same resume → skip LLM extraction
2. JD keyword cache      — same JD → skip LLM extraction
3. Run history           — past results
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)
DB_PATH = Config.DB_PATH


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS resume_cache (
                hash        TEXT PRIMARY KEY,
                keywords_json TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jd_cache (
                hash        TEXT PRIMARY KEY,
                keywords_json TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                job_title   TEXT,
                match_score REAL,
                state_json  TEXT NOT NULL
            );
        """)
    logger.info(f"DB initialised: {DB_PATH}")


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


# ── Resume cache ──────────────────────────────────────────────────────────────

def get_cached_resume(resume_text: str) -> dict | None:
    """Return cached keyword extraction for this resume, or None."""
    init_db()
    h = _hash(resume_text)
    with _conn() as conn:
        row = conn.execute(
            "SELECT keywords_json FROM resume_cache WHERE hash = ?", (h,)
        ).fetchone()
    if row:
        logger.info(f"Resume cache HIT (hash={h})")
        return json.loads(row["keywords_json"])
    logger.info(f"Resume cache MISS (hash={h})")
    return None


def cache_resume(resume_text: str, keywords: dict) -> None:
    init_db()
    h = _hash(resume_text)
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO resume_cache (hash, keywords_json, created_at) VALUES (?,?,?)",
            (h, json.dumps(keywords), datetime.utcnow().isoformat()),
        )
    logger.info(f"Resume cached (hash={h})")


# ── JD cache ──────────────────────────────────────────────────────────────────

def get_cached_jd(jd_text: str) -> dict | None:
    """Return cached keyword extraction for this JD, or None."""
    init_db()
    h = _hash(jd_text)
    with _conn() as conn:
        row = conn.execute(
            "SELECT keywords_json FROM jd_cache WHERE hash = ?", (h,)
        ).fetchone()
    if row:
        logger.info(f"JD cache HIT (hash={h})")
        return json.loads(row["keywords_json"])
    logger.info(f"JD cache MISS (hash={h})")
    return None


def cache_jd(jd_text: str, keywords: dict) -> None:
    init_db()
    h = _hash(jd_text)
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jd_cache (hash, keywords_json, created_at) VALUES (?,?,?)",
            (h, json.dumps(keywords), datetime.utcnow().isoformat()),
        )
    logger.info(f"JD cached (hash={h})")


# ── Run history ───────────────────────────────────────────────────────────────

def save_run(state: dict) -> int:
    init_db()

    def _serial(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, list):
            return [_serial(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _serial(v) for k, v in obj.items()}
        return obj

    job_title   = getattr(state.get("jd_keywords"), "role_title", None)
    match_score = getattr(state.get("match_score"), "overall_score", None)

    with _conn() as conn:
        cursor = conn.execute(
            "INSERT INTO runs (created_at, job_title, match_score, state_json) VALUES (?,?,?,?)",
            (datetime.utcnow().isoformat(), job_title, match_score,
             json.dumps(_serial(dict(state)), default=str)),
        )
    logger.info(f"Run saved id={cursor.lastrowid}")
    return cursor.lastrowid


def list_runs(limit: int = 20) -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, created_at, job_title, match_score FROM runs ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]