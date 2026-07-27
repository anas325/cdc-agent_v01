"""Supabase (Postgres) persistence: shared connection pool, the LangGraph
checkpointer, and a tiny DAO for the per-user run index.

The checkpointer replaces the old in-RAM MemorySaver so runs survive server
restarts. The `runs` table maps each thread_id to the user who owns it, giving
us a per-user "my runs" history to resume from.

Connections are configured for pgbouncer compatibility (autocommit + no
server-side prepared statements) so a Supabase session pooler works. Use the
session pooler / direct connection, NOT the transaction pooler (6543).
"""

from __future__ import annotations

import os
from functools import lru_cache

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_CONNECT_KWARGS = {
    "autocommit": True,
    "prepare_threshold": None,
    "row_factory": dict_row,
}


def _db_url() -> str:
    url = os.environ.get("SUPABASE_DB_URL", "")
    if not url:
        try:
            import streamlit as st

            url = st.secrets.get("SUPABASE_DB_URL", "")
        except Exception:
            url = ""
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL non configuré (Cloud secrets ou .streamlit/secrets.toml)."
        )
    return url


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """Process-wide connection pool. Opened lazily on first use."""
    pool = ConnectionPool(
        conninfo=_db_url(),
        max_size=5,
        kwargs=_CONNECT_KWARGS,
        open=True,
    )
    _ensure_runs_table(pool)
    return pool


def _ensure_runs_table(pool: ConnectionPool) -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                thread_id  uuid PRIMARY KEY,
                username   text NOT NULL,
                title      text,
                status     text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS runs_username_idx ON runs (username)")


@lru_cache(maxsize=1)
def get_checkpointer():
    """PostgresSaver backed by the shared pool. `setup()` is idempotent."""
    from langgraph.checkpoint.postgres import PostgresSaver

    saver = PostgresSaver(get_pool())
    saver.setup()
    return saver


# ---------------------------------------------------------------------------
# runs DAO
# ---------------------------------------------------------------------------


def create_run(thread_id: str, username: str, title: str | None) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO runs (thread_id, username, title, status)
            VALUES (%s, %s, %s, 'running')
            ON CONFLICT (thread_id) DO NOTHING
            """,
            (thread_id, username, title),
        )


def touch_run(thread_id: str, status: str) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            "UPDATE runs SET status = %s, updated_at = now() WHERE thread_id = %s",
            (status, thread_id),
        )


def list_runs(username: str) -> list[dict]:
    """Return the user's runs, newest activity first."""
    with get_pool().connection() as conn:
        cur = conn.execute(
            """
            SELECT thread_id, title, status, created_at, updated_at
            FROM runs
            WHERE username = %s
            ORDER BY updated_at DESC
            """,
            (username,),
        )
        return [dict(row) for row in cur.fetchall()]
