"""SQL access layer for the app + agent tools.

Connects to a Databricks SQL warehouse via `databricks-sql-connector` using the Databricks Apps
default service principal (`DATABRICKS_HOST` / `DATABRICKS_TOKEN` are injected at runtime).
All statements are parameterized with `?` placeholders to keep LLM-sourced writes safe.
"""
import os

from databricks import sql as dsql


def _env(name, default=None):
    return os.environ.get(name, default)


def _quote(name):
    return f"`{name}`"


class Database:
    def __init__(self, host=None, http_path=None, token=None, catalog=None, schema=None):
        self.host = host or _env("DATABRICKS_HOST", "").replace("https://", "").rstrip("/")
        self.http_path = http_path or _env("SQL_WAREHOUSE_PATH")
        self.token = token or _env("DATABRICKS_TOKEN")
        self.catalog = catalog or _env("CATALOG", "anime_night_planner")
        self.schema = schema or _env("SCHEMA", "default")

    def table(self, name):
        """Fully-qualified, quoted table name (handles reserved words like `groups`)."""
        return f"{_quote(self.catalog)}.{_quote(self.schema)}.{_quote(name)}"

    def _connect(self):
        if not self.http_path:
            raise RuntimeError(
                "SQL_WAREHOUSE_PATH is not set — add it to app.yaml (the warehouse's http path)."
            )
        return dsql.connect(
            server_hostname=self.host,
            http_path=self.http_path,
            access_token=self.token,
        )

    def query(self, sql, params=None):
        """Run a SELECT and return a list of dicts (lowercased column names)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                cols = [d[0].lower() for d in cur.description] if cur.description else []
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def execute(self, sql, params=None):
        """Run a DML statement (INSERT / UPDATE / MERGE)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
        return True

    def groups(self):
        return self.query(
            f"""
            SELECT g.group_id, g.name, collect_list(u.name) AS members
            FROM {self.table('groups')} g
            LEFT JOIN {self.table('group_members')} gm ON g.group_id = gm.group_id
            LEFT JOIN {self.table('users')} u ON gm.user_id = u.user_id
            GROUP BY g.group_id, g.name
            ORDER BY g.group_id
            """
        )

    def watchlist(self, group_id):
        return self.query(
            f"""
            SELECT w.anime_id, a.title, w.status, w.added_at,
                   a.score AS mal_score, a.episodes, a.genres
            FROM {self.table('watchlist_items')} w
            LEFT JOIN {self.table('anime')} a ON w.anime_id = a.anime_id
            WHERE w.group_id = ?
            ORDER BY w.added_at DESC
            """,
            [group_id],
        )

    def ratings(self, group_id):
        return self.query(
            f"""
            SELECT r.anime_id, a.title, r.score, r.comment, r.rated_at
            FROM {self.table('ratings')} r
            LEFT JOIN {self.table('anime')} a ON r.anime_id = a.anime_id
            WHERE r.group_id = ?
            ORDER BY r.rated_at DESC
            """,
            [group_id],
        )
