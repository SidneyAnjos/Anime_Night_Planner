"""SQL access layer for the app + agent tools.

Connects to a Databricks SQL warehouse via `databricks-sql-connector`. Two auth modes,
tried in order so the same code works both locally under the VS Code extension and as a
deployed Databricks App:

1. **App service principal (M2M)** — when `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET`
   are present (Databricks Apps injects these automatically for the app's SP), use
   `oauth_service_principal` via `databricks.sdk.core.Config`.
2. **Personal / token** — fall back to `DATABRICKS_TOKEN` (local dev with the VS Code
   extension, or a PAT). Use `access_token=` directly.

All statements are parameterized with `?` placeholders to keep LLM-sourced writes safe.
"""
import os

from databricks import sql as dsql


def _env(name, default=None):
    return os.environ.get(name, default)


def _quote(name):
    return f"`{name}`"


def _credentials_provider():
    """Return (kwargs, provider) for dsql.connect, preferring the app SP (M2M) when available."""
    client_id = _env("DATABRICKS_CLIENT_ID")
    client_secret = _env("DATABRICKS_CLIENT_SECRET")
    if client_id and client_secret:
        # M2M OAuth: Databricks App runtime injected SP credentials.
        from databricks.sdk.core import Config, oauth_service_principal

        host = _env("DATABRICKS_HOST", "").rstrip("/")
        if host and not host.startswith("http"):
            host = f"https://{host}"
        cfg = Config(host=host, client_id=client_id, client_secret=client_secret)
        return {"credentials_provider": lambda: oauth_service_principal(cfg)}
    if _env("DATABRICKS_TOKEN"):
        return {"access_token": _env("DATABRICKS_TOKEN")}
    return {}  # let the connector fall back to its default auth chain


class Database:
    def __init__(self, host=None, http_path=None, token=None, catalog=None, schema=None):
        self.host = (host or _env("DATABRICKS_HOST", "")).replace("https://", "").rstrip("/")
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
                "SQL_WAREHOUSE_PATH is not set — add it to databricks.yml env (the warehouse's http path)."
            )
        connect_kwargs = {
            "server_hostname": self.host,
            "http_path": self.http_path,
        }
        connect_kwargs.update(_credentials_provider())
        return dsql.connect(**connect_kwargs)

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
