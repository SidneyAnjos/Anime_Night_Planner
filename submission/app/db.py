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
Domain: Movie Night Planner (TMDB-sourced silver tables).
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


def _native(v):
    """Coerce a databricks-sql-connector value to a JSON/Python-native form.

    The connector returns ARRAY<...> columns as `numpy.ndarray` and typed numerics as numpy
    scalars. ndarrays break `json.dumps` without a default= AND raise "The truth value of an
    array with more than one element is ambiguous" on truthiness checks (which crashed the
    genre filter in tools.search_movies_semantic). Normalizing here means every tool and the
    app's Browse page receive plain lists / ints / floats — the same shape as any other DB.
    """
    if isinstance(v, (list, tuple)):
        return [_native(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return [_native(x) for x in v.tolist()]
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:  # numpy is a transitive dep of databricks-sql-connector; stay safe
        pass
    return v


class Database:
    def __init__(self, host=None, http_path=None, token=None, catalog=None, schema=None):
        self.host = (host or _env("DATABRICKS_HOST", "")).replace("https://", "").rstrip("/")
        self.http_path = http_path or _env("SQL_WAREHOUSE_PATH")
        self.token = token or _env("DATABRICKS_TOKEN")
        self.catalog = catalog or _env("CATALOG", "movie_night_planner")
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
        """Run a SELECT and return a list of dicts (lowercased column names).

        Every value is passed through `_native` so numpy arrays (ARRAY columns) and numpy
        scalars become plain lists / ints / floats — see `_native` for why that matters.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                cols = [d[0].lower() for d in cur.description] if cur.description else []
                return [dict(zip(cols, [_native(v) for v in row])) for row in cur.fetchall()]

    def execute(self, sql, params=None):
        """Run a DML statement (INSERT / UPDATE / MERGE)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
        return True

    def bulk_insert(self, table, columns, rows):
        """Insert many rows (list of tuples in `columns` order) in batches.

        Uses a parameterized multi-row VALUES list so fetch-sourced payloads are bound,
        not interpolated. Values come in as Python objects; the connector handles type
        conversion; `payload` columns are JSON strings.
        """
        if not rows:
            return 0
        col_sql = ", ".join(_quote(c) for c in columns)
        one_row = "(" + ", ".join(["?"] * len(columns)) + ")"
        total = 0
        batch = 500
        with self._connect() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(rows), batch):
                    chunk = rows[i:i + batch]
                    sql = (f"INSERT INTO {self.table(table)} ({col_sql}) VALUES "
                           + ", ".join([one_row] * len(chunk)))
                    params = [v for row in chunk for v in row]
                    cur.execute(sql, params)
                    total += len(chunk)
        return total

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
            SELECT w.movie_id, m.title, w.status, w.added_at,
                   m.vote_average AS tmdb_score, m.runtime, m.genres
            FROM {self.table('watchlist_items')} w
            LEFT JOIN {self.table('movies')} m ON w.movie_id = m.movie_id
            WHERE w.group_id = ?
            ORDER BY w.added_at DESC
            """,
            [group_id],
        )

    def ratings(self, group_id):
        return self.query(
            f"""
            SELECT r.movie_id, m.title, r.score, r.comment, r.rated_at
            FROM {self.table('ratings')} r
            LEFT JOIN {self.table('movies')} m ON r.movie_id = m.movie_id
            WHERE r.group_id = ?
            ORDER BY r.rated_at DESC
            """,
            [group_id],
        )
