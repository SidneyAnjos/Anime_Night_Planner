"""Semantic search over the anime library.

- `embed(text)` calls the same Mosaic AI embedding model used at index time
  (`databricks-bge-large-en`), so query vectors live in the same space.
- `search(text, ...)` queries the Delta-sync Vector Search index and returns parsed rows.
"""
import os

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient


def _env(name, default=None):
    return os.environ.get(name, default)


def _extract_rows(res):
    """VectorSearchClient.query_index returns either a dict or an object with .result.data_array."""
    if isinstance(res, dict):
        return res.get("result", {}).get("data_array", []) or []
    result = getattr(res, "result", None)
    return getattr(result, "data_array", None) or []


class VectorStore:
    def __init__(self, embed_endpoint=None, catalog=None, schema=None):
        self.embed_endpoint = embed_endpoint or _env("EMBED_ENDPOINT", "databricks-bge-large-en")
        catalog = catalog or _env("CATALOG", "anime_night_planner")
        schema = schema or _env("SCHEMA", "default")
        self.index_name = f"{catalog}.{schema}.anime_synopsis_index"
        self._ws = None
        self._client = None

    def _workspace(self):
        if self._ws is None:
            self._ws = WorkspaceClient()
        return self._ws

    def _vsc(self):
        if self._client is None:
            self._client = VectorSearchClient()
        return self._client

    def embed(self, text):
        resp = self._workspace().serving_endpoints.invoke(
            endpoint_name=self.embed_endpoint,
            inputs={"input": text[:5000]},
        )
        return resp.data[0]["embedding"]

    def search(self, query_text, columns=None, num_results=10):
        """Return a list of dicts: {'anime_id': ..., <columns...>, '_similarity': <distance>}.

        The Vector Search index returns the primary key first, the requested columns next, and the
        similarity distance last — mapped here into clean dicts.
        """
        columns = columns or ["title", "score", "episodes", "genres"]
        query_vector = self.embed(query_text)
        res = self._vsc().query_index(
            index_name=self.index_name,
            query_vector=query_vector,
            columns=columns,
            num_results=num_results,
        )
        rows = _extract_rows(res)
        parsed = []
        for row in rows:
            item = {"anime_id": row[0]}
            for name, val in zip(columns, row[1:-1]):
                item[name] = val
            item["_similarity"] = row[-1] if len(row) >= len(columns) + 2 else None
            parsed.append(item)
        return parsed
