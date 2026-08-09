"""Semantic search over the movie library.

Queries the AI Search / Vector Search Delta-sync index via the Databricks SDK
(`WorkspaceClient.vector_search_indexes`), so the app has no dependency on the
separate `databricks-vectorsearch` package. Query vectors are embedded with the
same Mosaic AI model used at index time (step 03 uses the same endpoint), so they
live in the same vector space.
"""
import os

from databricks.sdk import WorkspaceClient


def _env(name, default=None):
    return os.environ.get(name, default)


def _extract_rows(res):
    """query_index returns a QueryVectorIndexResponse whose .result.data_array holds the rows."""
    result = getattr(res, "result", None)
    return getattr(result, "data_array", None) or []


class VectorStore:
    def __init__(self, embed_endpoint=None, catalog=None, schema=None):
        self.embed_endpoint = embed_endpoint or _env("EMBED_ENDPOINT", "databricks-bge-large-en")
        catalog = catalog or _env("CATALOG", "movie_night_planner")
        schema = schema or _env("SCHEMA", "default")
        self.index_name = f"{catalog}.{schema}.movie_embeddings_index"
        self._ws = None

    def _workspace(self):
        if self._ws is None:
            self._ws = WorkspaceClient()
        return self._ws

    def embed(self, text):
        # serving_endpoints.query returns a QueryEndpointResponse whose .data is a list of
        # EmbeddingsV1ResponseEmbeddingElement (input order), each with an .embedding list.
        resp = self._workspace().serving_endpoints.query(
            name=self.embed_endpoint,
            input=[text[:5000]],
        )
        return list(resp.data[0].embedding)

    def search(self, query_text, columns=None, num_results=10):
        """Return a list of dicts: {'movie_id': ..., <columns...>, '_similarity': <distance>}.

        The index returns the primary key first, the requested columns next, and the similarity
        distance last — mapped here into clean dicts.
        """
        columns = columns or ["title", "vote_average", "runtime", "genres"]
        query_vector = self.embed(query_text)
        res = self._workspace().vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=columns,
            num_results=num_results,
            query_type="ANN",
            query_vector=query_vector,
        )
        rows = _extract_rows(res)
        parsed = []
        for row in rows:
            item = {"movie_id": row[0]}
            for name, val in zip(columns, row[1:-1]):
                item[name] = val
            item["_similarity"] = row[-1] if len(row) >= len(columns) + 2 else None
            parsed.append(item)
        return parsed
