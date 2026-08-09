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
        """Return a list of dicts: {'movie_id': ..., 'title', 'overview', 'vote_average',
        'runtime', 'genres', '_similarity': <distance>}.

        The Vector Search index only syncs the columns present on the source `movie_embeddings`
        table — `movie_id` (pk), `title`, `overview` — so we query those from the index and then
        enrich each match with the richer catalog attributes (`vote_average`, `runtime`, `genres`)
        from the `movies` table by movie_id. Asking the index for columns it doesn't sync raises
        "Requested columns to fetch are not present in index".
        """
        indexed_cols = ["movie_id", "title", "overview"]
        query_vector = self.embed(query_text)
        res = self._workspace().vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=indexed_cols,
            num_results=num_results,
            query_type="ANN",
            query_vector=query_vector,
        )
        rows = _extract_rows(res)
        parsed = []
        for row in rows:
            # Row layout: [movie_id, title, overview, <similarity>]. The requested columns
            # are returned in order with the similarity distance appended last.
            item = {}
            for name, val in zip(indexed_cols, row[:-1]):
                item[name] = val
            item["_similarity"] = row[-1] if len(row) == len(indexed_cols) + 1 else None
            parsed.append(item)

        # Enrich: join genres / runtime / vote_average from `movies` (not in the index).
        ids = [it["movie_id"] for it in parsed if it.get("movie_id") is not None]
        if ids:
            db = self._db_for_enrich()
            extra = db.query(
                f"SELECT movie_id, vote_average, runtime, genres FROM {db.table('movies')} "
                "WHERE movie_id IN (%s)" % ", ".join(["?"] * len(ids)),
                ids,
            )
            by_id = {e["movie_id"]: e for e in extra}
            for it in parsed:
                e = by_id.get(it["movie_id"])
                if e:
                    it["vote_average"] = e.get("vote_average")
                    it["runtime"] = e.get("runtime")
                    it["genres"] = e.get("genres")
        return parsed

    def _db_for_enrich(self):
        """Lazy Database instance for the `movies` enrichment join (avoids import cycle)."""
        import db as db_mod  # noqa: WPS433 (local import to avoid cycle at module load)
        if getattr(self, "_db", None) is None:
            self._db = db_mod.Database()
        return self._db
