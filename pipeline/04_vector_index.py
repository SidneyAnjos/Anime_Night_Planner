# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Vector Search Endpoint + Index (Movie Night Planner)
# MAGIC
# MAGIC Creates a **serverless Vector Search endpoint** and a **Delta-sync vector index**
# MAGIC (`movie_embeddings_index`) over the precomputed `movie_embeddings.embedding_vector` column
# MAGIC (self-managed embeddings from step 03), then runs a live sanity query to prove end-to-end
# MAGIC retrieval works.
# MAGIC
# MAGIC The app's `VectorStore` queries this same index and uses the same embedding endpoint, so
# MAGIC index-time and query-time vectors live in the same vector space.
# MAGIC
# MAGIC **Prerequisites:** Vector Search enabled in the workspace region + permission to create
# MAGIC serverless endpoints / indexes.
# COMMAND ----------
# MAGIC %python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    VectorIndexType,
    PipelineType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
)

w = WorkspaceClient()

# Force the Movie Night Planner catalog/schema explicitly. (In a multi-task job each task
# runs in its own Spark session, so 00_setup's USE CATALOG does not carry over. Relying on
# spark.catalog.currentCatalog()/currentDatabase() would resolve to the workspace default
# and point the index at the wrong catalog.)
catalog = "movie_night_planner"
schema = "default"
ENDPOINT_NAME = "movie_vector_search_endpoint"
INDEX_NAME = f"{catalog}.{schema}.movie_embeddings_index"
SOURCE_TABLE = f"{catalog}.{schema}.movie_embeddings"
print(f"Using catalog='{catalog}', schema='{schema}'")
print(f"Index: {INDEX_NAME}  (source: {SOURCE_TABLE})")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Ensure a serverless Vector Search endpoint exists
# MAGIC
# MAGIC databricks-sdk 0.125 exposes `vector_search_endpoints.list_endpoints()` (not `.list()`)
# MAGIC and `create_endpoint_and_wait()` for blocking until the endpoint is ready.
# COMMAND ----------
# MAGIC %python
import time

existing_endpoints = {e.name for e in w.vector_search_endpoints.list_endpoints()}
if ENDPOINT_NAME in existing_endpoints:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists.")
else:
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD,  # "STANDARD" is the serverless-backed endpoint type
    )
    print(f"Created Vector Search endpoint '{ENDPOINT_NAME}'.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Create the Delta-sync vector index
# MAGIC
# MAGIC Self-managed embeddings: `movie_embeddings.embedding_vector` already holds the 1024-dim
# MAGIC vector (computed in step 03). The index syncs the precomputed vector column
# MAGIC (`embedding_vector_columns`), not a text source column. A CONTINUOUS pipeline keeps the
# MAGIC index up to date as movie_embeddings changes.
# COMMAND ----------
# MAGIC %python
# Idempotency: trying create_index on an existing index throws "already exists", which
# we treat as success. (We don't gate on list_indexes — the returned `MiniVectorIndex.name`
# shape is unreliable across SDK versions, and a false "exists" would make us skip create
# then block 15 min on get_index for an index that isn't there.)
try:
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key="movie_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=SOURCE_TABLE,
            pipeline_type=PipelineType.CONTINUOUS,
            embedding_vector_columns=[EmbeddingVectorColumn(name="embedding_vector")],
        ),
    )
    print("Index create request sent.")
except Exception as exc:  # noqa: BLE001 - already exists / race -> fine
    msg = str(exc).lower()
    if "already" in msg or "exists" in msg:
        print(f"Index '{INDEX_NAME}' already exists.")
    else:
        print(f"create_index returned: {exc!r}")

# Poll readiness by reading get_index().status.ready (databricks-sdk 0.125 has no
# wait_get_index_ready helper). Vector Search provisioning can take several minutes.
print("Waiting for index to become ready (up to ~15 min)…")
ready = False
for attempt in range(90):  # 90 * 10s = 15 min
    try:
        idx = w.vector_search_indexes.get_index(index_name=INDEX_NAME)
        st = idx.status
        ready = bool(getattr(st, "ready", False))
        indexed = getattr(st, "indexed_row_count", None)
        msg = getattr(st, "message", "")
        print(f"  [{attempt * 10:>4}s] ready={ready} indexed_rows={indexed} msg={msg!r}")
        if ready:
            break
    except Exception as exc:  # noqa: BLE001 - index object may not be fetchable immediately after create
        print(f"  [{attempt * 10:>4}s] get_index not available yet: {exc!r}")
    time.sleep(10)

if not ready:
    raise RuntimeError(f"Vector index '{INDEX_NAME}' did not become ready within 15 min.")
print(f"Index '{INDEX_NAME}' is READY.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Sanity query (proves end-to-end retrieval)
# COMMAND ----------
# MAGIC %python
# Embed a real query with the same model used at index time.
# (serving_endpoints.query returns a QueryEndpointResponse whose .data is a list of
# EmbeddingsV1ResponseEmbeddingElement in input order, each with an .embedding list.)
resp = w.serving_endpoints.query(
    name="databricks-bge-large-en",
    input=["a heist thriller with a twist ending"],
)
query_vector = list(resp.data[0].embedding)

result = w.vector_search_indexes.query_index(
    index_name=INDEX_NAME,
    columns=["movie_id", "title", "overview"],
    num_results=5,
    query_type="ANN",
    query_vector=query_vector,
)
print("Top semantic matches for 'a heist thriller with a twist ending':")
for row in (result.result.data_array or []):
    print("  ", row)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Log the run
# COMMAND ----------
# MAGIC %python
spark.sql(f"""
    INSERT INTO {catalog}.{schema}.pipeline_log (run_id, step, rows, status, ts)
    SELECT 'index', '04_vector_index', 0, 'success', current_timestamp()
""")
print("04 vector index complete.")
