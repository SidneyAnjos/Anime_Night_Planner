# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Vector Search Endpoint + Index
# MAGIC
# MAGIC Creates a **serverless Vector Search endpoint** and a **Delta-sync vector index**
# MAGIC (`anime_synopsis_index`) over the precomputed `anime.embedding_vector` column, then runs a
# MAGIC live sanity query to prove end-to-end retrieval works.
# MAGIC
# MAGIC **Prerequisites:** Vector Search enabled in the workspace region + permission to create
# MAGIC serverless endpoints/indexes.
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

catalog = spark.catalog.currentCatalog()
schema = spark.catalog.currentDatabase()
ENDPOINT_NAME = "anime_vector_search_endpoint"
INDEX_NAME = f"{catalog}.{schema}.anime_synopsis_index"
SOURCE_TABLE = f"{catalog}.{schema}.anime"
print(f"Using catalog='{catalog}', schema='{schema}'")
print(f"Index: {INDEX_NAME}  (source: {SOURCE_TABLE})")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Ensure a serverless Vector Search endpoint exists
# COMMAND ----------
# MAGIC %python
existing_endpoints = {e.name for e in w.vector_search_endpoints.list()}
if ENDPOINT_NAME in existing_endpoints:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists.")
else:
    try:
        w.vector_search_endpoints.create_endpoint_and_wait(
            name=ENDPOINT_NAME,
            endpoint_type=EndpointType.STANDARD,  # "STANDARD" is the serverless-backed endpoint type
        )
        print(f"Created Vector Search endpoint '{ENDPOINT_NAME}'.")
    except TypeError:
        # Older SDK: create then poll readiness separately.
        w.vector_search_endpoints.create_endpoint(
            name=ENDPOINT_NAME,
            endpoint_type=EndpointType.STANDARD,
        )
        w.vector_search_endpoints.wait_get_endpoint_ready(ENDPOINT_NAME, timeout=900)
        print(f"Created Vector Search endpoint '{ENDPOINT_NAME}' (fallback path).")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Create the Delta-sync vector index
# MAGIC
# MAGIC We use **self-managed embeddings**: the `anime.embedding_vector` column already holds the
# MAGIC 1024-dim vector (computed in step 03). So the index syncs the precomputed vector column
# MAGIC (`embedding_vector_columns`), not a text source column.
# COMMAND ----------
# MAGIC %python
try:
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key="anime_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=SOURCE_TABLE,
            pipeline_type=PipelineType.CONTINUOUS,
            embedding_vector_columns=[EmbeddingVectorColumn(name="embedding_vector")],
        ),
    )
    print("Index create request sent.")
except Exception as exc:  # noqa: BLE001 - likely already exists
    print(f"create_index returned: {exc!r}")

w.vector_search_indexes.wait_get_index_ready(INDEX_NAME, timeout=900)
print("Index is READY.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Sanity query (proves end-to-end retrieval)
# COMMAND ----------
# MAGIC %python
# Embed a real query with the same model used at index time.
resp = w.serving_endpoints.invoke(
    endpoint_name="databricks-bge-large-en",
    inputs={"input": "an emotional romance with a sci-fi twist"},
)
query_vector = resp.data[0]["embedding"]

result = w.vector_search_indexes.query_index(
    index_name=INDEX_NAME,
    columns=["anime_id", "title", "score"],
    num_results=5,
    query_type="ANN",
    query_vector=query_vector,
)
print("Top semantic matches for 'an emotional romance with a sci-fi twist':")
for row in (result.result.data_array or []):
    print("  ", row)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Log the run
# COMMAND ----------
# MAGIC %python
spark.sql("""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'index', '04_vector_index', 0, 'success', current_timestamp()
""")
print("04 vector index complete.")
