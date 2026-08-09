# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Embeddings (Movie Night Planner)
# MAGIC
# MAGIC Builds a composite text corpus per movie — **overview + tagline + top keywords + top billed
# MAGIC cast + review snippets** — and embeds it with the Mosaic AI foundation model
# MAGIC `databricks-bge-large-en` (1024-dim, pay-as-you-go) into the `movie_embeddings` table.
# MAGIC
# MAGIC `movie_embeddings` is the **source table for the Vector Search index** (step 04), so the
# MAGIC query-side embedder in the app MUST use the same model — and it does: both call
# MAGIC `serving_endpoints.invoke("databricks-bge-large-en", …)`, keeping index + query vectors in the
# MAGIC same space.
# MAGIC
# MAGIC Only movies not already present in `movie_embeddings` are embedded, so re-runs are
# MAGIC incremental.
# MAGIC
# MAGIC **Prerequisite:** Foundation Model API (pay-as-you-go) enabled for the workspace.
# COMMAND ----------
# MAGIC %python
# In a multi-task job each task runs in its own Spark session, so the USE CATALOG done
# in 00_setup does NOT carry over. Force catalog/schema so the unqualified table
# references (movies, keywords, cast, reviews, movie_embeddings, pipeline_log) resolve.
spark.sql("USE CATALOG movie_night_planner")
spark.sql("USE movie_night_planner.default")
print("Using catalog='movie_night_planner', schema='default'")
# COMMAND ----------
# MAGIC %python
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

w = WorkspaceClient()
EMBED_ENDPOINT = "databricks-bge-large-en"
BATCH = 50
MAX_CHARS = 5000   # keep well under the model's token limit
TOP_KEYWORDS = 15
TOP_CAST = 15
TOP_REVIEWS = 5
REVIEW_CHARS = 600


def embed_texts(texts):
    """Embed a list of strings, returning one 1024-dim vector list per input (batched)."""
    if not texts:
        return []
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = [t[:MAX_CHARS] for t in texts[i:i + BATCH]]
        resp = w.serving_endpoints.invoke(
            endpoint_name=EMBED_ENDPOINT,
            inputs={"input": batch},
        )
        data = getattr(resp, "data", None)
        if not data:
            raise RuntimeError(f"No embedding data returned for batch {i // BATCH}")
        ordered = sorted(data, key=lambda d: d.get("embedding_index", 0))
        vectors.extend(d["embedding"] for d in ordered)
    return vectors


print(f"Embedding endpoint: {EMBED_ENDPOINT}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Build the composite text per movie
# MAGIC
# MAGIC Join `movies` with aggregated keywords, cast names and review snippets into one `text` column.
# COMMAND ----------
# MAGIC %python
kw_agg = (
    spark.table("keywords")
    .groupBy("movie_id")
    .agg(F.collect_list("keyword").alias("kw_list"))
)

cast_agg = (
    spark.table("cast")
    .filter(F.col("credit_order").isNotNull())
    .orderBy("movie_id", "credit_order")
    .groupBy("movie_id")
    .agg(F.collect_list("name").alias("cast_list"))
)

review_agg = (
    spark.table("reviews")
    .filter(F.col("content").isNotNull() & (F.length("content") > 0))
    .withColumn("snippet", F.substring("content", 1, REVIEW_CHARS))
    .groupBy("movie_id")
    .agg(F.collect_list("snippet").alias("review_list))
)

base = (
    spark.table("movies")
    .select(
        "movie_id", "title", "overview",
        F.coalesce("tagline", F.lit("")).alias("tagline"),
    )
)

text_df = (
    base
    .join(kw_agg, "movie_id", "left")
    .join(cast_agg, "movie_id", "left")
    .join(review_agg, "movie_id", "left")
    .select(
        "movie_id", "title", "overview",
        F.concat_ws(
            " | ",
            F.col("overview"),
            F.when(F.length("tagline") > 0, F.col("tagline")).otherwise(F.lit("")),
            F.when(F.size(F.coalesce("kw_list", F.array())).cast("int") > 0,
                   F.concat_ws(", ", F.slice(F.coalesce("kw_list", F.array()), 1, TOP_KEYWORDS))
                  ).otherwise(F.lit("")),
            F.when(F.size(F.coalesce("cast_list", F.array())).cast("int") > 0,
                   F.concat_ws(", ", F.slice(F.coalesce("cast_list", F.array()), 1, TOP_CAST))
                  ).otherwise(F.lit("")),
            F.when(F.size(F.coalesce("review_list", F.array())).cast("int") > 0,
                   F.concat_ws(" ", F.slice(F.coalesce("review_list", F.array()), 1, TOP_REVIEWS))
                  ).otherwise(F.lit("")),
        ).alias("text"),
    )
    .filter(F.col("overview").isNotNull() | (F.length("title") > 0))
)

# Only movies not already embedded.
need = (
    text_df.join(spark.table("movie_embeddings"), "movie_id", "left_anti")
)
rows = need.select("movie_id", "title", "overview", "text").collect()
print(f"Movies to embed: {len(rows)}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Embed + MERGE into `movie_embeddings`
# COMMAND ----------
# MAGIC %python
from pyspark.sql.types import StructType, StructField, LongType, StringType, ArrayType, FloatType, TimestampType

now_expr = F.current_timestamp()

if rows:
    texts = [r.text for r in rows]
    vecs = embed_texts(texts)
    if len(vecs) != len(rows):
        raise RuntimeError(f"Embedding count mismatch: {len(vecs)} vectors for {len(rows)} rows")

    updates = [
        (r.movie_id, r.title, r.overview, [float(x) for x in v], EMBED_ENDPOINT)
        for r, v in zip(rows, vecs)
    ]
    schema = StructType([
        StructField("movie_id", LongType(), True),
        StructField("title", StringType(), True),
        StructField("overview", StringType(), True),
        StructField("embedding_vector", ArrayType(FloatType()), True),
        StructField("embedding_model", StringType(), True),
    ])
    spark.createDataFrame(updates, schema=schema).createOrReplaceTempView("embed_updates")

    spark.sql("""
        MERGE INTO movie_embeddings AS t
        USING embed_updates AS s
        ON t.movie_id = s.movie_id
        WHEN MATCHED THEN UPDATE SET
            t.title = s.title,
            t.overview = s.overview,
            t.embedding_vector = s.embedding_vector,
            t.embedding_model = s.embedding_model,
            t.embedded_at = current_timestamp()
        WHEN NOT MATCHED THEN INSERT (
            movie_id, title, overview, embedding_vector, embedding_model, embedded_at
        ) VALUES (
            s.movie_id, s.title, s.overview, s.embedding_vector, s.embedding_model, current_timestamp()
        )
    """)
    print(f"Embedded movies: {len(updates)}")
else:
    print("No movies need embedding.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Log the run
# COMMAND ----------
# MAGIC %python
n_embedded = spark.sql("SELECT count(*) FROM movie_embeddings WHERE embedding_vector IS NOT NULL").first()[0]
spark.sql(f"""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'embed', '03_embed', {int(n_embedded)}, 'success', current_timestamp()
""")

print(f"movie_embeddings with vectors: {n_embedded}")
print("03 embed complete.")
