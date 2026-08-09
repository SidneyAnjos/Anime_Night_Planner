# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Embeddings
# MAGIC
# MAGIC Embeds the unstructured text corpus and stores vectors in `embedding_vector` columns:
# MAGIC
# MAGIC - **`anime.synopsis`** (falling back to `title` when a synopsis is missing) — primary RAG corpus
# MAGIC - **`reviews.review`** — user-submitted reviews (secondary corpus)
# MAGIC
# MAGIC Uses the Mosaic AI foundation model `databricks-bge-large-en` (1024-dim, pay-as-you-go) via the
# MAGIC Databricks SDK. Only rows with a `NULL` embedding are processed, so re-runs are incremental.
# MAGIC
# MAGIC **Prerequisite:** Foundation Model API (pay-as-you-go) must be enabled for the workspace.
# MAGIC If the endpoint is unavailable this notebook fails loudly — the fallback is to self-host a
# MAGIC small sentence-transformers model, but then the query side in the app must use the same model.
# COMMAND ----------
# MAGIC %python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
EMBED_ENDPOINT = "databricks-bge-large-en"
BATCH = 50
MAX_CHARS = 5000   # keep well under the model's token limit


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
# MAGIC ## 1. Embed `anime.synopsis`
# COMMAND ----------
# MAGIC %python
anime_rows = spark.sql("""
    SELECT anime_id, COALESCE(NULLIF(synopsis, ''), title) AS text
    FROM anime
    WHERE embedding_vector IS NULL
      AND COALESCE(synopsis, title) IS NOT NULL
""").collect()
print(f"Anime rows to embed: {len(anime_rows)}")

if anime_rows:
    texts = [r.text for r in anime_rows]
    vecs = embed_texts(texts)
    updates = [(r.anime_id, v) for r, v in zip(anime_rows, vecs)]
    spark.createDataFrame(updates, schema=["anime_id", "embedding_vector"]).createOrReplaceTempView("embed_updates")
    spark.sql("""
        MERGE INTO anime AS t
        USING embed_updates AS s
        ON t.anime_id = s.anime_id
        WHEN MATCHED THEN UPDATE SET t.embedding_vector = s.embedding_vector
    """)
    print(f"Embedded anime synopses: {len(updates)}")
else:
    print("No anime rows need embedding.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Embed `reviews.review`
# COMMAND ----------
# MAGIC %python
review_rows = spark.sql("""
    SELECT review_id, review AS text
    FROM reviews
    WHERE embedding_vector IS NULL
      AND review IS NOT NULL
      AND LENGTH(review) > 0
""").collect()
print(f"Review rows to embed: {len(review_rows)}")

if review_rows:
    vecs = embed_texts([r.text for r in review_rows])
    updates = [(r.review_id, v) for r, v in zip(review_rows, vecs)]
    spark.createDataFrame(updates, schema=["review_id", "embedding_vector"]).createOrReplaceTempView("review_embed_updates")
    spark.sql("""
        MERGE INTO reviews AS t
        USING review_embed_updates AS s
        ON t.review_id = s.review_id
        WHEN MATCHED THEN UPDATE SET t.embedding_vector = s.embedding_vector
    """)
    print(f"Embedded reviews: {len(updates)}")
else:
    print("No review rows need embedding.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Log the run
# COMMAND ----------
# MAGIC %python
n_embedded = spark.sql("SELECT count(*) FROM anime WHERE embedding_vector IS NOT NULL").first()[0]
n_embed_total = n_embedded + spark.sql("SELECT count(*) FROM reviews WHERE embedding_vector IS NOT NULL").first()[0]
spark.sql("""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'embed', '03_embed', ?, 'success', current_timestamp()
""", args=[n_embed_total])

print(f"anime with embeddings: {n_embedded} | reviews with embeddings: {spark.sql('SELECT count(*) FROM reviews WHERE embedding_vector IS NOT NULL').first()[0]}")
print("03 embed complete.")
