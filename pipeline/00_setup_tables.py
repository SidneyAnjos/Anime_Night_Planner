# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Catalog, Schema, Tables & Demo Seed (Movie Night Planner)
# MAGIC
# MAGIC Creates the Unity Catalog schema plus every bronze / silver / app-state table used by the
# MAGIC Movie Night Planner. Idempotent — safe to run repeatedly. Also seeds a demo group so the
# MAGIC app has data to render before the pipeline is ever run.
# MAGIC
# MAGIC Domain source: TMDB (The Movie Database) API. Data provided by TMDB — free for
# MAGIC non-commercial educational use with attribution.
# MAGIC
# MAGIC **Note:** `groups` is a reserved word in Spark SQL, so that table is always backtick-quoted.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Catalog & schema
# COMMAND ----------
# MAGIC %python
# Attempt a dedicated catalog first; fall back to `main` if the account denies catalog creation.
CATALOG = "movie_night_planner"
SCHEMA = "default"
FALLBACK_CATALOG = "main"


def ensure_catalog_schema():
    catalog = CATALOG
    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
        print(f"Catalog '{CATALOG}' ready (or already existed).")
    except Exception as e:  # noqa: BLE001 - fall back if CREATE CATALOG is not permitted
        print(f"Could not create catalog '{CATALOG}': {e!r}")
        print(f"Falling back to catalog '{FALLBACK_CATALOG}'.")
        catalog = FALLBACK_CATALOG
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{SCHEMA}")
    spark.sql(f"USE {catalog}.{SCHEMA}")
    print(f"Using catalog='{catalog}', schema='{SCHEMA}'")
    return catalog, SCHEMA


catalog, schema = ensure_catalog_schema()
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. DDL — all tables
# COMMAND ----------
# MAGIC %python
# Bronze tables keep the raw TMDB JSON payloads verbatim for lineage.
DDL = {
    "raw_movies": """
        CREATE TABLE IF NOT EXISTS raw_movies (
            tmdb_id      BIGINT,
            source_url   STRING,
            fetched_at   TIMESTAMP,
            payload      STRING
        ) USING DELTA
    """,
    "raw_credits": """
        CREATE TABLE IF NOT EXISTS raw_credits (
            tmdb_id      BIGINT,
            source_url   STRING,
            fetched_at   TIMESTAMP,
            payload      STRING
        ) USING DELTA
    """,
    "raw_keywords": """
        CREATE TABLE IF NOT EXISTS raw_keywords (
            tmdb_id      BIGINT,
            source_url   STRING,
            fetched_at   TIMESTAMP,
            payload      STRING
        ) USING DELTA
    """,
    "raw_reviews": """
        CREATE TABLE IF NOT EXISTS raw_reviews (
            tmdb_id      BIGINT,
            source_url   STRING,
            fetched_at   TIMESTAMP,
            payload      STRING
        ) USING DELTA
    """,
    "raw_providers": """
        CREATE TABLE IF NOT EXISTS raw_providers (
            tmdb_id      BIGINT,
            source_url   STRING,
            fetched_at   TIMESTAMP,
            payload      STRING
        ) USING DELTA
    """,
    "raw_genres": """
        CREATE TABLE IF NOT EXISTS raw_genres (
            genre_id     INT,
            source_url   STRING,
            fetched_at   TIMESTAMP,
            payload      STRING
        ) USING DELTA
    """,
    # Silver — the RAG + dashboard corpus.
    "movies": """
        CREATE TABLE IF NOT EXISTS movies (
            movie_id        BIGINT,
            title           STRING,
            original_title  STRING,
            overview        STRING,
            tagline         STRING,
            runtime         INT,
            vote_average    DOUBLE,
            vote_count      BIGINT,
            release_date    DATE,
            year            INT,
            poster_path     STRING,
            backdrop_path   STRING,
            genres          ARRAY<STRING>,
            language        STRING,
            status          STRING,
            updated_at      TIMESTAMP
        ) USING DELTA
    """,
    "cast": """
        CREATE TABLE IF NOT EXISTS cast (
            person_id    BIGINT,
            movie_id     BIGINT,
            name         STRING,
            character    STRING,
            credit_order INT
        ) USING DELTA
    """,
    "keywords": """
        CREATE TABLE IF NOT EXISTS keywords (
            movie_id     BIGINT,
            keyword_id   BIGINT,
            keyword      STRING
        ) USING DELTA
    """,
    "reviews": """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id    STRING,
            movie_id     BIGINT,
            author       STRING,
            rating       DOUBLE,
            content      STRING,
            created_at    STRING,
            url          STRING
        ) USING DELTA
    """,
    "providers": """
        CREATE TABLE IF NOT EXISTS providers (
            movie_id      BIGINT,
            country       STRING,
            provider_name STRING,
            provider_type STRING
        ) USING DELTA
    """,
    "genres": """
        CREATE TABLE IF NOT EXISTS genres (
            genre_id    INT,
            name        STRING
        ) USING DELTA
    """,
    # Embeddings — self-managed vectors, source table for the Vector Search index.
    "movie_embeddings": """
        CREATE TABLE IF NOT EXISTS movie_embeddings (
            movie_id          BIGINT,
            title             STRING,
            overview          STRING,
            embedding_vector  ARRAY<FLOAT>,
            embedding_model   STRING,
            embedded_at       TIMESTAMP
        ) USING DELTA
    """,
    # App-state — written by the agent via the SQL warehouse.
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            user_id      INT,
            name         STRING,
            preferences  ARRAY<STRING>
        ) USING DELTA
    """,
    "`groups`": """
        CREATE TABLE IF NOT EXISTS `groups` (
            group_id    INT,
            name        STRING,
            created_at  TIMESTAMP
        ) USING DELTA
    """,
    "group_members": """
        CREATE TABLE IF NOT EXISTS group_members (
            group_id    INT,
            user_id     INT,
            role        STRING
        ) USING DELTA
    """,
    "ratings": """
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id   BIGINT,
            group_id    INT,
            movie_id    BIGINT,
            user_id     INT,
            score       INT,
            comment     STRING,
            rated_at    TIMESTAMP
        ) USING DELTA
    """,
    "watchlist_items": """
        CREATE TABLE IF NOT EXISTS watchlist_items (
            item_id     BIGINT,
            group_id    INT,
            movie_id    BIGINT,
            status      STRING,
            added_by    INT,
            added_at    TIMESTAMP
        ) USING DELTA
    """,
    "recommendations": """
        CREATE TABLE IF NOT EXISTS recommendations (
            rec_id        BIGINT,
            group_id     INT,
            movie_id     BIGINT,
            reason       STRING,
            recommended_by STRING,
            recommended_at TIMESTAMP
        ) USING DELTA
    """,
    # Lineage / run metadata.
    "pipeline_log": """
        CREATE TABLE IF NOT EXISTS pipeline_log (
            run_id    STRING,
            step      STRING,
            rows      BIGINT,
            status    STRING,
            ts        TIMESTAMP
        ) USING DELTA
    """,
}

for name, stmt in DDL.items():
    spark.sql(stmt)
    print(f"OK  {name}")

# Enable Change Data Feed on movie_embeddings (the Vector Search Delta-sync source table).
# Idempotent: ALTER is safe even if the property is already set.
spark.sql(f"ALTER TABLE `{catalog}`.`{schema}`.movie_embeddings SET TBLPROPERTIES ('delta.enableChangeDataFeed' = true)")
print("OK  movie_embeddings (delta.enableChangeDataFeed = true)")

print("\nTables in current schema:")
for row in spark.sql("SHOW TABLES").collect():
    print(" -", row.tableName)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Demo seed (idempotent)
# MAGIC
# MAGIC Seeds one group with three members and a small watchlist referencing well-known TMDB
# MAGIC movie ids ( Shawshank=278, Godfather=238, Dark Knight=155, Pulp Fiction=680),
# MAGIC which appear in the popular-movie ingest, so joins resolve once the pipeline runs.
# COMMAND ----------
# MAGIC %python
def table_nonempty(name):
    """Return True if the table exists and has at least one row."""
    try:
        return spark.sql(f"SELECT count(*) AS n FROM {name}").first()["n"] > 0
    except Exception:  # noqa: BLE001 - table may not exist yet
        return False


if not table_nonempty("users"):
    spark.sql("""
        INSERT INTO users (user_id, name, preferences) VALUES
            (1, 'Alice', array('sci-fi', 'comedy')),
            (2, 'Bob',   array('action', 'thriller')),
            (3, 'Carol', array('drama', 'romance'))
    """)
    print("Seeded users.")

if not table_nonempty("`groups`"):
    spark.sql("""
        INSERT INTO `groups` (group_id, name, created_at)
        VALUES (1, 'Friday Night Crew', current_timestamp())
    """)
    print("Seeded groups.")

if not table_nonempty("group_members"):
    spark.sql("""
        INSERT INTO group_members (group_id, user_id, role) VALUES
            (1, 1, 'owner'),
            (1, 2, 'member'),
            (1, 3, 'member')
    """)
    print("Seeded group_members.")

if not table_nonempty("watchlist_items"):
    spark.sql("""
        INSERT INTO watchlist_items (item_id, group_id, movie_id, status, added_by, added_at) VALUES
            (1, 1, 278,  'watched', 1, current_timestamp()),
            (2, 1, 238,  'watched', 1, current_timestamp()),
            (3, 1, 155,  'watched', 2, current_timestamp()),
            (4, 1, 680,  'queued',  3, current_timestamp())
    """)
    print("Seeded watchlist_items.")

spark.sql("""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'setup', '00_setup_tables', 0, 'success', current_timestamp()
""")
print("pipeline_log entry written. 00 setup complete.")
