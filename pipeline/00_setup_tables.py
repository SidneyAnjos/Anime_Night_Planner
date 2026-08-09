# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Catalog, Schema, Tables & Demo Seed
# MAGIC
# MAGIC Creates the Unity Catalog schema plus every bronze / silver / app-state table used by the
# MAGIC Anime Night Planner. Idempotent — safe to run repeatedly. Also seeds a demo group so the
# MAGIC app has data to render before the pipeline is ever run.
# MAGIC
# MAGIC **Note:** `groups` is a reserved word in Spark SQL, so that table is always backtick-quoted.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Catalog & schema
# COMMAND ----------
# MAGIC %python
# Attempt a dedicated catalog first; fall back to `main` if the account denies catalog creation.
CATALOG = "anime_night_planner"
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
    print(f"Using catalog='{catalog}', schema='{schema}'")
    return catalog, SCHEMA


catalog, schema = ensure_catalog_schema()
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. DDL — all tables
# COMMAND ----------
# MAGIC %python
# Bronze tables keep the raw Jikan JSON payloads verbatim for lineage.
DDL = {
    "raw_anime": """
        CREATE TABLE IF NOT EXISTS raw_anime (
            id          BIGINT,
            source_url  STRING,
            fetched_at  TIMESTAMP,
            payload     STRING
        ) USING DELTA
    """,
    "raw_characters": """
        CREATE TABLE IF NOT EXISTS raw_characters (
            id          BIGINT,
            anime_id    BIGINT,
            source_url  STRING,
            fetched_at  TIMESTAMP,
            payload     STRING
        ) USING DELTA
    """,
    "raw_reviews": """
        CREATE TABLE IF NOT EXISTS raw_reviews (
            id          BIGINT,
            anime_id    BIGINT,
            source_url  STRING,
            fetched_at  TIMESTAMP,
            payload     STRING
        ) USING DELTA
    """,
    # Silver — the RAG + dashboard corpus.
    "anime": """
        CREATE TABLE IF NOT EXISTS anime (
            anime_id        BIGINT,
            title           STRING,
            title_english   STRING,
            synopsis        STRING,
            type            STRING,
            episodes        INT,
            score           DOUBLE,
            scored_by       BIGINT,
            year            INT,
            season          STRING,
            source          STRING,
            rating          STRING,
            status          STRING,
            genres          ARRAY<STRING>,
            embedding_vector ARRAY<FLOAT>,
            updated_at      TIMESTAMP
        ) USING DELTA
    """,
    "genres": """
        CREATE TABLE IF NOT EXISTS genres (
            genre_id    INT,
            name        STRING
        ) USING DELTA
    """,
    "anime_genres": """
        CREATE TABLE IF NOT EXISTS anime_genres (
            anime_id    BIGINT,
            genre_id    INT
        ) USING DELTA
    """,
    "characters": """
        CREATE TABLE IF NOT EXISTS characters (
            character_id    BIGINT,
            anime_id        BIGINT,
            name            STRING,
            role            STRING,
            favorites       BIGINT,
            about           STRING,
            embedding_vector ARRAY<FLOAT>
        ) USING DELTA
    """,
    "reviews": """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id       BIGINT,
            anime_id        BIGINT,
            author          STRING,
            score           DOUBLE,
            review          STRING,
            embedding_vector ARRAY<FLOAT>
        ) USING DELTA
    """,
    # App-state — written by the agent via the SQL warehouse.
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            user_id     INT,
            name        STRING,
            preferences ARRAY<STRING>
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
            group_id    INT,
            anime_id    BIGINT,
            score       INT,
            comment     STRING,
            rated_at    TIMESTAMP
        ) USING DELTA
    """,
    "watchlist_items": """
        CREATE TABLE IF NOT EXISTS watchlist_items (
            group_id    INT,
            anime_id    BIGINT,
            status      STRING,
            added_by    INT,
            added_at    TIMESTAMP
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

# Enable Change Data Feed on the `anime` table (required by the Delta-sync Vector Search index).
# Idempotent: ALTER is safe even if the property is already set.
spark.sql(f"ALTER TABLE `{catalog}`.`{schema}`.anime SET TBLPROPERTIES ('delta.enableChangeDataFeed' = true)")
print("OK  anime (delta.enableChangeDataFeed = true)")

print("\nTables in current schema:")
for row in spark.sql("SHOW TABLES").collect():
    print(" -", row.tableName)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Demo seed (idempotent)
# MAGIC
# MAGIC Seeds one group with three members and a small watchlist referencing well-known MAL ids
# MAGIC (Cowboy Bebop=1, FMA:Brotherhood=5114, Attack on Titan=16498, One Piece=21). These titles
# MAGIC appear in the top-anime ingest, so joins resolve once the pipeline runs.
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
            (1, 'Alice', array('romance', 'sci-fi')),
            (2, 'Bob',   array('action', 'comedy')),
            (3, 'Carol', array('slice of life', 'drama'))
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
        INSERT INTO watchlist_items (group_id, anime_id, status, added_by, added_at) VALUES
            (1, 1,     'watched', 1, current_timestamp()),
            (1, 5114,  'watched', 1, current_timestamp()),
            (1, 16498, 'watched', 2, current_timestamp()),
            (1, 21,    'queued',  3, current_timestamp())
    """)
    print("Seeded watchlist_items.")

spark.sql("""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'setup', '00_setup_tables', 0, 'success', current_timestamp()
""")
print("pipeline_log entry written. 00 setup complete.")
