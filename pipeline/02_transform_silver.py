# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Transform
# MAGIC
# MAGIC Parses the bronze JSON payloads and normalizes them into the silver tables
# MAGIC (`anime`, `genres`, `anime_genres`, `characters`, `reviews`).
# MAGIC
# MAGIC - `anime` is updated with a **MERGE** so an existing `embedding_vector` (written by step 03)
# MAGIC   is preserved across re-runs instead of being wiped by an overwrite.
# MAGIC - `characters`/`reviews` are fully derived from bronze → overwrite is safe; the embed step
# MAGIC   refills any null embeddings afterwards.
# MAGIC
# MAGIC Note: the Jikan `/anime/{id}/characters` payload has no character "about" text, so that
# MAGIC column stays null (the semantic corpus is synopses + user reviews).
# COMMAND ----------
# MAGIC %python
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType, DoubleType, ArrayType,
)
from pyspark.sql import functions as F

# --- Jikan anime payload schema (subset we care about) ---
anime_schema = StructType([
    StructField("mal_id", LongType(), True),
    StructField("title", StringType(), True),
    StructField("title_english", StringType(), True),
    StructField("synopsis", StringType(), True),
    StructField("type", StringType(), True),
    StructField("episodes", IntegerType(), True),
    StructField("score", DoubleType(), True),
    StructField("scored_by", LongType(), True),
    StructField("year", IntegerType(), True),
    StructField("season", StringType(), True),
    StructField("source", StringType(), True),
    StructField("rating", StringType(), True),
    StructField("status", StringType(), True),
    StructField("genres", ArrayType(StructType([
        StructField("mal_id", IntegerType(), True),
        StructField("name", StringType(), True),
    ])), True),
])

anime_src = (
    spark.table("raw_anime")
    .select(F.from_json("payload", anime_schema).alias("a"))
    .select("a.*")
    .filter(F.col("mal_id").isNotNull())
    .dropDuplicates(["mal_id"])
)
print(f"Parsed anime records: {anime_src.count()}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. `anime` — merge (preserves embedding_vector)
# COMMAND ----------
# MAGIC %python
anime_updates = anime_src.select(
    F.col("mal_id").alias("anime_id"),
    F.col("title"),
    F.col("title_english"),
    F.col("synopsis"),
    F.col("type"),
    F.col("episodes"),
    F.col("score"),
    F.col("scored_by"),
    F.col("year"),
    F.col("season"),
    F.col("source"),
    F.col("rating"),
    F.col("status"),
    F.coalesce(F.transform(F.col("genres"), lambda g: g["name"]), F.array().cast("array<string>")).alias("genres"),
)
anime_updates.createOrReplaceTempView("anime_updates")

spark.sql("""
    MERGE INTO anime AS t
    USING anime_updates AS s
    ON t.anime_id = s.anime_id
    WHEN MATCHED THEN UPDATE SET
        t.title = s.title,
        t.title_english = s.title_english,
        t.synopsis = s.synopsis,
        t.type = s.type,
        t.episodes = s.episodes,
        t.score = s.score,
        t.scored_by = s.scored_by,
        t.year = s.year,
        t.season = s.season,
        t.source = s.source,
        t.rating = s.rating,
        t.status = s.status,
        t.genres = s.genres,
        t.updated_at = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (
        anime_id, title, title_english, synopsis, type, episodes, score, scored_by,
        year, season, source, rating, status, genres, updated_at
    ) VALUES (
        s.anime_id, s.title, s.title_english, s.synopsis, s.type, s.episodes, s.score, s.scored_by,
        s.year, s.season, s.source, s.rating, s.status, s.genres, current_timestamp()
    )
""")
print("anime table merged.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. `genres` + `anime_genres` (relational dimensions)
# COMMAND ----------
# MAGIC %python
genres_df = (
    anime_src.select(F.explode_outer(F.col("genres")).alias("g"))
    .select(F.col("g.mal_id").alias("genre_id"), F.col("g.name").alias("name"))
    .filter("genre_id is not null")
    .distinct()
)
genres_df.write.mode("overwrite").saveAsTable("genres")
print(f"genres: {genres_df.count()} rows")

anime_genres_df = (
    anime_src.select(F.col("mal_id").alias("anime_id"), F.explode_outer(F.col("genres")).alias("g"))
    .select("anime_id", F.col("g.mal_id").alias("genre_id"))
    .filter("genre_id is not null")
    .distinct()
)
anime_genres_df.write.mode("overwrite").saveAsTable("anime_genres")
print(f"anime_genres: {anime_genres_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. `characters`
# COMMAND ----------
# MAGIC %python
char_schema = ArrayType(StructType([
    StructField("character", StructType([
        StructField("mal_id", LongType(), True),
        StructField("name", StringType(), True),
    ]), True),
    StructField("role", StringType(), True),
    StructField("favorites", LongType(), True),
]))

characters_df = (
    spark.table("raw_characters")
    .select(F.col("anime_id"), F.from_json("payload", char_schema).alias("c"))
    .select(F.col("anime_id"), F.explode_outer("c").alias("e"))
    .select(
        F.col("e.character.mal_id").alias("character_id"),
        F.col("anime_id"),
        F.col("e.character.name").alias("name"),
        F.col("e.role").alias("role"),
        F.col("e.favorites").alias("favorites"),
    )
    .filter(F.col("character_id").isNotNull())
    .dropDuplicates(["character_id", "anime_id"])
)
characters_df = characters_df.withColumn("about", F.lit(None).cast("string"))
characters_df = characters_df.withColumn("embedding_vector", F.lit(None).cast("array<float>"))
characters_df.write.mode("overwrite").saveAsTable("characters")
print(f"characters: {characters_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. `reviews`
# COMMAND ----------
# MAGIC %python
review_schema = ArrayType(StructType([
    StructField("mal_id", LongType(), True),
    StructField("review", StringType(), True),
    StructField("score", IntegerType(), True),
    StructField("user", StructType([StructField("username", StringType(), True)]), True),
]))

reviews_df = (
    spark.table("raw_reviews")
    .select(F.col("anime_id"), F.from_json("payload", review_schema).alias("r"))
    .select(F.col("anime_id"), F.explode_outer("r").alias("e"))
    .select(
        F.col("e.mal_id").alias("review_id"),
        F.col("anime_id"),
        F.col("e.user.username").alias("author"),
        F.col("e.score").cast("double").alias("score"),
        F.col("e.review").alias("review"),
    )
    .filter(F.col("review_id").isNotNull())
    .dropDuplicates(["review_id"])
)
reviews_df = reviews_df.withColumn("embedding_vector", F.lit(None).cast("array<float>"))
reviews_df.write.mode("overwrite").saveAsTable("reviews")
print(f"reviews: {reviews_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Log the run
# COMMAND ----------
# MAGIC %python
n_anime = spark.table("anime").count()
n_char = spark.table("characters").count()
n_rev = spark.table("reviews").count()
spark.sql("""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'transform', '02_transform_silver', ?, 'success', current_timestamp()
""", args=[n_anime + n_char + n_rev])

print(f"anime={n_anime} | characters={n_char} | reviews={n_rev} | genres={spark.table('genres').count()} | anime_genres={spark.table('anime_genres').count()}")
print("02 transform complete.")
