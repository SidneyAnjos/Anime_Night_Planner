# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Transform (Movie Night Planner)
# MAGIC
# MAGIC Parses the bronze JSON payloads (raw TMDB payloads from step 01) and normalizes them into
# MAGIC the silver tables: `movies`, `cast`, `keywords`, `reviews`, `providers`, `genres`.
# MAGIC
# MAGIC - `movies` is updated with a **MERGE** (idempotent) keyed on `movie_id`.
# MAGIC - `cast`, `keywords`, `reviews`, `providers`, `genres` are fully derived from bronze →
# MAGIC   overwrite is safe on each run.
# MAGIC
# MAGIC Source payloads:
# MAGIC - `/movie/{id}`           → movie detail (title, overview, tagline, runtime, genres, …)
# MAGIC - `/movie/{id}/credits`   → cast[] + crew[]
# MAGIC - `/movie/{id}/keywords`    → keywords[]
# MAGIC - `/movie/{id}/reviews`     → results[] of full reviews
# MAGIC - `/movie/{id}/watch/providers`→ results.US.flatrate / buy / rent providers
# MAGIC - `/genre/movie/list`      → id + name
# COMMAND ----------
# MAGIC %python
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType, DoubleType, ArrayType, DateType,
)
from pyspark.sql import functions as F

# --- TMDB /movie/{id} detail payload (the subset we index on) ---
movie_schema = StructType([
    StructField("id", LongType(), True),
    StructField("title", StringType(), True),
    StructField("original_title", StringType(), True),
    StructField("overview", StringType(), True),
    StructField("tagline", StringType(), True),
    StructField("runtime", IntegerType(), True),
    StructField("vote_average", DoubleType(), True),
    StructField("vote_count", LongType(), True),
    StructField("release_date", StringType(), True),            # "yyyy-MM-dd" or null
    StructField("poster_path", StringType(), True),
    StructField("backdrop_path", StringType(), True),
    StructField("original_language", StringType(), True),
    StructField("status", StringType(), True),
    StructField("genres", ArrayType(StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
    ])), True),
])

# Step 01 stores the full /movie/{id} detail payload in raw_movies.payload.
movie_src = (
    spark.table("raw_movies")
    .select(F.from_json("payload", movie_schema).alias("m"))
    .select("m.*")
    .filter(F.col("id").isNotNull())
    .dropDuplicates(["id"])
)
print(f"Parsed movie detail records: {movie_src.count()}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. `movies` — MERGE (idempotent keyed on movie_id)
# COMMAND ----------
# MAGIC %python
movies_updates = movie_src.select(
    F.col("id").alias("movie_id"),
    F.col("title"),
    F.col("original_title"),
    F.col("overview"),
    F.col("tagline"),
    F.col("runtime"),
    F.col("vote_average"),
    F.col("vote_count"),
    F.to_date(F.col("release_date")).alias("release_date"),
    F.year(F.to_date(F.col("release_date"))).alias("year"),
    F.col("poster_path"),
    F.col("backdrop_path"),
    F.coalesce(
        F.transform(F.col("genres"), lambda g: g["name"]),
        F.array().cast("array<string>"),
    ).alias("genres"),
    F.col("original_language").alias("language"),
    F.col("status"),
)
movies_updates.createOrReplaceTempView("movies_updates")

spark.sql("""
    MERGE INTO movies AS t
    USING movies_updates AS s
    ON t.movie_id = s.movie_id
    WHEN MATCHED THEN UPDATE SET
        t.title = s.title,
        t.original_title = s.original_title,
        t.overview = s.overview,
        t.tagline = s.tagline,
        t.runtime = s.runtime,
        t.vote_average = s.vote_average,
        t.vote_count = s.vote_count,
        t.release_date = s.release_date,
        t.year = s.year,
        t.poster_path = s.poster_path,
        t.backdrop_path = s.backdrop_path,
        t.genres = s.genres,
        t.language = s.language,
        t.status = s.status,
        t.updated_at = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (
        movie_id, title, original_title, overview, tagline, runtime, vote_average, vote_count,
        release_date, year, poster_path, backdrop_path, genres, language, status, updated_at
    ) VALUES (
        s.movie_id, s.title, s.original_title, s.overview, s.tagline, s.runtime, s.vote_average,
        s.vote_count, s.release_date, s.year, s.poster_path, s.backdrop_path, s.genres, s.language,
        s.status, current_timestamp()
    )
""")
print("movies table merged.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. `genres` (dimension; stable TMDB genre list)
# COMMAND ----------
# MAGIC %python
genre_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
])
genres_df = (
    spark.table("raw_genres")
    .select(F.from_json("payload", genre_schema).alias("g"))
    .select(F.col("g.id").alias("genre_id"), F.col("g.name").alias("name"))
    .filter("genre_id is not null")
    .dropDuplicates(["genre_id"])
)
genres_df.write.mode("overwrite").saveAsTable("genres")
print(f"genres: {genres_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. `cast` (top billed cast per movie)
# COMMAND ----------
# MAGIC %python
credits_schema = StructType([
    StructField("id", LongType(), True),
    StructField("cast", ArrayType(StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("character", StringType(), True),
        StructField("order", IntegerType(), True),
    ])), True),
])

cast_df = (
    spark.table("raw_credits")
    .select(F.col("tmdb_id"), F.from_json("payload", credits_schema).alias("c"))
    .select(F.col("tmdb_id").alias("movie_id"), F.explode_outer("c.cast").alias("e"))
    .select(
        F.col("e.id").alias("person_id"),
        F.col("movie_id"),
        F.col("e.name").alias("name"),
        F.col("e.character").alias("character"),
        F.col("e.order").alias("credit_order"),
    )
    .filter(F.col("person_id").isNotNull())
    .dropDuplicates(["person_id", "movie_id"])
)
cast_df.write.mode("overwrite").saveAsTable("cast")
print(f"cast: {cast_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. `keywords`
# COMMAND ----------
# MAGIC %python
keywords_schema = StructType([
    StructField("id", LongType(), True),
    StructField("keywords", ArrayType(StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
    ])), True),
])

keywords_df = (
    spark.table("raw_keywords")
    .select(F.col("tmdb_id"), F.from_json("payload", keywords_schema).alias("k"))
    .select(F.col("tmdb_id").alias("movie_id"), F.explode_outer("k.keywords").alias("e"))
    .select(
        F.col("movie_id"),
        F.col("e.id").alias("keyword_id"),
        F.col("e.name").alias("keyword"),
    )
    .filter(F.col("keyword_id").isNotNull())
    .dropDuplicates(["movie_id", "keyword_id"])
)
keywords_df.write.mode("overwrite").saveAsTable("keywords")
print(f"keywords: {keywords_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. `reviews` (explode the paginated reviews payload)
# COMMAND ----------
# MAGIC %python
reviews_schema = StructType([
    StructField("id", LongType(), True),
    StructField("page", IntegerType(), True),
    StructField("results", ArrayType(StructType([
        StructField("id", StringType(), True),
        StructField("author", StringType(), True),
        StructField("content", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("url", StringType(), True),
        StructField("author_details", StructType([StructField("rating", IntegerType(), True)]), True),
    ])), True),
])

reviews_df = (
    spark.table("raw_reviews")
    .select(F.col("tmdb_id"), F.from_json("payload", reviews_schema).alias("r"))
    .select(F.col("tmdb_id").alias("movie_id"), F.explode_outer("r.results").alias("e"))
    .select(
        F.col("e.id").alias("review_id"),
        F.col("movie_id"),
        F.col("e.author").alias("author"),
        F.col("e.author_details.rating").cast("double").alias("rating"),
        F.col("e.content").alias("content"),
        F.col("e.created_at").alias("created_at"),
        F.col("e.url").alias("url"),
    )
    .filter(F.col("review_id").isNotNull())
    .dropDuplicates(["review_id"])
)
reviews_df.write.mode("overwrite").saveAsTable("reviews")
print(f"reviews: {reviews_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. `providers` (US watch providers: flatrate / rent / buy)
# COMMAND ----------
# MAGIC %python
providers_schema = StructType([
    StructField("id", LongType(), True),
    StructField("results", StructType([
        StructField("US", StructType([
            StructField("flatrate", ArrayType(StructType([
                StructField("provider_name", StringType(), True),
            ])), True),
            StructField("rent", ArrayType(StructType([
                StructField("provider_name", StringType(), True),
            ])), True),
            StructField("buy", ArrayType(StructType([
                StructField("provider_name", StringType(), True),
            ])), True),
        ]), True),
    ]), True),
])

# Flatten the three provider lists into long form: (movie_id, country, provider_name, provider_type)
prov = (
    spark.table("raw_providers")
    .select(F.col("tmdb_id").alias("movie_id"), F.from_json("payload", providers_schema).alias("p"))
    .select("movie_id", "p.results")
)

flatrate = (
    prov.select("movie_id", F.explode_outer("results.US.flatrate").alias("e"))
    .select("movie_id", F.lit("US").alias("country"), F.col("e.provider_name").alias("provider_name"),
            F.lit("flatrate").alias("provider_type"))
)
rent = (
    prov.select("movie_id", F.explode_outer("results.US.rent").alias("e"))
    .select("movie_id", F.lit("US").alias("country"), F.col("e.provider_name").alias("provider_name"),
            F.lit("rent").alias("provider_type"))
)
buy = (
    prov.select("movie_id", F.explode_outer("results.US.buy").alias("e"))
    .select("movie_id", F.lit("US").alias("country"), F.col("e.provider_name").alias("provider_name"),
            F.lit("buy").alias("provider_type"))
)

providers_df = (
    flatrate.unionByName(rent)
    .unionByName(buy)
    .filter(F.col("provider_name").isNotNull())
    .dropDuplicates(["movie_id", "country", "provider_name", "provider_type"])
)
providers_df.write.mode("overwrite").saveAsTable("providers")
print(f"providers: {providers_df.count()} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Log the run
# COMMAND ----------
# MAGIC %python
n_movies = spark.table("movies").count()
n_cast = spark.table("cast").count()
n_kw = spark.table("keywords").count()
n_rev = spark.table("reviews").count()
n_prov = spark.table("providers").count()
n_gen = spark.table("genres").count()

spark.sql(f"""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'transform', '02_transform_silver',
           {n_movies + n_cast + n_kw + n_rev + n_prov + n_gen}, 'success', current_timestamp()
""")

print(f"movies={n_movies} | cast={n_cast} | keywords={n_kw} | reviews={n_rev} | "
      f"providers={n_prov} | genres={n_gen}")
print("02 transform complete.")
