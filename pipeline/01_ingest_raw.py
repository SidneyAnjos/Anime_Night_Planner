# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — TMDB Bronze Ingestion
# MAGIC
# MAGIC Fetches from the TMDB API (The Movie Database) and stores the **raw JSON payloads**
# MAGIC verbatim into bronze tables (`raw_movies`, `raw_credits`, `raw_keywords`, `raw_reviews`,
# MAGIC `raw_providers`, `raw_genres`) for lineage.
# MAGIC
# MAGIC - **Rate limits**: TMDB allows ~40 requests/10 seconds (~4 req/s). We throttle to ~0.25s
# MAGIC   between calls and retry with exponential backoff on 429/5xx.
# MAGIC - **Idempotent**: movie IDs already present in the bronze tables (within a 24h freshness
# MAGIC   window) are skipped, so re-runs only fetch what's missing.
# MAGIC - **API key**: read from Databricks secret scope `movie_night_planner` / key `tmdb_api_key`.
# MAGIC   Configure once via `databricks secrets create-scope ...` and `databricks secrets put ...`.
# COMMAND ----------
# MAGIC %python
import json
import time
import datetime
import requests
import os

BASE = "https://api.themoviedb.org/3"
# Capstone-scale scope (tune as desired):
POPULAR_MAX_PAGES = 25          # up to ~500 popular movies
ENRICHMENT_LIMIT = 200          # enrich top N movies with credits/keywords/reviews/providers
MIN_INTERVAL = 0.25             # seconds between requests (40 req/10s budget)
FRESHNESS_HOURS = 24            # re-fetch after this many hours


def get_secret(scope, key):
    """Read a secret from Databricks secret scope."""
    try:
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Secret {scope}/{key} not found: {e!r}")


class TMDBClient:
    def __init__(self, api_key, min_interval=MIN_INTERVAL, max_retries=5):
        self.api_key = api_key
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_ts = 0.0
        self.requests_made = 0

    def _throttle(self):
        elapsed = time.time() - self._last_ts
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_ts = time.time()

    def get(self, path, params=None, allow_missing=False):
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = requests.get(
                    f"{BASE}{path}",
                    params=params,
                    timeout=30,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 429 or resp.status_code >= 500:  # rate-limited / flaky edge
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                if resp.status_code == 404 and allow_missing:
                    return None
                resp.raise_for_status()
                self.requests_made += 1
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"    retry {path} in {wait}s ({exc!r})")
                time.sleep(wait)

    def paginated(self, path, page_fn, max_pages, **base_params):
        """Iterate a paginated endpoint, invoking page_fn(items) per page."""
        total = 0
        for page in range(1, max_pages + 1):
            data = self.get(path, params={**base_params, "page": page})
            items = (data or {}).get("results") or []
            if not items:
                break
            page_fn(items)
            total += len(items)
            if page >= (data or {}).get("total_pages", 0):
                break
        return total


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def fresh_cutoff(hours=FRESHNESS_HOURS):
    """Cutoff timestamp for idempotency: skip records fetched within this window."""
    return now_utc() - datetime.timedelta(hours=hours)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Get API key + track what we already have
# COMMAND ----------
# MAGIC %python
TMDB_API_KEY = get_secret("movie_night_planner", "tmdb_api_key")
print("TMDB API key loaded from secret scope.")


def existing_ids(table, col, hours=FRESHNESS_HOURS):
    """Return set of IDs in the bronze table with fetched_at within the freshness window."""
    cutoff = fresh_cutoff(hours)
    try:
        rows = spark.sql(
            f"SELECT DISTINCT {col} FROM {table} WHERE fetched_at >= ?"
        ).collect()
        return {r[0] for r in rows}
    except Exception:  # noqa: BLE001 - table may not exist yet
        return set()


seen_movies = existing_ids("raw_movies", "tmdb_id")
seen_credits = existing_ids("raw_credits", "tmdb_id")
seen_keywords = existing_ids("raw_keywords", "tmdb_id")
seen_reviews = existing_ids("raw_reviews", "tmdb_id")
seen_providers = existing_ids("raw_providers", "tmdb_id")
print(f"Already fresh in raw_movies: {len(seen_movies)} | credits: {len(seen_credits)} | "
      f"keywords: {len(seen_keywords)} | reviews: {len(seen_reviews)} | providers: {len(seen_providers)}")

client = TMDBClient(TMDB_API_KEY)

# Accumulators for new rows
raw_movies_rows = []
raw_credits_rows = []
raw_keywords_rows = []
raw_reviews_rows = []
raw_providers_rows = []


def add_movie_payload(payload, source_url):
    mid = payload.get("id")
    if mid is None or mid in seen_movies:
        return
    seen_movies.add(mid)
    raw_movies_rows.append((mid, source_url, now_utc(), json.dumps(payload)))


def add_credits_payload(payload, source_url):
    mid = payload.get("id")
    if mid is None or mid in seen_credits:
        return
    seen_credits.add(mid)
    raw_credits_rows.append((mid, source_url, now_utc(), json.dumps(payload)))


def add_keywords_payload(payload, source_url):
    mid = payload.get("id")
    if mid is None or mid in seen_keywords:
        return
    seen_keywords.add(mid)
    raw_keywords_rows.append((mid, source_url, now_utc(), json.dumps(payload)))


def add_reviews_payload(payload, source_url):
    mid = payload.get("id")
    if mid is None or mid in seen_reviews:
        return
    seen_reviews.add(mid)
    raw_reviews_rows.append((mid, source_url, now_utc(), json.dumps(payload)))


def add_providers_payload(payload, source_url):
    mid = payload.get("id")
    if mid is None or mid in seen_providers:
        return
    seen_providers.add(mid)
    raw_providers_rows.append((mid, source_url, now_utc(), json.dumps(payload)))


# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Fetch genre list (stable, cached long-term)
# COMMAND ----------
# MAGIC %python
genre_data = client.get("/genre/movie/list")
genres = genre_data.get("genres", [])
print(f"Genres fetched: {len(genres)}")
# Store in bronze for lineage
raw_genres_rows = []
for g in genres:
    raw_genres_rows.append((g["id"], f"{BASE}/genre/movie/list", now_utc(), json.dumps(g)))
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Fetch popular movies (primary source for our curated subset)
# COMMAND ----------
# MAGIC %python
def collect_movies(items):
    for it in items:
        add_movie_payload(it, it.get("url") or f"https://www.themoviedb.org/movie/{it.get('id')}")

n_popular = client.paginated("/discover/movie", collect_movies, POPULAR_MAX_PAGES, sort_by="popularity.desc")
print(f"Popular movies fetched: {n_popular} (new bronze rows so far: {len(raw_movies_rows)})")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Write raw_movies + raw_genres (bronze)
# COMMAND ----------
# MAGIC %python
def write_rows(table, rows, columns):
    if not rows:
        print(f"{table}: no new rows")
        return 0
    df = spark.createDataFrame(rows, schema=columns)
    df.write.mode("append").saveAsTable(table)
    return len(rows)


n_movies = write_rows("raw_movies", raw_movies_rows, ["tmdb_id", "source_url", "fetched_at", "payload"])
n_genres = write_rows("raw_genres", raw_genres_rows, ["genre_id", "source_url", "fetched_at", "payload"])
print(f"raw_movies: wrote {n_movies} rows | raw_genres: wrote {n_genres} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Enrich top movies with credits, keywords, reviews, providers
# COMMAND ----------
# MAGIC %python
# Enrich the top N movies from the popular fetch
enrich_ids = [mid for mid in list(seen_movies) if mid not in seen_credits][:ENRICHMENT_LIMIT]
print(f"Enriching {len(enrich_ids)} movies with credits/keywords/reviews/providers")

for i, mid in enumerate(enrich_ids):
    cdata = client.get(f"/movie/{mid}/credits", allow_missing=True)
    if cdata:
        add_credits_payload(cdata, f"{BASE}/movie/{mid}/credits")
    kdata = client.get(f"/movie/{mid}/keywords", allow_missing=True)
    if kdata:
        add_keywords_payload(kdata, f"{BASE}/movie/{mid}/keywords")
    rdata = client.get(f"/movie/{mid}/reviews", allow_missing=True)
    if rdata:
        add_reviews_payload(rdata, f"{BASE}/movie/{mid}/reviews")
    pdata = client.get(f"/movie/{mid}/watch/providers", allow_missing=True)
    if pdata:
        add_providers_payload(pdata, f"{BASE}/movie/{mid}/watch/providers")
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(enrich_ids)} done "
              f"(credits={len(raw_credits_rows)}, keywords={len(raw_keywords_rows)}, "
              f"reviews={len(raw_reviews_rows)}, providers={len(raw_providers_rows)})")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Write enrichment bronze tables + log the run
# COMMAND ----------
# MAGIC %python
n_credits = write_rows("raw_credits", raw_credits_rows, ["tmdb_id", "source_url", "fetched_at", "payload"])
n_keywords = write_rows("raw_keywords", raw_keywords_rows, ["tmdb_id", "source_url", "fetched_at", "payload"])
n_reviews = write_rows("raw_reviews", raw_reviews_rows, ["tmdb_id", "source_url", "fetched_at", "payload"])
n_providers = write_rows("raw_providers", raw_providers_rows, ["tmdb_id", "source_url", "fetched_at", "payload"])

total_written = n_movies + n_genres + n_credits + n_keywords + n_reviews + n_providers
spark.sql(f"""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'ingest', '01_ingest_raw', {int(total_written)}, 'success', current_timestamp()
""")

print(f"Requests made: {client.requests_made}")
print(f"Rows written: movies={n_movies}, genres={n_genres}, credits={n_credits}, "
      f"keywords={n_keywords}, reviews={n_reviews}, providers={n_providers}")
print("01 ingest complete.")