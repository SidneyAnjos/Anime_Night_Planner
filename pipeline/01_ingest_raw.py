# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Jikan Bronze Ingestion
# MAGIC
# MAGIC Fetches from the Jikan API (MyAnimeList wrapper) and stores the **raw JSON payloads**
# MAGIC verbatim into `raw_anime`, `raw_characters`, `raw_reviews` (bronze / lineage layer).
# MAGIC
# MAGIC - **Rate limits**: Jikan allows ~3 req/s and 60 req/min → we throttle to ~1.1s between
# MAGIC   calls and retry with exponential backoff on 429/503.
# MAGIC - **Idempotent**: anime ids already present in the bronze tables are skipped, so re-runs
# MAGIC   only fetch what is missing.
# COMMAND ----------
# MAGIC %python
import json
import time
import datetime
import requests

BASE = "https://api.jikan.moe/v4"
# Capstone-scale scope (tune as desired):
TOP_ANIME_MAX_PAGES = 28          # up to ~700 all-time top titles
SEASON_MAX_PAGES = 8              # per season
PRIOR_SEASONS = 2                 # current + the previous 2 seasons
CHARACTERS_REVIEWS_LIMIT = 250    # only enrich the top N anime with characters/reviews
MIN_INTERVAL = 1.1                # seconds between requests (60 req/min budget)


class JikanClient:
    def __init__(self, min_interval=MIN_INTERVAL, max_retries=5):
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
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = requests.get(
                    f"{BASE}{path}",
                    params=params or {},
                    timeout=30,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 429 or resp.status_code >= 500:   # rate-limited / flaky edge
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
            data = self.get(path, params={**base_params, "page": page, "limit": 25})
            items = (data or {}).get("data") or []
            if not items:
                break
            page_fn(items)
            total += len(items)
            if not (data or {}).get("pagination", {}).get("has_next_page"):
                break
        return total


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Track what we already have + fetch genres
# COMMAND ----------
# MAGIC %python
def existing_set(table, col):
    try:
        return {r[0] for r in spark.sql(f"SELECT DISTINCT {col} FROM {table}").collect()}
    except Exception:  # noqa: BLE001 - table may not exist yet
        return set()


seen_anime = existing_set("raw_anime", "id")
seen_char_anime = existing_set("raw_characters", "anime_id")
seen_rev_anime = existing_set("raw_reviews", "anime_id")
print(f"Already in raw_anime: {len(seen_anime)} ids | characters fetched for: {len(seen_char_anime)} | reviews fetched for: {len(seen_rev_anime)}")

client = JikanClient()

raw_anime_rows = []
raw_char_rows = []
raw_rev_rows = []


def add_anime_payload(payload, source_url):
    mid = payload.get("mal_id")
    if mid is None or mid in seen_anime:
        return
    seen_anime.add(mid)
    raw_anime_rows.append((mid, source_url, now_utc(), json.dumps(payload)))


genre_data = client.get("/genres/anime", params={"filter": "genres"})
genres = [(g["mal_id"], g["name"]) for g in genre_data["data"]]
print(f"Genres fetched: {len(genres)}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Fetch top anime + seasonal anime
# COMMAND ----------
# MAGIC %python
def collect_top(items):
    for it in items:
        add_anime_payload(it, it.get("url") or f"{BASE}/anime/{it.get('mal_id')}")

n_top = client.paginated("/top/anime", collect_top, TOP_ANIME_MAX_PAGES)
print(f"Top anime fetched: {n_top} (new bronze rows so far: {len(raw_anime_rows)})")

SEASON_ORDER = ["winter", "spring", "summer", "fall"]


def season_before(year, season):
    idx = SEASON_ORDER.index(season)
    if idx == 0:
        return year - 1, "fall"
    return year, SEASON_ORDER[idx - 1]


def prior_seasons(n):
    today = datetime.date.today()
    m = today.month
    season = "winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8) else "fall"
    year = today.year
    out = []
    for _ in range(n):
        year, season = season_before(year, season)
        out.append((year, season))
    return out


def collect_season(items):
    for it in items:
        add_anime_payload(it, it.get("url") or f"{BASE}/anime/{it.get('mal_id')}")

n_now = client.paginated("/seasons/now", collect_season, SEASON_MAX_PAGES)
print(f"Current season fetched: {n_now}")

n_season_total = 0
for year, season in prior_seasons(PRIOR_SEASONS):
    n = client.paginated(f"/seasons/{year}/{season}", collect_season, SEASON_MAX_PAGES)
    n_season_total += n
    print(f"Season {year} {season}: {n} titles")

print(f"Total bronze anime rows collected: {len(raw_anime_rows)}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Write raw_anime (bronze)
# COMMAND ----------
# MAGIC %python
def write_rows(table, rows, columns):
    if not rows:
        print(f"{table}: no new rows")
        return 0
    df = spark.createDataFrame(rows, schema=columns)
    df.write.mode("append").saveAsTable(table)
    return len(rows)


n_written_anime = write_rows("raw_anime", raw_anime_rows, ["id", "source_url", "fetched_at", "payload"])
print(f"raw_anime: wrote {n_written_anime} rows")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Fetch characters + reviews for the top titles
# COMMAND ----------
# MAGIC %python
top_ids = [mid for mid in list(seen_anime) if mid not in seen_char_anime][:CHARACTERS_REVIEWS_LIMIT]
print(f"Enriching {len(top_ids)} titles with characters/reviews")

for i, mid in enumerate(top_ids):
    cdata = client.get(f"/anime/{mid}/characters", allow_missing=True)
    if cdata and cdata.get("data"):
        raw_char_rows.append((mid, mid, f"{BASE}/anime/{mid}/characters", now_utc(), json.dumps(cdata["data"])))
    rdata = client.get(f"/anime/{mid}/reviews", allow_missing=True)
    if rdata and rdata.get("data"):
        raw_rev_rows.append((mid, mid, f"{BASE}/anime/{mid}/reviews", now_utc(), json.dumps(rdata["data"])))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(top_ids)} done (chars={len(raw_char_rows)}, reviews={len(raw_rev_rows)})")

print(f"Collected character payloads: {len(raw_char_rows)} | review payloads: {len(raw_rev_rows)}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Write raw_characters + raw_reviews + log the run
# COMMAND ----------
# MAGIC %python
n_written_char = write_rows("raw_characters", raw_char_rows, ["id", "anime_id", "source_url", "fetched_at", "payload"])
n_written_rev = write_rows("raw_reviews", raw_rev_rows, ["id", "anime_id", "source_url", "fetched_at", "payload"])
print(f"raw_characters: wrote {n_written_char} rows | raw_reviews: wrote {n_written_rev} rows")

spark.sql("""
    INSERT INTO pipeline_log (run_id, step, rows, status, ts)
    SELECT 'ingest', '01_ingest_raw', ?, 'success', current_timestamp()
""", args=[n_written_anime + n_written_char + n_written_rev])

print(f"Requests made: {client.requests_made} | anime rows: {n_written_anime} | char rows: {n_written_char} | review rows: {n_written_rev}")
print("01 ingest complete.")
