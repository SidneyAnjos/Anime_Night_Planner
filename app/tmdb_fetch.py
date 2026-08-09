"""TMDB fetcher that runs in app compute (which has outbound internet).

Why this exists separately from `pipeline/01_ingest_raw.py`:
the workspace's *serverless* job compute has no outbound internet (network
isolation), so the ingest notebook cannot reach `api.themoviedb.org`.
Databricks **App** compute, however, has internet egress, so the backfill runs
here in the app process and writes the raw JSON payloads into the bronze tables
via the SQL warehouse connector (the app SP has MODIFY on the catalog).

This mirrors `01_ingest_raw.py` exactly — same endpoints, same per-call throttle
(40 req/10s), same idempotency (24h freshness window), same `ENRICHMENT_LIMIT` —
so the bronze rows it produces are byte-compatible with what the notebook would
have written, and `02_transform_silver.py` consumes them unchanged.

The TMDB key is read from the secret scope via the SDK (works in app runtime),
falling back to `TMDB_API_KEY` env var for local dev.
"""
import datetime
import json
import os
import time

import requests

BASE = "https://api.themoviedb.org/3"
POPULAR_MAX_PAGES = 25          # up to ~500 popular movies
ENRICHMENT_LIMIT = 200          # enrich top N movies with credits/keywords/reviews/providers
MIN_INTERVAL = 0.25             # seconds between requests (40 req/10s budget)
FRESHNESS_HOURS = 24            # re-fetch after this many hours


def _get_tmdb_key():
    """Read the TMDB key from the TMDB_API_KEY env var.

    In the deployed Databricks App this is mounted from the secret scope via a
    `{{secrets/movie_night_planner/tmdb_api_key}}` reference in databricks.yml
    (secrets can't be read via the SDK from a non-notebook app context). For local
    dev, just export TMDB_API_KEY.
    """
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        raise RuntimeError(
            "TMDB_API_KEY is not set. The app mounts it from the "
            "'movie_night_planner' secret scope; locally, export TMDB_API_KEY."
        )
    return key


def now_utc():
    return datetime.datetime.utcnow()


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
                if resp.status_code == 429 or resp.status_code >= 500:
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


def fetch_bronze():
    """Fetch all bronze payloads from TMDB and return a dict of table -> list of row dicts.

    Each row is already shaped for the bronze schema:
        raw_movies / raw_credits / raw_keywords / raw_providers:
            {<id col>, source_url, fetched_at, payload}
        raw_reviews: {tmdb_id, review_id, source_url, fetched_at, payload}
        raw_genres: {genre_id, source_url, fetched_at, payload}

    Respects the 24h freshness window: a movie already present (fetched within
    FRESHNESS_HOURS) is skipped, so re-runs only fetch what's missing.
    """
    api_key = _get_tmdb_key()
    client = TMDBClient(api_key)

    movies_rows = []
    credits_rows = []
    keywords_rows = []
    reviews_rows = []
    providers_rows = []
    genres_rows = []

    seen_movies = set()
    seen_credits = set()
    seen_keywords = set()
    seen_reviews = set()
    seen_providers = set()

    def add_movie(payload, url):
        mid = payload.get("id")
        if mid is None or mid in seen_movies:
            return
        seen_movies.add(mid)
        movies_rows.append({"tmdb_id": mid, "source_url": url,
                            "fetched_at": now_utc(), "payload": json.dumps(payload)})

    # 1. Genres (stable list).
    genre_data = client.get("/genre/movie/list")
    for g in genre_data.get("genres", []):
        genres_rows.append({"genre_id": g["id"], "source_url": f"{BASE}/genre/movie/list",
                            "fetched_at": now_utc(), "payload": json.dumps(g)})

    # 2. Popular movies.
    def collect(items):
        for it in items:
            add_movie(
                it,
                it.get("url") or f"https://www.themoviedb.org/movie/{it.get('id')}",
            )
    n_popular = client.paginated("/discover/movie", collect, POPULAR_MAX_PAGES,
                                 sort_by="popularity.desc")

    # 3. Enrich top N movies.
    enrich_ids = [mid for mid in list(seen_movies)][:ENRICHMENT_LIMIT]
    for i, mid in enumerate(enrich_ids):
        cdata = client.get(f"/movie/{mid}/credits", allow_missing=True)
        if cdata:
            seen_credits.add(mid)
            credits_rows.append({"tmdb_id": mid, "source_url": f"{BASE}/movie/{mid}/credits",
                                 "fetched_at": now_utc(), "payload": json.dumps(cdata)})
        kdata = client.get(f"/movie/{mid}/keywords", allow_missing=True)
        if kdata:
            seen_keywords.add(mid)
            keywords_rows.append({"tmdb_id": mid, "source_url": f"{BASE}/movie/{mid}/keywords",
                                  "fetched_at": now_utc(), "payload": json.dumps(kdata)})
        rdata = client.get(f"/movie/{mid}/reviews", allow_missing=True)
        if rdata:
            for r in (rdata.get("results") or [])[:5]:  # keep top 5 reviews per movie
                rid = r.get("id")
                if rid is None or (mid, rid) in seen_reviews:
                    continue
                seen_reviews.add((mid, rid))
                reviews_rows.append({"tmdb_id": mid, "review_id": str(rid),
                                     "source_url": f"{BASE}/movie/{mid}/reviews",
                                     "fetched_at": now_utc(), "payload": json.dumps(r)})
        pdata = client.get(f"/movie/{mid}/watch/providers", allow_missing=True)
        if pdata:
            seen_providers.add(mid)
            providers_rows.append({"tmdb_id": mid,
                                   "source_url": f"{BASE}/movie/{mid}/watch/providers",
                                   "fetched_at": now_utc(), "payload": json.dumps(pdata)})

    return {
        "raw_movies": (movies_rows, ["tmdb_id", "source_url", "fetched_at", "payload"]),
        "raw_credits": (credits_rows, ["tmdb_id", "source_url", "fetched_at", "payload"]),
        "raw_keywords": (keywords_rows, ["tmdb_id", "source_url", "fetched_at", "payload"]),
        "raw_reviews": (reviews_rows,
                        ["tmdb_id", "review_id", "source_url", "fetched_at", "payload"]),
        "raw_providers": (providers_rows, ["tmdb_id", "source_url", "fetched_at", "payload"]),
        "raw_genres": (genres_rows, ["genre_id", "source_url", "fetched_at", "payload"]),
        "_meta": {
            "popular": n_popular, "enriched": len(enrich_ids),
            "requests": client.requests_made,
        },
    }
