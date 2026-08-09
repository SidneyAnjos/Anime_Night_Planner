"""Minimal rate-limited Jikan v4 client (app-side copy of the pipeline client).

The pipeline notebooks carry their own copy because Databricks Apps are deployed as a
self-contained container; a shared package would require bundling infrastructure.
"""
import time

import requests

BASE = "https://api.jikan.moe/v4"
MIN_INTERVAL = 1.1  # seconds between requests (Jikan: ~3 req/s and 60 req/min)
MAX_RETRIES = 5


class JikanClient:
    def __init__(self, min_interval=MIN_INTERVAL):
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, path, params=None, allow_missing=False):
        for attempt in range(MAX_RETRIES):
            elapsed = time.time() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.time()
            try:
                resp = requests.get(
                    f"{BASE}{path}",
                    params=params or {},
                    timeout=30,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 429 or resp.status_code >= 500:  # rate-limited / flaky edge
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                if resp.status_code == 404 and allow_missing:
                    return None
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        return None

    def top_anime(self, limit=10, filter=None):
        """Live trending list from Jikan (optionally filter='airing' for currently airing titles)."""
        data = self.get("/top/anime", params={"limit": min(limit, 25), "page": 1, **(filter and {"filter": filter} or {})})
        return (data or {}).get("data", [])[:limit]
