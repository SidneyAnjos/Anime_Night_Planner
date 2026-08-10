# Live verification results

These runs were executed against the live Databricks workspace (profile `SidneyAnjos`, catalog
`movie_night_planner.default`, serverless SQL warehouse) using the exact code in `app/` — the same code
the deployed Databricks App runs.

## 1. Agent end-to-end run — real durable writes

Prompt given to `Agent.run(...)`:

> *"Suggest a short thriller under 2 hours with a twist ending. Then add your single best pick to the
> group 1 watchlist as queued and record the recommendation."*

Observed tool-call sequence (native function-calling):

```
search_movies_semantic({'query': 'short thriller under 2 hours with a twist ending', 'max_runtime': 120, 'limit': 5})
search_movies_semantic({'query': 'thriller with a twist ending under 2 hours', 'max_runtime': 120, 'genre': 'thriller', 'limit': 5})
add_to_watchlist({'group_id': 1, 'movie_id': 1727780, 'status': 'queued'})
log_recommendation({'group_id': 1, 'movie_id': 1727780, 'reason': 'short thriller under 2 hours with a twist ending'})
```

Agent answer (matches the writes):

> *"I've added 'Borderline' to the watchlist for group 1 and recorded the recommendation."*

**DB delta verified after the run** (queries against `watchlist_items` and `recommendations`):

| Table | Before | After | Delta |
|-------|--------|-------|-------|
| `watchlist_items` | 5 | 6 | **+1** |
| `recommendations` | 1 | 2 | **+1** |

The `movie_id` written (1727780) is a real TMDB id returned by `search_movies_semantic` — not invented.
Repeated runs reproduced the same behavior (`+1` / `+1` each time).

## 2. Genre filter no longer crashes

Earlier, `search_movies_semantic(..., genre='thriller')` crashed with
`ValueError: The truth value of an array with more than one element is ambiguous` because the SQL
connector returns ARRAY columns as `numpy.ndarray`. Fixed by normalizing every query result at the data
layer (`db.Database.query()` → `_native()`: ndarray→list, np scalar→plain type). Re-run of the same call
with `genre='thriller'` returns clean JSON with genre lists.

## 3. App deployment

- Databricks App `movie-night-planner` — deployment **SUCCEEDED**, app **RUNNING**, compute **ACTIVE**,
  HTTP **200** at https://movie-night-planner-7474659512156367.aws.databricksapps.com.
- `app/app.yaml` carries the run command (`streamlit run app.py`) + env, so the Apps UI "Deploy" button
  and `databricks apps deploy` (which send no command/env) now serve correctly instead of crashing.
  Verified by creating a bare deployment (source only) → SUCCEEDED + RUNNING + HTTP 200.
- `TMDB_API_KEY` resolves via the app spec; `tmdb_fetch.py` also falls back to reading the key from the
  `movie_night_planner` secret scope through the app's service-principal auth if the env var is missing
  or an unresolved `{{secrets/...}}` literal.

## 4. Pipeline + data

- Silver `movies`: 505 rows, ~500 with runtime, ~498 with genres.
- `movie_embeddings`: 505 rows (1024-dim, `databricks-bge-large-en`).
- `movie_embeddings_index`: Vector Search index READY (Delta-sync).
- Bronze: 6 `raw_*` tables with `/movie/{id}` detail payloads (runtime + genres present), replacing an
  earlier `/discover/movie` list-payload shape that produced all-NULL runtimes.
