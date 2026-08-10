# Movie Night Planner

Capstone for the **"Rise of the AI Data Engineer"** boot camp. A group movie-recommendation agent built on
Databricks: it ingests movie data from the **TMDB API** (The Movie Database), embeds unstructured text
(plot + tagline + keywords + cast + reviews) into a **Vector Search** index, and lets a **native
function-calling agent** — served from a **Databricks App** — semantically search that library, pull up
trending titles, compare films, and **write back** to Lakebase (watchlist + ratings + recommendations). The
agent drives the Foundation Model's structured tool-calling API, so the writes it reports are real rows in
the warehouse — never a claimed-but-not-performed action.

Data provided by TMDB — free for non-commercial educational use with attribution.

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
   TMDB API             │            Databricks (Unity Catalog)          │
   /discover/movie,     │                                                 │
   /movie/{id},         │   Spark pipeline (serverless job)               │
   /credits, /keywords, │   ┌──────────┐   ┌──────────┐   ┌────────────┐ │
   /reviews,            │   │ 01 Bronze │──▶│ 02 Silver │──▶│ 03 Embed   │ │
   /watch/providers,    │   │ raw_*    │   │ movies /  │   │ movie_     │ │
   /genre/movie/list    │   │ (JSON)   │   │ cast /    │   │ embeddings  │ │
        │               │   └──────────┘   │ keywords │   └─────┬──────┘ │
        └──────────────▶│                  │ reviews / │         │        │
                        │                  │ providers │         ▼        │
                        │                  │ genres   │  04 Vector Search │
                        │                  └──────────┘  movie_embeddings│
                        │                               _index   ▲       │
                        │   ┌────────────────────────────────────┼───┐   │
                        │   │ Databricks App (Streamlit)         │   │   │
                        │   │  Agent tools                        │   │   │
                        │   │   search_movies_semantic ───────────┘   │   │
                        │   │   fetch_trending_movies (local DB)      │   │
                        │   │   get_movie_details / compare_movies    │   │
                        │   │   add_to_watchlist / log_group_rating ─┼──▶│ writes
                        │   │   log_recommendation                   │   │
                        │   │   get_group_history (persistence)      │   │
                        │   └───────────────────────────────────────┘   │
                        └─────────────────────────────────────────────────┘
```

## Data lineage (the story the capstone tells)

1. **Raw TMDB JSON** → stored verbatim in `raw_movies`, `raw_credits`, `raw_keywords`, `raw_reviews`,
   `raw_providers`, `raw_genres` (bronze, with `fetched_at` + source URL for lineage).
2. **Silver normalization** → `movies`, `cast`, `keywords`, `reviews`, `providers`, `genres`.
3. **Embeddings** → `movie_embeddings.embedding_vector` (Mosaic AI `databricks-bge-large-en`, 1024-dim)
   over plot + tagline + top keywords + top cast + review snippets.
4. **Vector index** → `movie_embeddings_index` (Delta-sync, self-managed vectors) enables semantic
   retrieval.
5. **Agent decision** → the agent retrieves by meaning ("a heist thriller with a twist ending"), honors
   length preference ("under two hours", max_runtime=120) and genre, and avoids re-recommending watched
   titles via `get_group_history`. It then writes back: add to watchlist, log a rating, record the
   recommendation.

## Repo layout

```
databricks.yml        # bundle: pipeline job + Databricks App
pipeline/             # Spark pipeline notebooks (Databricks cell-marker format)
  00_setup_tables     # catalog/schema + DDL + demo seed
  01_ingest_raw       # TMDB → bronze raw_* tables (rate-limited, idempotent)
  02_transform_silver # bronze → silver normalized tables
  03_embed            # Mosaic AI embeddings → movie_embeddings
  04_vector_index     # Vector Search endpoint + index
app/                  # Databricks App (self-contained Streamlit + agent)
  requirements.txt, app.py, db.py, vectorstore.py, tools.py, agent.py
```

## Prerequisites

- Databricks workspace with: **serverless compute** (jobs + warehouses), **Foundation Model API
  (pay-as-you-go)** for embeddings + chat, and **Vector Search** enabled in the region.
- Databricks CLI + VS Code Databricks extension (auth via metadata service / profile).
- A **SQL warehouse** (serverless) for app/agent DML — set its http path as `SQL_WAREHOUSE_PATH` in
  `databricks.yml`.
- A **TMDB API key** (v3) — get one free at https://www.themoviedb.org/settings/api.
- A **Databricks secret scope** holding the key so the ingest notebook can read it without the key
  landing in source.

## Setup & deploy

```bash
# 1. Create a Databricks secret scope + store your TMDB key (replace <KEY> with your v3 API key)
databricks secrets create-scope movie_night_planner --initial-manage-principal users
databricks secrets put-secret movie_night_planner tmdb_api_key --string-value "<KEY>"

# 2. Validate and deploy the bundle (creates pipeline job + app)
databricks bundle validate
databricks bundle deploy

# 3. Run the pipeline to populate tables + the vector index
databricks jobs run-now --job-id <job_id>

# 4. Deploy the app. Two equivalent ways:
#    (a) The Apps UI "Deploy" button or `databricks apps deploy <name>` now work —
#        `app/app.yaml` declares the command + env vars, so bare deployments run
#        Streamlit with the right env instead of crashing. (If the TMDB key shows up
#        as an unresolved "{{secrets/...}}" literal, tmdb_fetch.py falls back to
#        reading the secret via the app SP.)
#    (b) Or use the script (syncs app/ to the bundle path + deploys explicitly):
.venv/Scripts/python.exe deploy_app.py

# 5. Grant the app's service principal access (see notes in code / docs)
```

Then open the app URL from the Databricks Apps UI and ask the agent to plan a movie night.

## Notes

- The app makes **no live API calls** at serving time — everything comes from Unity Catalog, so the
  agent is fully reproducible offline and unaffected by TMDB rate limits. Live TMDB access happens only
  in step 01 of the scheduled pipeline.
- The query-side embedder (app `vectorstore.py`) and the index-time embedder (step 03) both call
  `serving_endpoints.invoke("databricks-bge-large-en", …)`, so index and query vectors share one space.
