# AI Anime Night Planner

Capstone for the **"Rise of the AI Data Engineer"** boot camp. A group anime-recommendation agent built on Databricks: it ingests anime data from the **Jikan API** (MyAnimeList wrapper), embeds unstructured text (synopses, characters, reviews) into a **Vector Search** index, and lets a **LangChain agent** — served from a **Databricks App** — semantically search that library, pull live trends, and **write back** to Lakebase (watchlist + ratings).

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
   Jikan API (MAL)      │            Databricks (Unity Catalog)          │
   /v4/top, /seasons,   │                                                 │
   /genres, /anime/{id} │   Spark pipeline (serverless job)               │
        │               │   ┌──────────┐   ┌──────────┐   ┌────────────┐ │
        └──────────────▶│   │ 01 Bronze │──▶│ 02 Silver │──▶│ 03 Embed   │ │
                       │   │ raw_*    │   │ anime /   │   │ embedding_  │ │
                       │   │ (JSON)   │   │ genres /  │   │ vector      │ │
                       │   └──────────┘   │ chars /   │   └─────┬──────┘ │
                       │                  │ reviews   │         │        │
                       │                  └──────────┘         ▼        │
                       │                             04 Vector Search    │
                       │                             anime_synopsis_index│
                       │                                       ▲          │
                       │   ┌────────────────────────────────────┼───┐    │
                       │   │ Databricks App (Streamlit)         │   │    │
                       │   │  Agent tools                       │   │    │
                       │   │   search_anime_semantic ───────────┘   │    │
                       │   │   fetch_trending_anime ──▶ Jikan       │    │
                       │   │   get_anime_details / compare_anime    │    │
                       │   │   add_to_watchlist / log_group_rating ─┼──▶│  writes
                       │   │   get_group_history (persistence)      │   │
                       │   └───────────────────────────────────────┘   │
                       └─────────────────────────────────────────────────┘
```

## Data lineage (the story the capstone tells)

1. **Raw Jikan JSON** → stored verbatim in `raw_anime`, `raw_characters`, `raw_reviews` (bronze, with `fetched_at` + source URL for lineage).
2. **Silver normalization** → `anime`, `genres`, `anime_genres`, `characters`, `reviews`.
3. **Embeddings** → `embedding_vector` (Mosaic AI `databricks-bge-large-en`, 1024-dim) on synopses (and optionally characters/reviews).
4. **Vector index** → `anime_synopsis_index` (Delta-sync) enables semantic retrieval.
5. **Agent decision** → the agent retrieves by meaning ("an emotional anime like *Your Lie in April* but sci-fi"), honors length preference ("a short 12-episode series"), and avoids re-recommending watched titles via `get_group_history`.

## Repo layout

```
databricks.yml        # bundle: pipeline job + Databricks App
pipeline/             # Spark pipeline notebooks (Databricks cell-marker format)
  00_setup_tables     # catalog/schema + DDL + demo seed
  01_ingest_raw       # Jikan → bronze raw_* tables (rate-limited, idempotent)
  02_transform_silver # bronze → silver normalized tables
  03_embed            # Mosaic AI embeddings → embedding_vector
  04_vector_index     # Vector Search endpoint + index
app/                  # Databricks App (self-contained Streamlit + agent)
  app.yaml, requirements.txt, app.py, db.py, vectorstore.py,
  jikan.py, tools.py, agent.py
```

## Prerequisites

- Databricks workspace with: **serverless compute** (jobs + warehouses), **Foundation Model API (pay-as-you-go)** for embeddings + chat, and **Vector Search** enabled in the region.
- Databricks CLI + VS Code Databricks extension (auth via metadata service, profile `SidneyAnjos`).
- A **SQL warehouse** (serverless) for app/agent DML.
- Jikan API is keyless — no secrets required.

## Setup & deploy

```bash
# 1. Validate and deploy the bundle (creates pipeline job + app)
databricks bundle validate
databricks bundle deploy

# 2. Run the pipeline to populate tables + vector index
databricks jobs run-now --job-id <job_id>

# 3. Deploy the app
databricks apps deploy anime-night-planner

# 4. Grant the app's service principal access (see notes in code / docs)
```

Then open the app URL from the Databricks Apps UI and ask the agent to plan a movie night.
