# Movie Night Planner — Capstone Submission

**Project**: A group movie-recommendation agent built on Databricks.
**Data source**: TMDB API (The Movie Database) — free for non-commercial educational use.
**Author**: Sidney Anjos
**Live app**: https://movie-night-planner-7474659512156367.aws.databricksapps.com (SSO-gated — see `evidence/` for a walkthrough)

---

## 1. How this project meets every requirement

| # | Requirement | Where it is implemented |
|---|-------------|--------------------------|
| 1 | **A data pipeline in Spark** | `pipeline/` — five Databricks notebooks (`00_setup_tables`, `01_ingest_raw`, `02_transform_silver`, `03_embed`, `04_vector_index`) wired into the `movie_pipeline_job` in `databricks.yml`. Bronze raw JSON → silver normalized tables → 1024-dim embeddings → Vector Search index. |
| 2 | **Integration with at least one third-party API** | **TMDB** (a suggested API): `/discover/movie`, `/movie/{id}`, `/credits`, `/keywords`, `/reviews`, `/watch/providers`, `/genre/movie/list`. Rate-limited (40 req/10s), idempotent, freshness-windowed. `app/tmdb_fetch.py` + `pipeline/01_ingest_raw.py`. |
| 3 | **Processing of unstructured data** | Plot + tagline + keywords + cast + review text embedded with Mosaic AI **`databricks-bge-large-en`** (1024-dim) into `movie_embeddings`, indexed by **Vector Search** for semantic retrieval. `pipeline/03_embed.py`, `app/vectorstore.py`, `app/tools.py` (`search_movies_semantic`). |
| 4 | **A Databricks App with a frontend** | `app/` — Streamlit app (Dashboard / Browse / Agent Chat / Admin) deployed as a Databricks App. `app/app.yaml` declares the run command + env so any deploy path serves correctly. |
| 5 | **An AI agent that does stuff** | `app/agent.py` — native function-calling agent on the Foundation Model chat endpoint. Tools in `app/tools.py` that **read** (semantic search, trending, details, compare, group history) and **write** (add_to_watchlist, log_group_rating, log_recommendation). Writes are real, parameterized rows — verified live (see `evidence/`). |

---

## 2. Architecture & data lineage

```
TMDB API ─▶ Spark pipeline ─▶ Unity Catalog ─▶ Databricks App
  /discover,        bronze      silver        movie_embeddings
  /movie/{id},      raw_movies  movies        └─▶ Vector Search index
  /credits, ...     raw_credits cast, ...      └─▶ Agent semantic search
```

1. **Bronze** — raw TMDB JSON payloads stored verbatim (`raw_movies`, `raw_credits`, `raw_keywords`,
   `raw_reviews`, `raw_providers`, `raw_genres`) with `fetched_at` + source URL for lineage.
   Critically, movies are fetched as `/movie/{id}` **detail** payloads (runtime + genres), not
   `/discover/movie` list payloads, so downstream filters work.
2. **Silver** — normalized, deduplicated tables (`movies`, `cast`, `keywords`, `reviews`, `providers`,
   `genres`). 505 movies, ~500 with runtime + genres.
3. **Embeddings** — `movie_embeddings.embedding_vector` (1024-dim) over plot + tagline + keywords +
   cast + review snippets.
4. **Vector index** — Delta-sync index enables semantic retrieval ("a heist thriller with a twist ending").
5. **Agent decision** — the agent retrieves by meaning, honors length/genre constraints, avoids
   re-recommending watched/queued titles via `get_group_history`, then **writes back**: watchlist,
   ratings, recommendations.

**Networking note**: this workspace's serverless job compute has no outbound egress, so TMDB ingestion
runs on the Databricks App's Admin page (app compute has egress); the pipeline runs transform → embed →
index over that bronze. See `README.md` → Notes.

---

## 3. What makes the agent's writes real

The agent uses **native OpenAI-style function-calling** on the Databricks Foundation Model
`chat/completions` endpoint, not free-text ReAct parsing. The model emits structured `tool_calls`
(`add_to_watchlist`, `log_group_rating`, `log_recommendation`) with real `movie_id`s returned by the
search tools; each call is executed against the SQL warehouse with parameterized statements, and the
model only reports an action after the tool returns success. Verified end-to-end: a full run produced
`+1 watchlist_items` row and `+1 recommendations` row in the warehouse (see `evidence/live_verification.md`).

---

## 4. How to run it yourself

Prereqs: a Databricks workspace with serverless compute, Foundation Model API, Vector Search, a SQL
warehouse, and a TMDB key in a secret scope (`movie_night_planner/tmdb_api_key`).

```bash
# 1. Deploy the bundle (pipeline job + app resource)
databricks bundle deploy

# 2. Build silver + embeddings + index from bronze, then deploy the app:
.venv/Scripts/python.exe deploy_app.py     # syncs app/ + deploys with command & env

# The Apps UI "Deploy" button also works — app/app.yaml carries the command + env.
```

Open the app URL, and in **Admin** run "Backfill bronze from TMDB" if bronze is empty (the only live-TMDB
step), then re-run the pipeline job.

---

## 5. Submission package manifest

```
submission/
├── SUBMISSION.md          # this document (requirements mapping, architecture, demo)
├── README.md              # project write-up (architecture, lineage, setup, deploy)
├── databricks.yml         # bundle: pipeline job + Databricks App resource
├── deploy_app.py          # one-command deploy helper (sync + deploy with command/env)
├── pipeline/              # Spark pipeline notebooks (cell-marker format)
│   ├── 00_setup_tables.py # catalog/schema + DDL + demo seed
│   ├── 01_ingest_raw.py   # TMDB → bronze raw_* (rate-limited, idempotent)
│   ├── 02_transform_silver.py
│   ├── 03_embed.py        # bge-large-en embeddings → movie_embeddings
│   └── 04_vector_index.py # Vector Search endpoint + index
├── app/                   # Databricks App (self-contained Streamlit + agent)
│   ├── app.py             # Streamlit frontend (Dashboard / Browse / Agent Chat / Admin)
│   ├── agent.py           # native function-calling agent loop
│   ├── tools.py           # agent tools: search/trending/details/compare + watchlist/rating/rec writes
│   ├── db.py              # SQL access layer (parameterized, numpy-normalized)
│   ├── vectorstore.py     # semantic search over the Vector Search index
│   ├── tmdb_fetch.py      # TMDB client + bronze fetch (detail payloads, rate-limited)
│   ├── app.yaml           # app spec: command + env (makes UI deploy work)
│   └── requirements.txt
└── evidence/              # demo walkthrough + live verification results
    ├── demo_script.md     # step-by-step recording script (screenshots / Loom)
    └── live_verification.md  # verified live agent run: tool calls + DB row deltas
```

---

## 6. Notes / honest limitations

- **Serverless egress**: TMDB ingestion must run in the app (not the serverless job) on this workspace —
  an environment constraint, handled by the Admin backfill.
- **Library scale**: the local library is ~500 popular movies — enough to demo semantic search + the
  agent, not the full TMDB catalog.
- **Single-writer assumption**: IDs are `max(id)+1` — fine for a single-writer agent/demo scale.
