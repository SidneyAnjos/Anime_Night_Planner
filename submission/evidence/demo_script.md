# Demo walkthrough (for screenshots / Loom recording)

The live app is SSO-gated, so reviewers outside your Databricks workspace can't click in. Record this
walkthrough (Loom / QuickTime, ~3–4 min) or take screenshots per step. Keep the camera on the browser and
speak over it briefly.

## 0. Setup sanity (optional, 10 s)
Open the **Databricks workspace** → **Workflows** → show the `movie_pipeline_job` with the last run green
(setup → transform → embed → index).

## 1. Browse — semantic search (45 s)
Open the app → **Browse**.
- Type `a heist thriller with a twist ending` → results come back with title, TMDB score, runtime, year, genres.
- Mention: *"every result comes from the local Unity Catalog library — no live API at serving time."*

## 2. Agent Chat — recommend + take real actions (90 s)
Open **Agent Chat** for a group.
- Ask: *"Suggest a short thriller under 2 hours with a twist ending."* The agent gives 3–5 options with
  reasons (semantic search, honoring `max_runtime` + genre).
- Ask: *"Add your best pick to this group's watchlist as queued and record the recommendation."*
- After it answers, open **Dashboard** and show the **new watchlist row** + **new recommendation row**
  that appeared — the point being the agent's claim matches a real DB write.
- Optionally: *"Rate it 8/10."* → the Ratings table updates.

## 3. Admin — backfill (30 s, optional)
Open **Admin** → show the bronze row counts → (optionally) click **Backfill bronze from TMDB** and show it
running. Mention this is the *only* live-TMDB step, and it runs here because serverless job compute has no
outbound egress.

## 4. Close (10 s)
"Architecture: TMDB → Spark bronze → silver → bge-large-en embeddings → Vector Search index; the app's
agent retrieves semantically and writes watchlist/ratings/recommendations to the warehouse."
