"""Agent tools: semantic search, local trending, detail/compare, and DB writes.

`init(db, vs)` must be called once at app startup to wire in the data layer.
Tools return JSON strings the LLM can read; the write tools are the "takes real actions" half of the
capstone requirement (watchlist + ratings + recommendations).

Domain: Movie Night Planner, backed by the TMDB silver tables populated by the Spark pipeline.
No live API calls are made at serving time — everything comes from Unity Catalog, so the agent is
fully reproducible offline.
"""
import json

from langchain_core.tools import tool


_db = None
_vs = None


def init(db, vs):
    global _db, _vs
    _db = db
    _vs = vs


def backfill_bronze_from_tmdb():
    """Fetch TMDB bronze data from app compute (has internet) and load it into the 6 bronze tables.

    Runs server-side in the Databricks App to work around the serverless job's lack of outbound
    internet. **Replaces** the bronze tables on each run (DELETE then INSERT) so a re-run never
    leaves stale payloads of a different shape (e.g. the old `/discover/movie` list payloads that
    lack runtime/genres) mixed with fresh `/movie/{id}` detail payloads. Returns a summary dict
    for display. NOT an @tool — this is an admin action, not an LLM call.
    """
    from tmdb_fetch import fetch_bronze
    data = fetch_bronze()
    meta = data.pop("_meta", {})
    written = {}
    for table, (rows, columns) in data.items():
        # Replace stale rows first so the fresh payloads are the ONLY rows for each movie.
        _db.execute(f"DELETE FROM {_db.table(table)}")
        # Bronze rows are dicts; convert to tuples in column order.
        tuple_rows = [tuple(r.get(c) for c in columns) for r in rows]
        written[table] = _db.bulk_insert(table, columns, tuple_rows)
    return {"written": written, "requests": meta.get("requests", 0),
            "popular": meta.get("popular", 0), "enriched": meta.get("enriched", 0)}


def _native_list(v):
    """Normalize a value to a JSON-native form for feeding back to the model.

    The databricks-sql-connector returns ARRAY<...> columns as `numpy.ndarray` and typed
    numerics as numpy scalars. An ndarray breaks `json.dumps` AND truth-tests on it raise
    "The truth value of an array with more than one element is ambiguous" — which made the
    genre filter in search_movies_semantic crash. Convert to plain lists/scalars here once.
    """
    if isinstance(v, (list, tuple)):
        return [_native_list(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return [_native_list(x) for x in v.tolist()]
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:  # numpy is a transitive dep of databricks-sql-connector; be safe
        pass
    return v


def _movies_by_ids(ids, extra_where=(), extra_params=()):
    """Fetch movie rows by id, preserving an external ordering via a CASE expression."""
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    where = f"m.movie_id IN ({placeholders})"
    if extra_where:
        where += " AND " + " AND ".join(extra_where)
    rows = _db.query(
        f"""
        SELECT m.movie_id, m.title, m.vote_average, m.runtime, m.genres, m.overview,
               m.release_date, m.year
        FROM {_db.table('movies')} m
        WHERE {where}
        """,
        list(ids) + list(extra_params),  # IN (...) placeholders precede the extra WHERE params
    )
    # Normalize numpy ARRAY/scalar values (see _native_list) so genre filtering, truthiness
    # checks and json.dumps all behave as with plain Python types.
    for r in rows:
        r["genres"] = _native_list(r.get("genres"))
    rank = {i: idx for idx, i in enumerate(ids)}
    rows.sort(key=lambda r: rank.get(r["movie_id"], len(ids)))
    return rows


def _summarize(rows):
    out = []
    for r in rows:
        ov = (r.get("overview") or "")
        out.append({
            "movie_id": r["movie_id"],
            "title": r["title"],
            "vote_average": r.get("vote_average"),
            "runtime": r.get("runtime"),
            "year": r.get("year"),
            "genres": r.get("genres"),
            "overview_snippet": ov[:300] + ("..." if len(ov) > 300 else ""),
        })
    return out


def _next_id(table_name, col):
    """max(id)+1 for a capstone-scale app (single-writer agent)."""
    row = _db.query(f"SELECT coalesce(max({col}), 0) + 1 AS nxt FROM {_db.table(table_name)}")
    return int(row[0]["nxt"]) if row else 1


@tool
def search_movies_semantic(query: str, max_runtime: int = 0, genre: str = "", limit: int = 5) -> str:
    """Semantically search the movie library by a natural-language description, e.g.
    'a heist thriller with a twist ending' or 'a feel-good 90s comedy'. Use this FIRST for any
    recommendation request. Optionally restrict by runtime (set max_runtime to e.g. 120 for a tight
    weeknight pick, in minutes) and by genre (single genre name like 'comedy' or 'horror'). Returns
    up to `limit` titles with movie_id, title, vote_average, runtime, year, genres and an overview
    snippet."""
    num_candidates = 100 if (max_runtime or genre) else limit
    candidates = _vs.search(query, columns=["title", "vote_average", "runtime", "genres"],
                            num_results=num_candidates * 4)
    ids = [int(c["movie_id"]) for c in candidates]
    if not ids:
        return json.dumps([])

    extra_where = []
    extra_params = []
    if max_runtime and max_runtime > 0:
        extra_where.append("m.runtime IS NOT NULL AND m.runtime <= ?")
        extra_params.append(int(max_runtime))
    # Genre filtering is done in Python below (movies.genres is ARRAY<STRING>), so the SQL fetch
    # stays a simple IN-list join regardless of warehouse SQL dialect.

    rows = _movies_by_ids(ids, extra_where, extra_params)
    if genre:
        g = genre.strip().lower()
        rows = [r for r in rows
                if r.get("genres") and any(g == str(x).lower() for x in r["genres"])]
    # Re-rank the (possibly filtered) rows by semantic similarity.
    sim = {int(c["movie_id"]): c["_similarity"] for c in candidates}
    rows.sort(key=lambda r: sim.get(r["movie_id"], float("inf")))
    return json.dumps(_summarize(rows[:limit]), ensure_ascii=False, default=str)


@tool
def fetch_trending_movies(limit: int = 10) -> str:
    """Return the most popular movies in the local library, ranked by TMDB vote count (a proxy for
    'trending'). Use for 'what's popular' / 'what's trending' style questions. Returns movie_id,
    title, vote_average, vote_count, year. No live API — everything comes from the local database."""
    rows = _db.query(
        f"""
        SELECT movie_id, title, vote_average, vote_count, year
        FROM {_db.table('movies')}
        WHERE vote_count IS NOT NULL
        ORDER BY vote_count DESC, vote_average DESC NULLS LAST
        LIMIT ?
        """,
        [int(limit)],
    )
    return json.dumps(rows, ensure_ascii=False, default=str)


@tool
def get_movie_details(title: str) -> str:
    """Look up full details for a specific movie by title (fuzzy match on title or original_title).
    Returns movie_id, title, overview, vote_average, runtime, genres, release_date, status. Use to
    answer questions about a specific film, or to obtain a movie_id for watchlist/rating actions."""
    rows = _db.query(
        f"""
        SELECT movie_id, title, original_title, overview, vote_average, runtime, genres,
               release_date, status
        FROM {_db.table('movies')}
        WHERE lower(title) LIKE lower(?) OR lower(original_title) LIKE lower(?)
        ORDER BY vote_average DESC NULLS LAST
        LIMIT 5
        """,
        [f"%{title}%", f"%{title}%"],
    )
    return json.dumps(rows, ensure_ascii=False, default=str)


@tool
def compare_movies(title_a: str, title_b: str) -> str:
    """Compare two movies by title. Returns their key stats (vote_average, runtime, genres, overview
    snippets, release year) so you can synthesize a comparison for the group."""
    out = {}
    for label, title in (("movie_a", title_a), ("movie_b", title_b)):
        rows = _db.query(
            f"""
            SELECT title, original_title, vote_average, runtime, genres, overview, release_date, year
            FROM {_db.table('movies')}
            WHERE lower(title) LIKE lower(?) OR lower(original_title) LIKE lower(?)
            ORDER BY vote_average DESC NULLS LAST LIMIT 1
            """,
            [f"%{title}%", f"%{title}%"],
        )
        r = rows[0] if rows else {}
        if r.get("overview"):
            r["overview"] = r["overview"][:300] + ("..." if len(r["overview"]) > 300 else "")
        out[label] = r
    return json.dumps(out, ensure_ascii=False, default=str)


@tool
def add_to_watchlist(group_id: int, movie_id: int, status: str = "queued") -> str:
    """Add a movie to a group's watchlist. status is 'queued' or 'watched'. This WRITES to the
    watchlist_items table. Use after the group agrees on a recommendation."""
    if status not in ("queued", "watched"):
        status = "queued"
    item_id = _next_id("watchlist_items", "item_id")
    _db.execute(
        f"""
        INSERT INTO {_db.table('watchlist_items')} (item_id, group_id, movie_id, status, added_by, added_at)
        VALUES (?, ?, ?, ?, 1, current_timestamp())
        """,
        [item_id, int(group_id), int(movie_id), status],
    )
    return f"Added movie {movie_id} to group {group_id} watchlist (status='{status}')."


@tool
def log_group_rating(group_id: int, movie_id: int, score: int, comment: str = "") -> str:
    """Log the group's rating (1-10) for a movie, with an optional comment. UPSERT: rating the
    same group+movie again overwrites the previous entry. This WRITES to the ratings table."""
    score = max(1, min(10, int(score)))
    existing = _db.query(
        f"""
        SELECT rating_id FROM {_db.table('ratings')}
        WHERE group_id = ? AND movie_id = ?
        """,
        [int(group_id), int(movie_id)],
    )
    if existing:
        _db.execute(
            f"""
            UPDATE {_db.table('ratings')}
            SET score = ?, comment = ?, rated_at = current_timestamp()
            WHERE group_id = ? AND movie_id = ?
            """,
            [score, comment, int(group_id), int(movie_id)],
        )
    else:
        rating_id = _next_id("ratings", "rating_id")
        _db.execute(
            f"""
            INSERT INTO {_db.table('ratings')}
                (rating_id, group_id, movie_id, score, comment, rated_at)
            VALUES (?, ?, ?, ?, ?, current_timestamp())
            """,
            [rating_id, int(group_id), int(movie_id), score, comment],
        )
    return f"Logged rating {score}/10 for movie {movie_id} by group {group_id}."


@tool
def log_recommendation(group_id: int, movie_id: int, reason: str = "") -> str:
    """Record that the agent recommended a specific movie to a group, with a short reason. This
    WRITES to the recommendations table and is how the capstone demonstrates the agent taking a
    durable action. Use after you settle on a pick for the group."""
    rec_id = _next_id("recommendations", "rec_id")
    _db.execute(
        f"""
        INSERT INTO {_db.table('recommendations')}
            (rec_id, group_id, movie_id, reason, recommended_by, recommended_at)
        VALUES (?, ?, ?, ?, 'agent', current_timestamp())
        """,
        [rec_id, int(group_id), int(movie_id), reason[:500]],
    )
    return f"Recorded recommendation of movie {movie_id} for group {group_id}."


@tool
def get_group_history(group_id: int) -> str:
    """Get the group's watchlist, ratings and past recommendations — which movies are
    queued/watched, how they were rated, and what the agent already recommended. CALL THIS before
    making recommendations so you never re-recommend a show the group has already watched, queued
    or been told to watch."""
    watchlist = _db.watchlist(int(group_id))
    ratings = _db.ratings(int(group_id))
    recs = _db.query(
        f"""
        SELECT r.movie_id, m.title, r.reason, r.recommended_at
        FROM {_db.table('recommendations')} r
        LEFT JOIN {_db.table('movies')} m ON r.movie_id = m.movie_id
        WHERE r.group_id = ?
        ORDER BY r.recommended_at DESC
        """,
        [int(group_id)],
    )
    return json.dumps({"watchlist": watchlist, "ratings": ratings, "recommendations": recs},
                      ensure_ascii=False, default=str)


TOOLS = [
    search_movies_semantic,
    fetch_trending_movies,
    get_movie_details,
    compare_movies,
    add_to_watchlist,
    log_group_rating,
    log_recommendation,
    get_group_history,
]
