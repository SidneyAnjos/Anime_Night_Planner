"""Agent tools: semantic search, live trends, detail/compare, and DB writes.

`init(db, vs)` must be called once at app startup to wire in the data layer.
Tools return JSON strings the LLM can read; write tools are the "takes real actions" half of the
capstone requirement (watchlist + ratings).
"""
import json

from langchain_core.tools import tool

from jikan import JikanClient

_db = None
_vs = None
_jikan = JikanClient()


def init(db, vs):
    global _db, _vs
    _db = db
    _vs = vs


def _anime_by_ids(ids, extra_where=(), extra_params=()):
    """Fetch anime rows by id, preserving an external ordering via a CASE expression."""
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    where = f"a.anime_id IN ({placeholders})"
    if extra_where:
        where += " AND " + " AND ".join(extra_where)
    rows = _db.query(
        f"""
        SELECT a.anime_id, a.title, a.title_english, a.score, a.episodes, a.genres, a.synopsis
        FROM {_db.table('anime')} a
        WHERE {where}
        """,
        list(ids) + list(extra_params),  # IN (...) placeholders precede the extra WHERE params
    )
    rank = {i: idx for idx, i in enumerate(ids)}
    rows.sort(key=lambda r: rank.get(r["anime_id"], len(ids)))
    return rows


def _summarize(rows):
    out = []
    for r in rows:
        syn = (r.get("synopsis") or "")
        out.append({
            "anime_id": r["anime_id"],
            "title": r["title"],
            "score": r.get("score"),
            "episodes": r.get("episodes"),
            "genres": r.get("genres"),
            "synopsis_snippet": syn[:300] + ("..." if len(syn) > 300 else ""),
        })
    return out


@tool
def search_anime_semantic(query: str, max_episodes: int = 0, genre: str = "", limit: int = 5) -> str:
    """Semantically search the anime library by a natural-language description, e.g.
    'an emotional romance with a sci-fi twist'. Use this FIRST for any recommendation request.
    Optionally restrict by episode length (set max_episodes to e.g. 12 for a short one-night
    series) and by genre (single genre name like 'sci-fi' or 'romance'). Returns up to `limit`
    titles with anime_id, title, score, episodes, genres and a synopsis snippet."""
    candidates = _vs.search(query, columns=["title", "score", "episodes", "genres"], num_results=100)
    ids = [int(c["anime_id"]) for c in candidates]
    if not ids:
        return json.dumps([])

    extra_where = []
    extra_params = []
    if max_episodes and max_episodes > 0:
        extra_where.append("a.episodes IS NOT NULL AND a.episodes <= ?")
        extra_params.append(int(max_episodes))
    if genre:
        extra_where.append("EXISTS (SELECT 1 FROM explode(a.genres) g WHERE lower(g.col) = lower(?))")
        extra_params.append(genre)

    rows = _anime_by_ids(ids, extra_where, extra_params)
    # Re-rank the (possibly filtered) rows by semantic similarity.
    sim = {int(c["anime_id"]): c["_similarity"] for c in candidates}
    rows.sort(key=lambda r: sim.get(r["anime_id"], float("inf")))
    return json.dumps(_summarize(rows[:limit]), ensure_ascii=False)


@tool
def fetch_trending_anime(limit: int = 10) -> str:
    """Fetch currently trending anime LIVE from the Jikan API (MyAnimeList), not from the local
    database. Returns up to `limit` currently-airing titles. Use for real-time trend questions."""
    items = _jikan.top_anime(limit=limit, filter="airing")
    return json.dumps([
        {"anime_id": it.get("mal_id"), "title": it.get("title"), "score": it.get("score"),
         "episodes": it.get("episodes"), "url": it.get("url")}
        for it in items
    ], ensure_ascii=False)


@tool
def get_anime_details(title: str) -> str:
    """Look up full details for a specific anime by title (fuzzy match on title or English title).
    Returns anime_id, synopsis, score, episodes, genres, status. Use to answer questions about a
    specific show, or to obtain an anime_id for watchlist/rating actions."""
    rows = _db.query(
        f"""
        SELECT anime_id, title, title_english, synopsis, score, episodes, genres, status
        FROM {_db.table('anime')}
        WHERE lower(title) LIKE lower(?) OR lower(title_english) LIKE lower(?)
        ORDER BY score DESC NULLS LAST
        LIMIT 5
        """,
        [f"%{title}%", f"%{title}%"],
    )
    return json.dumps(rows, ensure_ascii=False, default=str)


@tool
def compare_anime(title_a: str, title_b: str) -> str:
    """Compare two anime by title. Returns their key stats (score, episodes, genres, synopsis
    snippets) so you can synthesize a comparison for the group."""
    out = {}
    for label, title in (("anime_a", title_a), ("anime_b", title_b)):
        rows = _db.query(
            f"""
            SELECT title, title_english, score, episodes, genres, synopsis, status
            FROM {_db.table('anime')}
            WHERE lower(title) LIKE lower(?) OR lower(title_english) LIKE lower(?)
            ORDER BY score DESC NULLS LAST LIMIT 1
            """,
            [f"%{title}%", f"%{title}%"],
        )
        r = rows[0] if rows else {}
        if r.get("synopsis"):
            r["synopsis"] = r["synopsis"][:300] + ("..." if len(r["synopsis"]) > 300 else "")
        out[label] = r
    return json.dumps(out, ensure_ascii=False, default=str)


@tool
def add_to_watchlist(group_id: int, anime_id: int, status: str = "queued") -> str:
    """Add an anime to a group's watchlist. status is 'queued' or 'watched'. This WRITES to the
    watchlist_items table. Use after the group agrees on a recommendation."""
    if status not in ("queued", "watched"):
        status = "queued"
    _db.execute(
        f"""
        INSERT INTO {_db.table('watchlist_items')} (group_id, anime_id, status, added_by, added_at)
        VALUES (?, ?, ?, 1, current_timestamp())
        """,
        [int(group_id), int(anime_id), status],
    )
    return f"Added anime {anime_id} to group {group_id} watchlist (status='{status}')."


@tool
def log_group_rating(group_id: int, anime_id: int, score: int, comment: str = "") -> str:
    """Log the group's rating (1-10) for an anime, with an optional comment. UPSERT: rating the
    same group+anime again overwrites the previous entry. This WRITES to the ratings table."""
    score = max(1, min(10, int(score)))
    _db.execute(
        f"""
        MERGE INTO {_db.table('ratings')} t
        USING (SELECT ? AS group_id, ? AS anime_id) s
        ON t.group_id = s.group_id AND t.anime_id = s.anime_id
        WHEN MATCHED THEN UPDATE SET t.score = ?, t.comment = ?, t.rated_at = current_timestamp()
        WHEN NOT MATCHED THEN INSERT (group_id, anime_id, score, comment, rated_at)
            VALUES (s.group_id, s.anime_id, ?, ?, current_timestamp())
        """,
        [int(group_id), int(anime_id), score, comment, score, comment],
    )
    return f"Logged rating {score}/10 for anime {anime_id} by group {group_id}."


@tool
def get_group_history(group_id: int) -> str:
    """Get the group's watchlist and ratings history — which anime are queued/watched and how they
    were rated. CALL THIS before making recommendations so you never re-recommend a show the group
    has already watched or queued."""
    watchlist = _db.watchlist(int(group_id))
    ratings = _db.ratings(int(group_id))
    return json.dumps({"watchlist": watchlist, "ratings": ratings}, ensure_ascii=False, default=str)


TOOLS = [
    search_anime_semantic,
    fetch_trending_anime,
    get_anime_details,
    compare_anime,
    add_to_watchlist,
    log_group_rating,
    get_group_history,
]
