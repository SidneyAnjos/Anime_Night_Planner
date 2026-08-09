"""Movie Night Planner — Databricks App (Streamlit).

Pages:
- Dashboard  : the selected group's watchlist, ratings and recommendations (refreshes on every load).
- Browse     : keyword + semantic search over the movie library.
- Agent Chat : talk to the agent; it recommends, adds to the watchlist, logs ratings and records
               recommendations.
"""
import pandas as pd
import streamlit as st

import db as db_mod
import tools
import vectorstore
from agent import Agent


@st.cache_resource
def _backend():
    database = db_mod.Database()
    vs = vectorstore.VectorStore()
    tools.init(database, vs)
    agent = Agent()
    return database, vs, agent


def _fmt_genres(genres):
    if not genres:
        return ""
    if isinstance(genres, str):
        return genres
    return ", ".join(str(g) for g in genres)


def render_dashboard(database, group_id):
    st.header("Group dashboard")
    watchlist = database.watchlist(group_id)
    ratings = database.ratings(group_id)
    recs = database.query(
        f"""
        SELECT r.movie_id, m.title, r.reason, r.recommended_at
        FROM {database.table('recommendations')} r
        LEFT JOIN {database.table('movies')} m ON r.movie_id = m.movie_id
        WHERE r.group_id = ?
        ORDER BY r.recommended_at DESC
        """,
        [group_id],
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Queued", sum(1 for w in watchlist if w["status"] == "queued"))
    c2.metric("Watched", sum(1 for w in watchlist if w["status"] == "watched"))
    avg = pd.DataFrame(ratings)["score"].mean() if ratings else None
    c3.metric("Avg rating", f"{avg:.1f}" if avg is not None else "—")

    st.subheader("Watchlist")
    if watchlist:
        df = pd.DataFrame(watchlist)
        df["genres"] = df["genres"].map(_fmt_genres)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No watchlist items yet — ask the agent to plan a night.")

    st.subheader("Ratings")
    if ratings:
        st.dataframe(pd.DataFrame(ratings), use_container_width=True)
    else:
        st.info("No ratings yet — log one in the agent chat.")

    st.subheader("Recommendations")
    if recs:
        st.dataframe(pd.DataFrame(recs), use_container_width=True)
    else:
        st.info("The agent hasn't recorded any recommendations yet.")


def render_browse(database, vs):
    st.header("Browse movies")
    keyword = st.text_input("Keyword (title)", value="")
    sem = st.text_input("Semantic search (describe the mood, e.g. 'a heist thriller with a twist ending')",
                        value="")

    if sem:
        try:
            results = vs.search(sem, num_results=20)
            st.subheader("Semantic matches")
            rows = [{**r, "genres": _fmt_genres(r.get("genres"))} for r in results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Semantic search failed: {exc!r}")
        return

    sql = (
        f"SELECT movie_id, title, vote_average, runtime, year, genres "
        f"FROM {database.table('movies')}"
    )
    params = []
    if keyword:
        sql += " WHERE lower(title) LIKE lower(?) OR lower(original_title) LIKE lower(?)"
        params = [f"%{keyword}%", f"%{keyword}%"]
    sql += " ORDER BY vote_average DESC NULLS LAST, vote_count DESC NULLS LAST LIMIT 200"
    rows = database.query(sql, params)
    if not rows:
        st.info("No titles match. Try a different keyword or the semantic search box.")
        return
    df = pd.DataFrame(rows)
    df["genres"] = df["genres"].map(_fmt_genres)
    st.dataframe(df, use_container_width=True)


def render_admin(database):
    """Admin / backfill — fetches TMDB bronze from app compute (has outbound internet) because
    the serverless job compute is network-isolated and can't reach api.themoviedb.org."""
    st.header("Admin — backfill bronze from TMDB")
    st.caption(
        "Serverless job compute can't reach the public internet, so TMDB is fetched here in the "
        "app and written to the raw_* bronze tables. After this completes, kick off the pipeline "
        "(transform → embed → index) to populate silver + vectors."
    )

    counts = database.query(
        f"""
        SELECT 'raw_movies'    AS t, COUNT(*) AS n FROM {database.table('raw_movies')}    UNION ALL
        SELECT 'raw_credits'              , COUNT(*)   FROM {database.table('raw_credits')}   UNION ALL
        SELECT 'raw_keywords'             , COUNT(*)   FROM {database.table('raw_keywords')}  UNION ALL
        SELECT 'raw_reviews'              , COUNT(*)   FROM {database.table('raw_reviews')}   UNION ALL
        SELECT 'raw_providers'            , COUNT(*)   FROM {database.table('raw_providers')} UNION ALL
        SELECT 'raw_genres'               , COUNT(*)   FROM {database.table('raw_genres')}
        """
    )
    df = pd.DataFrame(counts)
    st.subheader("Current bronze counts")
    st.dataframe(df, use_container_width=True)

    if st.button("Backfill bronze from TMDB", type="primary"):
        with st.spinner("Fetching from TMDB and writing bronze… this takes a few minutes (~1300 calls)."):
            try:
                result = tools.backfill_bronze_from_tmdb()
                st.success("Backfill complete.")
                st.json(result)
                st.info("Now re-run the pipeline job (transform → embed → index) to build silver + vectors.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Backfill failed: {exc!r}")


def render_chat(database, agent, group_id):
    st.header("Agent chat")
    st.caption("Plan a night, get recommendations, add to the watchlist, log ratings.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. Suggest a short thriller under 2 hours for tonight — something with a twist.")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages[:-1]
                    ]
                    answer = agent.run(prompt, chat_history=history)
                except Exception as exc:  # noqa: BLE001
                    answer = f"Sorry, the agent hit an error: {exc!r}"
            st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


def main():
    st.set_page_config(page_title="Movie Night Planner", layout="wide")
    database, vs, agent = _backend()

    st.sidebar.title("🍿 Movie Night Planner")
    groups = database.groups()
    if not groups:
        st.warning("No groups found. Run the pipeline (pipeline/00_setup_tables) first.")
        return
    names = [g["name"] for g in groups]
    selected = st.sidebar.selectbox("Group", names)
    group_id = next(g["group_id"] for g in groups if g["name"] == selected)

    page = st.sidebar.radio("Page", ["Dashboard", "Browse", "Agent Chat", "Admin"])
    if page == "Dashboard":
        render_dashboard(database, group_id)
    elif page == "Browse":
        render_browse(database, vs)
    elif page == "Agent Chat":
        render_chat(database, agent, group_id)
    else:
        render_admin(database)


main()
