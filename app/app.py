"""Anime Night Planner — Databricks App (Streamlit).

Pages:
- Dashboard  : the selected group's watchlist, ratings and stats (refreshes on every load).
- Browse     : keyword + semantic search over the anime library.
- Agent Chat : talk to the agent; it recommends, adds to the watchlist, and logs ratings.
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


def render_browse(database, vs):
    st.header("Browse anime")
    keyword = st.text_input("Keyword (title / English title)", value="")
    sem = st.text_input("Semantic search (describe the mood, e.g. 'a cozy slice of life about food')", value="")

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
        f"SELECT anime_id, title, title_english, score, episodes, genres "
        f"FROM {database.table('anime')}"
    )
    params = []
    if keyword:
        sql += " WHERE lower(title) LIKE lower(?) OR lower(title_english) LIKE lower(?)"
        params = [f"%{keyword}%", f"%{keyword}%"]
    sql += " ORDER BY score DESC NULLS LAST LIMIT 200"
    rows = database.query(sql, params)
    if not rows:
        st.info("No titles match. Try a different keyword or the semantic search box.")
        return
    df = pd.DataFrame(rows)
    df["genres"] = df["genres"].map(_fmt_genres)
    st.dataframe(df, use_container_width=True)


def render_chat(database, agent, group_id):
    st.header("Agent chat")
    st.caption("Plan a night, get recommendations, add to watchlist, log ratings.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. Suggest a short sci-fi anime for tonight — nothing heavy.")
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
    st.set_page_config(page_title="Anime Night Planner", layout="wide")
    database, vs, agent = _backend()

    st.sidebar.title("🍿 Anime Night Planner")
    groups = database.groups()
    if not groups:
        st.warning("No groups found. Run the pipeline (pipeline/00_setup_tables) first.")
        return
    names = [g["name"] for g in groups]
    selected = st.sidebar.selectbox("Group", names)
    group_id = next(g["group_id"] for g in groups if g["name"] == selected)

    page = st.sidebar.radio("Page", ["Dashboard", "Browse", "Agent Chat"])
    if page == "Dashboard":
        render_dashboard(database, group_id)
    elif page == "Browse":
        render_browse(database, vs)
    else:
        render_chat(database, agent, group_id)


main()
