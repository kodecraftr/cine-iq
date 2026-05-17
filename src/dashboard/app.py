"""
Phase 7b — Streamlit Dashboard
================================
Visual interface for the movie recommender system.

Panels
------
  1. 🎬 Get Recommendations  — enter userId, see top-N recs with explanations
  2. 🔍 Find Similar Movies  — enter a movie title, see content-similar picks
  3. 📊 Your Taste Profile   — radar chart, decade heatmap, director affinities

Run
---
  streamlit run src/dashboard/app.py
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explainability"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "🎬 Movie Recommender",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    from ensemble      import HybridEnsemble
    from sentiment     import SentimentReRanker
    from content_based import ContentBasedFilter
    from explainer     import Explainer

    return {
        "ensemble"  : HybridEnsemble(alpha=0.6, beta=0.4),
        "sentiment" : SentimentReRanker(backend="vader", gamma=0.15),
        "cb"        : ContentBasedFilter.load(),
        "explainer" : Explainer(use_lime=False),   # LIME off for dashboard speed
    }

@st.cache_data(show_spinner=False)
def load_data():
    processed = Path(__file__).resolve().parents[2] / "data" / "processed"
    movies  = pd.read_csv(processed / "movies_merged.csv")
    ratings = pd.read_csv(processed / "ratings_clean.csv")
    return movies, ratings

def _parse(val):
    if isinstance(val, list): return val
    try: return ast.literal_eval(str(val))
    except: return []


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 Movie Recommender")
page = st.sidebar.radio(
    "Navigate",
    ["Get Recommendations", "Find Similar Movies", "Your Taste Profile"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Settings**")
alpha = st.sidebar.slider("Collaborative weight (α)", 0.1, 0.9, 0.6, 0.05)
gamma = st.sidebar.slider("Sentiment nudge (γ)",      0.0, 0.5, 0.15, 0.05)
n_recs = st.sidebar.number_input("# Recommendations", 5, 30, 10)

try:
    models = load_models()
    movies_df, ratings_df = load_data()
    models_ok = True
except Exception as e:
    st.error(f"Could not load models: {e}\nRun Phases 1–6 first.")
    models_ok = False


# ── PAGE 1: Recommendations ───────────────────────────────────────────────────
if page == "Get Recommendations":
    st.title("🎬 Personalised Recommendations")

    user_id = st.number_input("Enter your User ID", min_value=1, value=1, step=1)

    if st.button("Get Recommendations") and models_ok:
        with st.spinner("Generating recommendations …"):
            ens = models["ensemble"]
            ens.alpha, ens.beta = alpha, round(1 - alpha, 4)

            recs = ens.recommend(user_id=int(user_id), n=int(n_recs) * 3)

            reranker = models["sentiment"]
            reranker.gamma = gamma
            recs = reranker.rerank(recs).head(int(n_recs))

            recs = models["explainer"].explain_batch(int(user_id), recs)

        st.subheader(f"Top {int(n_recs)} picks for User {user_id}")

        for _, row in recs.iterrows():
            with st.expander(f"#{int(row['rank'])}  {row.get('title','Unknown')} "
                             f"({int(row['release_year']) if pd.notna(row.get('release_year')) else '?'}) "
                             f"⭐ {row.get('vote_average','?')}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Hybrid Score",    f"{row['hybrid_score']:.3f}")
                col2.metric("Sentiment Score", f"{row.get('sentiment_score', 0):.3f}")
                col3.metric("Final Score",     f"{row.get('final_score', row['hybrid_score']):.3f}")
                st.info(f"💡 {row.get('explanation', '')}")

        # score distribution bar chart
        fig = px.bar(
            recs, x="title", y="final_score",
            color="sentiment_score", color_continuous_scale="RdYlGn",
            title="Final Scores", labels={"final_score": "Score", "title": "Movie"}
        )
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)


# ── PAGE 2: Similar Movies ────────────────────────────────────────────────────
elif page == "Find Similar Movies":
    st.title("🔍 Find Similar Movies")

    all_titles = movies_df["title"].dropna().sort_values().unique().tolist()
    selected   = st.selectbox("Select a movie", all_titles)

    if st.button("Find Similar") and models_ok:
        row      = movies_df[movies_df["title"] == selected].iloc[0]
        movie_id = int(row["movieId"])

        with st.spinner("Computing similarities …"):
            similar = models["cb"].similar_movies(movie_id=movie_id, n=int(n_recs))

        st.subheader(f"Movies similar to **{selected}**")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                similar[["title", "release_year", "vote_average", "similarity_score"]],
                use_container_width=True
            )
        with col2:
            fig = px.bar(
                similar.head(10), x="similarity_score", y="title",
                orientation="h", title="Similarity Scores",
                color="similarity_score", color_continuous_scale="Blues"
            )
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)


# ── PAGE 3: Taste Profile ─────────────────────────────────────────────────────
elif page == "Your Taste Profile":
    st.title("📊 Your Taste Profile")

    user_id = st.number_input("Enter your User ID", min_value=1, value=1, step=1)

    if st.button("Show Profile") and models_ok:
        user_ratings = ratings_df[ratings_df["userId"] == int(user_id)]
        if user_ratings.empty:
            st.warning("No ratings found for this user.")
            st.stop()

        rated_movies = movies_df[movies_df["movieId"].isin(user_ratings["movieId"])].copy()
        rated_movies = rated_movies.merge(user_ratings[["movieId", "rating"]], on="movieId")

        # ── Genre Radar Chart ──
        st.subheader("🎭 Genre Preferences")
        rated_movies["genres_list"] = rated_movies["genres"].apply(_parse)
        genre_rows = []
        for _, r in rated_movies.iterrows():
            for g in r["genres_list"]:
                genre_rows.append({"genre": g, "rating": r["rating"]})
        genre_df = pd.DataFrame(genre_rows)

        if not genre_df.empty:
            genre_avg = genre_df.groupby("genre")["rating"].mean().reset_index()
            genre_avg = genre_avg[genre_avg["genre"] != "(no genres listed)"].nlargest(12, "rating")

            fig = go.Figure(go.Scatterpolar(
                r     = genre_avg["rating"],
                theta = genre_avg["genre"],
                fill  = "toself",
                line_color = "royalblue",
            ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])), title="Avg Rating by Genre")
            st.plotly_chart(fig, use_container_width=True)

        # ── Decade Heatmap ──
        st.subheader("📅 Ratings by Decade")
        rated_movies["decade"] = (rated_movies["release_year"] // 10 * 10).astype("Int64")
        decade_df = rated_movies.groupby("decade")["rating"].agg(["mean", "count"]).reset_index()
        decade_df.columns = ["decade", "avg_rating", "count"]
        decade_df = decade_df.dropna()

        fig2 = px.bar(
            decade_df, x="decade", y="avg_rating",
            color="count", color_continuous_scale="Viridis",
            title="Average Rating per Decade",
            labels={"avg_rating": "Avg Rating", "decade": "Decade", "count": "# Movies"}
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Director Affinity ──
        st.subheader("🎬 Favourite Directors")
        if "director" in rated_movies.columns:
            director_df = (
                rated_movies[rated_movies["director"].notna() & (rated_movies["director"] != "")]
                .groupby("director")["rating"]
                .agg(["mean", "count"])
                .reset_index()
            )
            director_df.columns = ["director", "avg_rating", "movies_rated"]
            director_df = director_df[director_df["movies_rated"] >= 2].nlargest(10, "avg_rating")

            fig3 = px.bar(
                director_df, x="avg_rating", y="director",
                orientation="h", color="avg_rating",
                color_continuous_scale="Burg",
                title="Top Directors (min. 2 movies rated)",
                labels={"avg_rating": "Avg Rating", "director": "Director"}
            )
            fig3.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig3, use_container_width=True)

        # ── Rating Distribution ──
        st.subheader("⭐ Your Rating Distribution")
        fig4 = px.histogram(
            user_ratings, x="rating", nbins=10,
            title="How You Rate Movies",
            color_discrete_sequence=["steelblue"]
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.success(f"Profile based on {len(user_ratings):,} ratings.")
