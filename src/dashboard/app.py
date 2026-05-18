"""
Cine IQ Streamlit Dashboard
===========================
Interactive movie recommendation interface built on the trained Cine IQ models.

Run
---
  streamlit run src/dashboard/app.py
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explainability"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Cine IQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .app-title { font-size: 2.25rem; font-weight: 800; margin-bottom: 0.1rem; }
    .muted { color: #667085; }
    .pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #eef2f6;
        color: #344054;
        font-size: 0.78rem;
        margin: 0.12rem 0.12rem 0.12rem 0;
    }
    .movie-card {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
        min-height: 230px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .movie-title { font-size: 1.08rem; font-weight: 750; margin-bottom: 0.15rem; }
    .movie-meta { color: #667085; font-size: 0.9rem; margin-bottom: 0.65rem; }
    .score-line { font-size: 0.9rem; color: #344054; margin-top: 0.35rem; }
    .reason-box {
        border-left: 3px solid #2e90fa;
        padding-left: 0.7rem;
        color: #344054;
        font-size: 0.92rem;
        margin-top: 0.75rem;
    }
    .watch-card {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 0.75rem;
        background: #fcfcfd;
        min-height: 112px;
    }
    .section-label {
        color: #475467;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading recommendation models...")
def load_models():
    from ensemble import HybridEnsemble
    from sentiment import SentimentReRanker
    from content_based import ContentBasedFilter
    from explainer import Explainer

    return {
        "ensemble": HybridEnsemble(alpha=0.6, beta=0.4),
        "sentiment": SentimentReRanker(backend="vader", gamma=0.15),
        "cb": ContentBasedFilter.load(),
        "explainer": Explainer(use_lime=False),
    }


@st.cache_data(show_spinner=False)
def load_data():
    processed = Path(__file__).resolve().parents[2] / "data" / "processed"
    movies = pd.read_csv(processed / "movies_merged.csv")
    ratings = pd.read_csv(processed / "ratings_clean.csv")
    return movies, ratings


def parse_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def build_profiles(ratings: pd.DataFrame) -> list[dict]:
    available = sorted(int(uid) for uid in ratings["userId"].dropna().unique())
    names = [
        ("Aarav", "Sci-fi, thrillers, and big-screen spectacle"),
        ("Maya", "Crime dramas and sharp character stories"),
        ("Kabir", "Animation, family adventures, and comfort watches"),
        ("Isha", "Music, romance, and modern drama"),
        ("Rohan", "Festival favorites and tense social thrillers"),
    ]

    profiles = []
    for i, user_id in enumerate(available[:5]):
        name, taste = names[i] if i < len(names) else (f"Viewer {i + 1}", "Mixed movie taste")
        profiles.append({"name": name, "taste": taste, "user_id": user_id})
    return profiles


def profile_watchlist(profile: dict, movies: pd.DataFrame, ratings: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    user_ratings = ratings[ratings["userId"] == profile["user_id"]]
    watchlist = (
        user_ratings.sort_values("rating", ascending=False)
        .head(limit)
        .merge(movies, on="movieId", how="left")
    )
    return watchlist


def movie_tags(row: pd.Series, limit: int = 3) -> str:
    genres = parse_list(row.get("genres", []))[:limit]
    return "".join(f'<span class="pill">{genre}</span>' for genre in genres)


def render_watch_shelf(watchlist: pd.DataFrame):
    cols = st.columns(3)
    for idx, (_, row) in enumerate(watchlist.iterrows()):
        with cols[idx % 3]:
            year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "Unknown"
            rating = row.get("rating", 0)
            st.markdown(
                f"""
                <div class="watch-card">
                    <div class="movie-title">{row.get("title", "Unknown")}</div>
                    <div class="movie-meta">{year} · rated {rating:.1f}/5</div>
                    {movie_tags(row)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def recommendation_cards(recs: pd.DataFrame):
    cols = st.columns(3)
    for idx, (_, row) in enumerate(recs.iterrows()):
        with cols[idx % 3]:
            year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "Unknown"
            vote = row.get("vote_average", 0)
            score = row.get("final_score", row.get("hybrid_score", 0))
            sentiment = row.get("sentiment_score", 0)
            explanation = row.get("explanation", "Recommended from this profile's watch history.")
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="section-label">Pick #{int(row.get("rank", idx + 1))}</div>
                    <div class="movie-title">{row.get("title", "Unknown")}</div>
                    <div class="movie-meta">{year} · audience score {vote}</div>
                    <div class="score-line">Match score: <b>{score:.3f}</b></div>
                    <div class="score-line">Audience signal: <b>{sentiment:+.3f}</b></div>
                    <div class="reason-box">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def generate_recommendations(models: dict, profile: dict, n_recs: int, alpha: float, gamma: float) -> pd.DataFrame:
    ensemble = models["ensemble"]
    ensemble.alpha = alpha
    ensemble.beta = round(1 - alpha, 4)

    recs = ensemble.recommend(user_id=profile["user_id"], n=n_recs * 3)
    reranker = models["sentiment"]
    reranker.gamma = gamma
    recs = reranker.rerank(recs).head(n_recs)
    return models["explainer"].explain_batch(profile["user_id"], recs)


try:
    models = load_models()
    movies_df, ratings_df = load_data()
    profiles = build_profiles(ratings_df)
    models_ok = bool(profiles)
except Exception as exc:
    st.error(f"Could not load Cine IQ models: {exc}\nRun preprocessing and model training first.")
    st.stop()


st.sidebar.markdown("<div class='app-title'>Cine IQ</div>", unsafe_allow_html=True)
st.sidebar.caption("Explainable movie recommendations")

selected_profile_name = st.sidebar.selectbox(
    "Viewer profile",
    [profile["name"] for profile in profiles],
)
profile = next(item for item in profiles if item["name"] == selected_profile_name)
st.sidebar.caption(profile["taste"])

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Browse",
    ["For You", "Discover Similar", "Taste Profile"],
)

st.sidebar.markdown("---")
n_recs = st.sidebar.slider("Number of picks", 3, 12, 6)
alpha = st.sidebar.slider("Personalization strength", 0.1, 0.9, 0.6, 0.05)
gamma = st.sidebar.slider("Audience sentiment boost", 0.0, 0.5, 0.15, 0.05)


st.markdown('<div class="app-title">Cine IQ</div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="muted">Now watching as <b>{profile["name"]}</b> · {profile["taste"]}</p>',
    unsafe_allow_html=True,
)


if page == "For You":
    watchlist = profile_watchlist(profile, movies_df, ratings_df)

    st.subheader("Continue from this watch history")
    render_watch_shelf(watchlist)

    st.subheader("Recommended next")
    with st.spinner("Ranking movies for this profile..."):
        recs = generate_recommendations(models, profile, int(n_recs), alpha, gamma)

    if recs.empty:
        st.warning("No recommendations are available for this profile yet.")
    else:
        recommendation_cards(recs)

        fig = px.bar(
            recs,
            x="title",
            y="final_score",
            color="sentiment_score",
            color_continuous_scale="RdYlGn",
            title="Recommendation ranking",
            labels={"final_score": "Match score", "title": "Movie"},
        )
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=20))
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)


elif page == "Discover Similar":
    st.subheader("Find movies like something you already enjoy")

    all_titles = movies_df["title"].dropna().sort_values().unique().tolist()
    selected = st.selectbox("Choose a movie", all_titles)
    row = movies_df[movies_df["title"] == selected].iloc[0]

    with st.spinner("Finding nearby movies..."):
        similar = models["cb"].similar_movies(movie_id=int(row["movieId"]), n=int(n_recs))

    hero_cols = st.columns([1, 2])
    with hero_cols[0]:
        st.markdown(
            f"""
            <div class="movie-card">
                <div class="section-label">Source title</div>
                <div class="movie-title">{row.get("title", "Unknown")}</div>
                <div class="movie-meta">{int(row["release_year"]) if pd.notna(row.get("release_year")) else "Unknown"} · audience score {row.get("vote_average", 0)}</div>
                {movie_tags(row)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_cols[1]:
        fig = px.bar(
            similar.head(10),
            x="similarity_score",
            y="title",
            orientation="h",
            title="Closest matches",
            color="similarity_score",
            color_continuous_scale="Blues",
            labels={"similarity_score": "Similarity", "title": ""},
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=350, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        similar[["title", "release_year", "vote_average", "similarity_score"]],
        use_container_width=True,
        hide_index=True,
    )


elif page == "Taste Profile":
    st.subheader(f"{profile['name']}'s taste profile")

    user_ratings = ratings_df[ratings_df["userId"] == profile["user_id"]]
    rated_movies = movies_df[movies_df["movieId"].isin(user_ratings["movieId"])].copy()
    rated_movies = rated_movies.merge(user_ratings[["movieId", "rating"]], on="movieId")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Movies watched", f"{len(user_ratings):,}")
    metric_cols[1].metric("Average rating", f"{user_ratings['rating'].mean():.2f}/5")
    metric_cols[2].metric("Top rating", f"{user_ratings['rating'].max():.1f}/5")
    metric_cols[3].metric("Genres touched", len({g for values in rated_movies["genres"].apply(parse_list) for g in values}))

    chart_cols = st.columns(2)

    with chart_cols[0]:
        rated_movies["genres_list"] = rated_movies["genres"].apply(parse_list)
        genre_rows = []
        for _, rated in rated_movies.iterrows():
            for genre in rated["genres_list"]:
                genre_rows.append({"genre": genre, "rating": rated["rating"]})
        genre_df = pd.DataFrame(genre_rows)

        if not genre_df.empty:
            genre_avg = genre_df.groupby("genre")["rating"].mean().reset_index()
            genre_avg = genre_avg.nlargest(10, "rating")
            fig = go.Figure(
                go.Scatterpolar(
                    r=genre_avg["rating"],
                    theta=genre_avg["genre"],
                    fill="toself",
                    line_color="#2e90fa",
                )
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 5])),
                title="Genre affinity",
                height=420,
                margin=dict(l=20, r=20, t=55, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        rated_movies["decade"] = (rated_movies["release_year"] // 10 * 10).astype("Int64")
        decade_df = rated_movies.groupby("decade")["rating"].agg(["mean", "count"]).reset_index()
        decade_df.columns = ["decade", "avg_rating", "count"]
        decade_df = decade_df.dropna()

        fig = px.bar(
            decade_df,
            x="decade",
            y="avg_rating",
            color="count",
            color_continuous_scale="Teal",
            title="Decade preferences",
            labels={"avg_rating": "Average rating", "decade": "Decade", "count": "Watched"},
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest rated movies")
    top_movies = rated_movies.sort_values("rating", ascending=False).head(8)
    st.dataframe(
        top_movies[["title", "release_year", "director", "rating", "vote_average"]],
        use_container_width=True,
        hide_index=True,
    )
