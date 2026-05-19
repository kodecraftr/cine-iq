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
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explainability"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
USER_REVIEWS_CSV = PROCESSED / "user_reviews.csv"
SENTIMENT_SCORES_CSV = PROCESSED / "sentiment_scores.csv"


st.set_page_config(
    page_title="Cine IQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #667085;
        --paper: #fffaf0;
        --surface: #ffffff;
        --line: #d9ded8;
        --teal: #2d6a6a;
        --teal-dark: #1f4f4f;
        --gold: #c58a2c;
        --coral: #b85c4b;
        --blue: #315d86;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(45, 106, 106, 0.10), rgba(247, 243, 234, 0) 260px),
            var(--paper);
        color: var(--ink);
    }
    section[data-testid="stSidebar"] {
        background: #18242f;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #b7c3cf !important;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .app-title {
        color: var(--ink);
        font-size: 2.4rem;
        font-weight: 850;
        letter-spacing: 0;
        margin-bottom: 0.1rem;
    }
    .muted { color: var(--muted); }
    .hero {
        border: 1px solid rgba(45, 106, 106, 0.18);
        border-radius: 10px;
        background: linear-gradient(135deg, #ffffff 0%, #f1f7f4 58%, #fff3dc 100%);
        padding: 1.2rem 1.35rem;
        margin: 0.8rem 0 1.2rem 0;
        box-shadow: 0 16px 36px rgba(23, 33, 43, 0.08);
    }
    .hero-eyebrow {
        color: var(--teal-dark);
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.08rem;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-copy {
        color: #425466;
        font-size: 0.98rem;
        margin: 0;
    }
    .stat-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin: 0.95rem 0 0.2rem 0;
    }
    .stat-card {
        border: 1px solid rgba(45, 106, 106, 0.16);
        border-radius: 8px;
        background: rgba(255,255,255,0.82);
        padding: 0.75rem 0.85rem;
    }
    .stat-label {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
    }
    .stat-value {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 0.1rem;
    }
    h2, h3 {
        color: var(--ink) !important;
        letter-spacing: 0;
    }
    h3 {
        margin-top: 1.1rem !important;
    }
    .pill {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: #edf4f1;
        color: var(--teal-dark);
        font-size: 0.78rem;
        font-weight: 650;
        margin: 0.12rem 0.12rem 0.12rem 0;
    }
    .movie-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.05rem;
        background: var(--surface);
        min-height: 250px;
        box-shadow: 0 12px 28px rgba(23, 33, 43, 0.07);
        border-top: 4px solid var(--teal);
    }
    .card-grid .movie-card:nth-child(3n+2) { border-top-color: var(--gold); }
    .card-grid .movie-card:nth-child(3n+3) { border-top-color: var(--blue); }
    .movie-title {
        color: var(--ink);
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .movie-meta {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.7rem;
    }
    .score-line {
        font-size: 0.92rem;
        color: #344054;
        margin-top: 0.38rem;
    }
    .reason-box {
        border-left: 3px solid var(--teal);
        padding-left: 0.7rem;
        color: #344054;
        font-size: 0.92rem;
        margin-top: 0.75rem;
        line-height: 1.5;
    }
    .watch-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem;
        background: #ffffff;
        min-height: 120px;
        box-shadow: 0 8px 22px rgba(23, 33, 43, 0.05);
    }
    .watch-card .movie-title {
        color: var(--ink);
    }
    .section-label {
        color: var(--coral);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 0.75rem;
        margin: 0.35rem 0 1.15rem 0;
    }
    .stDataFrame, div[data-testid="stPlotlyChart"] {
        background: #ffffff;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading recommendation models...")
def load_models():
    from ensemble import HybridEnsemble
    from sentiment import SentimentReRanker
    from sentiment import ReviewSentimentClassifier
    from content_based import ContentBasedFilter
    from explainer import Explainer

    try:
        review_classifier = ReviewSentimentClassifier.load()
    except FileNotFoundError:
        review_classifier = None

    return {
        "ensemble": HybridEnsemble(alpha=0.6, beta=0.4),
        "sentiment": SentimentReRanker(backend="vader", gamma=0.15),
        "review_classifier": review_classifier,
        "cb": ContentBasedFilter.load(),
        "explainer": Explainer(use_lime=False),
    }


@st.cache_data(show_spinner=False)
def load_data():
    processed = Path(__file__).resolve().parents[2] / "data" / "processed"
    movies = pd.read_csv(processed / "movies_merged.csv")
    ratings = pd.read_csv(processed / "ratings_clean.csv")
    ratings = ratings[ratings["movieId"].isin(set(movies["movieId"].astype(int)))].copy()
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
        user_ratings.merge(movies, on="movieId", how="inner")
        .sort_values("rating", ascending=False)
        .head(limit)
    )
    return watchlist


def movie_tags(row: pd.Series, limit: int = 3) -> str:
    genres = parse_list(row.get("genres", []))[:limit]
    return "".join(f'<span class="pill">{genre}</span>' for genre in genres)


def load_user_reviews() -> pd.DataFrame:
    columns = [
        "timestamp",
        "movieId",
        "title",
        "review",
        "sentiment_label",
        "sentiment_score",
        "confidence",
    ]
    if USER_REVIEWS_CSV.exists():
        return pd.read_csv(USER_REVIEWS_CSV)
    return pd.DataFrame(columns=columns)


def save_user_review(movie: pd.Series, review_text: str, prediction: dict) -> None:
    USER_REVIEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    sentiment_score = (
        prediction["positive_probability"]
        if prediction["label"] == "positive"
        else -prediction["negative_probability"]
    )
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "movieId": int(movie["movieId"]),
                "title": movie.get("title", "Unknown"),
                "review": review_text,
                "sentiment_label": prediction["label"],
                "sentiment_score": sentiment_score,
                "confidence": prediction["confidence"],
            }
        ]
    )
    existing = load_user_reviews()
    pd.concat([existing, row], ignore_index=True).to_csv(USER_REVIEWS_CSV, index=False)
    update_sentiment_scores()


def update_sentiment_scores() -> pd.DataFrame:
    reviews = load_user_reviews()
    if reviews.empty:
        if SENTIMENT_SCORES_CSV.exists():
            SENTIMENT_SCORES_CSV.unlink()
        return pd.DataFrame(columns=["movieId", "sentiment_score", "review_count"])

    scores = (
        reviews.groupby("movieId", as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), review_count=("review", "count"))
    )
    scores.to_csv(SENTIMENT_SCORES_CSV, index=False)
    return scores


def refresh_sentiment_reranker(models: dict, gamma: float) -> None:
    from sentiment import SentimentReRanker

    models["sentiment"] = SentimentReRanker(backend="vader", gamma=gamma)


def render_watch_shelf(watchlist: pd.DataFrame):
    cards = ['<div class="card-grid">']
    for _, row in watchlist.iterrows():
        year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "Unknown"
        rating = row.get("rating", 0)
        cards.append(
            f'<div class="watch-card">'
            f'<div class="movie-title">{row.get("title", "Unknown")}</div>'
            f'<div class="movie-meta">{year} &middot; rated {rating:.1f}/5</div>'
            f'{movie_tags(row)}'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def recommendation_cards(recs: pd.DataFrame):
    cards = ['<div class="card-grid">']
    for idx, (_, row) in enumerate(recs.iterrows()):
        year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "Unknown"
        vote = row.get("vote_average", 0)
        score = row.get("final_score", row.get("hybrid_score", 0))
        sentiment = row.get("sentiment_score", 0)
        sentiment_text = (
            f"<b>{sentiment:+.3f}</b>"
            if bool(row.get("sentiment_available", False))
            else "<b>not linked</b>"
        )
        explanation = row.get("explanation", "Recommended from this profile's watch history.")
        cards.append(
            f'<div class="movie-card">'
            f'<div class="section-label">Pick #{int(row.get("rank", idx + 1))}</div>'
            f'<div class="movie-title">{row.get("title", "Unknown")}</div>'
            f'<div class="movie-meta">{year} &middot; audience score {vote}</div>'
            f'<div class="score-line">Match score: <b>{score:.3f}</b></div>'
            f'<div class="score-line">Audience signal: {sentiment_text}</div>'
            f'<div class="reason-box">{explanation}</div>'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def generate_recommendations(models: dict, profile: dict, n_recs: int, alpha: float, gamma: float) -> pd.DataFrame:
    ensemble = models["ensemble"]
    ensemble.alpha = alpha
    ensemble.beta = round(1 - alpha, 4)

    recs = ensemble.recommend(user_id=profile["user_id"], n=n_recs * 3)
    reranker = models["sentiment"]
    reranker.gamma = gamma
    recs = reranker.rerank(recs).head(n_recs)
    recs["sentiment_available"] = reranker._scores is not None
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
    ["For You", "Discover Similar", "Review a Movie", "Taste Profile"],
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

profile_watch_count = len(ratings_df[ratings_df["userId"] == profile["user_id"]])
profile_avg = ratings_df[ratings_df["userId"] == profile["user_id"]]["rating"].mean()
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">Personalized discovery</div>
        <div class="hero-title">A watchlist-first recommender for {profile["name"]}</div>
        <p class="hero-copy">Cine IQ blends rating patterns, movie metadata, and audience sentiment to recommend what should feel natural to watch next.</p>
        <div class="stat-strip">
            <div class="stat-card">
                <div class="stat-label">Profile</div>
                <div class="stat-value">{profile["name"]}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Watched titles</div>
                <div class="stat-value">{profile_watch_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average rating</div>
                <div class="stat-value">{profile_avg:.2f}/5</div>
            </div>
        </div>
    </div>
    """,
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


elif page == "Review a Movie":
    st.subheader("Leave a movie review")
    st.caption("Cine IQ classifies the review and folds it into that movie's average audience sentiment.")

    all_titles = movies_df["title"].dropna().sort_values().unique().tolist()
    selected_title = st.selectbox("Movie", all_titles)
    selected_movie = movies_df[movies_df["title"] == selected_title].iloc[0]

    existing_reviews = load_user_reviews()
    movie_reviews = existing_reviews[existing_reviews["movieId"] == int(selected_movie["movieId"])]

    if not movie_reviews.empty:
        avg_sentiment = movie_reviews["sentiment_score"].mean()
        metric_cols = st.columns(3)
        metric_cols[0].metric("Submitted reviews", len(movie_reviews))
        metric_cols[1].metric("Average sentiment", f"{avg_sentiment:+.3f}")
        metric_cols[2].metric("Latest label", str(movie_reviews.iloc[-1]["sentiment_label"]).title())
    else:
        st.info("No user reviews have been submitted for this movie yet.")

    review_text = st.text_area(
        "Your review",
        height=180,
        placeholder="Write a real audience review for this selected movie...",
    )

    classifier = models.get("review_classifier")
    if classifier is None:
        st.warning("Review sentiment model is not trained yet. Run: python src/models/sentiment.py --train-review-classifier")
    elif st.button("Submit review", type="primary", disabled=not review_text.strip()):
        result = classifier.predict(review_text)
        save_user_review(selected_movie, review_text.strip(), result)
        refresh_sentiment_reranker(models, gamma)

        label = result["label"].title()
        confidence = result["confidence"]
        positive = result["positive_probability"]
        negative = result["negative_probability"]
        signed_score = positive if result["label"] == "positive" else -negative

        metric_cols = st.columns(3)
        metric_cols[0].metric("Review sentiment", label)
        metric_cols[1].metric("Confidence", f"{confidence:.1%}")
        metric_cols[2].metric("Stored score", f"{signed_score:+.3f}")
        st.success("Review saved. Recommendations now use the updated per-movie average sentiment.")

        fig = px.bar(
            pd.DataFrame(
                [
                    {"sentiment": "Positive", "probability": positive},
                    {"sentiment": "Negative", "probability": negative},
                ]
            ),
            x="sentiment",
            y="probability",
            color="sentiment",
            color_discrete_map={"Positive": "#2d6a6a", "Negative": "#b85c4b"},
            title="Sentiment probabilities",
            labels={"probability": "Probability", "sentiment": ""},
        )
        fig.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=55, b=20))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    updated_reviews = load_user_reviews()
    if not updated_reviews.empty:
        st.subheader("Recent user reviews powering re-ranking")
        recent = updated_reviews.sort_values("timestamp", ascending=False).head(10)
        st.dataframe(
            recent[["timestamp", "title", "sentiment_label", "sentiment_score", "confidence"]],
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
