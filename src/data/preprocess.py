"""
Phase 1 — Data Pipeline
========================
Merges MovieLens 25M + TMDB Metadata + IMDB Reviews into clean,
model-ready DataFrames.

Expected raw data layout
------------------------
data/raw/
  movielens/
    ratings.csv      — userId, movieId, rating, timestamp
    movies.csv       — movieId, title, genres
    links.csv        — movieId, imdbId, tmdbId
  tmdb/
    tmdb_5000_movies.csv   — id, title, genres, cast, crew, keywords, overview ...
    tmdb_5000_credits.csv  — movie_id, cast, crew
  imdb/
    IMDB Dataset.csv       — review, sentiment  (Kaggle 50 K dataset)

Outputs
-------
data/processed/movies_merged.csv   — one row per movie, rich metadata + soup
data/processed/ratings_clean.csv   — userId, movieId, rating (filtered)
data/processed/sentiment_scores.csv — movieId, sentiment_score
"""

import ast
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # project root
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_literal(val):
    """Parse a JSON-like string to a Python object; return [] on failure."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def _extract_names(obj, key: str = "name", limit: int | None = None) -> list[str]:
    """Extract `key` from a list of dicts (parsed from TMDB JSON columns)."""
    if not isinstance(obj, list):
        obj = _safe_literal(obj)
    names = [item[key] for item in obj if key in item]
    return names[:limit] if limit else names


def _extract_director(crew_obj) -> str:
    """Return the director's name from the TMDB crew list."""
    crew = crew_obj if isinstance(crew_obj, list) else _safe_literal(crew_obj)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def _clean_text(text: str) -> str:
    """Lowercase and remove spaces so 'Tom Hanks' → 'tomhanks' (avoids TF-IDF splitting)."""
    return str(text).lower().replace(" ", "")


# ── Step 1 : Load & clean MovieLens ───────────────────────────────────────────

def load_movielens(min_ratings_per_user: int = 20, min_ratings_per_movie: int = 10):
    """
    Load MovieLens ratings and movies; filter sparse users and movies.

    Returns
    -------
    ratings : pd.DataFrame  [userId, movieId, rating]
    ml_movies : pd.DataFrame  [movieId, title, ml_genres]
    """
    log.info("Loading MovieLens …")
    ratings_path = RAW / "movielens" / "ratings.csv"
    movies_path  = RAW / "movielens" / "movies.csv"
    links_path   = RAW / "movielens" / "links.csv"

    ratings  = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    ml_movies = pd.read_csv(movies_path)
    links    = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])

    # drop rows with missing tmdbId (needed to join TMDB)
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    # filter low-activity users and rarely-rated movies
    user_counts  = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()
    ratings = ratings[
        ratings["userId"].isin(user_counts[user_counts >= min_ratings_per_user].index) &
        ratings["movieId"].isin(movie_counts[movie_counts >= min_ratings_per_movie].index)
    ].copy()

    ml_movies = ml_movies.merge(links, on="movieId", how="left")
    ml_movies.rename(columns={"genres": "ml_genres"}, inplace=True)

    log.info(f"  ratings  : {len(ratings):,} rows | {ratings['userId'].nunique():,} users | {ratings['movieId'].nunique():,} movies")
    log.info(f"  ml_movies: {len(ml_movies):,} rows")
    return ratings, ml_movies


# ── Step 2 : Load & parse TMDB ────────────────────────────────────────────────

def load_tmdb():
    """
    Load TMDB movies + credits; parse JSON columns into flat Python lists.

    Returns
    -------
    tmdb : pd.DataFrame  [id, title, genres, cast, director, keywords, overview, vote_average, release_year]
    """
    log.info("Loading TMDB …")
    movies_path  = RAW / "tmdb" / "tmdb_5000_movies.csv"
    credits_path = RAW / "tmdb" / "tmdb_5000_credits.csv"

    tmdb     = pd.read_csv(movies_path)
    credits  = pd.read_csv(credits_path)

    # credits uses 'movie_id' in some Kaggle versions; normalise
    if "movie_id" in credits.columns:
        credits.rename(columns={"movie_id": "id"}, inplace=True)

    tmdb = tmdb.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    # parse JSON strings → Python lists / strings
    tmdb["genres"]   = tmdb["genres"].apply(lambda x: _extract_names(x))
    tmdb["keywords"] = tmdb["keywords"].apply(lambda x: _extract_names(x))
    tmdb["cast"]     = tmdb["cast"].apply(lambda x: _extract_names(x, limit=5))   # top-5 cast
    tmdb["director"] = tmdb["crew"].apply(_extract_director)

    # keep useful columns only
    tmdb["release_year"] = pd.to_datetime(tmdb["release_date"], errors="coerce").dt.year
    tmdb = tmdb[[
        "id", "title", "genres", "cast", "director",
        "keywords", "overview", "vote_average", "release_year"
    ]].copy()

    tmdb.dropna(subset=["overview"], inplace=True)
    tmdb["id"] = tmdb["id"].astype(int)

    log.info(f"  tmdb: {len(tmdb):,} movies")
    return tmdb


# ── Step 3 : Merge MovieLens ↔ TMDB ──────────────────────────────────────────

def merge_movielens_tmdb(ml_movies: pd.DataFrame, tmdb: pd.DataFrame) -> pd.DataFrame:
    """
    Join on tmdbId == tmdb.id.

    Returns
    -------
    merged : pd.DataFrame  — one row per movie with both ML and TMDB fields
    """
    log.info("Merging MovieLens ↔ TMDB …")
    merged = ml_movies.merge(
        tmdb,
        left_on="tmdbId",
        right_on="id",
        how="inner"
    )
    # prefer TMDB title (richer); fall back to ML title
    merged["title"] = merged["title_y"].fillna(merged["title_x"])
    merged.drop(columns=["title_x", "title_y", "id"], inplace=True)

    log.info(f"  merged: {len(merged):,} movies with full metadata")
    return merged


# ── Step 4 : Build the "soup" for content-based filtering ────────────────────

def build_soup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate genres + keywords + cast + director + overview into a single
    text column used by TF-IDF in Phase 3.

    Proper nouns are squashed (no spaces) so 'Tom Hanks' is treated as one token.
    """
    log.info("Building content soup …")

    def _row_soup(row) -> str:
        genres   = " ".join(_clean_text(g) for g in row["genres"])
        keywords = " ".join(_clean_text(k) for k in row["keywords"])
        cast     = " ".join(_clean_text(c) for c in row["cast"])
        director = _clean_text(row["director"]) + " " + _clean_text(row["director"])  # weight ×2
        overview = str(row["overview"]).lower()
        return f"{genres} {keywords} {cast} {director} {overview}"

    df["soup"] = df.apply(_row_soup, axis=1)
    return df


# ── Step 5 : IMDB sentiment scores ───────────────────────────────────────────

def compute_sentiment_scores() -> pd.DataFrame | None:
    """
    Aggregate IMDB review sentiments per movie into a single float in [-1, 1].

    The raw IMDB 50K dataset has columns: review, sentiment ('positive'/'negative').
    We map positive → +1, negative → -1, then average per movie title.

    NOTE: The Kaggle IMDB dataset does NOT contain movie IDs — it only has free-text
    reviews.  A best-effort fuzzy title match is attempted at the end of this function
    to link rows to movieId.  For a production system, replace with a dataset that
    includes IMDb tt-IDs and join via MovieLens links.csv.

    Returns None if the IMDB CSV is not found (sentiment re-ranker will be skipped).
    """
    imdb_path = RAW / "imdb" / "IMDB Dataset.csv"
    if not imdb_path.exists():
        log.warning("IMDB dataset not found — skipping sentiment scores.")
        return None

    log.info("Computing IMDB sentiment scores …")
    imdb = pd.read_csv(imdb_path)

    # normalise column names (Kaggle spellings vary)
    imdb.columns = [c.lower().strip() for c in imdb.columns]
    if "sentiment" not in imdb.columns:
        log.warning("  'sentiment' column missing — skipping.")
        return None

    imdb["score"] = imdb["sentiment"].map({"positive": 1, "negative": -1})
    imdb = imdb.dropna(subset=["score"])

    # aggregate: mean sentiment score per movie
    # (movie title column varies by dataset version)
    title_col = next((c for c in imdb.columns if "title" in c), None)
    if title_col:
        sentiment_scores = (
            imdb.groupby(title_col)["score"]
            .mean()
            .reset_index()
            .rename(columns={title_col: "title", "score": "sentiment_score"})
        )
    else:
        # no title column — return overall score as a placeholder
        log.warning("  No title column in IMDB data; cannot link to movies.")
        return None

    log.info(f"  sentiment_scores: {len(sentiment_scores):,} movies")
    return sentiment_scores


# ── Step 6 : Save outputs ─────────────────────────────────────────────────────

def save_outputs(
    merged: pd.DataFrame,
    ratings: pd.DataFrame,
    sentiment: pd.DataFrame | None,
) -> None:
    movies_out    = PROCESSED / "movies_merged.csv"
    ratings_out   = PROCESSED / "ratings_clean.csv"
    sentiment_out = PROCESSED / "sentiment_scores.csv"

    merged.to_csv(movies_out, index=False)
    log.info(f"Saved → {movies_out}")

    ratings.to_csv(ratings_out, index=False)
    log.info(f"Saved → {ratings_out}")

    if sentiment is not None:
        sentiment.to_csv(sentiment_out, index=False)
        log.info(f"Saved → {sentiment_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    min_ratings_per_user: int = 20,
    min_ratings_per_movie: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full Phase 1 pipeline.

    Returns
    -------
    movies_merged : pd.DataFrame
    ratings_clean : pd.DataFrame
    """
    ratings, ml_movies = load_movielens(min_ratings_per_user, min_ratings_per_movie)
    tmdb               = load_tmdb()
    merged             = merge_movielens_tmdb(ml_movies, tmdb)
    merged             = build_soup(merged)
    sentiment          = compute_sentiment_scores()
    save_outputs(merged, ratings, sentiment)

    log.info("✅ Phase 1 complete.")
    return merged, ratings


if __name__ == "__main__":
    run_pipeline()
