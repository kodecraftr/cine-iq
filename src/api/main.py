"""
Phase 7a — FastAPI Serving Layer
==================================
Exposes two endpoints:

  GET /recommend?user_id=42&n=10&gamma=0.15&alpha=0.6
      → top-N personalised recommendations with explanations

  GET /similar?movie_id=238&n=10
      → content-similar movies for a given movie

Run
---
  uvicorn src.api.main:app --reload --port 8000

  # then in browser / curl:
  curl "http://localhost:8000/recommend?user_id=42&n=5"
  curl "http://localhost:8000/similar?movie_id=238&n=5"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explainability"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from ensemble    import HybridEnsemble
from sentiment   import SentimentReRanker
from content_based import ContentBasedFilter
from explainer   import Explainer

app = FastAPI(
    title       = "Movie Recommender API",
    description = "Hybrid recommendation engine with sentiment re-ranking and explainability",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── lazy-load models once at startup ──────────────────────────────────────────

@app.on_event("startup")
def load_models():
    app.state.ensemble  = HybridEnsemble(alpha=0.6, beta=0.4)
    app.state.sentiment = SentimentReRanker(backend="vader", gamma=0.15)
    app.state.cb        = ContentBasedFilter.load()
    app.state.explainer = Explainer(use_lime=True)


# ── response schemas ──────────────────────────────────────────────────────────

class MovieRec(BaseModel):
    rank            : int
    movieId         : int
    title           : str | None
    release_year    : float | None
    vote_average    : float | None
    collab_score    : float
    content_score   : float
    hybrid_score    : float
    sentiment_score : float
    final_score     : float
    explanation     : str


class SimilarMovie(BaseModel):
    movieId          : int
    title            : str | None
    release_year     : float | None
    vote_average     : float | None
    similarity_score : float


# ── /recommend ────────────────────────────────────────────────────────────────

@app.get("/recommend", response_model=list[MovieRec])
def recommend(
    user_id : int   = Query(..., description="MovieLens userId"),
    n       : int   = Query(10,  description="Number of recommendations", ge=1, le=50),
    alpha   : float = Query(0.6, description="Collaborative weight (0–1)"),
    gamma   : float = Query(0.15,description="Sentiment nudge strength"),
):
    """
    Return top-N personalised movie recommendations for a user.
    Pipeline: SVD → Hybrid Ensemble → Sentiment Re-rank → Explain
    """
    try:
        ens = app.state.ensemble
        ens.alpha = alpha
        ens.beta  = round(1.0 - alpha, 4)

        # 1. hybrid recommendations
        recs = ens.recommend(user_id=user_id, n=n * 3)   # over-fetch for re-ranking headroom
        if recs.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}")

        # 2. sentiment re-ranking
        reranker = app.state.sentiment
        reranker.gamma = gamma
        recs = reranker.rerank(recs)
        recs = recs.head(n).copy()

        # 3. explainability
        recs = app.state.explainer.explain_batch(user_id=user_id, recommendations=recs)

        # 4. fill missing columns with defaults
        for col in ["sentiment_score", "final_score", "explanation"]:
            if col not in recs.columns:
                recs[col] = 0.0 if col != "explanation" else ""

        recs["rank"] = range(1, len(recs) + 1)

        return recs.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /similar ──────────────────────────────────────────────────────────────────

@app.get("/similar", response_model=list[SimilarMovie])
def similar(
    movie_id : int = Query(..., description="TMDB movieId"),
    n        : int = Query(10,  description="Number of similar movies", ge=1, le=50),
):
    """
    Return top-N content-similar movies for a given movie.
    Uses TF-IDF cosine similarity on the soup column.
    """
    try:
        result = app.state.cb.similar_movies(movie_id=movie_id, n=n)
        if result.empty:
            raise HTTPException(status_code=404, detail=f"movieId {movie_id} not found")
        return result.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}
