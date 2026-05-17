"""
Phase 5 — Sentiment-Aware Re-Ranker
=====================================
Loads pre-computed per-movie sentiment scores (from Phase 1) and uses them
to re-rank the hybrid ensemble output.

  final_score = hybrid_score * (1 + γ * sentiment_score)

where sentiment_score ∈ [-1, 1] and γ controls how strongly sentiment
nudges the ranking (default γ = 0.15).

Two sentiment backends are supported:
  - VADER   (fast, CPU, rule-based NLP — good default)
  - DistilBERT (slower, GPU-optional, more accurate on nuanced reviews)

Usage
-----
  from src.models.sentiment import SentimentReRanker
  reranker = SentimentReRanker(backend="vader")
  reranker.rerank(recommendations_df)

  # or compute fresh scores from raw reviews:
  reranker.score_reviews(reviews_df)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED     = ROOT / "data" / "processed"
SENTIMENT_CSV = PROCESSED / "sentiment_scores.csv"


# ── VADER backend ──────────────────────────────────────────────────────────────

def _score_with_vader(reviews: pd.DataFrame, text_col: str = "review") -> pd.Series:
    """
    Score a DataFrame of reviews with VADER.

    Returns
    -------
    pd.Series of compound scores in [-1, 1], index aligned with input DataFrame.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    return reviews[text_col].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])


# ── DistilBERT backend ─────────────────────────────────────────────────────────

def _score_with_distilbert(
    reviews   : pd.DataFrame,
    text_col  : str = "review",
    batch_size: int = 32,
) -> pd.Series:
    """
    Score reviews with HuggingFace DistilBERT sentiment pipeline.
    Maps POSITIVE → +score, NEGATIVE → -score.

    Returns
    -------
    pd.Series of signed confidence scores in [-1, 1].
    """
    from transformers import pipeline
    log.info("Loading DistilBERT sentiment pipeline …")
    pipe = pipeline(
        "sentiment-analysis",
        model     = "distilbert-base-uncased-finetuned-sst-2-english",
        truncation= True,
        max_length= 512,
    )

    texts  = reviews[text_col].fillna("").tolist()
    scores = []
    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        results = pipe(batch)
        for r in results:
            sign = 1 if r["label"] == "POSITIVE" else -1
            scores.append(sign * r["score"])
        if i % (batch_size * 10) == 0:
            log.info(f"  Scored {i + len(batch)}/{len(texts)} reviews …")

    return pd.Series(scores, index=reviews.index)


# ── Main class ─────────────────────────────────────────────────────────────────

class SentimentReRanker:
    """
    Re-ranks a recommendations DataFrame using per-movie sentiment scores.

    Parameters
    ----------
    backend : "vader" | "distilbert"
    gamma   : float  — how strongly sentiment nudges the final score (default 0.15)
    """

    def __init__(self, backend: str = "vader", gamma: float = 0.15):
        if backend not in ("vader", "distilbert"):
            raise ValueError("backend must be 'vader' or 'distilbert'")
        self.backend  = backend
        self.gamma    = gamma
        self._scores  = self._load_precomputed()

    # ── loading ────────────────────────────────────────────────────────────────

    def _load_precomputed(self) -> pd.Series | None:
        """
        Load sentiment_scores.csv produced in Phase 1.
        Returns a Series indexed by movieId (or title if no movieId available).
        """
        if not SENTIMENT_CSV.exists():
            log.warning("sentiment_scores.csv not found — sentiment re-ranking will be skipped.")
            return None

        df = pd.read_csv(SENTIMENT_CSV)
        log.info(f"Loaded {len(df):,} pre-computed sentiment scores.")

        if "movieId" in df.columns:
            return df.set_index("movieId")["sentiment_score"]
        elif "title" in df.columns:
            return df.set_index("title")["sentiment_score"]
        else:
            log.warning("sentiment_scores.csv has no 'movieId' or 'title' column.")
            return None

    # ── compute fresh scores from raw reviews ─────────────────────────────────

    def score_reviews(
        self,
        reviews   : pd.DataFrame,
        text_col  : str  = "review",
        id_col    : str  = "movieId",
    ) -> pd.Series:
        """
        Compute and aggregate sentiment scores from a raw reviews DataFrame.

        Parameters
        ----------
        reviews  : DataFrame with at least text_col and id_col columns
        text_col : column containing review text
        id_col   : column to group by (movieId or title)

        Returns
        -------
        pd.Series  index=id_col values, values=mean sentiment score in [-1, 1]
        """
        log.info(f"Scoring {len(reviews):,} reviews with {self.backend} …")

        if self.backend == "vader":
            reviews = reviews.copy()
            reviews["_score"] = _score_with_vader(reviews, text_col)
        else:
            reviews = reviews.copy()
            reviews["_score"] = _score_with_distilbert(reviews, text_col)

        agg = reviews.groupby(id_col)["_score"].mean()
        self._scores = agg
        log.info(f"Computed sentiment scores for {len(agg):,} movies.")
        return agg

    # ── re-ranking ────────────────────────────────────────────────────────────

    def rerank(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Re-rank a recommendations DataFrame using sentiment signal.

        Expects `recommendations` to have columns: movieId, hybrid_score
        Returns the same DataFrame with two new columns:
          - sentiment_score  : per-movie sentiment in [-1, 1]  (0.0 if unknown)
          - final_score      : hybrid_score * (1 + γ * sentiment_score)

        Rows are sorted by final_score descending and rank is updated.
        """
        recs = recommendations.copy()

        if self._scores is None:
            log.warning("No sentiment scores available — returning original ranking.")
            recs["sentiment_score"] = 0.0
            recs["final_score"]     = recs["hybrid_score"]
            return recs

        # map sentiment scores; default to 0.0 (neutral) for unseen movies
        if "movieId" in recs.columns and self._scores.index.dtype in (int, float, "int64"):
            recs["sentiment_score"] = recs["movieId"].map(self._scores).fillna(0.0)
        elif "title" in recs.columns:
            recs["sentiment_score"] = recs["title"].map(self._scores).fillna(0.0)
        else:
            recs["sentiment_score"] = 0.0

        # apply sentiment nudge
        recs["final_score"] = recs["hybrid_score"] * (1 + self.gamma * recs["sentiment_score"])

        # re-sort and update rank
        recs = recs.sort_values("final_score", ascending=False).reset_index(drop=True)
        recs["rank"] = range(1, len(recs) + 1)

        return recs

    # ── convenience: full pipeline ────────────────────────────────────────────

    @classmethod
    def from_raw_reviews(
        cls,
        reviews_path : str | Path,
        text_col     : str  = "review",
        id_col       : str  = "movieId",
        backend      : str  = "vader",
        gamma        : float = 0.15,
    ) -> "SentimentReRanker":
        """
        Build a SentimentReRanker by scoring a raw CSV of reviews from scratch.
        Useful when the Phase 1 pipeline didn't produce sentiment_scores.csv.
        """
        instance = cls(backend=backend, gamma=gamma)
        reviews  = pd.read_csv(reviews_path)
        instance.score_reviews(reviews, text_col=text_col, id_col=id_col)
        return instance


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse, json
    parser = argparse.ArgumentParser(description="Phase 5 — Sentiment Re-Ranker")
    parser.add_argument("--backend",      default="vader", choices=["vader", "distilbert"])
    parser.add_argument("--gamma",        type=float, default=0.15)
    parser.add_argument("--reviews-csv",  default=None, help="Path to raw reviews CSV for fresh scoring")
    args = parser.parse_args()

    reranker = SentimentReRanker(backend=args.backend, gamma=args.gamma)

    if args.reviews_csv:
        reviews = pd.read_csv(args.reviews_csv)
        scores  = reranker.score_reviews(reviews)
        print(scores.head(20))
    else:
        # smoke test: create a dummy recommendations df
        dummy_recs = pd.DataFrame({
            "rank"         : [1, 2, 3, 4, 5],
            "movieId"      : [278, 238, 424, 389, 807],
            "title"        : ["Shawshank", "Godfather", "Schindler", "12 Angry", "Se7en"],
            "hybrid_score" : [0.92, 0.88, 0.85, 0.80, 0.78],
        })
        print("Before re-ranking:")
        print(dummy_recs.to_string(index=False))
        reranked = reranker.rerank(dummy_recs)
        print("\nAfter sentiment re-ranking:")
        print(reranked.to_string(index=False))


if __name__ == "__main__":
    main()
