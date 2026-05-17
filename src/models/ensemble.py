"""
Phase 4 — Weighted Hybrid Ensemble
====================================
Blends collaborative (SVD) + content-based (TF-IDF cosine) scores into a
single ranked recommendation list.

  score = α * collab_score + β * content_score   (α + β = 1)

Default: α=0.6, β=0.4  (collaborative slightly favoured; tune via MLflow)

Exposes
-------
  hybrid_recommend(user_id, n, liked_movie_ids)  →  pd.DataFrame
  score_candidates(user_id, candidate_ids)       →  pd.DataFrame

Usage
-----
  python src/models/ensemble.py --user-id 42 --n 10

  from src.models.ensemble import HybridEnsemble
  ens = HybridEnsemble()
  ens.recommend(user_id=42, n=10)
"""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from collaborative  import CollaborativeFilter
from content_based  import ContentBasedFilter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RATINGS_PATH = PROCESSED / "ratings_clean.csv"


class HybridEnsemble:
    """
    Weighted ensemble of collaborative + content-based scores.

    Parameters
    ----------
    alpha : float  weight for collaborative score  (default 0.6)
    beta  : float  weight for content-based score  (default 0.4)
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        if not np.isclose(alpha + beta, 1.0):
            raise ValueError(f"alpha + beta must equal 1.0, got {alpha + beta}")
        self.alpha = alpha
        self.beta  = beta

        log.info("Loading sub-models …")
        self.cf = CollaborativeFilter.load()
        self.cb = ContentBasedFilter.load()
        self._ratings = pd.read_csv(RATINGS_PATH)
        log.info(f"Ensemble ready  (α={alpha}, β={beta})")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _user_liked_movies(self, user_id: int, rating_threshold: float = 3.5) -> list[int]:
        """Return movieIds the user rated ≥ threshold (their taste profile)."""
        user_df = self._ratings[self._ratings["userId"] == user_id]
        liked   = user_df[user_df["rating"] >= rating_threshold]["movieId"].tolist()
        return liked

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        """Normalise a series to [0, 1]; handle zero-range edge case."""
        mn, mx = series.min(), series.max()
        if np.isclose(mn, mx):
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    # ── scoring ────────────────────────────────────────────────────────────────

    def score_candidates(
        self,
        user_id       : int,
        candidate_ids : list[int],
    ) -> pd.DataFrame:
        """
        Score a list of candidate movieIds for a given user.

        Returns
        -------
        pd.DataFrame  [movieId, collab_score, content_score, hybrid_score]  sorted desc
        """
        # ── collaborative scores ──
        collab_scores = pd.Series(
            {mid: self.cf.predict_rating(user_id, mid) for mid in candidate_ids},
            name="collab_raw"
        )

        # ── content scores (average over liked movies) ──
        liked = self._user_liked_movies(user_id)
        if liked:
            content_scores = self.cb.content_score_for_user(liked, candidate_ids)
        else:
            content_scores = pd.Series(0.0, index=candidate_ids)
        content_scores.name = "content_raw"

        df = pd.DataFrame({"collab_raw": collab_scores, "content_raw": content_scores})
        df = df.dropna()

        # ── normalise each sub-score to [0, 1] before blending ──
        df["collab_score"]  = self._minmax(df["collab_raw"])
        df["content_score"] = self._minmax(df["content_raw"])

        # ── weighted blend ──
        df["hybrid_score"] = self.alpha * df["collab_score"] + self.beta * df["content_score"]
        df.index.name = "movieId"
        df = df.reset_index()
        df["movieId"] = df["movieId"].astype(int)

        return df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    # ── top-N recommendation ──────────────────────────────────────────────────

    def recommend(
        self,
        user_id : int,
        n       : int = 10,
    ) -> pd.DataFrame:
        """
        Generate top-N hybrid recommendations for a user.

        Strategy
        --------
        1. Get top-200 candidates from collaborative filter (fast, personalised)
        2. Score each candidate with the ensemble (adds content signal)
        3. Return top-N after hybrid re-ranking

        Returns
        -------
        pd.DataFrame  [rank, movieId, title, collab_score, content_score, hybrid_score]
        """
        log.info(f"Generating hybrid recommendations for user {user_id} …")

        # Step 1: candidate generation from SVD
        collab_candidates = self.cf.top_n(user_id=user_id, n=200)
        candidate_ids     = collab_candidates["movieId"].tolist()

        if not candidate_ids:
            log.warning("No candidates from collaborative filter (cold-start user?).")
            return pd.DataFrame()

        # Step 2: hybrid scoring
        scored = self.score_candidates(user_id, candidate_ids)

        # Step 3: top-N
        top = scored.head(n).copy()
        top["rank"] = range(1, len(top) + 1)

        # attach titles
        movie_map = self.cb.movies[["movieId", "title", "release_year", "vote_average"]].drop_duplicates("movieId")
        top = top.merge(movie_map, on="movieId", how="left")

        cols = ["rank", "movieId", "title", "release_year", "vote_average",
                "collab_score", "content_score", "hybrid_score"]
        return top[cols].reset_index(drop=True)

    # ── weight tuning via MLflow ───────────────────────────────────────────────

    def tune_weights(
        self,
        user_sample : int   = 50,
        n_candidates: int   = 100,
        alpha_values: list  = None,
    ) -> dict:
        """
        Grid search over alpha values (beta = 1 - alpha).
        Uses Precision@K as a proxy metric (how many top-K recommendations
        were actually rated ≥ 4 by the user).

        Returns best alpha/beta pair.
        """
        if alpha_values is None:
            alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        log.info("Tuning ensemble weights …")
        mlflow.set_experiment("movie-recommender-ensemble")

        # sample users with enough ratings
        user_counts = self._ratings["userId"].value_counts()
        eligible    = user_counts[user_counts >= 30].index.tolist()
        sample_users = np.random.choice(eligible, size=min(user_sample, len(eligible)), replace=False)

        best_alpha, best_precision = 0.6, -1.0

        for alpha in alpha_values:
            beta = round(1.0 - alpha, 4)
            self.alpha = alpha
            self.beta  = beta

            precisions = []
            for uid in sample_users:
                try:
                    # ground truth: movies rated ≥ 4 by this user
                    user_df    = self._ratings[self._ratings["userId"] == uid]
                    liked_set  = set(user_df[user_df["rating"] >= 4.0]["movieId"])

                    recs       = self.recommend(user_id=uid, n=10)
                    rec_ids    = set(recs["movieId"].tolist())
                    precision  = len(liked_set & rec_ids) / max(len(rec_ids), 1)
                    precisions.append(precision)
                except Exception:
                    continue

            mean_precision = float(np.mean(precisions)) if precisions else 0.0
            log.info(f"  α={alpha:.1f} β={beta:.1f} → Precision@10 = {mean_precision:.4f}")

            with mlflow.start_run(run_name=f"ensemble_alpha_{alpha}"):
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("beta",  beta)
                mlflow.log_metric("precision_at_10", mean_precision)

            if mean_precision > best_precision:
                best_precision = mean_precision
                best_alpha     = alpha

        best_beta = round(1.0 - best_alpha, 4)
        log.info(f"Best weights → α={best_alpha}, β={best_beta}  (Precision@10={best_precision:.4f})")
        self.alpha = best_alpha
        self.beta  = best_beta
        return {"alpha": best_alpha, "beta": best_beta, "precision_at_10": best_precision}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 — Hybrid Ensemble")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--n",       type=int, default=10)
    parser.add_argument("--alpha",   type=float, default=0.6)
    parser.add_argument("--tune",    action="store_true", help="Run weight tuning")
    args = parser.parse_args()

    ens = HybridEnsemble(alpha=args.alpha, beta=round(1 - args.alpha, 4))

    if args.tune:
        best = ens.tune_weights()
        print(f"\nBest weights: {best}")

    recs = ens.recommend(user_id=args.user_id, n=args.n)
    print(f"\nTop-{args.n} recommendations for user {args.user_id}:")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
