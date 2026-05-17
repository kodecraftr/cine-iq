"""
Phase 6 — Explainability Layer
================================
Surfaces a human-readable reason for every recommendation using two strategies:

  1. Rule-based templates  (fast, always available)
     "Recommended because you liked Inception, which shares genre Sci-Fi
      and director Christopher Nolan with Interstellar."

  2. LIME feature attribution  (richer, model-level explanation)
     Shows which TF-IDF terms (genres / keywords / cast) contributed most
     to the content similarity score.

Exposes
-------
  explain(user_id, movie_id, liked_movie_ids)  →  str
  explain_batch(user_id, recommendations_df)   →  pd.DataFrame  (adds 'explanation' col)

Usage
-----
  from src.explainability.explainer import Explainer
  exp = Explainer()
  exp.explain(user_id=42, movie_id=278, liked_movie_ids=[550, 680])
"""

import ast
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MOVIES_PATH  = PROCESSED / "movies_merged.csv"
RATINGS_PATH = PROCESSED / "ratings_clean.csv"


def _parse_list(val) -> list:
    """Parse a stringified list or return as-is."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


class Explainer:
    """
    Generates natural-language explanations for movie recommendations.

    Parameters
    ----------
    use_lime : bool  — whether to attempt LIME explanations (requires content model)
    """

    def __init__(self, use_lime: bool = True):
        self.movies   = self._load_movies()
        self.ratings  = self._load_ratings()
        self.use_lime = use_lime
        self._cb      = None   # lazy-loaded content model for LIME

        log.info("Explainer ready.")

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_movies(self) -> pd.DataFrame:
        movies = pd.read_csv(MOVIES_PATH)
        movies["genres"]   = movies["genres"].apply(_parse_list)
        movies["keywords"] = movies["keywords"].apply(_parse_list)
        movies["cast"]     = movies["cast"].apply(_parse_list)
        return movies.set_index("movieId")

    def _load_ratings(self) -> pd.DataFrame:
        return pd.read_csv(RATINGS_PATH)

    def _get_movie(self, movie_id: int) -> pd.Series | None:
        return self.movies.loc[movie_id] if movie_id in self.movies.index else None

    def _user_top_rated(self, user_id: int, threshold: float = 3.5, limit: int = 10) -> list[int]:
        """Return the user's highest-rated movieIds."""
        df = self.ratings[self.ratings["userId"] == user_id]
        df = df[df["rating"] >= threshold].sort_values("rating", ascending=False)
        return df["movieId"].head(limit).tolist()

    # ── shared signal extraction ───────────────────────────────────────────────

    def _shared_signals(self, ref_movie: pd.Series, target_movie: pd.Series) -> dict:
        """Find overlapping genres, keywords, cast, director between two movies."""
        shared_genres   = list(set(ref_movie["genres"])   & set(target_movie["genres"]))
        shared_keywords = list(set(ref_movie["keywords"]) & set(target_movie["keywords"]))
        shared_cast     = list(set(ref_movie["cast"])     & set(target_movie["cast"]))
        same_director   = (
            ref_movie.get("director", "") == target_movie.get("director", "")
            and ref_movie.get("director", "") != ""
        )
        return {
            "genres"       : shared_genres,
            "keywords"     : shared_keywords,
            "cast"         : shared_cast,
            "same_director": same_director,
            "director"     : ref_movie.get("director", ""),
        }

    # ── rule-based explanation ─────────────────────────────────────────────────

    def _rule_based_explain(
        self,
        user_id   : int,
        movie_id  : int,
        liked_ids : list[int],
    ) -> str:
        """
        Generate a readable explanation by finding the most similar liked movie
        and listing shared attributes.
        """
        target = self._get_movie(movie_id)
        if target is None:
            return "Recommended based on your rating history."

        target_title = target.get("title", f"Movie {movie_id}")

        # find the liked movie with most shared signals
        best_ref, best_signals, best_score = None, {}, -1
        for liked_id in liked_ids:
            ref = self._get_movie(liked_id)
            if ref is None:
                continue
            signals = self._shared_signals(ref, target)
            score   = (
                len(signals["genres"])   * 3 +
                len(signals["cast"])     * 2 +
                len(signals["keywords"]) * 1 +
                (5 if signals["same_director"] else 0)
            )
            if score > best_score:
                best_score   = score
                best_ref     = ref
                best_signals = signals

        if best_ref is None:
            return f"Recommended because it matches your overall taste profile."

        ref_title = best_ref.get("title", "a movie you liked")
        parts     = []

        if best_signals["same_director"]:
            parts.append(f"directed by {best_signals['director']}")

        if best_signals["genres"]:
            genre_str = ", ".join(best_signals["genres"][:3])
            parts.append(f"in the {genre_str} genre{'s' if len(best_signals['genres']) > 1 else ''}")

        if best_signals["cast"]:
            cast_str = " and ".join(best_signals["cast"][:2])
            parts.append(f"starring {cast_str}")

        if best_signals["keywords"]:
            kw_str = ", ".join(best_signals["keywords"][:2])
            parts.append(f"with themes of {kw_str}")

        if parts:
            reason = ", and ".join(parts)
            return f"Because you liked \"{ref_title}\" — \"{target_title}\" is also {reason}."
        else:
            return f"Based on your interest in \"{ref_title}\" and similar movies."

    # ── LIME explanation ───────────────────────────────────────────────────────

    def _lime_explain(self, movie_id: int, n_features: int = 6) -> str:
        """
        Use LIME to identify which soup tokens drove the content similarity score.
        Falls back gracefully if LIME or content model is unavailable.
        """
        try:
            from lime.lime_text import LimeTextExplainer
            import sys
            sys.path.insert(0, str(ROOT / "src" / "models"))
            from content_based import ContentBasedFilter

            if self._cb is None:
                self._cb = ContentBasedFilter.load()

            cb    = self._cb
            idx   = cb.movie_id_to_idx.get(movie_id)
            if idx is None:
                return ""

            def _predict_fn(texts: list[str]) -> np.ndarray:
                """Score each perturbed text against all other movies (mean sim)."""
                vecs = cb._vectorizer.transform(texts)
                sims = (vecs @ cb.tfidf_matrix.T).toarray().mean(axis=1)
                # return shape (n_texts, 2) as LIME expects probability-like output
                return np.column_stack([1 - sims, sims])

            explainer = LimeTextExplainer(class_names=["irrelevant", "similar"])
            target_soup = self.movies.loc[movie_id, "soup"] if movie_id in self.movies.index else ""
            if not target_soup:
                return ""

            exp = explainer.explain_instance(
                target_soup,
                _predict_fn,
                num_features = n_features,
                num_samples  = 500,
            )
            top_features = exp.as_list(label=1)  # label 1 = "similar"
            if not top_features:
                return ""

            positive = [f for f, w in top_features if w > 0][:4]
            if positive:
                return f"Key similarity signals: {', '.join(positive)}."
            return ""

        except Exception as e:
            log.debug(f"LIME explanation failed: {e}")
            return ""

    # ── public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        user_id          : int,
        movie_id         : int,
        liked_movie_ids  : list[int] | None = None,
    ) -> str:
        """
        Generate a full explanation for a recommendation.

        Parameters
        ----------
        user_id         : the user receiving the recommendation
        movie_id        : the recommended movie
        liked_movie_ids : optional override; if None, inferred from ratings

        Returns
        -------
        str — human-readable explanation
        """
        if liked_movie_ids is None:
            liked_movie_ids = self._user_top_rated(user_id)

        rule_text = self._rule_based_explain(user_id, movie_id, liked_movie_ids)

        lime_text = ""
        if self.use_lime:
            lime_text = self._lime_explain(movie_id)

        if lime_text:
            return f"{rule_text} {lime_text}"
        return rule_text

    def explain_batch(
        self,
        user_id         : int,
        recommendations : pd.DataFrame,
        liked_movie_ids : list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Add an 'explanation' column to a recommendations DataFrame.

        Parameters
        ----------
        user_id         : the user
        recommendations : output from HybridEnsemble.recommend() or SentimentReRanker.rerank()
        liked_movie_ids : optional override

        Returns
        -------
        pd.DataFrame with 'explanation' column appended
        """
        if liked_movie_ids is None:
            liked_movie_ids = self._user_top_rated(user_id)

        recs = recommendations.copy()
        recs["explanation"] = recs["movieId"].apply(
            lambda mid: self.explain(user_id, int(mid), liked_movie_ids)
        )
        return recs


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6 — Explainability")
    parser.add_argument("--user-id",   type=int, default=1)
    parser.add_argument("--movie-id",  type=int, default=None)
    parser.add_argument("--no-lime",   action="store_true")
    args = parser.parse_args()

    exp = Explainer(use_lime=not args.no_lime)
    liked = exp._user_top_rated(args.user_id)
    print(f"User {args.user_id} liked movies: {liked[:5]}")

    if args.movie_id:
        explanation = exp.explain(args.user_id, args.movie_id, liked)
        print(f"\nExplanation for movie {args.movie_id}:")
        print(explanation)
    else:
        # explain for the first liked movie as a demo
        if liked:
            for mid in liked[:3]:
                print(f"\n[movieId={mid}] {exp.explain(args.user_id, mid, liked)}")


if __name__ == "__main__":
    main()
