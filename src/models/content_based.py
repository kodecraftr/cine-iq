"""
Phase 3 — Content-Based Filtering (TF-IDF + Cosine Similarity)
===============================================================
Builds a movie similarity matrix from the 'soup' column created in Phase 1
(genres + keywords + cast + director + overview) and exposes:

  similar_movies(movie_id, n)          →  pd.DataFrame
  content_score(movie_id, candidate_ids) →  pd.Series

Usage
-----
  python src/models/content_based.py

  from src.models.content_based import ContentBasedFilter
  cb = ContentBasedFilter.load()
  cb.similar_movies(movie_id=238, n=10)   # 238 = The Godfather in TMDB
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = MODEL_DIR / "content_based.pkl"
MOVIES_PATH = PROCESSED / "movies_merged.csv"


class ContentBasedFilter:
    """
    TF-IDF content-based recommender.

    Attributes
    ----------
    tfidf_matrix  : sparse matrix (n_movies × n_features)
    movies        : pd.DataFrame  [movieId, title, soup, ...]
    movie_id_to_idx : dict  movieId → row index in tfidf_matrix
    idx_to_movie_id : dict  row index → movieId
    """

    def __init__(self):
        self.tfidf_matrix     = None
        self.movies           = None
        self.movie_id_to_idx  = {}
        self.idx_to_movie_id  = {}
        self._vectorizer      = None

    # ── build ──────────────────────────────────────────────────────────────────

    def build(
        self,
        max_features : int = 25_000,
        ngram_range  : tuple = (1, 2),
        min_df       : int = 2,
    ) -> "ContentBasedFilter":
        """
        Fit TF-IDF on the soup column and cache the full similarity-ready matrix.

        Parameters
        ----------
        max_features : vocabulary cap (25 K is a good balance of speed vs quality)
        ngram_range  : unigrams + bigrams capture 'science fiction' as one token
        min_df       : ignore terms that appear in fewer than 2 movies
        """
        log.info("Building content-based model …")
        movies = pd.read_csv(MOVIES_PATH)

        # drop rows without a soup
        movies = movies.dropna(subset=["soup"]).reset_index(drop=True)
        movies["soup"] = movies["soup"].fillna("").astype(str)

        self.movies = movies

        # build index maps
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies["movieId"])}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}

        log.info(f"  Fitting TF-IDF on {len(movies):,} movies …")
        vectorizer = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            min_df       = min_df,
            stop_words   = "english",
            sublinear_tf = True,      # apply log(1+tf) — dampens high-freq terms
        )
        self._vectorizer  = vectorizer
        self.tfidf_matrix = vectorizer.fit_transform(movies["soup"])

        log.info(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        self.save()
        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def _get_idx(self, movie_id: int) -> int:
        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"movieId {movie_id} not found in content model.")
        return self.movie_id_to_idx[movie_id]

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """
        Return the top-N most similar movies to a given movie.

        Uses linear_kernel (equivalent to cosine similarity for L2-normalised
        TF-IDF vectors) — much faster than full cosine_similarity on large matrices.

        Returns
        -------
        pd.DataFrame  [movieId, title, similarity_score]
        """
        idx = self._get_idx(movie_id)

        # compute similarity of this movie against all others (returns 1-D array)
        movie_vec  = self.tfidf_matrix[idx]                   # (1, n_features)
        sim_scores = linear_kernel(movie_vec, self.tfidf_matrix).flatten()

        # exclude self, sort descending
        sim_scores[idx] = -1
        top_indices = np.argsort(sim_scores)[::-1][:n]

        results = pd.DataFrame({
            "movieId"          : [self.idx_to_movie_id[i] for i in top_indices],
            "similarity_score" : [round(sim_scores[i], 4)  for i in top_indices],
        })

        results = results.merge(
            self.movies[["movieId", "title", "genres", "release_year", "vote_average"]],
            on="movieId", how="left"
        )
        return results[["movieId", "title", "release_year", "vote_average", "similarity_score"]].reset_index(drop=True)

    def content_score(self, movie_id: int, candidate_ids: list[int]) -> pd.Series:
        """
        Return a content similarity score for each candidate movie relative to
        a reference movie.  Used by the Phase 4 ensemble.

        Parameters
        ----------
        movie_id      : the reference movie (e.g. a movie the user liked)
        candidate_ids : list of movieIds to score

        Returns
        -------
        pd.Series  index=candidate_ids, values=similarity_score  (0 – 1)
        """
        ref_idx  = self._get_idx(movie_id)
        ref_vec  = self.tfidf_matrix[ref_idx]

        # only compute for candidates that exist in the model
        valid = [(mid, self.movie_id_to_idx[mid]) for mid in candidate_ids if mid in self.movie_id_to_idx]
        if not valid:
            return pd.Series(dtype=float)

        cand_ids, cand_idxs = zip(*valid)
        cand_matrix = self.tfidf_matrix[list(cand_idxs)]
        scores      = linear_kernel(ref_vec, cand_matrix).flatten()

        return pd.Series(scores, index=list(cand_ids))

    def content_score_for_user(self, liked_movie_ids: list[int], candidate_ids: list[int]) -> pd.Series:
        """
        Aggregate content similarity across multiple liked movies (user profile).
        Score = mean similarity across all liked movies.

        Used by the ensemble when we have a full user rating history.
        """
        all_scores = []
        for mid in liked_movie_ids:
            if mid in self.movie_id_to_idx:
                s = self.content_score(mid, candidate_ids)
                all_scores.append(s)

        if not all_scores:
            return pd.Series(0.0, index=candidate_ids)

        return pd.concat(all_scores, axis=1).mean(axis=1)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        payload = {
            "tfidf_matrix"    : self.tfidf_matrix,
            "movies"          : self.movies,
            "movie_id_to_idx" : self.movie_id_to_idx,
            "idx_to_movie_id" : self.idx_to_movie_id,
            "vectorizer"      : self._vectorizer,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        log.info(f"Content model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ContentBasedFilter":
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Run build() first.")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls()
        instance.tfidf_matrix     = payload["tfidf_matrix"]
        instance.movies           = payload["movies"]
        instance.movie_id_to_idx  = payload["movie_id_to_idx"]
        instance.idx_to_movie_id  = payload["idx_to_movie_id"]
        instance._vectorizer      = payload["vectorizer"]
        log.info(f"Content model loaded ← {path}")
        return instance


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 — Build Content-Based Filter")
    parser.add_argument("--max-features", type=int, default=25_000)
    parser.add_argument("--movie-id",     type=int, default=None, help="Smoke-test with this movieId")
    args = parser.parse_args()

    cb = ContentBasedFilter()
    cb.build(max_features=args.max_features)

    if args.movie_id:
        log.info(f"Similar movies to {args.movie_id}:")
        print(cb.similar_movies(args.movie_id, n=10).to_string(index=False))
    else:
        # default smoke test with first movie in dataset
        first_id = cb.movies["movieId"].iloc[0]
        title    = cb.movies["title"].iloc[0]
        log.info(f"Smoke test — similar to '{title}' (movieId={first_id}):")
        print(cb.similar_movies(first_id, n=10).to_string(index=False))


if __name__ == "__main__":
    main()
