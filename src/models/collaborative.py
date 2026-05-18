"""
Phase 2 — Collaborative Filtering (SVD)
========================================
Trains a matrix-factorisation model (SVD via Surprise) on the cleaned
MovieLens ratings, evaluates it, and exposes two clean interfaces:

  predict_rating(user_id, movie_id)  →  float
  top_n(user_id, n)                  →  pd.DataFrame

MLflow is used to track every experiment run automatically.

Usage
-----
  # train + save
  python src/models/collaborative.py

  # from another module
  from src.models.collaborative import CollaborativeFilter
  cf = CollaborativeFilter.load()
  cf.top_n(user_id=42, n=10)
"""

import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import GridSearchCV, train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = MODEL_DIR / "svd_model.pkl"
MOVIES_PATH = PROCESSED / "movies_merged.csv"
RATINGS_PATH = PROCESSED / "ratings_clean.csv"

# ── CollaborativeFilter class ──────────────────────────────────────────────────

class CollaborativeFilter:
    """
    Wrapper around Surprise SVD.

    Attributes
    ----------
    model      : trained SVD algorithm
    trainset   : the full Surprise Trainset (for anti-test-set generation)
    movie_map  : pd.DataFrame  [movieId, title]  — for display purposes
    """

    def __init__(self):
        self.model     = None
        self.trainset  = None
        self.movie_map = None

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self, sample_frac: float = 1.0):
        """
        Load ratings_clean.csv into a Surprise Dataset.

        Parameters
        ----------
        sample_frac : float
            Fraction of ratings to use (useful for quick experiments; default = all).
        """
        log.info("Loading ratings …")
        ratings = pd.read_csv(RATINGS_PATH)

        if sample_frac < 1.0:
            ratings = ratings.sample(frac=sample_frac, random_state=42)
            log.info(f"  Sampled {len(ratings):,} ratings ({sample_frac:.0%})")
        else:
            log.info(f"  {len(ratings):,} ratings loaded")

        # Surprise needs to know the rating scale
        reader  = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
        dataset = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

        # load movie titles for display
        self.movie_map = pd.read_csv(MOVIES_PATH, usecols=["movieId", "title"]).drop_duplicates("movieId")

        return dataset, ratings

    # ── hyperparameter tuning ─────────────────────────────────────────────────

    def tune(self, sample_frac: float = 0.2) -> dict:
        """
        Run a small GridSearchCV on a sample of the data to find good
        SVD hyperparameters.  Results are logged to MLflow.

        Parameters
        ----------
        sample_frac : fraction of ratings to use during tuning (default 20 %)

        Returns
        -------
        best_params : dict
        """
        log.info("Tuning SVD hyperparameters …")
        dataset, _ = self._load_data(sample_frac=sample_frac)

        param_grid = {
            "n_factors": [50, 100, 150],
            "n_epochs" : [20, 30],
            "lr_all"   : [0.005, 0.010],
            "reg_all"  : [0.02, 0.10],
        }

        gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
        gs.fit(dataset)

        best_params = gs.best_params["rmse"]
        best_rmse   = gs.best_score["rmse"]
        best_mae    = gs.best_score["mae"]

        log.info(f"  Best RMSE : {best_rmse:.4f}")
        log.info(f"  Best MAE  : {best_mae:.4f}")
        log.info(f"  Best params: {best_params}")

        with mlflow.start_run(run_name="svd_tuning"):
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_rmse", best_rmse)
            mlflow.log_metric("cv_mae",  best_mae)

        return best_params

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        n_factors : int   = 100,
        n_epochs  : int   = 30,
        lr_all    : float = 0.005,
        reg_all   : float = 0.02,
        test_size : float = 0.2,
        sample_frac: float = 1.0,
    ) -> dict:
        """
        Train SVD on ratings_clean.csv, evaluate on a held-out test set,
        and persist the model to disk.

        Returns
        -------
        metrics : dict  {rmse, mae}
        """
        log.info("Training SVD …")
        dataset, _ = self._load_data(sample_frac=sample_frac)

        trainset, testset = train_test_split(dataset, test_size=test_size, random_state=42)

        model = SVD(
            n_factors = n_factors,
            n_epochs  = n_epochs,
            lr_all    = lr_all,
            reg_all   = reg_all,
            random_state = 42,
            verbose   = False,
        )
        model.fit(trainset)

        # evaluate
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae  = accuracy.mae(predictions,  verbose=False)
        log.info(f"  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")

        # store on self
        self.model    = model
        self.trainset = trainset

        # log to MLflow
        params = dict(n_factors=n_factors, n_epochs=n_epochs,
                      lr_all=lr_all, reg_all=reg_all, test_size=test_size)
        with mlflow.start_run(run_name="svd_training"):
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae",  mae)

        # save model
        self.save()
        metrics = {"rmse": rmse, "mae": mae}
        return metrics

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating a user would give a specific movie.

        Returns
        -------
        float : predicted rating (within the original rating scale)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")
        pred = self.model.predict(uid=user_id, iid=movie_id)
        return round(pred.est, 4)

    def top_n(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Return the top-N movie recommendations for a user.

        Strategy: generate predictions for ALL movies the user has NOT yet
        rated (anti-test-set), sort by predicted rating descending.

        Returns
        -------
        pd.DataFrame  columns: [movieId, title, predicted_rating]
        """
        if self.model is None or self.trainset is None:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        raw_user_id = user_id
        try:
            uid_inner = self.trainset.to_inner_uid(raw_user_id)
        except ValueError:
            raw_user_id = str(user_id)
            uid_inner = self.trainset.to_inner_uid(raw_user_id)

        # movies already rated by this user (inner ids)
        rated_inner = set(iid for (iid, _) in self.trainset.ur[uid_inner])

        # all movie inner ids
        all_items = set(self.trainset.all_items())

        # anti-test set: movies not yet rated
        anti_testset = [
            (raw_user_id, self.trainset.to_raw_iid(iid), 0)   # (uid, iid, placeholder_rating)
            for iid in all_items - rated_inner
        ]

        predictions = self.model.test(anti_testset)
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_preds   = predictions[:n]

        results = pd.DataFrame({
            "movieId"          : [int(p.iid) for p in top_preds],
            "predicted_rating" : [round(p.est, 4) for p in top_preds],
        })

        # join titles
        if self.movie_map is not None:
            results = results.merge(self.movie_map, on="movieId", how="left")
            results = results[["movieId", "title", "predicted_rating"]]

        return results.reset_index(drop=True)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        payload = {
            "model"    : self.model,
            "trainset" : self.trainset,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        log.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CollaborativeFilter":
        """Load a previously trained model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Run train() first.")
        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance           = cls()
        instance.model     = payload["model"]
        instance.trainset  = payload["trainset"]

        # re-load movie map (lightweight CSV)
        if MOVIES_PATH.exists():
            instance.movie_map = pd.read_csv(
                MOVIES_PATH, usecols=["movieId", "title"]
            ).drop_duplicates("movieId")

        log.info(f"Model loaded ← {path}")
        return instance


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 — Train Collaborative Filter")
    parser.add_argument("--tune",         action="store_true", help="Run hyperparameter tuning first")
    parser.add_argument("--sample-frac",  type=float, default=1.0,  help="Fraction of ratings to use (0-1)")
    parser.add_argument("--n-factors",    type=int,   default=100)
    parser.add_argument("--n-epochs",     type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=0.005)
    parser.add_argument("--reg",          type=float, default=0.02)
    parser.add_argument("--test-size",    type=float, default=0.2)
    args = parser.parse_args()

    mlflow.set_experiment("movie-recommender-svd")

    cf = CollaborativeFilter()

    if args.tune:
        best = cf.tune(sample_frac=max(args.sample_frac, 0.2))
        # override CLI args with tuned values
        args.n_factors = best["n_factors"]
        args.n_epochs  = best["n_epochs"]
        args.lr        = best["lr_all"]
        args.reg       = best["reg_all"]

    metrics = cf.train(
        n_factors  = args.n_factors,
        n_epochs   = args.n_epochs,
        lr_all     = args.lr,
        reg_all    = args.reg,
        test_size  = args.test_size,
        sample_frac = args.sample_frac,
    )

    log.info(f"Final metrics → RMSE: {metrics['rmse']:.4f}  MAE: {metrics['mae']:.4f}")

    # quick smoke test
    log.info("Smoke test — top-10 for user 1:")
    recs = cf.top_n(user_id=1, n=10)
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
