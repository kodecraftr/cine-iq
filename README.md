# Cine IQ: Explainable Hybrid Movie Recommendation Engine

Cine IQ is an open, explainable movie recommendation system built for the IIT Guwahati Coding Club Even Semester Project. It combines collaborative filtering, TF-IDF content similarity, weighted hybrid ranking, sentiment-aware re-ranking, and natural-language explanations so that recommendations are personalized and interpretable.

## Project Status

The repository covers the core Cine IQ requirements:

| Requirement | Implementation |
| --- | --- |
| Hybrid recommendation engine | `src/models/ensemble.py` blends SVD collaborative scores with TF-IDF content scores. |
| Collaborative filtering and SVD | `src/models/collaborative.py` trains and serves a Surprise SVD model. |
| Content-based filtering | `src/models/content_based.py` builds TF-IDF vectors from metadata soup and uses cosine similarity. |
| Sentiment-aware re-ranking | `src/models/sentiment.py` supports VADER and DistilBERT review sentiment scoring. |
| Explainability layer | `src/explainability/explainer.py` provides rule-based explanations and optional LIME signals. |
| User taste dashboard | `src/dashboard/app.py` provides Streamlit pages for recommendations, similar movies, and taste profile charts. |
| API serving | `src/api/main.py` exposes `/recommend`, `/similar`, and `/health` through FastAPI. |
| Experiment tracking | Training and tuning scripts log metrics to MLflow. |

Important note: the large raw datasets and trained model pickle files are not committed because they are too large for normal GitHub hosting. Download the datasets listed below, run the pipeline, then train the models.

## Architecture

```text
Raw data
  MovieLens 25M ratings + movies
  TMDB metadata
  IMDb review sentiment
        |
        v
src/data/preprocess.py
  cleaned ratings, merged movie metadata, content soup, sentiment scores
        |
        v
Model layer
  CollaborativeFilter: Surprise SVD
  ContentBasedFilter: TF-IDF + cosine similarity
  HybridEnsemble: weighted blend of collaborative and content scores
  SentimentReRanker: VADER/DistilBERT score adjustment
  Explainer: rule templates and optional LIME terms
        |
        v
Serving layer
  FastAPI endpoints
  Streamlit dashboard
```

## Repository Structure

```text
cine-iq/
  src/
    api/main.py                    FastAPI service
    dashboard/app.py               Streamlit dashboard
    data/preprocess.py             dataset cleaning and merging pipeline
    explainability/explainer.py    recommendation explanations
    models/collaborative.py        SVD collaborative model
    models/content_based.py        TF-IDF content model
    models/ensemble.py             hybrid recommender
    models/sentiment.py            sentiment re-ranker
  notebooks/
    01_phase1_eda.ipynb
    02_phase2_collab.ipynb
  reports/
    cine_iq_report.md
    cine_iq_report.pdf             generated report
  scripts/
    generate_report_pdf.py
  requirements.txt
```

## Datasets

Download and place the files in this layout:

```text
data/raw/
  movielens/
    ratings.csv
    movies.csv
    links.csv
  tmdb/
    tmdb_5000_movies.csv
    tmdb_5000_credits.csv
  imdb/
    IMDB Dataset.csv
```

Sources:

| Dataset | Purpose | Link |
| --- | --- | --- |
| MovieLens 25M | user ratings and movie IDs | https://grouplens.org/datasets/movielens/25m/ |
| TMDB 5000 Movies + Credits | genres, cast, crew, overview, keywords | https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata |
| IMDb 50K Reviews | sentiment model experimentation | https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews |

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the data and model pipeline:

```bash
python src/data/preprocess.py
python src/models/collaborative.py --sample-frac 0.1
python src/models/content_based.py
python src/models/ensemble.py --user-id 42 --n 10
python src/models/sentiment.py
python src/explainability/explainer.py --user-id 42 --no-lime
```

For full training, remove `--sample-frac 0.1` after confirming the quick run works.

## Running the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Example requests:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommend?user_id=42&n=10"
curl "http://localhost:8000/similar?movie_id=238&n=10"
```

## Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open the Streamlit URL, usually `http://localhost:8501`.

## MLflow Tracking

The collaborative and ensemble scripts log metrics and parameters to MLflow:

```bash
mlflow ui
```

Then open `http://localhost:5000`.

## Evaluation Targets

| Component | Metric |
| --- | --- |
| SVD collaborative model | RMSE and MAE on held-out MovieLens ratings |
| Hybrid ensemble | Precision@10 during weight tuning |
| Sentiment re-ranker | rank shifts and final score changes |
| Dashboard/API | smoke tests for model loading and endpoint responses |

## Deliverables

| Deliverable | File or link |
| --- | --- |
| Public GitHub codebase | Push this repository to a public GitHub repo after committing the local changes. |
| Demo video | Record the dashboard/API flow and upload it to YouTube or Google Drive. |
| Report | `reports/cine_iq_report.pdf` and `reports/cine_iq_report.md` |

## Known Limitations

- Raw MovieLens/TMDB/IMDb datasets are not included in Git because they are large and externally licensed.
- Saved model files are generated locally under `models/saved/` after training.
- The common IMDb 50K review dataset does not reliably include movie IDs, so title-level sentiment joins are best-effort unless a review dataset with stable movie IDs is used.
- Full MovieLens 25M training can be slow on CPU. Use `--sample-frac` for quick checks and full data for final metrics.
