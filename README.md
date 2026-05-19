# Cine IQ: Explainable Hybrid Movie Recommendation Engine

Cine IQ is an explainable movie recommendation web app built for the IIT Guwahati Coding Club Even Semester Project. It ranks movies with a weighted hybrid recommender, explains why each title was recommended, and lets users add real reviews that become an audience-sentiment signal for future rankings.

The current app is designed as a movie recommendation website rather than a plain ML dashboard. It includes sign-in with demo viewer profiles, a searchable movie catalog with embedded movie detail pages and reviews, and a user profile page with taste analytics.

## Current Features

| Area | Implementation |
| --- | --- |
| Sign-in flow | Streamlit sign-in screen with demo viewer profiles backed by MovieLens user histories. |
| Home recommendations | Personalized watch-history shelf and ranked recommendations. |
| Search | Catalog search for title, genre, director, and cast lookup, with selected movie details shown on the same page. |
| Movie details | Overview, year, director, cast, genres, keywords, similar movies, user reviews, and review form inside Search. |
| User profile | Watch history, average rating, top genre, favorite decade, genre radar, decade preferences, director affinity, and actor affinity. |
| Hybrid ranking | `HybridEnsemble` blends collaborative filtering and content-based scores. |
| Collaborative filtering | Surprise SVD matrix factorization trained from MovieLens ratings. |
| Content-based filtering | TF-IDF over metadata soup with cosine similarity. |
| Review sentiment | IMDb 50K trains the positive/negative review classifier; user-submitted reviews create movie-level sentiment scores. |
| Sentiment re-ranking | Average sentiment per movie is folded into final recommendations through `SentimentReRanker`. |
| Explainability | Rule-based explanations describe genre, director, cast, keyword, and history-based reasons. |
| API serving | FastAPI exposes health, recommendation, and similar-movie endpoints. |

## Recommendation Workflow

```text
MovieLens ratings + TMDB metadata
        |
        v
Preprocessing
  movies_merged.csv
  ratings_clean.csv
        |
        v
Model training
  SVD collaborative model
  TF-IDF content model
  IMDb-trained review sentiment classifier
        |
        v
Hybrid recommendation
  collaborative score + content score
        |
        v
User reviews in the app
  review text -> positive/negative sentiment -> per-movie average sentiment
        |
        v
Final ranking
  hybrid score adjusted by audience sentiment
        |
        v
Streamlit website + FastAPI endpoints
```

Important distinction: the IMDb 50K dataset is used only to train the review sentiment classifier. Cine IQ does not use IMDb titles for movie re-ranking. Re-ranking uses reviews submitted inside the app for a specific movie, stored locally in `data/processed/user_reviews.csv`, then aggregated into `data/processed/sentiment_scores.csv`.

## Repository Structure

```text
cine-iq/
  src/
    api/main.py                    FastAPI service
    dashboard/app.py               Streamlit website
    data/preprocess.py             dataset cleaning and merging pipeline
    explainability/explainer.py    recommendation explanations
    models/collaborative.py        SVD collaborative model
    models/content_based.py        TF-IDF content model
    models/ensemble.py             hybrid recommender
    models/sentiment.py            review classifier and sentiment re-ranker
  notebooks/
    01_phase1_eda.ipynb
    02_phase2_collab.ipynb
  reports/
    cine_iq_report.md
    cine_iq_report.pdf
  requirements.txt
  README.md
```

Large datasets, generated model files, submitted reviews, and local verification logs are ignored by Git. They are generated locally during setup and training.

## Datasets

Place the required files in this layout:

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

| Dataset | Purpose | Source |
| --- | --- | --- |
| MovieLens 25M | Ratings, user histories, and movie IDs | https://grouplens.org/datasets/movielens/25m/ |
| TMDB 5000 Movies + Credits | Movie metadata, cast, crew, overview, keywords | https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata |
| IMDb 50K Reviews | Train positive/negative review sentiment classifier | https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews |

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training From Scratch

Run the full preprocessing and training workflow:

```bash
python src/data/preprocess.py
python src/models/collaborative.py
python src/models/content_based.py
python src/models/sentiment.py --train-review-classifier
python src/models/ensemble.py --user-id 1 --n 10
python src/explainability/explainer.py --user-id 1 --no-lime
```

For a quick smoke test of SVD training, use a sample first:

```bash
python src/models/collaborative.py --sample-frac 0.1
```

For the final demo, train without `--sample-frac` so the app runs on the full processed data. The saved SVD model can be large because it stores the trained factorization for the full ratings matrix.

Generated artifacts:

```text
data/processed/movies_merged.csv
data/processed/ratings_clean.csv
models/saved/svd_model.pkl
models/saved/content_based.pkl
models/saved/review_sentiment.pkl
```

When users submit reviews in the Streamlit app, these local files may also appear:

```text
data/processed/user_reviews.csv
data/processed/sentiment_scores.csv
```

## Running the Streamlit App

```bash
streamlit run src/dashboard/app.py
```

Open the local Streamlit URL, usually:

```text
http://localhost:8501
```

Suggested demo flow:

1. Sign in as one of the demo viewers.
2. Show the Home page recommendations and watch-history shelf.
3. Open Search, find a movie, and show its detail section on the same page.
4. Show overview, cast, metadata, similar movies, and user reviews.
5. Submit a positive or negative review for that movie.
6. Return to Home and explain that submitted reviews update the movie-level sentiment signal used in re-ranking.
7. Open My Profile and show the taste analytics: genre radar, decade preference, director affinity, actor affinity, and watch history.

## Running the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Example requests:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommend?user_id=1&n=10"
curl "http://localhost:8000/similar?movie_id=1&n=10"
```

## MLflow Tracking

The collaborative and ensemble scripts log metrics and parameters to MLflow:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

## Evaluation

| Component | Metric or check |
| --- | --- |
| SVD collaborative model | RMSE and MAE on held-out MovieLens ratings. |
| Content-based model | Similarity sanity checks for known movies. |
| Hybrid ensemble | Ranked recommendation quality and Precision@K during tuning. |
| Review sentiment classifier | Accuracy on IMDb positive/negative review validation data. |
| Sentiment re-ranker | Rank shifts when submitted reviews create positive or negative audience signals. |
| Streamlit app | Sign-in, Home, Search with movie details/reviews, and My Profile flows. |
| API | `/health`, `/recommend`, and `/similar` smoke tests. |

## Deliverables

| Deliverable | Status |
| --- | --- |
| Public GitHub codebase | This repository contains the source code and README. |
| Demo video | Record the Streamlit flow above and upload to YouTube or Google Drive. |
| Report | See `reports/cine_iq_report.md` and `reports/cine_iq_report.pdf`. |

## Known Limitations

- Raw datasets and saved model files are not committed because they are large and externally licensed.
- Full MovieLens 25M SVD training can take time on CPU and produces a large local model file.
- User-submitted reviews are stored locally in CSV files for the demo; a production app would use a database and real authentication.
- The app uses demo viewer profiles derived from available MovieLens users, not real account registration.
