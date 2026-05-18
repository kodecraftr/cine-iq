# Cine IQ Submission Checklist

## Requirement Coverage

| Cine IQ requirement | Status | Evidence |
| --- | --- | --- |
| Hybrid recommendation engine | Covered | `src/models/ensemble.py` |
| Collaborative filtering with SVD | Covered | `src/models/collaborative.py` |
| Content-based filtering with TF-IDF and cosine similarity | Covered | `src/models/content_based.py` |
| Sentiment-aware re-ranker with VADER/DistilBERT | Covered | `src/models/sentiment.py` |
| Streamlit user taste dashboard | Covered | `src/dashboard/app.py` |
| Explainability layer with readable reasons and optional LIME | Covered | `src/explainability/explainer.py` |
| FastAPI serving with `/recommend` and `/similar` | Covered | `src/api/main.py` |
| Plotly charts | Covered | `src/dashboard/app.py` |
| MLflow tracking | Covered | `src/models/collaborative.py`, `src/models/ensemble.py` |
| Well structured README | Covered | `README.md` |
| Detailed 2-3 page report | Covered | `reports/cine_iq_report.pdf` |
| Small public demo video | To be recorded | Record the dashboard/API flow and upload to YouTube or Google Drive |

## Remaining Account-Dependent Steps

These cannot be completed from this local session without your GitHub and video hosting account access.

1. Create a public GitHub repository, for example `cine-iq`.
2. Commit and push this local folder:

```bash
git add README.md requirements.txt .gitignore SUBMISSION_CHECKLIST.md reports scripts demo
git commit -m "Prepare Cine IQ submission deliverables"
git remote set-url origin https://github.com/<your-username>/cine-iq.git
git push -u origin main
```

3. Record a short demo video and upload it to YouTube or Google Drive.
4. Put the public video link in your submission form and, if desired, in the README deliverables table.
5. Submit the GitHub repository link and `reports/cine_iq_report.pdf`.

## Notes for Evaluators

- Large raw datasets are excluded from Git and should be downloaded using the README instructions.
- Model files under `models/saved/` are generated after running preprocessing and training.
- The report PDF is generated from `reports/cine_iq_report.md` using `scripts/generate_report_pdf.py`.
- The demo video should show the README/project structure, dashboard pages, and optionally the FastAPI docs page.
