# A3 - Predicting Car Price (Classification, MLflow, CI/CD)

AT82.03 Machine Learning - Assignment 3  
Student: **<YOUR NAME> (st127304)**

## What this project does
Predicts which **price class (0-3)** a used car falls into, using a multinomial logistic regression written from scratch
(with an optional Ridge/L2 penalty and hand-written classification metrics).

| Class | Meaning |
|---|---|
| 0 | cheapest 25% of training cars |
| 1 | lower-middle 25% |
| 2 | upper-middle 25% |
| 3 | most expensive 25% |

## Repository layout
```
.
├── A3_Predicting_Car_Price.ipynb   # Tasks 1 & 2 + MLflow logging/registration (Task 3, objectives 1-2)
├── Cars.csv                        # dataset
├── app/                            # the web application
│   ├── app.py                      # Dash app
│   ├── car_model.py                # packaged model (preprocessing + softmax)
│   ├── requirements.txt
│   └── model_artifacts/            # best model of the MLflow experiment (weights + preprocessing values)
├── tests/test_model.py             # 2 unit tests
├── deploy/docker-compose.yaml      # compose file used on the ml-brain server
├── screenshots/                    # MLflow + CI/CD screenshots
├── Dockerfile
└── .github/workflows/ci-cd.yml     # GitHub Actions: test on every push, then deploy if tests pass
```

## Task 1 - classification metrics from scratch
Price is bucketed into 4 quartile classes. `accuracy`, per-class `precision`/`recall`/`f1`, `macro_*` and `weighted_*`
were added to the `LogisticRegression` class and checked against `sklearn.metrics.classification_report`.
*Support* = the number of true samples of each class.

## Task 2 - Ridge logistic regression
`use_penalty` / `lambda_` add `lambda * sum(W^2)` to the loss (intercept excluded). Results: <PASTE 2-3 SENTENCES>.

## Task 3 - MLflow, deployment, CI/CD
Per the TA's announcement, MLflow is run **locally** (`sqlite:///mlflow.db`) and screenshots are provided.

* **Experiment:** `st127304-a3` - 4 runs (no penalty + Ridge 0.0001 / 0.001 / 0.01); dataset not logged; model saved with every run.
* **Registered model:** `st127304-a3-model` (alias `staging`) - best run: `<RUN NAME>`, macro F1 = `<VALUE>`
* **Unit tests:** `pytest` - (1) the model takes the expected input, (2) the output has the expected shape
* **CI/CD:** every push runs the tests; if they pass, the image is built, pushed to Docker Hub and deployed to the ml-brain server

### Screenshots
| | |
|---|---|
| MLflow experiment runs | `screenshots/mlflow_runs.png` |
| Best run's saved model | `screenshots/mlflow_run_model.png` |
| Registered model (staging) | `screenshots/mlflow_registered_model.png` |
| Green GitHub Actions run | `screenshots/github_actions.png` |
| Deployed website | `screenshots/website.png` |

### Run locally
```bash
pip install -r app/requirements.txt
cd app && python app.py            # open http://localhost:8050
# or with Docker
docker build -t arpanoj/a3-car-price:latest . && docker run -p 8050:8050 arpanoj/a3-car-price:latest
```
Tests: `pip install pytest && pytest -v` &nbsp;|&nbsp; MLflow page: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

### Live app
<https://web-st127304-a3.ml.brain.cs.ait.ac.th>
