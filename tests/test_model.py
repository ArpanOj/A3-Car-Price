"""Two unit tests required by A3 Task 3: (1) expected INPUT, (2) expected OUTPUT shape."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from car_model import CarPriceClassifier, RAW_COLUMNS

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "app", "model_artifacts")


@pytest.fixture(scope="module")
def model():
    return CarPriceClassifier.from_dir(ARTIFACTS)       # real trained weights, no network needed


def sample(n=1):
    row = dict(name="Maruti", year=2018, km_driven=40000, fuel="Petrol", seller_type="Individual",
               transmission="Manual", owner=1, mileage=20.0, engine=1197.0, max_power=82.0, seats=5)
    return pd.DataFrame([row] * n)


def test_model_takes_expected_input(model):
    """(1) The model accepts the expected raw columns and turns them into the feature matrix it was trained on."""
    df = sample(3)
    assert list(df.columns) == RAW_COLUMNS                       # the input has the expected columns
    X = model._preprocess(df)
    assert X.shape == (3, model.W.shape[0])                      # (rows, n_features + intercept) = rows of W
    assert not np.isnan(X).any()                                 # no missing values reach the model
    assert np.all(X[:, 0] == 1)                                  # intercept column is 1


def test_model_output_has_expected_shape(model):
    """(2) One predicted class per input row, each a valid class 0..k-1 (and probabilities sum to 1)."""
    df = sample(5)
    pred = model.predict(None, df)
    assert pred.shape == (5,)
    assert set(pred).issubset(set(range(model.W.shape[1])))
    proba = model.predict_proba(df)
    assert proba.shape == (5, model.W.shape[1])
    assert np.allclose(proba.sum(axis=1), 1.0)
