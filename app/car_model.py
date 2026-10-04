"""
car_model.py  -  the packaged model used by the notebook (logging), the Dash app and the unit tests.

It wraps ONLY numbers (the weight matrix W) and a small JSON of preprocessing values, so it does not
depend on pickling notebook classes. It takes RAW car features (like a user would type) and returns
a price class 0-3.
"""
import json
import numpy as np
import pandas as pd
try:                                   # MLflow is only needed when LOGGING the model in the notebook;
    import mlflow.pyfunc               # the web app, the unit tests and CI work without it
    _PythonModel = mlflow.pyfunc.PythonModel
except ImportError:
    _PythonModel = object

# Raw columns the model expects (same as A1 after cleaning, without selling_price)
RAW_COLUMNS = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',
               'owner', 'mileage', 'engine', 'max_power', 'seats']
NUM_FEATURES = ['year', 'km_driven', 'owner', 'mileage', 'engine', 'max_power', 'seats']
CAT_FEATURES = ['name', 'fuel', 'seller_type', 'transmission']


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class CarPriceClassifier(_PythonModel):
    """MLflow 'pyfunc' model: raw car features in -> price class (0,1,2,3) out."""

    # ---- used by MLflow when the model is loaded from the registry ----
    def load_context(self, context):
        self.W = np.load(context.artifacts["weights"])                 # (n_features + 1, k)
        with open(context.artifacts["preprocess"]) as f:
            self.pp = json.load(f)

    # ---- used by unit tests / local fallback: build the model from a local folder ----
    @classmethod
    def from_dir(cls, folder):
        obj = cls()
        obj.W = np.load(f"{folder}/weights.npy")
        with open(f"{folder}/preprocess.json") as f:
            obj.pp = json.load(f)
        return obj

    def _preprocess(self, df):
        """Raw table -> numeric matrix (n_rows, n_features + 1) exactly like training."""
        df = df.copy()
        for c in RAW_COLUMNS:                                          # add any missing column as empty
            if c not in df.columns:
                df[c] = np.nan
        df = df[RAW_COLUMNS]
        for c in NUM_FEATURES:                                         # blanks -> training median
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.pp["medians"][c])
        for c in CAT_FEATURES:                                         # blanks -> training mode
            df[c] = df[c].replace("", np.nan).fillna(self.pp["modes"][c]).astype(str)
        mean = np.array(self.pp["scaler_mean"])
        scale = np.array(self.pp["scaler_scale"])
        df[NUM_FEATURES] = (df[NUM_FEATURES].astype(float) - mean) / scale   # same standardisation as training
        df = pd.get_dummies(df, columns=CAT_FEATURES, dtype=float)     # one-hot
        df = df.reindex(columns=self.pp["columns"], fill_value=0.0)    # same columns/order as training
        X = df.to_numpy(dtype=float)
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)   # intercept column

    def predict_proba(self, df):
        return softmax(self._preprocess(df) @ self.W)

    def predict(self, context, model_input, params=None):
        return np.argmax(self.predict_proba(model_input), axis=1)
