"""
app.py - Dash web app for A3. The user types car details and the app shows the predicted price class.

The model is loaded from app/model_artifacts (the best model of the MLflow experiment, saved by the notebook).
"""
import os
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
from car_model import CarPriceClassifier

MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")      # optional, e.g. "models:/st127304-a3-model@staging" (needs a reachable MLflow)
HERE = os.path.dirname(os.path.abspath(__file__))

CLASS_TEXT = {0: "Class 0 - cheapest 25% of cars", 1: "Class 1 - lower-middle", 2: "Class 2 - upper-middle", 3: "Class 3 - most expensive 25%"}


def load_model():
    """Default: the local copy of the best model (app/model_artifacts, written by the notebook).
    Optional: set MLFLOW_MODEL_URI to load a registered model from an MLflow server instead."""
    if MLFLOW_MODEL_URI:
        try:
            import mlflow
            return mlflow.pyfunc.load_model(MLFLOW_MODEL_URI), f"MLflow ({MLFLOW_MODEL_URI})"
        except Exception as e:
            print("Could not load from MLflow, using the local copy:", e)
    return CarPriceClassifier.from_dir(os.path.join(HERE, "model_artifacts")), "local copy of the best MLflow model"


model, source = load_model()


def predict_class(df):
    """The local model and an MLflow-loaded model have different predict() signatures; hide that here."""
    if isinstance(model, CarPriceClassifier):
        return model.predict(None, df)
    return model.predict(df)

def field(label, comp):
    return html.Div([html.Label(label, style={"fontWeight": "600"}), comp], style={"marginBottom": "10px"})

app = Dash(__name__)
server = app.server
app.layout = html.Div(style={"maxWidth": "560px", "margin": "30px auto", "fontFamily": "sans-serif"}, children=[
    html.H2("Car price class predictor"),
    html.P(f"Model source: {source}", style={"color": "#666"}),
    field("Brand", dcc.Input(id="name", value="Maruti", type="text", style={"width": "100%"})),
    field("Year", dcc.Input(id="year", value=2018, type="number", style={"width": "100%"})),
    field("Kilometres driven", dcc.Input(id="km_driven", value=40000, type="number", style={"width": "100%"})),
    field("Fuel", dcc.Dropdown(["Diesel", "Petrol"], "Petrol", id="fuel", clearable=False)),
    field("Seller type", dcc.Dropdown(["Individual", "Dealer", "Trustmark Dealer"], "Individual", id="seller_type", clearable=False)),
    field("Transmission", dcc.Dropdown(["Manual", "Automatic"], "Manual", id="transmission", clearable=False)),
    field("Owner (1=first ... 4=fourth & above)", dcc.Input(id="owner", value=1, type="number", min=1, max=4, style={"width": "100%"})),
    field("Mileage (kmpl) - leave blank if unknown", dcc.Input(id="mileage", type="number", style={"width": "100%"})),
    field("Engine (CC) - leave blank if unknown", dcc.Input(id="engine", type="number", style={"width": "100%"})),
    field("Max power (bhp) - leave blank if unknown", dcc.Input(id="max_power", type="number", style={"width": "100%"})),
    field("Seats - leave blank if unknown", dcc.Input(id="seats", type="number", style={"width": "100%"})),
    html.Button("Predict", id="go", n_clicks=0),
    html.H3(id="result", style={"marginTop": "20px"}),
])

FIELDS = ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "seats"]

@app.callback(Output("result", "children"), Input("go", "n_clicks"), [State(f, "value") for f in FIELDS])
def predict(n_clicks, *values):
    if not n_clicks:
        return ""
    row = dict(zip(FIELDS, values))                      # blanks arrive as None -> filled with training medians in the model
    pred = predict_class(pd.DataFrame([row]))
    return f"Predicted: {CLASS_TEXT[int(pred[0])]}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=False)
