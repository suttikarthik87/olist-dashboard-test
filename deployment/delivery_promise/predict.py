"""Scoring for the C1 app, kept free of Streamlit so it can be checked on its own (and reused by an
API later).

    from predict import load_model, score
    parts = score(load_model(), orders)     # orders: a DataFrame holding the 17 model inputs

score() returns each step of the promise: the Log OLS estimate, the forest estimate, their average
on log(1 + days), and the promise (the average raised by the margin, rounded up). These are the same
operations as DeliveryPromise.promise() in src/c1_model.py, split out so the app can show them.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# The team repository keeps the model and c1_model.py beside this file; the Phase 3 folder keeps them in models/ and src/
MODEL_PATH = next(p for p in (HERE / "c1_delivery_promise.pkl", ROOT / "models" / "c1_delivery_promise.pkl") if p.exists())
sys.path.insert(0, str(ROOT / "src"))   # the pickle refers to the module c1_model
sys.path.insert(0, str(HERE))
from c1_model import promise_days  # noqa: E402

# Brazil's five official regions, by customer state (used for the region table)
REGION = {**dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "North"),
          **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Northeast"),
          **dict.fromkeys(["DF", "GO", "MS", "MT"], "Centre-West"),
          **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Southeast"),
          **dict.fromkeys(["PR", "RS", "SC"], "South")}


def load_model(path=MODEL_PATH):
    """The saved ensemble, exactly as tested in the notebook."""
    return joblib.load(path)


def score(model, orders):
    """Each step of the promise for every order, in the order given."""
    X = orders[model.features]
    with warnings.catch_warnings():
        # A category never seen in training (e.g. a rare product category) is encoded as "other"
        # or as all zeros, as in the notebook; scikit-learn warns each time, which is expected.
        warnings.filterwarnings("ignore", message="Found unknown categories")
        ols = model.ols.predict(X)
        forest = model.forest.predict(X)
    log_estimate = (np.log1p(ols) + np.log1p(forest)) / 2
    return pd.DataFrame({"ols": ols, "forest": forest, "estimate": np.expm1(log_estimate),
                         "promise": promise_days(log_estimate, model.log_margin)}, index=orders.index)
