"""The final C1 model: the pipelines chosen in c1_02 and c1_03, and the promise built from them.

Kept in a module, rather than in a notebook, so that the fitted model can be saved with joblib and
loaded again: pickle cannot save functions defined inside a notebook or as lambdas.

    from c1_model import build_ols, build_forest, DeliveryPromise
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PAYMENT = ["payment_type", "max_installments"]       # left out, as decided in c1_02 Table 12
FOREST_SETTINGS = {"n_estimators": 200, "max_features": 0.2, "min_samples_leaf": 10}   # c1_03 Table 2


def to_float(x):
    """Yes/no columns as 1.0 and 0.0."""
    return x.astype(float)


def name_dummy(feature, value):
    """Dummy column names, with the merged rare values called "other"."""
    return f"{feature}_{'other' if value == 'infrequent_sklearn' else value}"


def _columns(spec):
    numeric = [c for c in spec["numeric"] if c not in PAYMENT]
    grouped = [c for c in spec["categorical"] if c not in PAYMENT and c != "customer_state"]
    return numeric, spec["boolean"], grouped


def build_ols(spec, reference):
    """The chosen OLS version (c1_02): log days, log features, payment left out, one reference
    value dropped per category, values under 1% merged into "other"."""
    numeric, boolean, grouped = _columns(spec)
    prepare = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"),
                                  FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                  StandardScaler()), numeric),
        ("yes/no", FunctionTransformer(to_float, feature_names_out="one-to-one"), boolean),
        ("state", OneHotEncoder(drop=[reference["customer_state"]], handle_unknown="ignore",
                                sparse_output=False), ["customer_state"]),
        ("grouped", OneHotEncoder(drop=[reference[c] for c in grouped], min_frequency=0.01,
                                  handle_unknown="infrequent_if_exist", sparse_output=False,
                                  feature_name_combiner=name_dummy), grouped),
    ], verbose_feature_names_out=False)
    return TransformedTargetRegressor(make_pipeline(prepare, LinearRegression()),
                                      func=np.log1p, inverse_func=np.expm1)


def build_forest(spec, seed):
    """The chosen random forest (c1_03): raw days and raw features, every dummy kept, values
    under 1% merged into "other"."""
    numeric, boolean, grouped = _columns(spec)
    prepare = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("yes/no", FunctionTransformer(to_float, feature_names_out="one-to-one"), boolean),
        ("state", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["customer_state"]),
        ("grouped", OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist",
                                  sparse_output=False, feature_name_combiner=name_dummy), grouped),
    ], verbose_feature_names_out=False)
    return make_pipeline(prepare, RandomForestRegressor(**FOREST_SETTINGS, n_jobs=-1, random_state=seed))


def promise_days(log_estimate, log_margin):
    """A promise in whole days: the estimate on log(1 + days) raised by the margin, rounded up."""
    return np.ceil(np.expm1(log_estimate + log_margin))


class DeliveryPromise:
    """The final model: the average of OLS and the forest on log(1 + days), raised by one margin
    and rounded up to whole days."""

    def __init__(self, ols, forest, log_margin, features):
        self.ols, self.forest, self.log_margin, self.features = ols, forest, log_margin, list(features)

    def log_estimate(self, orders):
        X = orders[self.features]
        return (np.log1p(self.ols.predict(X)) + np.log1p(self.forest.predict(X))) / 2

    def estimate(self, orders):
        """The typical delivery time, in days."""
        return np.expm1(self.log_estimate(orders))

    def promise(self, orders):
        """The promised delivery time, in whole days."""
        return promise_days(self.log_estimate(orders), self.log_margin)
