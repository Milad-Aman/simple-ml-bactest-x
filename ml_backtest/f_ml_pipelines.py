"""
ML pipelines for classification-based strategies.

Provides Standardised Logistic Regression wrapped with sigmoid calibration.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def log_reg():
    """Create a calibrated classifier by name.

    Returns:
        A scikit-learn estimator supporting `fit` and `predict_proba`.
    """
    base = Pipeline([("scaler", StandardScaler(with_mean=True)),
                         ("clf", LogisticRegression(max_iter=1000))])
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return model

def fit_predict_proba(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    """Fit model on (X_train, y_train) and return class-1 probabilities for X_test."""
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:,1]

