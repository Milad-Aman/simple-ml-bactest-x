import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fit_scaler_on_train(X_train: pd.DataFrame):
    scaler = StandardScaler()
    Xtr = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index, columns=X_train.columns
    )
    return scaler, Xtr

def transform_with_scaler(X: pd.DataFrame, scaler: StandardScaler):
    return pd.DataFrame(
        scaler.transform(X), index=X.index, columns=X.columns
    )

def orthogonalise_on_base(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    base_cols: list,
    target_cols: list | None = None,
    add_intercept: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Residualise target_cols on base_cols using TRAIN ONLY, apply to TEST.
    Returns transformed (residualised) X_train, X_test and dict of betas.
    """
    base_cols = [c for c in base_cols if c in X_train.columns]
    if not base_cols:
        return X_train, X_test, {}

    if target_cols is None:
        target_cols = [c for c in X_train.columns if c not in base_cols]
    else:
        target_cols = [c for c in target_cols if c in X_train.columns and c not in base_cols]

    Xtr = X_train.copy()
    Xte = X_test.copy()
    betas = {}

    # design matrices
    Ztr = Xtr[base_cols].values
    Zte = Xte[base_cols].values
    if add_intercept:
        Ztr = np.c_[np.ones(len(Xtr)), Ztr]
        Zte = np.c_[np.ones(len(Xte)), Zte]

    for col in target_cols:
        ytr = Xtr[col].values
        beta, *_ = np.linalg.lstsq(Ztr, ytr, rcond=None)
        betas[col] = beta
        # replace with residuals
        Xtr[col] = ytr - Ztr @ beta
        Xte[col] = Xte[col].values - Zte @ beta

    return Xtr, Xte, betas
