"""
Runs test for independence of return signs.

Computes the Wald-Wolfowitz runs test on the sign of a return series, returning
the z-score and a two-sided p-value. If all signs are the same, results are NaN.
"""

import numpy as np
import pandas as pd
from math import erf, sqrt

def runs_test(returns: pd.Series):
    """Wald-Wolfowitz runs test on return signs.

    Args:
        returns (pd.Series): Per-period returns (sign is used).

    Returns:
        Tuple[float, float]: (z-score, two-sided p-value). NaN if all signs equal.
    """
    x = (returns > 0).astype(int).values
    n1 = x.sum()
    n2 = len(x) - n1
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    R = 1 + (x[1:] != x[:-1]).sum()
    mu = 1 + 2*n1*n2/(n1+n2)
    var = (2*n1*n2 * (2*n1*n2 - n1 - n2)) / (((n1+n2)**2) * (n1+n2 - 1))
    z = (R - mu) / np.sqrt(var) if var > 0 else float("nan")
    p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2)))) if np.isfinite(z) else float("nan")
    return float(z), float(p)
