"""
Monte Carlo permutation test utilities for performance statistics.

This module estimates a p-value for an observed Sharpe ratio under the null
hypothesis that returns are randomly ordered. To partially preserve short-term
dependence, it supports block permutation where contiguous blocks of returns are
shuffled rather than individual observations.

Key functions:
- `block_permute`: Shuffle an array by blocks to retain within-block structure.
- `mc_permutation_pvalue`: Compute a p-value that a randomised (block-permuted)
  return series achieves a Sharpe at least as large as the observed one.
"""

import numpy as np
import pandas as pd
from .c_metrics import sharpe

def block_permute(arr, block, rng):
    """Shuffle a 1-D array by contiguous blocks of size `block`.

    If `block` is 1 or less, performs a full random permutation of the array.
    Otherwise, the array is partitioned into contiguous blocks; if its length is
    not divisible by `block`, the array is padded by repeating the tail elements
    so that the final reshape is valid. Blocks are then shuffled and flattened,
    and the result is truncated back to the original length.

    Args:
        arr (np.ndarray): One-dimensional array to permute.
        block (int): Block length for shuffling; use >1 to preserve local order.
        rng (np.random.Generator): Random number generator to use.

    Returns:
        np.ndarray: Block-permuted array of the same length as `arr`.
    """
    n = len(arr)
    if block <= 1:
        return rng.permutation(arr)
    pad = (-n) % block
    if pad > 0:
        arr = np.concatenate([arr, arr[-pad:]])
    blocks = arr.reshape(-1, block)
    rng.shuffle(blocks, axis=0)
    return blocks.reshape(-1)[:n]

def mc_permutation_pvalue(oos_returns: pd.Series, n_iter: int = 2000, block: int = 5, seed: int = 0) -> float:
    """Estimate a Monte Carlo permutation p-value for the Sharpe ratio.

    The observed Sharpe of `oos_returns` is compared to the distribution of
    Sharpes computed from `n_iter` block-permuted versions of the series. The
    returned p-value is the proportion of permutations with Sharpe >= observed,
    using a standard +1 pseudo-count in numerator/denominator for stability.

    Args:
        oos_returns (pd.Series): Out-of-sample log returns (or pnl per period).
        n_iter (int): Number of Monte Carlo permutations to run.
        block (int): Block size for `block_permute`; set to 1 for iid shuffle.
        seed (int): Seed for the NumPy RNG used in permutations.

    Returns:
        float: Estimated p-value in [0, 1].
    """
    rng = np.random.default_rng(seed)
    base = sharpe(oos_returns)
    arr = oos_returns.values.copy()
    cnt = 0
    for _ in range(n_iter):
        perm = block_permute(arr, block, rng)
        s = sharpe(pd.Series(perm, index=oos_returns.index))
        if s >= base:
            cnt += 1
    p = (cnt + 1) / (n_iter + 1)
    return float(p)
