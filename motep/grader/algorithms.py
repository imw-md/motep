"""Module for algorithms."""

import logging
from itertools import combinations
from math import comb

import numpy as np

logger = logging.getLogger(__name__)


def find_active_set_exhaustive(matrix: np.ndarray) -> np.ndarray:
    """Find the active set.

    Returns
    -------
    indices : np.ndarray
        Indices for the active set.

    Raises
    ------
    RuntimeError

    """
    nrows, ncols = matrix.shape

    # Choose rows (configurations)
    # This is preliminarily implemented only in an exhausive manner.
    # This is therefore valid so far only for a small `data_in`
    # and for a low level `potential_final`.
    if comb(nrows, ncols) > 2**24:  # 16777216
        msg = "too large possible combinations of rows"
        raise RuntimeError(msg, comb(nrows, ncols))
    det_max = 0.0
    indices = np.arange(nrows)
    for _ in combinations(range(nrows), ncols):
        indices_checked = np.array(_)
        submatrix = matrix[indices_checked]
        det = np.abs(np.linalg.det(submatrix))
        if det > det_max:
            indices = indices_checked
            det_max = det

    return indices


def find_active_set_maxvol(
    matrix: np.ndarray,
    rng: np.random.Generator,
    *,
    maxiter: int = 100_100,
) -> np.ndarray:
    """Find the active set.

    Returns
    -------
    indices : np.ndarray
        Indices for the active set.

    """
    nrows, ncols = matrix.shape

    indices = rng.choice(nrows, ncols, replace=False)
    flags = np.zeros(nrows, dtype=bool)
    flags[indices] = True

    tolerance = 1e-9
    for _ in range(maxiter):
        submatrix = matrix[flags]
        c = np.linalg.lstsq(submatrix.T, matrix.T, rcond=None)[0].T
        i, j = np.divmod(np.argmax(np.abs(c)), ncols)
        if np.abs(c[i, j]) < 1.0 + tolerance:
            break
        k = np.where(flags)[0][j]  # row/column in the original matrix
        flags[[k, i]] = flags[[i, k]]
    else:
        cmax = np.abs(c[i, j])
        msg = (
            f"Maxvol algorithm did not converge within {_} iterations. "
            f"Current c-max: {cmax}"
        )
        logger.warning(msg)

    return np.where(flags)[0]
