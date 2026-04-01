"""Module for MaxVol algorithms."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from math import comb

import numpy as np
from scipy.linalg import qr

from motep.setting import DataclassFromAny

logger = logging.getLogger(__name__)


def _init_maxvol_first(matrix: np.ndarray) -> np.ndarray:
    return np.arange(matrix.shape[1])


def _init_maxvol_last(matrix: np.ndarray) -> np.ndarray:
    return np.arange(matrix.shape[0] - matrix.shape[1], matrix.shape[0])


def _init_maxvol_qr(matrix: np.ndarray) -> np.ndarray:
    return qr(matrix.T, pivoting=True)[-1][: matrix.shape[1]]


def _init_maxvol_random(matrix: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(matrix.shape[0], matrix.shape[1], replace=False)


class InitMethod(StrEnum):
    """Initialization method for the indices for the MaxVol calculation."""

    FIRST = "first"
    LAST = "last"
    QR = "qr"
    RANDOM = "random"


_INIT_METHODS: dict[InitMethod, Callable] = {
    InitMethod.FIRST: _init_maxvol_first,
    InitMethod.LAST: _init_maxvol_last,
    InitMethod.RANDOM: _init_maxvol_random,
    InitMethod.QR: _init_maxvol_qr,
}


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        msg = "matrix must be 2-dimensional"
        raise ValueError(msg)
    nrows, ncols = matrix.shape
    if nrows < ncols:
        msg = "matrix must satisfy nrows >= ncols"
        raise ValueError(msg)


def _validate_indices(matrix: np.ndarray, indices: np.ndarray) -> None:
    if indices.ndim != 1:
        msg = "indices must be 1-dimensional"
        raise ValueError(msg)
    nrows, ncols = matrix.shape
    if indices.size != ncols:
        msg = "indices length must be ncols"
        raise ValueError(msg)
    if len(np.unique(indices)) != indices.size:
        msg = "indices must be unique"
        raise ValueError(msg)
    if np.any(indices < 0) or np.any(indices >= nrows):
        msg = "indices must be 0 <= indices < nrows"
        raise ValueError(msg)


def _exhaust(matrix: np.ndarray) -> np.ndarray:
    """Find the MaxVol indices exhaustively.

    Returns
    -------
    indices : np.ndarray
        MaxVol indices.

    Raises
    ------
    RuntimeError

    """
    _validate_matrix(matrix)
    nrows, ncols = matrix.shape

    # Choose rows (configurations)
    # This is preliminarily implemented only in an exhausive manner.
    # This is therefore valid so far only for a small `configurations.initlal`
    # and for a low level `potentials.final`.
    if comb(nrows, ncols) > 2**24:  # 16777216
        msg = "too large possible combinations of rows"
        raise RuntimeError(msg, comb(nrows, ncols))
    slogdet_max = -np.inf
    indices = np.arange(ncols)
    for _ in combinations(range(nrows), ncols):
        indices_checked = np.array(_, dtype=int)
        submatrix = matrix[indices_checked]
        sign, slogdet = np.linalg.slogdet(submatrix)  # for numerical stability
        if sign == 0.0:
            continue
        if slogdet > slogdet_max:
            indices = indices_checked
            slogdet_max = slogdet

    return indices


def _maxvol(
    matrix: np.ndarray,
    indices: np.ndarray,
    *,
    threshold: float = 1e-9,
    maxiter: int = 100_100,
) -> np.ndarray:
    """Find the MaxVol indices.

    Returns
    -------
    indices : np.ndarray
        MaxVol indices.

    """
    _validate_matrix(matrix)
    _validate_indices(matrix, indices)
    nrows, ncols = matrix.shape
    selected = np.array(indices, dtype=int, copy=True)
    in_selected = np.zeros(nrows, dtype=bool)
    in_selected[selected] = True

    c = _calc_c(matrix, selected)
    for _ in range(maxiter):
        i, j = np.divmod(np.argmax(np.abs(c)), ncols)
        cmax = np.abs(c[i, j])
        if cmax - 1.0 < threshold:
            break
        if in_selected[i]:
            break
        k = selected[j]
        in_selected[k] = False
        in_selected[i] = True
        selected[j] = i
        _update_c(c, i, j)
    else:
        msg = (
            f"Maxvol algorithm did not converge within {maxiter} iterations. "
            f"Current c-max: {cmax}"
        )
        logger.warning(msg)

    return selected


def _calc_c(matrix: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Calculate c explicitly based on c @ matrix = matrix[selected].

    Returns
    -------
    c: np.ndarray

    """
    return np.linalg.lstsq(matrix[selected].T, matrix.T, rcond=None)[0].T


def _update_c(c: np.ndarray, i: np.int_, j: np.int_) -> None:
    """Modify c based on the rank-1 update.

    https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    """
    row = c[i, :].copy()
    row[j] -= 1.0
    col = c[:, j].copy()
    c -= np.outer(col, row) / c[i, j]


class FindMethod(StrEnum):
    """Finding method for the indices for he MaxVol calculation."""

    EXHAUST = "exhaust"
    MAXVOL = "maxvol"


@dataclass
class MaxVolSetting(DataclassFromAny):
    """MaxVol setting."""

    algorithm: FindMethod = FindMethod.MAXVOL
    init_method: InitMethod = InitMethod.QR
    threshold: float = 1e-9
    maxiter: int = 100_000


@dataclass
class MaxVol:
    """MaxVol algorithm."""

    algorithm: FindMethod = FindMethod.MAXVOL
    init_method: InitMethod = InitMethod.QR
    rng: np.random.Generator | None = None
    init_fn: Callable[..., np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        """Set up the initialization method.

        Raises
        ------
        ValueError

        """
        try:
            self.init_fn = _INIT_METHODS[self.init_method]
        except KeyError as err:
            msg = f"Unknown init method: {self.init_method}"
            raise ValueError(msg) from err

    def run(
        self,
        matrix: np.ndarray,
        *,
        threshold: float = 1e-9,
        maxiter: int = 100_100,
    ) -> np.ndarray:
        """Find the indices for the MaxVol calculation.

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError

        """
        _validate_matrix(matrix)
        if self.algorithm == FindMethod.EXHAUST:
            return _exhaust(matrix)
        if self.algorithm == FindMethod.MAXVOL:
            if self.init_method == InitMethod.RANDOM:
                indices = self.init_fn(matrix, rng=self.rng)
            else:
                indices = self.init_fn(matrix)
            return _maxvol(matrix, indices, threshold=threshold, maxiter=maxiter)
        raise ValueError(self.algorithm)
