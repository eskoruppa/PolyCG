"""
Stiffness-matrix rescaling transforms.

These helpers rescale selected degrees of freedom (DOFs) of a block-structured
stiffness matrix via a congruence (similarity) transform

    K -> D K D

where ``D`` is a diagonal matrix carrying ``sqrt(factor)`` on the entries of the
rescaled DOFs and unity elsewhere. As a consequence:

- diagonal stiffness entries of a rescaled DOF are multiplied by ``factor``,
- couplings between a rescaled and an unscaled DOF are multiplied by
  ``sqrt(factor)``,
- couplings between two rescaled DOFs are multiplied by ``factor``,
- all other entries are unchanged.

The matrix is assumed to be block-structured with ``ndims`` degrees of freedom
per site (``ndims=6`` for SE(3): the rotational and translational components of
each base-pair step), so DOF ``k`` lives on indices ``k, k+ndims, k+2*ndims, ...``.

Rescaling only affects the stiffness (the curvature of the energy around the
ground state); it leaves the ground state itself unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy as sp
from scipy.sparse import spmatrix


__all__ = ["rescale_kth", "rescale_stiff", "rescale_stiff_dofs"]


def _check_matrix(stiff: np.ndarray | spmatrix, ndims: int) -> int:
    """Validate that ``stiff`` is a square 2D matrix whose size is a multiple of ``ndims``."""
    if not hasattr(stiff, "shape") or len(stiff.shape) != 2:
        raise TypeError("stiff must be a 2D numpy array or a scipy sparse matrix.")
    n, m = stiff.shape
    if n != m:
        raise ValueError(f"stiff must be square, got shape {stiff.shape}.")
    if n == 0:
        raise ValueError("stiff must be non-empty.")
    if n % ndims != 0:
        raise ValueError(f"stiff dimension must be a multiple of {ndims}, got {n}.")
    return n


def _check_factor(factor: float) -> float:
    """Validate and coerce a single positive, finite rescaling factor."""
    if not isinstance(factor, (int, float, np.number)):
        raise TypeError("factor must be a real number.")
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("factor must be finite (not inf/NaN).")
    if factor <= 0:
        raise ValueError(f"factor must be a positive number, encountered factor = {factor}")
    return factor


def _apply_congruence(stiff: np.ndarray | spmatrix, d: np.ndarray) -> np.ndarray | spmatrix:
    """Return ``D @ stiff @ D`` for the diagonal ``D = diag(d)``, preserving the input type."""
    if sp.sparse.issparse(stiff):
        fmt = getattr(stiff, "format", "csr") or "csr"
        D = sp.sparse.diags(d, format=fmt)
        return D @ stiff @ D
    stiff = np.asarray(stiff)
    return (d[:, None] * stiff) * d[None, :]


def rescale_kth(
    stiff: np.ndarray | spmatrix,
    k: int,
    factor: float,
    ndims: int = 6,
) -> np.ndarray | spmatrix:
    """
    Rescale the ``k``-th degree of freedom of every site by ``factor``.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix whose dimension is a multiple of ``ndims``.
    k : int
        Degree-of-freedom index in the range ``0 .. ndims-1``.
    factor : float
        Positive rescaling factor for the diagonal stiffness entries of the DOF.
    ndims : int, default=6
        Number of degrees of freedom per site.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix (same type as input).
    """
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an int.")
    k = int(k)
    if not (0 <= k < ndims):
        raise ValueError(f"k must be an integer in the range 0..{ndims - 1}.")
    factor = _check_factor(factor)
    n = _check_matrix(stiff, ndims)

    d = np.ones(n, dtype=float)
    d[k::ndims] = np.sqrt(factor)
    return _apply_congruence(stiff, d)


def rescale_stiff(
    stiff: np.ndarray | spmatrix,
    factor: float,
    entries: Sequence[int] | None = None,
    ndims: int = 6,
) -> np.ndarray | spmatrix:
    """
    Rescale selected degrees of freedom by a common ``factor``.

    If ``entries`` is None or empty, the whole matrix is multiplied by ``factor``.
    Otherwise the listed DOFs (each in ``0 .. ndims-1``) are rescaled.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix whose dimension is a multiple of ``ndims``.
    factor : float
        Positive rescaling factor.
    entries : sequence of int or None, optional
        Degree-of-freedom indices to rescale. None/empty rescales all uniformly.
    ndims : int, default=6
        Number of degrees of freedom per site.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix (same type as input).
    """
    factor = _check_factor(factor)
    n = _check_matrix(stiff, ndims)

    if entries is None or len(entries) == 0:
        return stiff * factor

    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("entries must be a sequence of ints (e.g., [0, 3, 5]) or None.")

    cleaned: list[int] = []
    for i, k in enumerate(entries):
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"entries[{i}] must be an int, got {type(k)}.")
        k = int(k)
        if not (0 <= k < ndims):
            raise ValueError(f"entries[{i}] must be in range 0..{ndims - 1}, got {k}.")
        cleaned.append(k)

    sqfac = np.sqrt(factor)
    d = np.ones(n, dtype=float)
    for k in sorted(set(cleaned)):
        d[k::ndims] = sqfac
    return _apply_congruence(stiff, d)


def rescale_stiff_dofs(
    stiff: np.ndarray | spmatrix,
    factors: Sequence[float],
    ndims: int = 6,
) -> np.ndarray | spmatrix:
    """
    Rescale each degree of freedom by its own factor in a single congruence transform.

    This is the per-DOF generalisation of :func:`rescale_stiff`: ``factors`` is a
    length-``ndims`` vector giving an independent positive factor for each DOF.
    Building the diagonal once and applying ``D K D`` a single time is more
    efficient than rescaling DOFs one at a time.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix whose dimension is a multiple of ``ndims``.
    factors : sequence of float
        Length-``ndims`` vector of positive per-DOF rescaling factors. A factor of
        1.0 leaves the corresponding DOF unchanged.
    ndims : int, default=6
        Number of degrees of freedom per site.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix (same type as input).
    """
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 1 or len(factors) != ndims:
        raise ValueError(
            f"factors must be a 1D sequence of length {ndims}, got shape {factors.shape}."
        )
    if not np.all(np.isfinite(factors)):
        raise ValueError("all factors must be finite (not inf/NaN).")
    if np.any(factors <= 0):
        raise ValueError(f"all factors must be positive, got {factors}.")

    n = _check_matrix(stiff, ndims)

    d = np.ones(n, dtype=float)
    for k in range(ndims):
        d[k::ndims] = np.sqrt(factors[k])
    return _apply_congruence(stiff, d)
