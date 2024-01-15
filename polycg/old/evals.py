import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from .SO3 import so3


def covariance_matrix(vals: np.ndarray, gs=None, subtract_gs=True) -> np.ndarray:
    """Calculate the covariance matrix

    Args:
        vals (np.ndarray): Values (M,N) where M is the number of snapshots and N the number of degrees of freedom.
        gs (_type_, optional): Groundstate that is to be subtracted. Defaults to None, in which case the groundstate is estimated by the mean (if subtract_gs is True)
        subtract_T0 (bool, optional): Set whether or not to subtract the groundstate. Defaults to True.

    Returns:
        np.ndarray: Returns covariance matrix
    """
    if subtract_gs:
        if gs is None:
            gs = np.mean(vals, axis=0)
        vals = vals - gs
    cov = np.zeros((len(vals[0]),) * 2)
    for val in vals:
        cov += np.outer(val, val)
    cov /= len(vals)
    return cov
