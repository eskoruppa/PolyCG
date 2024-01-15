import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .SO3 import so3
from .pyConDec.pycondec import cond_jit






##########################################################################################################
##########################################################################################################
############### Change Splitting between static and dynamic components ###################################
##########################################################################################################
##########################################################################################################


def splittransform_group2algebra(Theta_groundstate: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) to lie algebra splitting
    representation R = exp(hat(Theta_0) + hat(Delta')). Linear transformation T transforms Delta
    into Delta' as T*Delta = Delta'.

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians. (3N array expected)

    Returns:
        float: Linear transformation matrix T (3Nx3N) that transforms Delta into Delta': T*Delta = Delta'
    """
    N = len(Theta_groundstate)
    T = np.zeros((N,) * 2)
    for i in range(N // 3):
        T0 = Theta_groundstate[i * 3 : (i + 1) * 3]
        T[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = so3.splittransform_group2algebra(
            T0
        )
    return T


def splittransform_algebra2group(Theta_groundstate: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in lie algebra splitting representation R = exp(hat(Theta_0) + hat(Delta')) to group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) t. Linear transformation T transforms Delta
    into Delta' as T'*Delta' = Delta. Currently this is defined as the inverse of the transformation
    defined in the method splittransform_group2algebra

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians. (3N array expected)

    Returns:
        float: Linear transformation matrix T' (3Nx3N) that transforms Delta into Delta': T'*Delta' = Delta
    """
    N = len(Theta_groundstate)
    T = np.zeros((N,) * 2)
    for i in range(N // 3):
        T0 = Theta_groundstate[i * 3 : (i + 1) * 3]
        T[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = so3.splittransform_algebra2group(
            T0
        )
    return T


##########################################################################################################
##########################################################################################################
############### Unit Conversions #########################################################################
##########################################################################################################
##########################################################################################################


def fifth2rad(val: Any) -> float:
    """
    convert single value from fifth radians into radians
    """
    return val / 5


def fifth2deg(val: Any) -> float:
    """
    convert single value from fifth radians into degrees
    """
    return val * 180 / np.pi / 5


def gs2rad(gs: np.ndarray, only_rot=False) -> np.ndarray:
    """
    convert ground state vector from fifth radians into radians. If only_rot is True it will assume that all values are rotations.
    Otherwise it is assumed that each block has 6 entries
    """
    return _gsconf(fifth2rad, gs, only_rot=only_rot)


def gs2deg(gs: np.ndarray, only_rot=False) -> np.ndarray:
    """
    convert ground state vector from fifth radians into degrees. If only_rot is True it will assume that all values are rotations.
    Otherwise it is assumed that each block has 6 entries
    """
    return _gsconf(fifth2deg, gs, only_rot=only_rot)


def _gsconf(convfunc: Callable, gs: np.ndarray, only_rot=False):
    gs = np.copy(gs)
    if not only_rot:
        if len(gs) % 6 != 0:
            raise ValueError(
                f"Unexpected dimension of gs. Expecting multiple of 6, but received {len(gs)} (len(gs)%6={len(gs)%6})."
            )
        N = len(gs) // 6
        for i in range(N):
            gs[6 * i : 6 * i + 3] = convfunc(gs[6 * i : 6 * i + 3])
        return gs
    return convfunc(gs)


def stiff2rad(stiff: np.ndarray, only_rot=False) -> np.ndarray:
    """
    convert stiffness matrix from fifth radians into radians. If only_rot is True it will assume that all values are rotations.
    Otherwise it is assumed that each block has 6 entries
    """
    return _stiffconf(fifth2rad(1), stiff, only_rot=only_rot)


def stiff2deg(stiff: np.ndarray, only_rot=False) -> np.ndarray:
    """
    convert stiffness matrix from fifth radians into degrees. If only_rot is True it will assume that all values are rotations.
    Otherwise it is assumed that each block has 6 entries
    """
    return _stiffconf(fifth2deg(1), stiff, only_rot=only_rot)


def _stiffconf(fac: float, stiff: np.ndarray, only_rot=False):
    stiff = np.copy(stiff)
    if not only_rot:
        if len(stiff) % 6 != 0:
            raise ValueError(
                f"Unexpected dimension of gs. Expecting multiple of 6, but received {len(stiff)} (len(gs)%6={len(stiff)%6})."
            )
        N = len(stiff) // 6
        for i in range(N):
            stiff[6 * i : 6 * i + 3, :] /= fac
            stiff[:, 6 * i : 6 * i + 3] /= fac
        return stiff
    return stiff / fac**2


##########################################################################################################
##########################################################################################################
############### Convert stiffness and groundstate between different definitions of rotation DOFS #########
##########################################################################################################
##########################################################################################################


def rotbps_algebra2group(
    gs_euler: np.ndarray, stiff_euler: np.ndarray
) -> np.ndarray:
    """Converts stiffness matrix for fluctuations split in so3 to fluctuations split in SO3.

    Args:
        gs (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix for fluctuations split in so3 using the Euler map definition for vector components. Expressed in arbitrary units but e

    Returns:
        Tuple[np.ndarray,np.ndarray]: Stiffness matrix for fluctuations split in SO3.
    """
    Ta2g_full = splittransform_algebra2group(gs_euler)
    Ta2g_full_inv = np.linalg.inv(Ta2g_full)
    # stiff_euler_group = np.matmul(Ta2g_full_inv.T,np.matmul( stiff_euler, Ta2g_full_inv))
    stiff_euler_group = Ta2g_full_inv.T @ stiff_euler @ Ta2g_full_inv
    return stiff_euler_group


def rotbps_group2algebra(
    gs_euler: np.ndarray, stiff_euler: np.ndarray
) -> np.ndarray:
    """Converts stiffness matrix for fluctuations split in SO3 to fluctuations split in so3.

    Args:
        gs (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix for fluctuations split in SO3 using the Euler map definition for vector components. Expressed in arbitrary units but e

    Returns:
        Tuple[np.ndarray,np.ndarray]: Stiffness matrix for fluctuations split in so3.
    """
    Ta2g_full = splittransform_group2algebra(gs_euler)
    Ta2g_full_inv = np.linalg.inv(Ta2g_full)
    # stiff_euler_algebra = np.matmul(Ta2g_full_inv.T,np.matmul( stiff_euler, Ta2g_full_inv))
    stiff_euler_algebra = Ta2g_full_inv.T @ stiff_euler @ Ta2g_full_inv
    return stiff_euler_algebra