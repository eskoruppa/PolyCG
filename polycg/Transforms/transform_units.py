import numpy as np
from typing import List, Tuple, Callable, Any, Dict

##########################################################################################################
##########################################################################################################
############### Unit Conversions #########################################################################
##########################################################################################################
##########################################################################################################

def conversion(array: np.ndarray, factor: float, block_dim: int = 6, dofs: List[int] = None):
    dims = len(array.shape)
    # check if valid shape
    if dims > 2:
        raise ValueError(f'Conversion only supports vectors and matrices. Encountered array of shape {array.shape}.')
    # if everthing is to be converted
    carray = np.copy(array)
    if dofs is None:
        for i in range(dims):
            carray *= factor
        return carray
    # reduce to be converted degrees of freedom    
    dofs = sorted(list(set([dof%block_dim for dof in dofs])))
    for dof in dofs:
        carray[dof::block_dim] *= factor
        if dims == 2:
            carray[:,dof::block_dim] *= factor
    return carray
            
    
# def fifth2rad(val: Any) -> float:
#     """
#     convert single value from fifth radians into radians
#     """
#     return val / 5


# def fifth2deg(val: Any) -> float:
#     """
#     convert single value from fifth radians into degrees
#     """
#     return val * 180 / np.pi / 5


# def gs2rad(gs: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert ground state vector from fifth radians into radians. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _gsconf(fifth2rad, gs, only_rot=only_rot)


# def gs2deg(gs: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert ground state vector from fifth radians into degrees. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _gsconf(fifth2deg, gs, only_rot=only_rot)


# def _gsconf(convfunc: Callable, gs: np.ndarray, only_rot=False):
#     gs = np.copy(gs)
#     if not only_rot:
#         if len(gs) % 6 != 0:
#             raise ValueError(
#                 f"Unexpected dimension of gs. Expecting multiple of 6, but received {len(gs)} (len(gs)%6={len(gs)%6})."
#             )
#         N = len(gs) // 6
#         for i in range(N):
#             gs[6 * i : 6 * i + 3] = convfunc(gs[6 * i : 6 * i + 3])
#         return gs
#     return convfunc(gs)


# def stiff2rad(stiff: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert stiffness matrix from fifth radians into radians. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _stiffconf(fifth2rad(1), stiff, only_rot=only_rot)


# def stiff2deg(stiff: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert stiffness matrix from fifth radians into degrees. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _stiffconf(fifth2deg(1), stiff, only_rot=only_rot)


# def _stiffconf(fac: float, stiff: np.ndarray, only_rot=False):
#     stiff = np.copy(stiff)
#     if not only_rot:
#         if len(stiff) % 6 != 0:
#             raise ValueError(
#                 f"Unexpected dimension of gs. Expecting multiple of 6, but received {len(stiff)} (len(gs)%6={len(stiff)%6})."
#             )
#         N = len(stiff) // 6
#         for i in range(N):
#             stiff[6 * i : 6 * i + 3, :] /= fac
#             stiff[:, 6 * i : 6 * i + 3] /= fac
#         return stiff
#     return stiff / fac**2