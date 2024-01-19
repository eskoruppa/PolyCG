import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .SO3 import so3
from .pyConDec.pycondec import cond_jit


##########################################################################################################
############### Conversion between Euler vectors and rotation matrices ###################################
##########################################################################################################


def eulers2rotmats(eulers: np.ndarray) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    if eulers.shape[-1] != 3:
        raise ValueError(f"Expected set of 3-vectors. Instead received shape {eulers.shape}")
    rotmats = np.zeros(eulers.shape + (3,))
    if len(eulers.shape) > 2:
        for i in range(len(eulers)):
            rotmats[i] = eulers2rotmats(eulers[i])
        return rotmats
    for i, euler in enumerate(eulers):
        rotmats[i] = so3.euler2rotmat(euler)
    return rotmats


def rotmats2eulers(rotmats: np.ndarray) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectrs (...,N,3)
    """
    eulers = np.zeros(rotmats.shape[:-1])
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            eulers[i] = rotmats2eulers(rotmats[i])
        return eulers
    for i, rotmat in enumerate(rotmats):
        eulers[i] = so3.rotmat2euler(rotmat)
    return eulers


# def fluctrotmats2rotmats_euler(eulers_gs: np.ndarray, drotmats: np.ndarray, static_left=True):
#     """Converts collection of fluctuating component rotation matrices into collection of full rotation matrices (including static components)

#     Converts groundstate into set of static rotation matrices {S}. Then generates rotation matrices as
#     R = S*D
#     (for static_left = True, with D the indicidual fluctuating component rotation matrices)

#     Args:
#         eulers_gs (np.ndarray): Groundstate euler vectors (dim: N,3)
#         drotmats (np.ndarray): Collection of fluctuating component rotation matrices (dim: ...,N,3,3)
#         static_left (bool): Specifies whether static component of the rotation matrix is defined to be on the left or right.
#                             Defaults to True (left definition).

#     Returns:
#         np.ndarray: collection of rotation matrices (...,N,3,3)
#     """

#     def _dynamicrotmats2rotmats(
#         eulers_gs_rotmat: np.ndarray, drotmats: np.ndarray
#     ) -> np.ndarray:
#         rotmats = np.zeros(drotmats.shape)
#         if len(drotmats.shape) > 3:
#             for i in range(len(drotmats)):
#                 rotmats[i] = _dynamicrotmats2rotmats(eulers_gs_rotmat, drotmats[i])
#             return rotmats

#         if static_left:
#             # left multiplication of static rotation matrix
#             for i, drotmat in enumerate(drotmats):
#                 rotmats[i] = np.matmul(eulers_gs_rotmat[i], drotmat)
#         else:
#             # right multiplication of static rotation matrix
#             for i, drotmat in enumerate(drotmats):
#                 rotmats[i] = np.matmul(drotmat, eulers_gs_rotmat[i])
#         return rotmats

#     eulers_gs_rotmat = eulers2rotmats(eulers_gs)
#     return _dynamicrotmats2rotmats(eulers_gs_rotmat, drotmats)


# def rotmats2fluctrotmats_euler(eulers_gs: np.ndarray, rotmats: np.ndarray, static_left=True):
#     """Extracts dynamic component rotation matrices from collection of full rotation matrices (including static components)

#     Args:
#         eulers_gs (np.ndarray): Groundstate euler vectors (dim: N,3)
#         rotmats (np.ndarray): Collection of rotation matrices (dim: ...,N,3,3)
#         static_left (bool): Specifies whether static component of the rotation matrix is defined to be on the left or right.
#                             Defaults to True (left definition).

#     Returns:
#         np.ndarray: collection of dynamic component rotation matrices (...,N,3,3)
#     """

#     def _rotmats2fluctrotmats(
#         eulers_gs_rotmat: np.ndarray, rotmats: np.ndarray
#     ) -> np.ndarray:
#         drotmats = np.zeros(rotmats.shape)
#         if len(rotmats.shape) > 3:
#             for i in range(len(rotmats)):
#                 rotmats[i] = _rotmats2fluctrotmats(eulers_gs_rotmat, rotmats[i])
#             return rotmats

#         if static_left:
#             # left multiplication of static rotation matrix
#             for i, rotmat in enumerate(rotmats):
#                 drotmats[i] = np.matmul(eulers_gs_rotmat[i].T, rotmat)
#         else:
#             # right multiplication of static rotation matrix
#             for i, rotmat in enumerate(rotmats):
#                 drotmats[i] = np.matmul(rotmat, eulers_gs_rotmat[i].T)
#         return drotmats

#     eulers_gs_rotmat = eulers2rotmats(eulers_gs)
#     return _rotmats2fluctrotmats(eulers_gs_rotmat, rotmats)


# def eulers2rotmats_SO3fluct(
#     eulers_gs: np.ndarray, eulers_fluct: np.ndarray, static_left=True
# ) -> np.ndarray:
#     """Converts configuration of euler vectors into collection of rotation matrices

#     Args:
#         eulers_gs (np.ndarray): Groundstate euler vectors (N,3)
#         eulers_fluct (np.ndarray): Collection of fluctuating component euler vectors (...,N,3)
#         static_left (bool): Specifies whether static component of the rotation matrix is defined to be on the left or right.
#                             Defaults to True (left definition).

#     Returns:
#         np.ndarray: collection of rotation matrices (...,N,3,3)
#     """

#     def _eulers2rotmats_SO3fluct(
#         eulers_gs_rotmat: np.ndarray, eulers_fluct: np.ndarray
#     ) -> np.ndarray:
#         rotmats = np.zeros(eulers_fluct.shape + (3,))
#         if len(eulers_fluct.shape) > 2:
#             for i in range(len(eulers_fluct)):
#                 rotmats[i] = _eulers2rotmats_SO3fluct(eulers_gs_rotmat, eulers_fluct[i])
#             return rotmats

#         if static_left:
#             # left multiplication of static rotation matrix
#             for i, euler in enumerate(eulers_fluct):
#                 rotmats[i] = np.matmul(eulers_gs_rotmat[i], so3.euler2rotmat(euler))
#         else:
#             # right multiplication of static rotation matrix
#             for i, euler in enumerate(eulers_fluct):
#                 rotmats[i] = np.matmul(so3.euler2rotmat(euler), eulers_gs_rotmat[i])
#         return rotmats

#     eulers_gs_rotmat = eulers2rotmats(eulers_gs)
#     return _eulers2rotmats_SO3fluct(eulers_gs_rotmat, eulers_fluct)


# # def rotmats2eulers_SO3fluct(eulers_gs: np.ndarray, rotmats: np.ndarray) -> np.ndarray:
# #     """Converts configuration of rotation matrices into collection of fluctuating components of rotation matrices

# #     Args:
# #         eulers_gs (np.ndarray): Groundstate euler vectors (dim: (N,3))
# #         rotmats (np.ndarray): collection of rotation matrices (dim: (...,N,3,3))
# #         static_left (bool): Specifies whether static component of the rotation matrix is defined to be on the left or right.
# #                             Defaults to True (left definition).

# #     Returns:
# #         np.ndarray: collection of fluctuating components of  (dim: (...,N,3,3))
# #     """


##########################################################################################################
############### Conversion between Cayley vectors and rotation matrices ###################################
##########################################################################################################


def cayleys2rotmats(cayleys: np.ndarray) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    if cayleys.shape[-1] != 3:
        raise ValueError(f"Expected set of 3-vectors. Instead received shape {cayleys.shape}")
    
    rotmats = np.zeros(cayleys.shape + (3,))
    if len(cayleys.shape) > 2:
        for i in range(len(cayleys)):
            rotmats[i] = cayleys2rotmats(cayleys[i])
        return rotmats
    for i, cayley in enumerate(cayleys):
        rotmats[i] = so3.cayley2rotmat(cayley)
    return rotmats


def rotmats2cayleys(rotmats: np.ndarray) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectors (...,N,3)
    """
    cayleys = np.zeros(rotmats.shape[:-1])
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            cayleys[i] = rotmats2cayleys(rotmats[i])
        return cayleys
    for i, rotmat in enumerate(rotmats):
        cayleys[i] = so3.rotmat2cayley(rotmat)
    return cayleys

##########################################################################################################
############### Conversion between vectors and rotation matrices #########################################
##########################################################################################################


def vecs2rotmats(vecs: np.ndarray, rotation_map: str = "euler") -> np.ndarray:
    """Converts configuration of vectors into collection of rotation matrices

    Args:
        vecs (np.ndarray): Collection of rotational vectors (...,N,3)
        rotation_map (str): selected map between rotation rotation coordinates and rotation matrix.
                Options:    - cayley: default cnDNA map (Cayley map)
                            - euler:  Axis angle representation.

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    if rotation_map == "euler":
        return eulers2rotmats(vecs)
    elif rotation_map == "cayley":
        return cayleys2rotmats(vecs)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


def rotmats2vecs(rotmats: np.ndarray, rotation_map: str = "euler") -> np.ndarray:
    """Converts collection of rotation matrices into collection of vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)
        rotation_map (str): selected map between rotation rotation coordinates and rotation matrix.
                Options:    - cayley: default cnDNA map (Cayley map)
                            - euler:  Axis angle representation.

    Returns:
        np.ndarray: Collection of vectors (...,N,3)
    """
    if rotation_map == "euler":
        return rotmats2eulers(rotmats)
    elif rotation_map == "cayley":
        return rotmats2cayleys(rotmats)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


##########################################################################################################
############### Conversion between rotation matrices and triads ##########################################
##########################################################################################################


def rotmats2triads(rotmats: np.ndarray, first_triad=None) -> np.ndarray:
    """Converts collection of rotation matrices into collection of triads

    Args:
        rotmats (np.ndarray): set of rotation matrices that constitute the local junctions in the chain of triads. (...,N,3,3)
        first_triad (None or np.ndarray): rotation of first triad. Should be none or single triad. For now only supports identical rotation for all snapshots.

    Returns:
        np.ndarray: set of triads (...,N+1,3,3)
    """
    sh = list(rotmats.shape)
    sh[-3] += 1
    triads = np.zeros(tuple(sh))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            triads[i] = rotmats2triads(rotmats[i])
        return triads

    if first_triad is None:
        first_triad = np.eye(3)
    assert first_triad.shape == (
        3,
        3,
    ), f"invalid shape of triad {first_triad.shape}. Triad shape needs to be (3,3)."

    triads[0] = first_triad
    for i, rotmat in enumerate(rotmats):
        triads[i + 1] = np.matmul(triads[i], rotmat)
    return triads


def triads2rotmats(triads: np.ndarray) -> np.ndarray:
    """Converts set of triads into set of rotation matrices

    Args:
        triads (np.ndarray): set of triads (...,N+1,3,3)

    Returns:
        np.ndarray: set of rotation matrices (...,N,3,3)
    """
    sh = list(triads.shape)
    sh[-3] -= 1
    rotmats = np.zeros(tuple(sh))
    if len(triads.shape) > 3:
        for i in range(len(triads)):
            rotmats[i] = triads2rotmats(triads[i])
        return rotmats

    for i in range(len(triads) - 1):
        rotmats[i] = np.matmul(triads[i].T, triads[i + 1])
    return rotmats



##########################################################################################################
############### Generate positions from triads ###########################################################
##########################################################################################################

def triads2positions(triads: np.ndarray, disc_len=0.34) -> np.ndarray:
    """generates a set of position vectors from a set of triads

    Args:
        triads (np.ndarray): set of trads (...,N,3,3)
        disc_len (float): discretization length

    Returns:
        np.ndarray: set of position vectors (...,N,3)
    """
    pos = np.zeros(triads.shape[:-1])
    if len(triads.shape) > 3:
        for i in range(len(triads)):
            pos[i] = triads2positions(triads[i])
        return pos
    pos[0] = np.zeros(3)
    for i in range(len(triads) - 1):
        pos[i + 1] = pos[i] + triads[i, :, 2] * disc_len
    return pos