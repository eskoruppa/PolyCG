from __future__ import annotations

import numpy as np

from ..SO3 import so3
from ..pyConDec.pycondec import cond_jit


##########################################################################################################
############### Conversion between Euler vectors and rotation matrices ###################################
##########################################################################################################


def euler2rotmat_so3(
    eulers: np.ndarray  # shape (..., N, 3): Euler angles (axis-angle)
) -> np.ndarray:  # shape (..., N, 3, 3): rotation matrices
    """
    Convert SO(3) Euler vectors to 3x3 rotation matrices.
    
    Uses axis-angle (Euler) representation to generate rotation matrices.
    Recursively handles higher-dimensional arrays.
    
    Args:
        eulers: Euler vectors (3-component rotation vectors).
    
    Returns:
        3x3 rotation matrices.
    
    Raises:
        ValueError: If last dimension is not 3, or if 4-vectors provided (use SE(3) functions instead).
    """
    if eulers.shape[-1] != 3:
        if eulers.shape[-1] == 4:
            raise ValueError(f"Expected set of 3-vectors. Instead received set of 4-vectors. For the the corresponding transformation in se3 please use euler2rotmat or se3_euler2rotmat.")
        raise ValueError(f"Expected set of 3-vectors. Instead received shape {eulers.shape}")
    rotmats = np.zeros(eulers.shape + (3,))
    if len(eulers.shape) > 2:
        for i in range(len(eulers)):
            rotmats[i] = euler2rotmat_so3(eulers[i])
        return rotmats
    for i, euler in enumerate(eulers):
        rotmats[i] = so3.euler2rotmat(euler)
    return rotmats


def rotmat2euler_so3(
    rotmats: np.ndarray  # shape (..., N, 3, 3): rotation matrices
) -> np.ndarray:  # shape (..., N, 3): Euler angles
    """
    Convert SO(3) rotation matrices to Euler vectors.
    
    Extracts axis-angle (Euler) representation from 3x3 rotation matrices.
    Inverse operation of euler2rotmat_so3.
    
    Args:
        rotmats: 3x3 rotation matrices.
    
    Returns:
        Euler vectors (axis-angle representation).
    
    Raises:
        ValueError: If matrices are not 3x3, or if 4x4 matrices provided (use SE(3) functions instead).
    """
    if rotmats.shape[-1] != 3:
        if rotmats.shape[-2:] == (4,4):
            raise ValueError(f"Expected set of 3x3-matrices. Received set of 4x4-matrices. For the the corresponding transformation in se3 please use rotmat2euler or se3_rotmat2euler.")
        raise ValueError(f"Expected set of 3x3-matrices. Instead received shape {rotmats.shape}.")
    eulers = np.zeros(rotmats.shape[:-1])
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            eulers[i] = rotmat2euler_so3(rotmats[i])
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


def cayley2rotmat(
    cayleys: np.ndarray  # shape (..., N, 3): Cayley vectors
) -> np.ndarray:  # shape (..., N, 3, 3): rotation matrices
    """
    Convert Cayley vectors to 3x3 rotation matrices.
    
    Uses Cayley map (default cnDNA parametrization) to generate rotation matrices.
    The Cayley map provides an alternative to Euler angles for rotation parametrization.
    
    Args:
        cayleys: Cayley vectors (3-component).
    
    Returns:
        3x3 rotation matrices.
    
    Raises:
        ValueError: If last dimension is not 3.
    """
    if cayleys.shape[-1] != 3:
        raise ValueError(f"Expected set of 3-vectors. Instead received shape {cayleys.shape}")
    
    rotmats = np.zeros(cayleys.shape + (3,))
    if len(cayleys.shape) > 2:
        for i in range(len(cayleys)):
            rotmats[i] = cayley2rotmat(cayleys[i])
        return rotmats
    for i, cayley in enumerate(cayleys):
        rotmats[i] = so3.cayley2rotmat(cayley)
    return rotmats


def rotmat2cayley(
    rotmats: np.ndarray  # shape (..., N, 3, 3): rotation matrices
) -> np.ndarray:  # shape (..., N, 3): Cayley vectors
    """
    Convert rotation matrices to Cayley vectors.
    
    Extracts Cayley parametrization from 3x3 rotation matrices.
    Inverse operation of cayley2rotmat.
    
    Args:
        rotmats: 3x3 rotation matrices.
    
    Returns:
        Cayley vectors (3-component).
    """
    cayleys = np.zeros(rotmats.shape[:-1])
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            cayleys[i] = rotmat2cayley(rotmats[i])
        return cayleys
    for i, rotmat in enumerate(rotmats):
        cayleys[i] = so3.rotmat2cayley(rotmat)
    return cayleys

##########################################################################################################
############### Conversion between vectors and rotation matrices #########################################
##########################################################################################################


def vec2rotmat_so3(
    vecs: np.ndarray,  # shape (..., N, 3): rotation parameter vectors
    rotation_map: str = "euler"
) -> np.ndarray:  # shape (..., N, 3, 3): rotation matrices
    """
    Convert rotation vectors to matrices using specified parametrization.
    
    Flexible conversion function supporting different rotation parametrizations.
    Dispatches to appropriate conversion based on rotation_map.
    
    Args:
        vecs: 3-component rotation vectors.
        rotation_map: Rotation parametrization - "euler" for Euler angles or "cayley" for Cayley map.
    
    Returns:
        3x3 rotation matrices.
    
    Raises:
        ValueError: If rotation_map is not "euler" or "cayley".
    """
    if rotation_map == "euler":
        return euler2rotmat_so3(vecs)
    elif rotation_map == "cayley":
        return cayley2rotmat(vecs)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


def rotmats2vecs_so3(
    rotmats: np.ndarray,  # shape (..., N, 3, 3): rotation matrices
    rotation_map: str = "euler"
) -> np.ndarray:  # shape (..., N, 3): rotation parameter vectors
    """
    Convert rotation matrices to vectors using specified parametrization.
    
    Flexible conversion function supporting different rotation parametrizations.
    Inverse of vec2rotmat_so3.
    
    Args:
        rotmats: 3x3 rotation matrices.
        rotation_map: Rotation parametrization - "euler" for Euler angles or "cayley" for Cayley map.
    
    Returns:
        3-component rotation vectors.
    
    Raises:
        ValueError: If rotation_map is not "euler" or "cayley".
    """
    if rotation_map == "euler":
        return rotmat2euler_so3(rotmats)
    elif rotation_map == "cayley":
        return rotmat2cayley(rotmats)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


##########################################################################################################
############### Conversion between rotation matrices and triads ##########################################
##########################################################################################################


def rotmat2triad(
    rotmats: np.ndarray,  # shape (..., N, 3, 3): rotation matrices
    first_triad=None  # shape (3, 3) or None: initial triad orientation
) -> np.ndarray:  # shape (..., N+1, 3, 3): triads (coordinate frames)
    """
    Convert rotation matrices to triads (coordinate frames).
    
    Accumulates rotations to build a chain of local coordinate frames.
    Each triad represents the orientation of a reference frame in 3D space.
    
    Args:
        rotmats: Local rotation matrices connecting consecutive triads.
        first_triad: Initial triad orientation (default: identity).
    
    Returns:
        Chain of triads (N+1 triads for N rotations).
    
    Raises:
        AssertionError: If first_triad is not a 3x3 matrix.
    """
    sh = list(rotmats.shape)
    sh[-3] += 1
    triads = np.zeros(tuple(sh))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            triads[i] = rotmat2triad(rotmats[i])
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


def triad2rotmat(
    triads: np.ndarray  # shape (..., N+1, 3, 3): triads (coordinate frames)
) -> np.ndarray:  # shape (..., N, 3, 3): rotation matrices
    """
    Convert chain of triads to local rotation matrices.
    
    Inverse operation of rotmat2triad. Computes the rotation matrices
    connecting consecutive triads in a chain.
    
    Args:
        triads: Chain of coordinate frames (N+1 triads).
    
    Returns:
        Local rotation matrices (N rotations for N+1 triads).
    """
    sh = list(triads.shape)
    sh[-3] -= 1
    rotmats = np.zeros(tuple(sh))
    if len(triads.shape) > 3:
        for i in range(len(triads)):
            rotmats[i] = triad2rotmat(triads[i])
        return rotmats

    for i in range(len(triads) - 1):
        rotmats[i] = np.matmul(triads[i].T, triads[i + 1])
    return rotmats



##########################################################################################################
############### Generate positions from triads ###########################################################
##########################################################################################################

def triad2position(
    triads: np.ndarray,  # shape (..., N, 3, 3): coordinate triads
    disc_len: float = 0.34
) -> np.ndarray:  # shape (..., N, 3): position vectors
    """
    Generate position vectors from coordinate triads.
    
    Computes positions by accumulating displacements along the local z-axis
    (third column) of each triad. Default discretization length is 0.34 nm,
    typical for DNA base pair spacing.
    
    Args:
        triads: Coordinate frames (triads) defining local orientations.
        disc_len: Discretization length (spacing between positions) in nm.
    
    Returns:
        Position vectors in 3D space.
    """
    pos = np.zeros(triads.shape[:-1])
    if len(triads.shape) > 3:
        for i in range(len(triads)):
            pos[i] = triad2position(triads[i])
        return pos
    pos[0] = np.zeros(3)
    for i in range(len(triads) - 1):
        pos[i + 1] = pos[i] + triads[i, :, 2] * disc_len
    return pos