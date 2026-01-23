from __future__ import annotations

import numpy as np

from ..SO3 import so3
from ..pyConDec.pycondec import cond_jit


##########################################################################################################
############### Conversion between Euler vectors and rotation matrices ###################################
##########################################################################################################


def euler2rotmat(
    eulers: np.ndarray,  # shape (..., N, 3) or (..., N, 6): Euler angles
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 3, 3) or (..., N, 4, 4): rotation/transformation matrices
    """
    Convert Euler vector configuration to rotation/transformation matrices.
    
    Handles both SO(3) (3-vectors, rotation only) and SE(3) (6-vectors, rotation + translation)
    representations. Recursively processes higher-dimensional arrays.
    
    Args:
        eulers: Collection of Euler vectors. Last dimension must be 3 (SO(3)) or 6 (SE(3)).
        rotation_first: For 6-vectors, if True, first 3 components are rotation (Euler angles).
    
    Returns:
        Collection of rotation matrices (3x3 for SO(3)) or transformation matrices (4x4 for SE(3)).
    
    Raises:
        ValueError: If last dimension is not 3 or 6.
    """

    if eulers.shape[-1] == 6:
        use_se3 = True
        matshape = (4,4)
    elif eulers.shape[-1] == 3:
        use_se3 = False
        matshape = (3,3)
    else:
        raise ValueError(f"Expected set of 3- or 6-vectors. Instead received shape {eulers.shape}")
        
    rotmats = np.zeros(tuple(list(eulers.shape)[:-1]) + matshape)
    if len(eulers.shape) > 2:
        for i in range(len(eulers)):
            rotmats[i] = euler2rotmat(eulers[i],rotation_first=rotation_first)
        return rotmats
    
    if use_se3:
        for i, euler in enumerate(eulers):
            rotmats[i] = so3.se3_euler2rotmat(euler, rotation_first=rotation_first)
    else:
        for i, euler in enumerate(eulers):
            rotmats[i] = so3.euler2rotmat(euler)  
    return rotmats


def rotmat2euler(
    rotmats: np.ndarray,  # shape (..., N, 3, 3) or (..., N, 4, 4): rotation/transformation matrices
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 3) or (..., N, 6): Euler vectors
    """
    Convert rotation/transformation matrices to Euler vectors.
    
    Inverse operation of euler2rotmat. Handles both SO(3) (3x3) and SE(3) (4x4) matrices.
    Recursively processes higher-dimensional arrays.
    
    Args:
        rotmats: Collection of rotation matrices (3x3) or transformation matrices (4x4).
        rotation_first: For 4x4 matrices, if True, rotation components come first in output.
    
    Returns:
        Collection of Euler vectors (3-vectors for SO(3), 6-vectors for SE(3)).
    
    Raises:
        ValueError: If matrix dimensions are not 3x3 or 4x4.
    """
    if rotmats.shape[-2:] == (4,4):
        use_se3 = True
        vecsize = 6
    elif rotmats.shape[-2:] == (3,3):
        use_se3 = False
        vecsize = 3
    else:
        raise ValueError(f"Expected set of 3x3 or 4x4 matrices. Instead received shape {rotmats.shape}")
       
    eulers = np.zeros(tuple(list(rotmats.shape)[:-2])+(vecsize,))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            eulers[i] = rotmat2euler(rotmats[i],rotation_first=rotation_first)
        return eulers

    if use_se3:
        for i, rotmat in enumerate(rotmats):
            eulers[i] = so3.se3_rotmat2euler(rotmat,rotation_first=rotation_first)
    else:
        for i, rotmat in enumerate(rotmats):
            eulers[i] = so3.rotmat2euler(rotmat)
    return eulers


def euler2rotmat_se3(
    eulers: np.ndarray,  # shape (..., N, 6) or (6,): SE(3) Euler vectors
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 4, 4) or (4, 4): SE(3) transformation matrices
    """
    Convert SE(3) Euler vectors to 4x4 transformation matrices.
    
    Specialized function for SE(3) transformations (rotation + translation).
    Handles single vectors or collections, recursively processing higher dimensions.
    
    Args:
        eulers: SE(3) Euler vectors with 6 components (3 rotation + 3 translation).
        rotation_first: If True, first 3 components are rotation (Euler angles).
    
    Returns:
        4x4 homogeneous transformation matrices representing SE(3) transformations.
    """
    # if eulers.shape[-1] != 6:
    #     raise ValueError(f"Expected set of 6-vectors. Instead received shape {eulers.shape}")
    if eulers.shape == (6,):
        return so3.se3_euler2rotmat(eulers,rotation_first=rotation_first)
    
    rotmats = np.zeros(tuple(list(eulers.shape)[:-1]) + (4,4))
    if len(eulers.shape) > 2:
        for i in range(len(eulers)):
            rotmats[i] = euler2rotmat_se3(eulers[i],rotation_first=rotation_first)
        return rotmats
    for i, euler in enumerate(eulers):
        rotmats[i] = so3.se3_euler2rotmat(euler, rotation_first=rotation_first)
    return rotmats


def rotmat2euler_se3(
    rotmats: np.ndarray,  # shape (..., N, 4, 4) or (4, 4): SE(3) transformation matrices
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 6) or (6,): SE(3) Euler vectors
    """
    Convert SE(3) transformation matrices to Euler vectors.
    
    Inverse operation of euler2rotmat_se3. Extracts rotation and translation
    components from 4x4 homogeneous transformation matrices.
    
    Args:
        rotmats: 4x4 SE(3) transformation matrices.
        rotation_first: If True, rotation components come first in output vectors.
    
    Returns:
        6-component vectors with rotation (Euler angles) and translation.
    """
    if rotmats.shape == (4,4):
        return so3.se3_rotmat2euler(rotmats,rotation_first=rotation_first)
    
    eulers = np.zeros(tuple(list(rotmats.shape)[:-2])+(6,))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            eulers[i] = rotmat2euler_se3(rotmats[i],rotation_first=rotation_first)
        return eulers
    for i, rotmat in enumerate(rotmats):
        eulers[i] = so3.se3_rotmat2euler(rotmat,rotation_first=rotation_first)
    return eulers


##########################################################################################################
############### Conversion between Cayley vectors and rotation matrices ###################################
##########################################################################################################


def cayley2rotmat_se3(
    cayleys: np.ndarray,  # shape (..., N, 6) or (6,): Cayley vectors
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 4, 4) or (4, 4): SE(3) transformation matrices
    """
    Convert SE(3) Cayley vectors to 4x4 transformation matrices.
    
    Uses Cayley map for rotation parametrization instead of Euler angles.
    The Cayley map is the default parametrization in cnDNA.
    
    Args:
        cayleys: SE(3) Cayley vectors with 6 components (3 rotation + 3 translation).
        rotation_first: If True, first 3 components are rotation (Cayley parameters).
    
    Returns:
        4x4 homogeneous transformation matrices.
    """
    # if cayleys.shape[-1] != 3:
    #     raise ValueError(f"Expected set of 3-vectors. Instead received shape {cayleys.shape}")
    if cayleys.shape == (6,):
        return so3.se3_cayley2rotmat(cayleys,rotation_first=rotation_first)
    
    rotmats = np.zeros(tuple(list(cayleys.shape)[:-1]) + (4,4))
    if len(cayleys.shape) > 2:
        for i in range(len(cayleys)):
            rotmats[i] = cayley2rotmat_se3(cayleys[i],rotation_first=rotation_first)
        return rotmats
    for i, cayley in enumerate(cayleys):
        rotmats[i] = so3.se3_cayley2rotmat(cayley,rotation_first=rotation_first)
    return rotmats


def rotmat2cayley_se3(
    rotmats: np.ndarray,  # shape (..., N, 4, 4) or (4, 4): SE(3) transformation matrices
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 6) or (6,): Cayley vectors
    """
    Convert SE(3) transformation matrices to Cayley vectors.
    
    Inverse operation of cayley2rotmat_se3. Extracts rotation (using Cayley map)
    and translation components from 4x4 homogeneous transformation matrices.
    
    Args:
        rotmats: 4x4 SE(3) transformation matrices.
        rotation_first: If True, rotation components come first in output vectors.
    
    Returns:
        6-component vectors with rotation (Cayley parameters) and translation.
    """
    if rotmats.shape == (4,4):
        return so3.se3_rotmat2cayley(rotmats,rotation_first=rotation_first)
    
    cayleys = np.zeros(tuple(list(rotmats.shape)[:-2])+(6,))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            cayleys[i] = rotmat2cayley_se3(rotmats[i],rotation_first=rotation_first)
        return cayleys
    for i, rotmat in enumerate(rotmats):
        cayleys[i] = so3.se3_rotmat2cayley(rotmat,rotation_first=rotation_first)
    return cayleys

##########################################################################################################
############### Conversion between vectors and rotation matrices #########################################
##########################################################################################################


def vecs2rotmats_se3(
    vecs: np.ndarray,  # shape (..., N, 6): SE(3) parameter vectors
    rotation_map: str = "euler",
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 4, 4): SE(3) transformation matrices
    """
    Convert SE(3) vectors to transformation matrices using specified rotation map.
    
    Flexible conversion function that supports different rotation parametrizations.
    Dispatches to appropriate conversion function based on rotation_map.
    
    Args:
        vecs: SE(3) parameter vectors (6 components).
        rotation_map: Rotation parametrization - "euler" for Euler angles or "cayley" for Cayley map.
        rotation_first: If True, first 3 components are rotation parameters.
    
    Returns:
        4x4 homogeneous transformation matrices.
    
    Raises:
        ValueError: If rotation_map is not "euler" or "cayley".
    """
    if rotation_map == "euler":
        return euler2rotmat_se3(vecs,rotation_first=rotation_first)
    elif rotation_map == "cayley":
        return cayley2rotmat_se3(vecs,rotation_first=rotation_first)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


def rotmat2vec_se3(
    rotmats: np.ndarray,  # shape (..., N, 4, 4): SE(3) transformation matrices
    rotation_map: str = "euler",
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 6): SE(3) parameter vectors
    """
    Convert SE(3) transformation matrices to vectors using specified rotation map.
    
    Flexible conversion function that supports different rotation parametrizations.
    Dispatches to appropriate conversion function based on rotation_map.
    
    Args:
        rotmats: 4x4 SE(3) transformation matrices.
        rotation_map: Rotation parametrization - "euler" for Euler angles or "cayley" for Cayley map.
        rotation_first: If True, rotation components come first in output vectors.
    
    Returns:
        6-component parameter vectors (rotation + translation).
    
    Raises:
        ValueError: If rotation_map is not "euler" or "cayley".
    """
    if rotation_map == "euler":
        return rotmat2euler_se3(rotmats,rotation_first=rotation_first)
    elif rotation_map == "cayley":
        return rotmat2cayley_se3(rotmats,rotation_first=rotation_first)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


##########################################################################################################
############### Conversion between rotation matrices and triads ##########################################
##########################################################################################################


def rotmat2triad_se3(
    rotmats: np.ndarray,  # shape (..., N, 4, 4): SE(3) transformation matrices
    first_triad=None,  # shape (4, 4) or None: initial triad orientation
    midstep_trans: bool = False
) -> np.ndarray:  # shape (..., N+1, 4, 4): SE(3) triads
    """
    Convert SE(3) transformation matrices to triads (coordinate frames).
    
    Accumulates transformations to build a chain of local coordinate frames (triads).
    Each triad represents the orientation and position of a reference frame.
    
    Args:
        rotmats: Local transformation matrices connecting consecutive triads.
        first_triad: Initial triad orientation (default: identity, positioned at origin).
        midstep_trans: If True, use midstep frame for translations.
    
    Returns:
        Chain of triads (N+1 triads for N transformations).
    
    Raises:
        ValueError: If rotmats are not 4x4 matrices.
    """
    if rotmats.shape[-2:] != (4,4):
        raise ValueError(f'Invalid shape of rotmats. Expected set of 4x4 matrices, but reseived ndarray of shape {rotmats.shape}')
    
    sh = list(rotmats.shape)
    sh[-3] += 1
    triads = np.zeros(tuple(sh))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            triads[i] = rotmat2triad_se3(rotmats[i])
        return triads

    if first_triad is None:
        first_triad = np.eye(4)
    assert first_triad.shape == (
        4,
        4,
    ), f"invalid shape of triad {first_triad.shape}. Triad shape needs to be (4,4)."

    triads[0] = first_triad
    
    if not midstep_trans:
        for i, rotmat in enumerate(rotmats):
            triads[i + 1] = np.matmul(triads[i], rotmat)
    else:
        for i, rotmat in enumerate(rotmats):
            triads[i + 1] = so3.se3_triadxrotmat_midsteptrans(triads[i], rotmat)
    return triads


def triad2rotmat_se3(
    triads: np.ndarray,  # shape (..., N+1, 4, 4): SE(3) triads
    midstep_trans: bool = False
) -> np.ndarray:  # shape (..., N, 4, 4): SE(3) transformation matrices
    """
    Convert chain of triads to local transformation matrices.
    
    Inverse operation of rotmat2triad_se3. Computes the transformation matrices
    connecting consecutive triads in a chain.
    
    Args:
        triads: Chain of coordinate frames (N+1 triads).
        midstep_trans: If True, use midstep frame for translations.
    
    Returns:
        Local transformation matrices (N transformations for N+1 triads).
    
    Raises:
        ValueError: If triads are not 4x4 matrices.
    """
    if triads.shape[-2:] != (4,4):
        raise ValueError(f'Invalid shape of triads. Expected set of 4x4 matrices, but reseived ndarray of shape {triads.shape}')
    
    sh = list(triads.shape)
    sh[-3] -= 1
    rotmats = np.zeros(tuple(sh))
    if len(triads.shape) > 3:
        for i in range(len(triads)):
            rotmats[i] = triad2rotmat_se3(triads[i])
        return rotmats

    if not midstep_trans:
        for i in range(len(triads) - 1):
            rotmats[i] = so3.se3_inverse(triads[i]) @ triads[i + 1]
    else:
        for i in range(len(triads) - 1):
            rotmats[i] = so3.se3_triads2rotmat_midsteptrans(triads[i], triads[i + 1])
    return rotmats


##########################################################################################################
######### Conversion of rotation matrices between midstep and normal definition of translations ##########
##########################################################################################################

def transformations_midstep2triad_se3(
    se3_gs: np.ndarray  # shape (..., N, 4, 4): SE(3) matrices with midstep translations
) -> np.ndarray:  # shape (..., N, 4, 4): SE(3) matrices with triad translations
    """
    Convert SE(3) transformations from midstep to triad translation definition.
    
    Transforms translation components from midstep frame (halfway between triads)
    to standard triad frame. This affects how translations are interpreted.
    
    Args:
        se3_gs: SE(3) transformation matrices with translations in midstep frame.
    
    Returns:
        SE(3) transformation matrices with translations in triad frame.
    """
    midgs = np.zeros(se3_gs.shape)
    if len(se3_gs.shape) > 3:
        for i in range(len(se3_gs)):
            midgs[i] = transformations_midstep2triad_se3(se3_gs[i])
        return midgs

    for i, g in enumerate(se3_gs):
        midgs[i] = so3.se3_transformation_midstep2triad(g)
    return midgs  

def transformations_triad2midstep_se3(
    se3_midgs: np.ndarray  # shape (..., N, 4, 4): SE(3) matrices with triad translations
) -> np.ndarray:  # shape (..., N, 4, 4): SE(3) matrices with midstep translations
    """
    Convert SE(3) transformations from triad to midstep translation definition.
    
    Inverse operation of transformations_midstep2triad_se3. Transforms translation
    components from standard triad frame to midstep frame.
    
    Args:
        se3_midgs: SE(3) transformation matrices with translations in triad frame.
    
    Returns:
        SE(3) transformation matrices with translations in midstep frame.
    """
    gs = np.zeros(se3_midgs.shape)
    if len(se3_midgs.shape) > 3:
        for i in range(len(se3_midgs)):
            gs[i] = transformations_triad2midstep_se3(se3_midgs[i])
        return gs

    for i, midg in enumerate(se3_midgs):
        gs[i] = so3.se3_transformation_triad2midstep(midg)
    return gs  


##########################################################################################################
######### Inversion of elements of SE3 ###################################################################
##########################################################################################################

def invert_se3(
    se3s: np.ndarray  # shape (..., N, 4, 4): SE(3) transformation matrices
) -> np.ndarray:  # shape (..., N, 4, 4): inverted SE(3) matrices
    """
    Compute inverses of SE(3) transformation matrices.
    
    For homogeneous transformation matrices, computes the proper inverse
    that accounts for both rotation and translation components.
    
    Args:
        se3s: SE(3) transformation matrices to invert.
    
    Returns:
        Inverse transformation matrices.
    """
    inverses = np.zeros(se3s.shape)
    if len(se3s.shape) > 3:
        for i in range(len(se3s)):
            inverses[i] = transformations_midstep2triad_se3(se3s[i])
        return inverses

    for i, se3 in enumerate(se3s):
        inverses[i] = so3.se3_transformation_midstep2triad(se3)
    return inverses      

def invert(
    rotmats: np.ndarray  # shape (..., N, 3, 3) or (..., N, 4, 4): rotation/transformation matrices
) -> np.ndarray:  # shape (..., N, 3, 3) or (..., N, 4, 4): inverted matrices
    """
    Compute inverses of rotation or transformation matrices.
    
    Handles both SO(3) (3x3 rotation matrices) and SE(3) (4x4 transformation matrices).
    For SO(3), uses transpose; for SE(3), uses proper SE(3) inverse.
    
    Args:
        rotmats: Rotation matrices (3x3) or transformation matrices (4x4).
    
    Returns:
        Inverse matrices (same dimensions as input).
    
    Raises:
        ValueError: If matrix dimensions are not 3x3 or 4x4.
    """
    if rotmats.shape != (3,3) and rotmats.shape != (4,4):
        raise ValueError(f'Invalid rotmats dimension {rotmats.shape}. Should be 3x3 or 4x4.')
    
    inverses = np.zeros(rotmats.shape)
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            inverses[i] = invert(rotmats[i])
        return inverses

    if rotmats.shape[-2:] == (4,4):
        for i, SE3 in enumerate(rotmats):
            inverses[i] = so3.se3_transformation_midstep2triad(SE3)
    else:
        for i, SO3 in enumerate(rotmats):
            inverses[i] = SO3.T        
    return inverses      

##########################################################################################################
############### Generate positions from triads ###########################################################
##########################################################################################################

def triads2positions(
    triads: np.ndarray,  # shape (..., N, 3, 3) or (..., N, 4, 4): coordinate triads
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
            pos[i] = triads2positions(triads[i])
        return pos
    pos[0] = np.zeros(3)
    for i in range(len(triads) - 1):
        pos[i + 1] = pos[i] + triads[i, :, 2] * disc_len
    return pos