import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .SO3 import so3
from .pyConDec.pycondec import cond_jit


##########################################################################################################
############### Conversion between Euler vectors and rotation matrices ###################################
##########################################################################################################


def se3_eulers2rotmats(eulers: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    # if eulers.shape[-1] != 6:
    #     raise ValueError(f"Expected set of 6-vectors. Instead received shape {eulers.shape}")
    
    rotmats = np.zeros(tuple(list(eulers.shape)[:-1]) + (4,4))
    if len(eulers.shape) > 2:
        for i in range(len(eulers)):
            rotmats[i] = se3_eulers2rotmats(eulers[i],rotation_first=rotation_first)
        return rotmats
    for i, euler in enumerate(eulers):
        rotmats[i] = so3.se3_euler2rotmat(euler, rotation_first=rotation_first)
    return rotmats


def se3_rotmats2eulers(rotmats: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectrs (...,N,3)
    """
    eulers = np.zeros(tuple(list(rotmats.shape)[:-2])+(6,))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            eulers[i] = se3_rotmats2eulers(rotmats[i],rotation_first=rotation_first)
        return eulers
    for i, rotmat in enumerate(rotmats):
        eulers[i] = so3.se3_rotmat2euler(rotmat,rotation_first=rotation_first)
    return eulers


##########################################################################################################
############### Conversion between Cayley vectors and rotation matrices ###################################
##########################################################################################################


def se3_cayleys2rotmats(cayleys: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    # if cayleys.shape[-1] != 3:
    #     raise ValueError(f"Expected set of 3-vectors. Instead received shape {cayleys.shape}")
    
    rotmats = np.zeros(tuple(list(cayleys.shape)[:-1]) + (4,4))
    if len(cayleys.shape) > 2:
        for i in range(len(cayleys)):
            rotmats[i] = se3_cayleys2rotmats(cayleys[i],rotation_first=rotation_first)
        return rotmats
    for i, cayley in enumerate(cayleys):
        rotmats[i] = so3.se3_cayley2rotmat(cayley,rotation_first=rotation_first)
    return rotmats


def se3_rotmats2cayleys(rotmats: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectors (...,N,3)
    """
    cayleys = np.zeros(tuple(list(rotmats.shape)[:-2])+(6,))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            cayleys[i] = se3_rotmats2cayleys(rotmats[i],rotation_first=rotation_first)
        return cayleys
    for i, rotmat in enumerate(rotmats):
        cayleys[i] = so3.se3_rotmat2cayley(rotmat,rotation_first=rotation_first)
    return cayleys

##########################################################################################################
############### Conversion between vectors and rotation matrices #########################################
##########################################################################################################


def se3_vecs2rotmats(vecs: np.ndarray, rotation_map: str = "euler", rotation_first: bool = True) -> np.ndarray:
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
        return se3_eulers2rotmats(vecs,rotation_first=rotation_first)
    elif rotation_map == "cayley":
        return se3_cayleys2rotmats(vecs,rotation_first=rotation_first)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


def se3_rotmats2vecs(rotmats: np.ndarray, rotation_map: str = "euler", rotation_first: bool = True) -> np.ndarray:
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
        return se3_rotmats2eulers(rotmats,rotation_first=rotation_first)
    elif rotation_map == "cayley":
        return se3_rotmats2cayleys(rotmats,rotation_first=rotation_first)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


##########################################################################################################
############### Conversion between rotation matrices and triads ##########################################
##########################################################################################################


def se3_rotmats2triads(rotmats: np.ndarray, first_triad=None, midstep_trans: bool = False) -> np.ndarray:
    """Converts collection of se3 matrices into collection of se3-triads

    Args:
        rotmats (np.ndarray): set of rotation matrices that constitute the local junctions in the chain of triads. (...,N,4,4)
        first_triad (None or np.ndarray): rotation of first triad. Should be none or single triad. For now only supports identical rotation for all snapshots.

    Returns:
        np.ndarray: set of triads (...,N+1,3,3)
    """
    sh = list(rotmats.shape)
    sh[-3] += 1
    triads = np.zeros(tuple(sh))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            triads[i] = se3_rotmats2triads(rotmats[i])
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


def se3_triads2rotmats(triads: np.ndarray, midstep_trans: bool = False) -> np.ndarray:
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
            rotmats[i] = se3_triads2rotmats(triads[i])
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

def se3_transformations_midstep2triad(gs: np.ndarray) -> np.ndarray:
    midgs = np.zeros(gs.shape)
    if len(gs.shape) > 3:
        for i in range(len(gs)):
            midgs[i] = se3_transformations_midstep2triad(gs[i])
        return midgs

    for i, g in enumerate(gs):
        midgs[i] = so3.se3_transformation_midsteptrans2normal(g)
    return midgs  

def se3_transformations_triad2midstep(midgs: np.ndarray) -> np.ndarray:
    gs = np.zeros(midgs.shape)
    if len(midgs.shape) > 3:
        for i in range(len(midgs)):
            gs[i] = se3_transformations_triad2midstep(midgs[i])
        return gs

    for i, midg in enumerate(midgs):
        gs[i] = so3.se3_transformation_normal2midsteptrans(midg)
    return gs  

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