import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from ..SO3 import so3
from ..pyConDec.pycondec import cond_jit


##########################################################################################################
############### Conversion between Euler vectors and rotation matrices ###################################
##########################################################################################################


def euler2rotmat(eulers: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
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


def rotmat2euler(rotmats: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectrs (...,N,3)
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


def euler2rotmat_se3(eulers: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
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


def rotmat2euler_se3(rotmats: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectrs (...,N,3)
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


def cayley2rotmat_se3(cayleys: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
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


def rotmat2cayley_se3(rotmats: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectors (...,N,3)
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


def vecs2rotmats_se3(vecs: np.ndarray, rotation_map: str = "euler", rotation_first: bool = True) -> np.ndarray:
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
        return euler2rotmat_se3(vecs,rotation_first=rotation_first)
    elif rotation_map == "cayley":
        return cayley2rotmat_se3(vecs,rotation_first=rotation_first)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


def rotmat2vec_se3(rotmats: np.ndarray, rotation_map: str = "euler", rotation_first: bool = True) -> np.ndarray:
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
        return rotmat2euler_se3(rotmats,rotation_first=rotation_first)
    elif rotation_map == "cayley":
        return rotmat2cayley_se3(rotmats,rotation_first=rotation_first)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


##########################################################################################################
############### Conversion between rotation matrices and triads ##########################################
##########################################################################################################


def rotmat2triad_se3(rotmats: np.ndarray, first_triad=None, midstep_trans: bool = False) -> np.ndarray:
    """Converts collection of se3 matrices into collection of se3-triads

    Args:
        rotmats (np.ndarray): set of rotation matrices that constitute the local junctions in the chain of triads. (...,N,4,4)
        first_triad (None or np.ndarray): rotation of first triad. Should be none or single triad. For now only supports identical rotation for all snapshots.

    Returns:
        np.ndarray: set of triads (...,N+1,3,3)
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


def triad2rotmat_se3(triads: np.ndarray, midstep_trans: bool = False) -> np.ndarray:
    """Converts set of triads into set of rotation matrices

    Args:
        triads (np.ndarray): set of triads (...,N+1,3,3)

    Returns:
        np.ndarray: set of rotation matrices (...,N,3,3)
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

def transformations_midstep2triad_se3(se3_gs: np.ndarray) -> np.ndarray:
    midgs = np.zeros(se3_gs.shape)
    if len(se3_gs.shape) > 3:
        for i in range(len(se3_gs)):
            midgs[i] = transformations_midstep2triad_se3(se3_gs[i])
        return midgs

    for i, g in enumerate(se3_gs):
        midgs[i] = so3.se3_transformation_midstep2triad(g)
    return midgs  

def transformations_triad2midstep_se3(se3_midgs: np.ndarray) -> np.ndarray:
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

def invert_se3(se3s: np.ndarray) -> np.ndarray:
    inverses = np.zeros(se3s.shape)
    if len(se3s.shape) > 3:
        for i in range(len(se3s)):
            inverses[i] = transformations_midstep2triad_se3(se3s[i])
        return inverses

    for i, se3 in enumerate(se3s):
        inverses[i] = so3.se3_transformation_midstep2triad(se3)
    return inverses      

def invert(rotmats: np.ndarray) -> np.ndarray:
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