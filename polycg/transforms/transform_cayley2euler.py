from __future__ import annotations

import numpy as np
from ..SO3 import so3
from .transform_statevec import statevec2vecs

##########################################################################################################
##########################################################################################################
############### Conversion between Euler and Cayley (Rodrigues) coordinates ##############################
##########################################################################################################
##########################################################################################################


def euler2cayley(
    eulers: np.ndarray,  # shape (..., N, 3) or (..., N, 6): Euler vectors
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 3) or (..., N, 6): Cayley vectors
    """
    Convert Euler vectors (axis-angle) to Cayley vectors (Rodrigues).
    
    Transforms rotation parametrization from Euler (axis-angle) to Cayley (Rodrigues).
    Translation components (if present) remain unchanged. Handles both SO(3) (3-vectors)
    and SE(3) (6-vectors) representations.
    
    Args:
        eulers: Euler vectors. Can be 3-vectors (rotation only) or 6-vectors (rotation + translation).
        rotation_first: For 6-vectors, if True, first 3 components are rotation.
    
    Returns:
        Cayley vectors with same shape as input. Translation components unchanged.
    
    Raises:
        ValueError: If vectors are not 3 or 6 dimensional.
    """
    if eulers.shape[-1] == 3:
        translations_included = False
    elif eulers.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {eulers.shape}")

    if len(eulers.shape) == 1:
        return so3.euler2cayley(euler)
    
    cayleys = np.copy(eulers)
    if len(eulers.shape) > 2:
        for i in range(len(eulers)):
            cayleys[i] = euler2cayley(eulers[i])
        return cayleys
    
    if not translations_included:
        for i, euler in enumerate(eulers):
            cayleys[i] = so3.euler2cayley(euler)
    else:
        if rotation_first:
            for i, euler in enumerate(eulers):
                cayleys[i,:3] = so3.euler2cayley(euler[:3])
        else:
            for i, euler in enumerate(eulers):
                cayleys[i,3:] = so3.euler2cayley(euler[3:])   
    return cayleys


def cayley2euler(
    cayleys: np.ndarray,  # shape (..., N, 3) or (..., N, 6): Cayley vectors
    rotation_first: bool = True
) -> np.ndarray:  # shape (..., N, 3) or (..., N, 6): Euler vectors
    """
    Convert Cayley vectors (Rodrigues) to Euler vectors (axis-angle).
    
    Inverse operation of euler2cayley. Transforms rotation parametrization from
    Cayley (Rodrigues) to Euler (axis-angle). Translation components (if present)
    remain unchanged.
    
    Args:
        cayleys: Cayley vectors. Can be 3-vectors (rotation only) or 6-vectors (rotation + translation).
        rotation_first: For 6-vectors, if True, first 3 components are rotation.
    
    Returns:
        Euler vectors with same shape as input. Translation components unchanged.
    
    Raises:
        ValueError: If vectors are not 3 or 6 dimensional.
    """
    if cayleys.shape[-1] == 3:
        translations_included = False
    elif cayleys.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {eulers.shape}")

    if len(cayleys.shape) == 1:
        return so3.cayley2euler(cayleys)

    eulers = np.copy(cayleys)
    if len(cayleys.shape) > 2:
        for i in range(len(cayleys)):
            eulers[i] = cayley2euler(cayleys[i])
        return eulers
    
    if not translations_included:
        for i, cayley in enumerate(cayleys):
            eulers[i] = so3.cayley2euler(cayley)
    else:
        if rotation_first:
            for i, cayley in enumerate(cayleys):
                eulers[i,:3] = so3.cayley2euler(cayley[:3])
        else:
            for i, cayley in enumerate(cayleys):
                eulers[i,3:] = so3.cayley2euler(cayley[3:])   
    return eulers


def cayley2euler_lintrans(
    groundstate_cayleys: np.ndarray,  # shape (N, 3) or (N, 6): groundstate Cayley vectors
    rotation_first: bool = True
) -> np.ndarray:  # shape (N*3, N*3) or (N*6, N*6): linear transformation matrix
    """
    Linearize Cayley to Euler transformation around groundstate.
    
    Computes the Jacobian (linear approximation) of the cayley2euler transformation
    around a given groundstate. Useful for transforming covariances and stiffness matrices.
    
    Args:
        groundstate_cayleys: Groundstate Cayley vectors defining linearization point.
        rotation_first: For 6-vectors, if True, first 3 components are rotation.
    
    Returns:
        Linear transformation matrix (Jacobian) for small deviations from groundstate.
    
    Raises:
        ValueError: If vectors are not 3 or 6 dimensional, or if shape is higher than 2D.
    """
    if groundstate_cayleys.shape[-1] == 3:
        translations_included = False
    elif groundstate_cayleys.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_cayleys.shape}")

    if len(groundstate_cayleys.shape) > 2:
        raise ValueError(f'Unexpected shape of groundstate_cayleys {groundstate_cayleys.shape}')
    if len(groundstate_cayleys.shape) == 1:
        groundstate_cayleys = np.array([groundstate_cayleys])

    dim = len(groundstate_cayleys)*3 
    if translations_included:
        dim *= 2  
    # trans = np.zeros((dim,) * 2)
    trans = np.eye(dim)
        
    if not translations_included:
        for i, vec in enumerate(groundstate_cayleys):
            trans[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = so3.cayley2euler_linearexpansion(vec)
    else:
        if rotation_first:
            for i, vec in enumerate(groundstate_cayleys):
                trans[
                    6*i:6*i+3, 6*i:6*i+3
                ] = so3.cayley2euler_linearexpansion(vec[:3])
        else:
            for i, vec in enumerate(groundstate_cayleys):
                trans[
                    6*i+3 : 6*i+6, 6*i+3 : 6*i+6
                ] = so3.cayley2euler_linearexpansion(vec[3:])
    return trans


def euler2cayley_lintrans(
    groundstate_eulers: np.ndarray,  # shape (N, 3) or (N, 6): groundstate Euler vectors
    rotation_first: bool = True
) -> np.ndarray:  # shape (N*3, N*3) or (N*6, N*6): linear transformation matrix
    """
    Linearize Euler to Cayley transformation around groundstate.
    
    Computes the Jacobian (linear approximation) of the euler2cayley transformation
    around a given groundstate. Inverse operation of cayley2euler_lintrans.
    
    Args:
        groundstate_eulers: Groundstate Euler vectors defining linearization point.
        rotation_first: For 6-vectors, if True, first 3 components are rotation.
    
    Returns:
        Linear transformation matrix (Jacobian) for small deviations from groundstate.
    
    Raises:
        ValueError: If vectors are not 3 or 6 dimensional, or if shape is higher than 2D.
    """
    if groundstate_eulers.shape[-1] == 3:
        translations_included = False
    elif groundstate_eulers.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_eulers.shape}")

    if len(groundstate_eulers.shape) > 2:
        raise ValueError(f'Unexpected shape of groundstate_eulers {groundstate_eulers.shape}')
    if len(groundstate_eulers.shape) == 1:
        groundstate_eulers = np.array([groundstate_eulers])

    dim = len(groundstate_eulers)*3 
    if translations_included:
        dim *= 2  
    # trans = np.zeros((dim,) * 2)
    trans = np.eye(dim)
    
    if not translations_included:
        for i, vec in enumerate(groundstate_eulers):
            trans[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = so3.euler2cayley_linearexpansion(vec)
    else:
        if rotation_first:
            for i, vec in enumerate(groundstate_eulers):
                trans[
                    6*i:6*i+3, 6*i:6*i+3
                ] = so3.euler2cayley_linearexpansion(vec[:3])
        else:
            for i, vec in enumerate(groundstate_eulers):
                trans[
                    6*i+3 : 6*i+6, 6*i+3 : 6*i+6
                ] = so3.euler2cayley_linearexpansion(vec[3:])
    return trans


##########################################################################################################
##########################################################################################################
############### Convert stiffnessmatrix between different definitions of rotation DOFS ###################
##########################################################################################################
##########################################################################################################

def cayley2euler_stiffmat(
    groundstate_cayley: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in Cayley parametrization
    stiff: np.ndarray,  # shape (N*ndims, N*ndims): stiffness matrix in Cayley coordinates
    rotation_first: bool = True
) -> np.ndarray:  # shape (N*ndims, N*ndims): stiffness matrix in Euler coordinates
    """
    Transform stiffness matrix from Cayley to Euler parametrization.
    
    Converts stiffness matrix between rotation parametrizations using linearized
    coordinate transformation. Assumes fluctuations are dominated by groundstate.
    
    Args:
        groundstate_cayley: Groundstate in Cayley (Rodrigues) coordinates.
        stiff: Stiffness matrix in Cayley parametrization.
        rotation_first: For 6-vectors, if True, first 3 components are rotation.
    
    Returns:
        Stiffness matrix transformed to Euler (axis-angle) parametrization.
    """ 
    # Tc2e = cayley2euler_lintrans(groundstate_cayley,rotation_first=rotation_first)
    # Tc2e_inv = np.linalg.inv(Tc2e)
    groundstate_euler = cayley2euler(groundstate_cayley, rotation_first=rotation_first)
    Tc2e_inv = euler2cayley_lintrans(groundstate_euler,  rotation_first=rotation_first)
    
    # stiff_euler = np.matmul(Tc2e_inv.T,np.matmul(stiff,Tc2e_inv))
    stiff_euler = Tc2e_inv.T @ stiff @ Tc2e_inv
    return stiff_euler

def euler2cayley_stiffmat(
    groundstate_euler: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in Euler parametrization
    stiff: np.ndarray,  # shape (N*ndims, N*ndims): stiffness matrix in Euler coordinates
    rotation_first: bool = True
) -> np.ndarray:  # shape (N*ndims, N*ndims): stiffness matrix in Cayley coordinates
    """
    Transform stiffness matrix from Euler to Cayley parametrization.
    
    Inverse operation of cayley2euler_stiffmat. Converts stiffness matrix between
    rotation parametrizations using linearized coordinate transformation.
    
    Args:
        groundstate_euler: Groundstate in Euler (axis-angle) coordinates.
        stiff: Stiffness matrix in Euler parametrization.
        rotation_first: For 6-vectors, if True, first 3 components are rotation.
    
    Returns:
        Stiffness matrix transformed to Cayley (Rodrigues) parametrization.
    """
    # Tc2e = euler2cayley_lintrans(groundstate_euler,rotation_first=rotation_first)
    # Tc2e_inv = np.linalg.inv(Tc2e)
    groundstate_cayley = euler2cayley(groundstate_euler,  rotation_first=rotation_first)
    Tc2e_inv =  cayley2euler_lintrans(groundstate_cayley, rotation_first=rotation_first)
        
    # stiff_euler = np.matmul(Tc2e_inv.T,np.matmul(stiff,Tc2e_inv))
    stiff_euler = Tc2e_inv.T @ stiff @ Tc2e_inv
    return stiff_euler