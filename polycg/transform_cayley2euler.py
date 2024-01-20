import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from .SO3 import so3
from .transform_statevec import statevec2vecs

##########################################################################################################
##########################################################################################################
############### Conversion between Euler and Cayley (Rodrigues) coordinates ##############################
##########################################################################################################
##########################################################################################################


def euler2cayley(eulers: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts set of Euler vectors (axis angle rotation vectors) into Rodrigues
    vectors (Cayley vectors)

    Args:
        eulers (np.ndarray): set of Euler vectors (Nx3 or Nx6). If the vectors are 6-vectors translations are assumed to be included
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of freedom if this variable is set to True. (default: True)

    Returns:
        np.ndarray: set of Rodrigues vectors (including the unchanged translational degrees of freedom)
    """
    if eulers.shape[-1] == 3:
        translations_included = False
    elif eulers.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {eulers.shape}")

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


def cayley2euler(cayleys: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Converts set of rodrigues vectors (Cayley vectors) into Euler vectors (axis angle
    rotation vectors)

    Args:
        cayleys (np.ndarray): set of Cayley vectors (Nx3 or Nx6). If the vectors are 6-vectors translations are assumed to be included
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        np.ndarray: set of Euler vectors (including the unchanged translational degrees of freedom)
    """
    if cayleys.shape[-1] == 3:
        translations_included = False
    elif cayleys.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {eulers.shape}")

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
    groundstate_cayleys: np.ndarray, 
    rotation_first: bool = True
    ) -> np.ndarray:
    """Linearization of the transformation from Cayley to Euler vector around a given
    groundstate vector

    Args:
        groundstate_cayleys (np.ndarray): set of groundstate Cayley vectors (Nx3 or Nx6) around which the transformation is linearly expanded. If the vectors are 6-vectors translations are assumed to be included
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if groundstate_cayleys.shape[-1] == 3:
        translations_included = False
    elif groundstate_cayleys.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_cayleys.shape}")

    if len(groundstate_cayleys.shape) != 2:
        raise ValueError(f'Expected array of shape (N,3) or (N,6), encountered {groundstate_cayleys.shape}')

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
    groundstate_eulers: np.ndarray, 
    rotation_first: bool = True
    ) -> np.ndarray:
    """Linearization of the transformation from Euler to Cayley vector around a
    given groundstate vector

    Args:
        eulers (np.ndarray): set of groundstate Euler vectors (Nx3 or Nx6) around which the transformation is linearly expanded. If the vectors are 6-vectors translations are assumed to be included
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if groundstate_eulers.shape[-1] == 3:
        translations_included = False
    elif groundstate_eulers.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_eulers.shape}")

    if len(groundstate_eulers.shape) != 2:
        raise ValueError(f'Expected array of shape (N,3) or (N,6), encountered {groundstate_eulers.shape}')

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
    groundstate_cayley: np.ndarray, 
    stiff: np.ndarray, 
    rotation_first: bool = True
    ) -> np.ndarray:
    """Converts stiffness matrix from Cayley map representation to Euler map representation. Transformation of 
    stiffness matrix assumes the magnitude of the rotation vector to be dominated by the groundstate.

    Args:
        groundstate_cayley (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix expressed in arbitrary units
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """ 
    # Tc2e = cayley2euler_lintrans(groundstate_cayley,rotation_first=rotation_first)
    # Tc2e_inv = np.linalg.inv(Tc2e)
    groundstate_euler = cayley2euler(groundstate_cayley, rotation_first=rotation_first)
    Tc2e_inv = euler2cayley_lintrans(groundstate_euler,  rotation_first=rotation_first)
    
    # stiff_euler = np.matmul(Tc2e_inv.T,np.matmul(stiff,Tc2e_inv))
    stiff_euler = Tc2e_inv.T @ stiff @ Tc2e_inv
    return stiff_euler

def euler2cayley_stiffmat(
    groundstate_euler: np.ndarray, 
    stiff: np.ndarray, 
    rotation_first: bool = True
    ) -> np.ndarray:
    """Converts stiffness matrix from Euler map representation to Cayley map representation. Transformation of 
    stiffness matrix assumes the magnitude of the rotation vector to be dominated by the groundstate.

    Args:
        groundstate_euler (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix expressed in arbitrary units
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """
    # Tc2e = euler2cayley_lintrans(groundstate_euler,rotation_first=rotation_first)
    # Tc2e_inv = np.linalg.inv(Tc2e)
    groundstate_cayley = euler2cayley(groundstate_euler,  rotation_first=rotation_first)
    Tc2e_inv =  cayley2euler_lintrans(groundstate_cayley, rotation_first=rotation_first)
        
    # stiff_euler = np.matmul(Tc2e_inv.T,np.matmul(stiff,Tc2e_inv))
    stiff_euler = Tc2e_inv.T @ stiff @ Tc2e_inv
    return stiff_euler