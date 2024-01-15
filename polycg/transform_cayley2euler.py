import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from .SO3 import so3
from .transform_state import statevec2vecs

##########################################################################################################
##########################################################################################################
############### Conversion between Euler and Cayley (Rodrigues) coordinates ##############################
##########################################################################################################
##########################################################################################################


def eulers2cayleys(eulers: np.ndarray, rotations_first: bool = True) -> np.ndarray:
    """Converts set of Euler vectors (axis angle rotation vectors) into Rodrigues
    vectors (Cayley vectors)

    Args:
        eulers (np.ndarray): set of Euler vectors (Nx3 or Nx6). If the vectors are 6-vectors translations are assumed to be included
        rotations_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of freedom if this variable is set to True. (default: True)

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
            cayleys[i] = eulers2cayleys(eulers[i])
        return cayleys
    
    if not translations_included:
        for i, euler in enumerate(eulers):
            cayleys[i] = so3.euler2cayley(euler)
    else:
        if rotations_first:
            for i, euler in enumerate(eulers):
                cayleys[i,:3] = so3.euler2cayley(euler[:3])
        else:
            for i, euler in enumerate(eulers):
                cayleys[i,3:] = so3.euler2cayley(euler[3:])   
    return cayleys


def cayleys2eulers(cayleys: np.ndarray, rotations_first: bool = True) -> np.ndarray:
    """Converts set of rodrigues vectors (Cayley vectors) into Euler vectors (axis angle
    rotation vectors)

    Args:
        cayleys (np.ndarray): set of Cayley vectors (Nx3 or Nx6). If the vectors are 6-vectors translations are assumed to be included
        rotations_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

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
            eulers[i] = cayleys2eulers(cayleys[i])
        return eulers
    
    if not translations_included:
        for i, cayley in enumerate(cayleys):
            eulers[i] = so3.cayley2euler(cayley)
    else:
        if rotations_first:
            for i, cayley in enumerate(cayleys):
                eulers[i,:3] = so3.euler2cayley(cayley[:3])
        else:
            for i, cayley in enumerate(cayleys):
                eulers[i,3:] = so3.euler2cayley(cayley[3:])   
    return eulers


def cayleys2eulers_lintrans(cayleys: np.ndarray, rotations_first: bool = True) -> np.ndarray:
    """Linearization of the transformation from Cayley to Euler vector around a given
    groundstate vector

    Args:
        cayleys (np.ndarray): set of Cayley vectors (Nx3 or Nx6) around which the transformation is linearly expanded. If the vectors are 6-vectors translations are assumed to be included
        rotations_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if cayleys.shape[-1] == 3:
        translations_included = False
    elif cayleys.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {cayleys.shape}")

    if len(cayleys.shape) != 2:
        raise ValueError(f'Expected array of shape (N,3) or (N,6), encountered {cayleys.shape}')

    dim = len(cayleys)*3 * ()
    if translations_included:
        dim *= 2  
    trans = np.zeros((dim,) * 2)
    
    if not translations_included:
        for i, vec in enumerate(cayleys):
            trans[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = so3.cayley2euler_linearexpansion(vec)
    else:
        if rotations_first:
            for i, vec in enumerate(cayleys):
                trans[
                    6*i:6*i+3, 6*i:6*i+3
                ] = so3.cayley2euler_linearexpansion(vec[:3])
        else:
            for i, vec in enumerate(cayleys):
                trans[
                    6*i+3 : 6*i+6, 6*i+3 : 6*i+6
                ] = so3.cayley2euler_linearexpansion(vec[3:])
    return trans


def eulers2cayleys_lintrans(eulers: np.ndarray, rotations_first: bool = True) -> np.ndarray:
    """Linearization of the transformation from Euler to Cayley vector around a
    given groundstate vector

    Args:
        eulers (np.ndarray): set of Euler vectors (Nx3 or Nx6) around which the transformation is linearly expanded. If the vectors are 6-vectors translations are assumed to be included
        rotations_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if eulers.shape[-1] == 3:
        translations_included = False
    elif eulers.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {eulers.shape}")

    if len(eulers.shape) != 2:
        raise ValueError(f'Expected array of shape (N,3) or (N,6), encountered {eulers.shape}')

    dim = len(eulers)*3 * ()
    if translations_included:
        dim *= 2  
    trans = np.zeros((dim,) * 2)
    
    if not translations_included:
        for i, vec in enumerate(eulers):
            trans[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = so3.euler2cayley_linearexpansion(vec)
    else:
        if rotations_first:
            for i, vec in enumerate(eulers):
                trans[
                    6*i:6*i+3, 6*i:6*i+3
                ] = so3.euler2cayley_linearexpansion(vec[:3])
        else:
            for i, vec in enumerate(eulers):
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
    rotations_first: bool = True
    ) -> np.ndarray:
    """Converts stiffness matrix from Cayley map representation to Euler map representation. Transformation of 
    stiffness matrix assumes the magnitude of the rotation vector to be dominated by the groundstate.

    Args:
        groundstate_cayley (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix expressed in arbitrary units
        rotations_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """
    
    Tc2e = cayleys2eulers_lintrans(groundstate_cayley)
    Tc2e_inv = np.linalg.inv(Tc2e)
    # stiff_euler = np.matmul(Tc2e_inv.T,np.matmul(stiff,Tc2e_inv))
    stiff_euler = Tc2e_inv.T @ stiff @ Tc2e_inv
    return stiff_euler

def euler2cayley_stiffmat(
    groundstate_euler: np.ndarray, 
    stiff: np.ndarray, 
    rotations_first: bool = True
    ) -> np.ndarray:
    """Converts stiffness matrix from Euler map representation to Cayley map representation. Transformation of 
    stiffness matrix assumes the magnitude of the rotation vector to be dominated by the groundstate.

    Args:
        groundstate_euler (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix expressed in arbitrary units
        rotations_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """
    Tc2e = eulers2cayleys_lintrans(groundstate_euler)
    Tc2e_inv = np.linalg.inv(Tc2e)
    # stiff_euler = np.matmul(Tc2e_inv.T,np.matmul(stiff,Tc2e_inv))
    stiff_euler = Tc2e_inv.T @ stiff @ Tc2e_inv
    return stiff_euler