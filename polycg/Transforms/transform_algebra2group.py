import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from ..SO3 import so3
from .transform_statevec import statevec2vecs
from .transform_SE3 import euler2rotmat, rotmat2euler, invert

##########################################################################################################
########## Conversion of splitting of static and dynamic components at the lie algebra ###################
########## (vector) and lie group (matrix) level                                       ###################
##########################################################################################################

# def algebra2group(
#     algebra_groundstate: np.ndarray,
#     algebra_dynamic: np.ndarray,
#     rotation_first: bool = True
# ) -> np.ndarray:
#     """Converts the dynamic (rotational) components of basepair step coordinates from lie algebra level splitting (vector) to lie group
#     level splitting (matrix). All rotational components need to be in in euler parametrization.

#     Args:
#         algebra_groundstate (np.ndarray): set of static components (Nx3 or Nx6). If the vectors are 6-vectors translations are assumed to be included
#         algebra_dynamic (np.ndarray): set of dynamic components (Nx3 or Nx6). If the vectors are 6-vectors translations are assumed to be included
#         rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of freedom if this variable is set to True. (default: True)

#     Returns:
#         np.ndarray: set of static components (Nx3 or Nx6) in group (matrix splitting) definition   
#     """
    
#     if algebra_dynamic.shape[-2:] != algebra_groundstate.shape:
#         raise ValueError(f'Incompatible shape of algebra_groundstate ({algebra_groundstate.shape}) and algebra_dynamic ({algebra_dynamic.shape}).')
    
#     if algebra_groundstate.shape[-1] == 3:
#         translations_included = False
#     elif algebra_groundstate.shape[-1] == 6:
#         translations_included = True
#     else:
#         raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {eulers.shape}")
    
    
    
#     def alg2gro(
#         rotmats_groundstate: np.ndarray,
#         rotmats_full: np.ndarray,
#         rotation_first: bool = True,
#         translation_included: bool = True
#     ) -> np.ndarray:
        
#         while len(rotmats_full.shape) > 3:
            
#             2+3
    
    
#     full_bps = algebra_groundstate + algebra_dynamic
#     rotmats_full = euler2rotmat(full_bps,rotation_first=rotation_first)
#     rotmats_gs   = euler2rotmat(algebra_groundstate,rotation_first=rotation_first)
#     inverse_gs   = invert(rotmats_gs)
    
#     if translations_included:
    
    
##########################################################################################################
########## Linear transformations between algebra and group definition of fluctuations ###################
##########################################################################################################

def algebra2group_lintrans(
    groundstate_algebra: np.ndarray, 
    rotation_first: bool = True,
    translation_as_midstep: bool = False
    ) -> np.ndarray:
    """Linearization of the transformation of dynamic components from algebra (vector) to group (matrix) splitting between static and dynamic parts. Optionally the transformations from midstep triad definition to triad definition of the translational component may also be included. 

    Args:
        groundstate_algebra (np.ndarray): set of groundstate Euler vectors (Nx3 or Nx6) around which the transformation is linearly expanded. If the vectors are 6-vectors translations are assumed to be included
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of freedom
        translation_as_midstep (bool): If True, the translational component is assumed to defined in the midstep triad frame. The translational component of resulting vectors will be defined in the standard SE3 definition assuming that the splitting between static and dynamic compoent occures at the level of the group (SE3): 
        g=s*d.
        
            / R   v \   / S   s \  / D   d \     / SD  Sd+s \
        g =           =                       = 
            \ 0   1 /   \ 0   1 /  \ 0   1 /     \ 0     1  /
                
    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if groundstate_algebra.shape[-1] == 3:
        translations_included = False
    elif groundstate_algebra.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_algebra.shape}")

    if len(groundstate_algebra.shape) != 2:
        raise ValueError(f'Expected array of shape (N,3) or (N,6), encountered {groundstate_algebra.shape}')

    # define index selection
    if translations_included and rotation_first:
        rot_from = 0
        rot_to   = 3
        trans_from = 3
        trans_to   = 6
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        rot_from = 3
        rot_to   = 6
        trans_from = 0
        trans_to   = 3
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    # initialize transformation matrix
    dim = len(groundstate_algebra)*3 
    if translations_included:
        dim *= 2  
    HX = np.eye(dim)
    
    if not translations_included:
        for i, vec in enumerate(groundstate_algebra):
            HX[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = so3.splittransform_algebra2group(vec)
    else:
        for i, vec in enumerate(groundstate_algebra):
            Omega_0 = vec[rotslice]
            zeta_0  = vec[transslice]
            H = so3.splittransform_algebra2group(Omega_0)
            sid = 6*i
            rfr = sid+rot_from
            rto = sid+rot_to
            tfr = sid+trans_from
            tto = sid+trans_to
            
            HX[rfr:rto,rfr:rto] = H
            
            if translation_as_midstep:
                sqrtS_transp = so3.euler2rotmat(-0.5*Omega_0)
                zeta_0_hat_transp = so3.hat_map(-zeta_0)    
                
                # HX[tfr:tto,rfr:rto] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H
                H_half = so3.splittransform_algebra2group(0.5*Omega_0)
                HX[tfr:tto,rfr:rto] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H_half
                
                HX[tfr:tto,tfr:tto] = sqrtS_transp
            else:
                ST = so3.euler2rotmat(-Omega_0)
                HX[tfr:tto,tfr:tto] = ST
                
    return HX


def group2algebra_lintrans(
    groundstate_group: np.ndarray, 
    rotation_first: bool = True,
    translation_as_midstep: bool = False
    ) -> np.ndarray:
    """Linearization of the transformation of dynamic components from group (matrix) to algebra (vector) splitting between static and dynamic parts. Optionally the translational component may be expressed in terms of the midstep triad. 

    Args:
        groundstate_group (np.ndarray): set of groundstate Euler vectors (Nx3 or Nx6) around which the transformation is linearly expanded. If the vectors are 6-vectors translations are assumed to be included
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of freedom
        translation_as_midstep (bool): If True, the translational component of the final state will be expressed in terms of the midstep triad.    
    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if groundstate_group.shape[-1] == 3:
        translations_included = False
    elif groundstate_group.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_group.shape}")

    if len(groundstate_group.shape) != 2:
        raise ValueError(f'Expected array of shape (N,3) or (N,6), encountered {groundstate_group.shape}')

    # define index selection
    if translations_included and rotation_first:
        rot_from = 0
        rot_to   = 3
        trans_from = 3
        trans_to   = 6
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        rot_from = 3
        rot_to   = 6
        trans_from = 0
        trans_to   = 3
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    # initialize transformation matrix
    dim = len(groundstate_group)*3 
    if translations_included:
        dim *= 2  
    HX_inv = np.eye(dim)
    
    if not translations_included:
        for i, vec in enumerate(groundstate_group):
            HX_inv[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = so3.splittransform_group2algebra(vec)
    else:
        for i, vec in enumerate(groundstate_group):
            # Phi_0 = Omega_0
            Phi_0 = vec[rotslice]
            s     = vec[transslice]
            
            H_inv = so3.splittransform_group2algebra(Phi_0)
            sid = 6*i
            rfr = sid+rot_from
            rto = sid+rot_to
            tfr = sid+trans_from
            tto = sid+trans_to
            
            HX_inv[rfr:rto,rfr:rto] = H_inv
            
            if translation_as_midstep:
                sqrtS = so3.euler2rotmat(0.5*Phi_0)
                zeta_0 = sqrtS.T @ s
                zeta_0_hat_transp = so3.hat_map(-zeta_0)    
                
                # HX_inv[tfr:tto,rfr:rto] = -0.5 * zeta_0_hat_transp
                H_half = so3.splittransform_algebra2group(0.5*Phi_0)
                HX_inv[tfr:tto,rfr:rto] = -0.5 * zeta_0_hat_transp @ H_half @ H_inv
                
                HX_inv[tfr:tto,tfr:tto] = sqrtS
            else:
                S = so3.euler2rotmat(Phi_0)
                HX_inv[tfr:tto,tfr:tto] = S
                
    return HX_inv


##########################################################################################################
##########################################################################################################
############### Convert stiffnessmatrix between different definitions of rotation DOFS ###################
##########################################################################################################
##########################################################################################################

def algebra2group_stiffmat(
    groundstate_algebra: np.ndarray, 
    stiff_algebra: np.ndarray, 
    rotation_first: bool = True,
    translation_as_midstep: bool = False
    ) -> np.ndarray:
    """Converts stiffness matrix from algebra-level (vector) splitting between static and dynamic component to group-level (matrix) splitting. Optionally, the transformations from midstep triad definition to triad definition of the translational component may also be included.  

    Args:
        groundstate_algebra (np.ndarray): groundstate expressed in algebra (vector) splitting definition. The rotational component is the same in the algebra and group definition.
        stiff_algebra (np.ndarray): stiffness matrix expressed in arbitrary units
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of freedom
        translation_as_midstep (bool): If True, the translational component of the final state will be expressed in terms of the midstep triad.    

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """ 
    
    HX = algebra2group_lintrans(
        groundstate_algebra,
        rotation_first=rotation_first,
        translation_as_midstep=translation_as_midstep
    )
    HX_inv = np.linalg.inv(HX)
    
    # if translation_as_midstep:
    #     groundstate_group = np.copy(groundstate_algebra)
    #     for i,vec in enumerate(groundstate_group):
    #         if rotation_first:
    #             Phi_0 = vec[:3]
    #             zeta_0 = vec[3:]
    #             sqrtS = so3.euler2rotmat(0.5*Phi_0)
    #             s = sqrtS @ zeta_0
    #             groundstate_group[i,3:] = s
    #         else:
    #             Phi_0  = vec[3:]
    #             zeta_0 = vec[:3]
    #             sqrtS = so3.euler2rotmat(0.5*Phi_0)
    #             s = sqrtS @ zeta_0
    #             groundstate_group[i,:3] = s
    #     test_HX_inv = group2algebra_lintrans(
    #         groundstate_group,
    #         rotation_first=rotation_first,
    #         translation_as_midstep=translation_as_midstep
    #     )
    #     if np.abs(np.sum(HX_inv-test_HX_inv)) > 1e-10:
    #         raise ValueError(f'Inconsistent definition of H_inv. Discrepancy: {np.abs(np.sum(HX_inv-test_HX_inv))}')
    #     else:
    #         print('no problem')
    
    stiff_group = HX_inv.T @ stiff_algebra @ HX_inv
    return stiff_group


def group2algebra_stiffmat(
    groundstate_group: np.ndarray, 
    stiff_group: np.ndarray, 
    rotation_first: bool = True,
    translation_as_midstep: bool = False
    ) -> np.ndarray:
    """Converts stiffness matrix from group-level (matrix) splitting between static and dynamic component to algebra-level (vector) splitting. Optionally, the transformations from midstep triad definition to triad definition of the translational component may also be included. I.e. the final 
    definition will assume a midstep triad definition of the translational component. 

    Args:
        groundstate_group (np.ndarray): groundstate expressed in group (matrix) splitting definition. The rotational component is the same in the algebra and group definition.
        stiff_group (np.ndarray): stiffness matrix expressed in arbitrary units
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of freedom
        translation_as_midstep (bool): If True, the translational component of the final state will be expressed in terms of the midstep triad.    

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """ 
    
    HX_inv = group2algebra_lintrans(
        groundstate_group,
        rotation_first=rotation_first,
        translation_as_midstep=translation_as_midstep
    )
    HX = np.linalg.inv(HX_inv)
    stiff_algebra = HX.T @ stiff_group @ HX
    return stiff_algebra

