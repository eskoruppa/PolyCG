from __future__ import annotations

import numpy as np

from ..SO3 import so3
from ..pyConDec.pycondec import cond_jit


##########################################################################################################
############# Transformation between midsteptriad and triad definitions of translations ##################
##########################################################################################################

def midstep2triad(
    vecs: np.ndarray,  # shape (..., 6): Euler ground state vectors
    rotation_map: str = 'euler', 
    rotation_first: bool = True
    ) -> np.ndarray:  # shape (..., 6): Transformed vectors
    """Transform translations from midstep triad frame to standard triad frame.
    
    The translations are defined in the rotated frame of the midstep triad. This method 
    transforms these translations into the SE(3) translations in the frame of triad i. 
    The rotational vectors of these two definitions are the same, so this method acts as 
    the identity on the rotational component. For the translational component:
    1) Applies the rotational displacement of half the rotation
    2) Accounts for the radial displacement of the midstep triad
    
    Args:
        vecs: Euler ground state vectors with rotation and translation components.
              Can be a single 6-vector or array of shape (N, 6) or higher dimensions.
        rotation_map: Rotation parametrization to use, either 'euler' or 'cayley'.
        rotation_first: If True, first 3 coordinates are rotational degrees of freedom,
                       otherwise last 3 are rotational.

    Returns:
        Transformed ground state vectors with same shape as input. Rotational components
        unchanged, translational components transformed to triad frame.
        
    Raises:
        ValueError: If vectors don't have 6 components in last dimension or invalid rotation_map.
    """
    if vecs.shape[-1] != 6:
        raise ValueError(f"Expected set of 6-vectors. Instead received shape {vecs.shape}")
    
    nvecs = np.copy(vecs)
    if len(vecs.shape) > 2:
        for i, vec in enumerate(vecs):
            nvecs[i] = midstep2triad(vec,rotation_map=rotation_map,rotation_first=rotation_first)
        return nvecs
    
    if rotation_first:  
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    if rotation_map == 'euler':
        def sqrt_rot(vrot: np.ndarray) -> np.ndarray:
            return so3.euler2rotmat(0.5*vrot)
        vrot2sqrt_rot = sqrt_rot
        # vec2rotmat = so3.euler2rotmat
    elif rotation_map == 'cayley':
        def sqrt_rot(vrot: np.ndarray) -> np.ndarray:
            return so3.euler2rotmat(0.5*so3.cayley2euler(vrot))
        vrot2sqrt_rot = sqrt_rot
        # vec2rotmat = so3.cayley2rotmat
    else:
        raise ValueError(f'Invalid rotation_method "{rotation_map}".')
    
    if len(vecs.shape) == 1:
        vrot   = vecs[rotslice]
        vtrans = vecs[transslice]
        sqrt_rotmat = vrot2sqrt_rot(vrot)
        nvecs[transslice] = sqrt_rotmat @ vtrans
        return nvecs
    else:
        for i, vec in enumerate(vecs):
            vrot   = vec[rotslice]
            vtrans = vec[transslice]
            # rotmat = vec2rotmat(vrot)
            # sqrt_rotmat = so3.sqrt_rot(rotmat) 
            # print(f'diff = {np.abs(np.sum(sqrt_rotmat-vrot2sqrt_rot(vrot)))}')
            sqrt_rotmat = vrot2sqrt_rot(vrot)
            nvecs[i,transslice] = sqrt_rotmat @ vtrans
        return nvecs


def triad2midstep(
    vecs: np.ndarray,  # shape (..., 6): Euler ground state vectors
    rotation_map: str = 'euler', 
    rotation_first: bool = True
    ) -> np.ndarray:  # shape (..., 6): Transformed vectors
    """Transform translations from standard triad frame to midstep triad frame.
    
    The translations are defined in the frame of triad i. This method transforms these 
    translations into the frame of the midstep triad. The rotational vectors of these two 
    definitions are the same, so this method acts as the identity on the rotational component. 
    For the translational component:
    1) Applies the rotational displacement of half the rotation
    2) Accounts for the radial displacement of the midstep triad
    
    Args:
        vecs: Euler ground state vectors with rotation and translation components.
              Can be a single 6-vector or array of shape (N, 6) or higher dimensions.
        rotation_map: Rotation parametrization to use, either 'euler' or 'cayley'.
        rotation_first: If True, first 3 coordinates are rotational degrees of freedom,
                       otherwise last 3 are rotational.

    Returns:
        Transformed ground state vectors with same shape as input. Rotational components
        unchanged, translational components transformed to midstep triad frame.
        
    Raises:
        ValueError: If vectors don't have 6 components in last dimension or invalid rotation_map.
    """
    if vecs.shape[-1] != 6:
        raise ValueError(f"Expected set of 6-vectors. Instead received shape {vecs.shape}")
    
    nvecs = np.copy(vecs)
    if len(vecs.shape) > 2:
        for i, vec in enumerate(vecs):
            nvecs[i] = triad2midstep(vec,rotation_map=rotation_map,rotation_first=rotation_first)
        return nvecs
    
    if rotation_first:  
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    if rotation_map == 'euler':
        def sqrt_rot(vrot: np.ndarray) -> np.ndarray:
            return so3.euler2rotmat(0.5*vrot)
        vrot2sqrt_rot = sqrt_rot
        # vec2rotmat = so3.euler2rotmat
    elif rotation_map == 'cayley':
        def sqrt_rot(vrot: np.ndarray) -> np.ndarray:
            return so3.euler2rotmat(0.5*so3.cayley2euler(vrot))
        vrot2sqrt_rot = sqrt_rot
        # vec2rotmat = so3.cayley2rotmat
    else:
        raise ValueError(f'Invalid rotation_method "{rotation_map}".')
    
    if len(vecs.shape) == 1:
        vrot   = vecs[rotslice]
        vtrans = vecs[transslice]
        sqrt_rotmat = vrot2sqrt_rot(vrot)
        nvecs[transslice] = sqrt_rotmat.T @ vtrans
        return nvecs
        
    else:
        for i, vec in enumerate(vecs):
            vrot   = vec[rotslice]
            vtrans = vec[transslice]
            # rotmat = vec2rotmat(vrot)
            # sqrt_rotmat = so3.sqrt_rot(rotmat) 
            sqrt_rotmat = vrot2sqrt_rot(vrot)
            nvecs[i,transslice] = sqrt_rotmat.T @ vtrans
        return nvecs
    
    

# The following methods are depricated 
##########################################################################################################
###### Linearization of transformation between midsteptriad and triad definitions of translations ########
##########################################################################################################

from warnings import warn

def midstep2triad_lintrans(
    groundstate_euler: np.ndarray, 
    rotation_first: bool = True, 
    split_fluctuations: str = 'vector',
    groundstate_definition: str = 'midstep'
    ) -> np.ndarray:
    """Linearization of transformation from midsteptriad- to triad-definitions of translations. The rotational component needs to be expressed in euler coordinates.
    """
    warn('This method is deprecated and possibly generates unexpected results.', DeprecationWarning, stacklevel=2)
    
    if split_fluctuations not in ['vector','matrix', 'so3', 'SO3']:
        raise ValueError(f'Invalid split_fluctutations method "{split_fluctuations}". Should be either "vector" or "so3" for splitting in so3 or "matrix" or "SO3" for splitting in SO3.')
    if groundstate_definition not in ['midstep','triad']:
        raise ValueError(f'Invalid groundstate_definition method "{groundstate_definition}". Should be either "midstep" or "triad".')
    if groundstate_euler.shape[-1] != 6:
        raise ValueError(f"Expected set of 6-vectors or a single 6-vector. Instead received shape {groundstate_euler.shape}")
    
    if split_fluctuations == 'vector':
        split_fluctuations = 'so3' 
    if split_fluctuations == 'matrix':
        split_fluctuations = 'SO3' 
    
    ###################################################
    ###################################################
    if len(groundstate_euler.shape) == 1:
        if rotation_first:
            Omega0 = groundstate_euler[:3]
            zeta0  = groundstate_euler[3:]
        else:
            Omega0 = groundstate_euler[3:]
            zeta0  = groundstate_euler[:3]
            
        sqrt_rotmat  = so3.euler2rotmat(0.5*Omega0) 
        if groundstate_definition != 'midstep':
            zeta0 = sqrt_rotmat.T @ zeta0
        
        zeta0_hat    = so3.hat_map(zeta0)
        Hm = np.eye(6)
        Hm[3:,3:] = sqrt_rotmat
        crosscoup = 0.5*sqrt_rotmat @ zeta0_hat.T
        # if split_fluctuations == 'so3':
        #     crosscoup = crosscoup @ so3.splittransform_algebra2group(Omega0)  
        Hm[3:,:3] = crosscoup
        Hm[:3,:3] = so3.splittransform_group2algebra(Omega0)
        return Hm

    ###################################################
    ###################################################
    
    if len(groundstate_euler.shape) > 2:
        raise ValueError(f'Expected array of shape (N,6), encountered {groundstate_euler.shape}')

    Hm = np.zeros((len(groundstate_euler)*6 ,)*2)
    for i, gs_euler in enumerate(groundstate_euler):
        Hm[i*6:(i+1)*6,i*6:(i+1)*6] = midstep2triad_lintrans(
            gs_euler,
            rotation_first=rotation_first,
            split_fluctuations=split_fluctuations,
            groundstate_definition=groundstate_definition)
    
    return Hm


def triad2midstep_lintrans(
    groundstate_euler: np.ndarray, 
    rotation_first: bool = True, 
    split_fluctuations: str = 'vector',
    groundstate_definition: str = 'triad'
    ) -> np.ndarray:
    """Linearization of transformation from midsteptriad- to triad-definitions of translations. The rotational component needs to be expressed in euler coordinates.
    """
    warn('This method is deprecated and possibly generates unexpected results.', DeprecationWarning, stacklevel=2)
    
    # if split_fluctuations not in ['vector','matrix', 'so3', 'SO3']:
    #     raise ValueError(f'Invalid split_fluctutations method "{split_fluctuations}". Should be either "vector" or "so3" for splitting in so3 or "matrix" or "SO3" for splitting in SO3.')
    if groundstate_definition not in ['midstep','triad']:
        raise ValueError(f'Invalid groundstate_definition method "{groundstate_definition}". Should be either "midstep" or "triad".')
    if groundstate_euler.shape[-1] != 6:
        raise ValueError(f"Expected set of 6-vectors or a single 6-vector. Instead received shape {groundstate_euler.shape}")
    
    # if split_fluctuations == 'vector':
    #     split_fluctuations = 'so3' 
    # if split_fluctuations == 'matrix':
    #     split_fluctuations = 'SO3' 
    
    ###################################################
    ###################################################
    if len(groundstate_euler.shape) == 1:
        if rotation_first:
            Omega0 = groundstate_euler[:3]
            zeta0  = groundstate_euler[3:]
        else:
            Omega0 = groundstate_euler[3:]
            zeta0  = groundstate_euler[:3]
            
        sqrt_rotmat  = so3.euler2rotmat(0.5*Omega0) 
        if groundstate_definition != 'midstep':
            zeta0 = sqrt_rotmat.T @ zeta0
        
        zeta0_hat    = so3.hat_map(zeta0)
        H = np.eye(6)
        H[3:,3:] = sqrt_rotmat.T
                
        crosscoup = 0.5 * zeta0_hat
        # if split_fluctuations == 'so3':
        #     print('so3')
        #     crosscoup = crosscoup @ so3.splittransform_algebra2group(Omega0) 
        H[3:,:3] = crosscoup
        return H

    ###################################################
    ###################################################
    
    if len(groundstate_euler.shape) > 2:
        raise ValueError(f'Expected array of shape (N,6), encountered {groundstate_euler.shape}')

    H = np.zeros((len(groundstate_euler)*6 ,)*2)
    for i, gs_euler in enumerate(groundstate_euler):
        H[i*6:(i+1)*6,i*6:(i+1)*6] = triad2midstep_lintrans(
            gs_euler,
            rotation_first=rotation_first,
            split_fluctuations=split_fluctuations,
            groundstate_definition=groundstate_definition)
    
    return H


##########################################################################################################
######################## Transformation of stiffness matrices ############################################
##########################################################################################################



def midstep2triad_stiffmat(
    groundstate_euler: np.ndarray, 
    stiff_midstep: np.ndarray, 
    split_fluctuations: str = 'vector',
    groundstate_definition: str = 'midstep',
    rotation_first: bool = True 
    ) -> np.ndarray:
    """Converts stiffness matrix from midstep definition of translations to SE(3) (triad) definition. The rotational
    component has to be expressed in euler coordinates. 

    Args:
        groundstate_euler (np.ndarray): Nx6 dimensional ground state. The rotational components have to be expressed in euler coordinates.
        Translational component of the ground state can either be expressed in the coodinate
        system of the midstep-triad (groundstate_definition = 'midstep') or in the coodinate system of the first triad of the repsective pair
        (groundstate_definition = 'triad').
        stiff_midstep (np.ndarray): stiffness matrix in euler coodinates. Static and dynamic components may either be split at the lie algebra level (so(3) -> split_fluctuations = 'vector), or at the lie
        group level (SO(3) split_fluctuations = 'matrix').
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """ 
    warn('This method is deprecated and possibly generates unexpected results.', DeprecationWarning, stacklevel=2)
    
    Tt2m = triad2midstep_lintrans(
        groundstate_euler,
        rotation_first=rotation_first,
        split_fluctuations=split_fluctuations,
        groundstate_definition=groundstate_definition)
    return Tt2m.T @ stiff_midstep @ Tt2m


def triad2midstep_stiffmat(
    groundstate_euler: np.ndarray, 
    stiff_triad: np.ndarray, 
    split_fluctuations: str = 'vector',
    groundstate_definition: str = 'midstep',
    rotation_first: bool = True 
    ) -> np.ndarray:
    """Converts stiffness matrix from SE(3) (triad) definition of translations to midstep definition. The rotational
    component has to be expressed in euler coordinates. 

    Args:
        groundstate_euler (np.ndarray): Nx6 dimensional ground state. The rotational components have to be expressed in euler coordinates.
        Translational component of the ground state can either be expressed in the coodinate
        system of the midstep-triad (groundstate_definition = 'midstep') or in the coodinate system of the first triad of the repsective pair
        (groundstate_definition = 'triad').
        stiff_midstep (np.ndarray): stiffness matrix in euler coodinates. Static and dynamic components may either be split at the lie algebra level (so(3) -> split_fluctuations = 'vector), or at the lie
        group level (SO(3) split_fluctuations = 'matrix').
        rotation_first (bool): If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational degrees of fre

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """ 
    warn('This method is deprecated and possibly generates unexpected results.', DeprecationWarning, stacklevel=2)
    
    Tm2t = midstep2triad_lintrans(
        groundstate_euler,
        rotation_first=rotation_first,
        split_fluctuations=split_fluctuations,
        groundstate_definition=groundstate_definition)
    return Tm2t.T @ stiff_triad @ Tm2t