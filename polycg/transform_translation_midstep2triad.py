import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .SO3 import so3
from .pyConDec.pycondec import cond_jit


##########################################################################################################
############# Transformation between midsteptriad and triad definitions of translations ##################
##########################################################################################################

def translation_midstep2triad(vecs: np.ndarray, rotation_map: str = 'euler', rotation_first: bool = True):
    if vecs.shape[-1] != 6:
        raise ValueError(f"Expected set of 6-vectors. Instead received shape {vecs.shape}")
    
    nvecs = np.copy(vecs)
    if len(vecs.shape) > 2:
        for i, vec in enumerate(vecs):
            nvecs[i] = translation_midstep2triad(vec)
        return nvecs
    
    if rotation_first:  
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    if rotation_map == 'euler':
        vec2rotmat = so3.euler2rotmat
        # rotmat2vec = so3.rotmat2euler
    elif rotation_map == 'cayley':
        vec2rotmat = so3.cayley2rotmat
        # rotmat2vec = so3.rotmat2cayley
    else:
        raise ValueError(f'Invalid rotation_method "{rotation_map}".')
     
    for i, vec in enumerate(vecs):
        vrot   = vec[rotslice]
        vtrans = vec[transslice]
        rotmat = vec2rotmat(vrot)
        sqrt_rotmat = so3.sqrt_rot(rotmat) 
        nvecs[i,transslice] = sqrt_rotmat @ vtrans
    return nvecs

def translation_triad2midstep(vecs: np.ndarray, rotation_map: str = 'euler', rotation_first: bool = True):
    if vecs.shape[-1] != 6:
        raise ValueError(f"Expected set of 6-vectors. Instead received shape {vecs.shape}")
    
    nvecs = np.copy(vecs)
    if len(vecs.shape) > 2:
        for i, vec in enumerate(vecs):
            nvecs[i] = translation_triad2midstep(vec)
        return nvecs
    
    if rotation_first:  
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    if rotation_map == 'euler': 
        vec2rotmat = so3.euler2rotmat
        # rotmat2vec = so3.rotmat2euler
    elif rotation_map == 'cayley':
        vec2rotmat = so3.cayley2rotmat
        # rotmat2vec = so3.rotmat2cayley
    else:
        raise ValueError(f'Invalid rotation_map "{rotation_map}".')
     
    for i, vec in enumerate(vecs):
        vrot   = vec[rotslice]
        vtrans = vec[transslice]
        rotmat = vec2rotmat(vrot)
        sqrt_rotmat = so3.sqrt_rot(rotmat) 
        nvecs[i,transslice] = sqrt_rotmat.T @ vtrans
    return nvecs
    

##########################################################################################################
###### Linearization of transformation between midsteptriad and triad definitions of translations ########
##########################################################################################################


def midstep2triad_lintrans(
    groundstate_euler: np.ndarray, 
    rotations_first: bool = True, 
    split_fluctuations: str = 'vector',
    groundstate_definition: str = 'midstep'
    ) -> np.ndarray:
    """Linearization of transformation from midsteptriad- to triad-definitions of translations. The rotational component needs to be expressed in euler coordinates.
    """
    
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
        if rotations_first:
            Omega0 = groundstate_euler[:3]
            zeta0  = groundstate_euler[3:]
        else:
            Omega0 = groundstate_euler[3:]
            zeta0  = groundstate_euler[:3]
            
        if groundstate_definition != 'midstep':
            zeta0 = sqrt_rotmat.T @ zeta0
        
        sqrt_rotmat  = so3.euler2rotmat(0.5*Omega0) 
        zeta0_hat    = so3.hat_map(zeta0)
        Hm = np.eye(6)
        Hm[3:,3:] = sqrt_rotmat
        
        crosscoup = 0.5*sqrt_rotmat @ zeta0_hat.T
        if split_fluctuations == 'so3':
            crosscoup = crosscoup @ so3.splittransform_algebra2group(Omega0)     
        Hm[3:,:3] = crosscoup
        return Hm

    ###################################################
    ###################################################
    
    if len(groundstate_euler.shape) > 2:
        raise ValueError(f'Expected array of shape (N,6), encountered {groundstate_euler.shape}')

    Hm = np.zeros((len(groundstate_euler)*6 ,)*2)
    for i, gs_euler in enumerate(groundstate_euler):
        Hm[i*6:(i+1)*6,i*6:(i+1)*6] = midstep2triad_lintrans(
            gs_euler,
            rotations_first=rotations_first,
            split_fluctuations=split_fluctuations)
    
    return Hm


def triad2midstep_lintrans(
    groundstate_euler: np.ndarray, 
    rotations_first: bool = True, 
    split_fluctuations: str = 'vector',
    groundstate_definition: str = 'midstep'
    ) -> np.ndarray:
    """Linearization of transformation from midsteptriad- to triad-definitions of translations. The rotational component needs to be expressed in euler coordinates.
    """
    
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
        if rotations_first:
            Omega0 = groundstate_euler[:3]
            zeta0  = groundstate_euler[3:]
        else:
            Omega0 = groundstate_euler[3:]
            zeta0  = groundstate_euler[:3]
            
        if groundstate_definition != 'midstep':
            zeta0 = sqrt_rotmat.T @ zeta0
        
        sqrt_rotmat  = so3.euler2rotmat(0.5*Omega0) 
        zeta0_hat    = so3.hat_map(zeta0)
        H = np.eye(6)
        H[3:,3:] = sqrt_rotmat.T
        
        crosscoup = 0.5 * zeta0_hat
        if split_fluctuations == 'so3':
            crosscoup = crosscoup @ so3.splittransform_algebra2group(Omega0) 
        H[3:,:3] = crosscoup
        return H

    ###################################################
    ###################################################
    
    if len(groundstate_euler.shape) > 2:
        raise ValueError(f'Expected array of shape (N,6), encountered {groundstate_euler.shape}')

    H = np.zeros((len(groundstate_euler)*6 ,)*2)
    for i, gs_euler in enumerate(groundstate_euler):
        H[i*6:(i+1)*6,i*6:(i+1)*6] = midstep2triad_lintrans(
            gs_euler,
            rotations_first=rotations_first,
            split_fluctuations=split_fluctuations)
    
    return H
