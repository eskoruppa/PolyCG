from __future__ import annotations

import numpy as np

from .SO3 import so3
from .transforms.transform_SO3 import euler2rotmat_so3, rotmat2euler_so3
from .transforms.transform_SE3 import euler2rotmat_se3, rotmat2euler_se3


def composite_groundstate(
    groundstate: np.ndarray  # shape (N, 3) for SO(3) or (N, 6) for SE(3)
) -> np.ndarray:  # shape (3,) for SO(3) or (6,) for SE(3): composite parameters
    """
    Compute the composite groundstate by accumulating rotation matrices.
    
    For SO(3) (rotation-only), composes N rotation matrices by sequential multiplication
    and returns the resulting Euler angles. For SE(3) (rotation and translation),
    composes N transformation matrices and returns the composite parameters.
    
    Args:
        groundstate: Array of shape (N, 3) for rotations only (Euler angles) or
                     (N, 6) for rotations and translations (Euler + translation vectors).
    
    Returns:
        Composite parameters as Euler angles (3,) for SO(3) or Euler + translation (6,) for SE(3).
    
    Raises:
        ValueError: If groundstate has invalid dimension (not 3 or 6 per row).
    """
    if len(groundstate[0]) == 3:
        # rotations only
        smats = euler2rotmat_so3(groundstate)
        saccu = np.eye(3)
        for smat in smats:
            saccu = saccu @ smat
        return so3.rotmat2euler(saccu)
        
    elif len(groundstate[0]) == 6:
        # rotations and translations
        smats = euler2rotmat_se3(groundstate)
        saccu = np.eye(4)
        for smat in smats:
            saccu = saccu @ smat
        return rotmat2euler_se3(saccu)
    else:
        raise ValueError(f'Invalid dimension of groundstate vectors {len(groundstate[0])}.')


def composite_matrix(
    groundstate: np.ndarray,  # shape (N, 3) or (N, 6): SE(3) or SO(3) parameters
    substitute_block: int = -1  # index of block to substitute (-1 for last)
) -> np.ndarray:  # shape (N*ndims, N*ndims): composite transformation matrix
    """
    Build a composite transformation matrix by substituting one block row.
    
    Creates an identity matrix of size (N*ndims, N*ndims) and replaces one block row
    with the composite block computed from the groundstate parameters. This is used
    in coarse-graining operations to relate local parameters to composite parameters.
    
    Args:
        groundstate: Array of shape (N, 3) for SO(3) or (N, 6) for SE(3) parameters.
        substitute_block: Index of the block row to substitute with composite block.
                         Negative indices count from the end (default -1 for last block).
    
    Returns:
        Composite transformation matrix of shape (N*ndims, N*ndims).
    """
    comp_block = composite_block(groundstate)    
    ndims = len(comp_block)
    N = len(groundstate)
    mat = np.eye(N*ndims)
    if substitute_block < 0:
        substitute_block = N + substitute_block
    mat[substitute_block*ndims:(substitute_block+1)*ndims,:] = comp_block
    return mat    


def inv_composite_matrix(
    groundstate: np.ndarray,  # shape (N, 3) or (N, 6): SE(3) or SO(3) parameters
    substitute_block: int = -1  # index of block to substitute (-1 for last)
) -> np.ndarray:  # shape (N*ndims, N*ndims): inverse composite transformation matrix
    """
    Build the inverse composite transformation matrix.
    
    Creates an identity matrix and substitutes one block row with the inverse composite
    block. The inverse is computed by negating all but the last block of columns in the
    composite block, which corresponds to the inverse transformation.
    
    Args:
        groundstate: Array of shape (N, 3) for SO(3) or (N, 6) for SE(3) parameters.
        substitute_block: Index of the block row to substitute with inverse composite block.
                         Negative indices count from the end (default -1 for last block).
    
    Returns:
        Inverse composite transformation matrix of shape (N*ndims, N*ndims).
    """
    comp_block = composite_block(groundstate)    
    ndims = len(comp_block)
    N = len(groundstate)
    mat = np.eye(N*ndims)
    if substitute_block < 0:
        substitute_block = N + substitute_block
    comp_block[:,:(N-1)*ndims] *= -1 
    mat[substitute_block*ndims:(substitute_block+1)*ndims,:] = comp_block
    return mat    
    
    
def composite_block(
    groundstate: np.ndarray  # shape (N, 3) or (N, 6): SE(3) or SO(3) parameters
) -> np.ndarray:  # shape (ndims, N*ndims): composite block matrix
    """
    Compute the composite block matrix for coarse-graining operations.
    
    This is the core function for building composite transformation blocks. It computes
    how N local parameters (rotations or rotations+translations) relate to a single
    composite parameter through accumulated transformations. The algorithm pre-computes
    all accumulated rotation matrices and coupling terms efficiently.
    
    For SO(3) (rotation-only), returns a (3, N*3) block containing accumulated rotation
    transposes. For SE(3) (rotation+translation), returns a (6, N*6) block containing
    both rotation blocks, translation blocks, and coupling blocks that account for the
    interaction between rotations and translations.
    
    Args:
        groundstate: Array of shape (N, 3) for rotations only or (N, 6) for
                     rotations and translations.
    
    Returns:
        Composite block matrix of shape (3, N*3) for SO(3) or (6, N*6) for SE(3).
    
    Raises:
        ValueError: If groundstate does not have shape (N, 3) or (N, 6).
    """
    if len(groundstate.shape) != 2 or groundstate.shape[-1] not in [3,6]:
        raise ValueError(f'groundstate is expected to have dimension Nx3 (rotation only) or Nx6 rotation and translation. Instead received shape {groundstate.shape}')
        
    if groundstate.shape[-1] == 3:
        include_translation = False
        ndims = 3    
    else:
        include_translation = True    
        ndims = 6
    
    N = len(groundstate)
    
    comp_block  = np.zeros((ndims,N*ndims))
    rots  = groundstate[:,:3]
    trans = groundstate[:,3:]
    
    # initiate accumulative rotations
    smats = euler2rotmat_so3(rots)
    Saccus = np.zeros((N+1,3,3))
    Saccu = np.eye(3)
    Saccus[-1] = Saccu
    for i in range(1,N+1):
        Saccu = smats[-i] @ Saccu
        Saccus[-1-i] = Saccu
    
    # assign rotation blocks
    for k in range(N):
        comp_block[:3,k*ndims:k*ndims+3] = Saccus[k+1].T
       
    # if translations are not included 
    if not include_translation:
        return comp_block
    
    # initiate accumulative translations
    hmaccus = np.zeros((N,3,3))
    hmaccu = np.zeros((3,3))
    hmaccus[-1] = hmaccu
    for i in range(1,N):
        hmaccu += so3.hat_map(-Saccus[-i-1].T @ trans[-i])
        hmaccus[-1-i] = hmaccu
    
    # assign translation and coupling blocks
    for k in range(N):
        coup = hmaccus[k] @ Saccus[k+1].T     
        comp_block[3:,k*ndims:k*ndims+3] = coup 
        comp_block[3:,k*ndims+3:k*ndims+6] = Saccus[k+1].T    
    
    return comp_block
    
    
def test_compblock(
    groundstate: np.ndarray  # shape (N, 6): SE(3) parameters
) -> np.ndarray:  # shape (6, N*6): test composite block
    rots  = groundstate[:,:3]
    trans = groundstate[:,3:]
    ndims = 6
    N = len(groundstate)
    comp_block  = np.zeros((ndims,N*ndims))
    for k in range(N):
        Saccu = get_Saccu(rots,k+1,N-1)
        comp_block[:3,k*ndims:k*ndims+3]   = Saccu.T
        comp_block[3:,k*ndims+3:k*ndims+6] = Saccu.T
        hm = np.zeros((3,3))
        for l in range(k+1,N):
           hm += so3.hat_map(-get_Saccu(rots,l,N-1).T @ trans[l])
        comp_block[3:,k*ndims:k*ndims+3] = hm @ Saccu.T
    return comp_block


def test_combblock(
    groundstate: np.ndarray  # shape (N, 6): SE(3) parameters
) -> np.ndarray:  # shape (6, N*6): test composite block
    rots  = groundstate[:,:3]
    trans = groundstate[:,3:]
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    comp_block  = np.zeros((ndims,N*ndims))
    for k in range(i,j+1):
        Saccu = get_Saccu(rots,k+1,j)
        comp_block[:3,k*6:k*6+3]   = Saccu.T
        comp_block[3:,k*6+3:k*6+6] = Saccu.T
        
        coup = np.zeros((3,3))
        for l in range(k+1,j+1):
            coup += so3.hat_map(-get_Saccu(rots,l,j).T @ trans[l])
        coup = coup @ Saccu.T
        comp_block[3:,k*6:k*6+3] = coup
    return comp_block


def alterate_summation(
    groundstate: np.ndarray  # shape (N, 6): SE(3) parameters
) -> np.ndarray:  # shape (6, N*6): alternate summation composite block
    """
    Compute composite block using an alternate summation order.
    
    This is an alternative implementation of the composite block calculation that
    uses a different loop structure for the coupling terms. It produces the same
    result as composite_block but may be useful for testing or understanding the
    mathematical structure of the transformation.
    
    Args:
        groundstate: Array of shape (N, 6) containing rotation and translation parameters.
    
    Returns:
        Composite block matrix of shape (6, N*6).
    """
    rots  = groundstate[:,:3]
    trans = groundstate[:,3:]
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    comp_block  = np.zeros((ndims,N*ndims))
    for k in range(i,j+1):
        Saccu = get_Saccu(rots,k+1,j)
        comp_block[:3,k*6:k*6+3]   = Saccu.T
        comp_block[3:,k*6+3:k*6+6] = Saccu.T
    for l in range(i,j+1):
        for k in range(i,l):
            term = so3.hat_map(-get_Saccu(rots,l,j).T @ trans[l]) @ get_Saccu(rots,k+1,j).T
            comp_block[3:,k*6:k*6+3] += term
    return comp_block


def get_Saccu(
    rots: np.ndarray,  # shape (N, 3): rotation parameters
    i: int,  # start index
    j: int  # end index
) -> np.ndarray:  # shape (3, 3): accumulated rotation matrix
    """
    Accumulate rotation matrices from index i to j (inclusive).
    
    Sequentially multiplies rotation matrices converted from Euler angles for the
    specified index range. Returns the product R_i @ R_{i+1} @ ... @ R_j.
    
    Args:
        rots: Array of shape (N, 3) containing Euler angles for N rotations.
        i: Starting index (inclusive).
        j: Ending index (inclusive).
    
    Returns:
        Accumulated rotation matrix of shape (3, 3).
    """
    saccu = np.eye(3)
    for k in range(i,j+1):
        saccu = saccu @ so3.euler2rotmat(rots[k])
    return saccu
    