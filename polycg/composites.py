import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .SO3 import so3
from .transform_SO3 import euler2rotmat_so3


def composite_matrix(
    groundstate: np.ndarray,
    substitute_block: int = -1
) -> np.ndarray:
    comp_block = composite_block(groundstate)
    comp_block = test_combblock(groundstate)
    # test_comb = test_combblock(groundstate)
    # test_comp = test_compblock(groundstate)
    
    # alt_block = alterate_summation(groundstate)
        
    # print('\n\n')
    # print('COMP')
    # print(groundstate)
    # print(comp_block)
    # print('\n\n')
    # print(alt_block)
    # print('\n\n')
    # print(comp_block-alt_block)
    # print('\n\n')
    
    # sys.exit()
    
    ndims = len(comp_block)
    N = len(groundstate)
    mat = np.eye(N*ndims)
    if substitute_block < 0:
        substitute_block = N + substitute_block
    mat[substitute_block*ndims:(substitute_block+1)*ndims,:] = comp_block
    return mat    


def inv_composite_matrix(
    groundstate: np.ndarray,
    substitute_block: int = -1
) -> np.ndarray:
    # comp_block = composite_block(groundstate)
    # TESTING HERE
    comp_block = test_combblock(groundstate)
    #######################
    # print('\n\n')
    # print('INV')
    # print(groundstate)
    # print(comp_block)
    # print('\n\n')
    
    ndims = len(comp_block)
    N = len(groundstate)
    mat = np.eye(N*ndims)
    if substitute_block < 0:
        substitute_block = N + substitute_block
    comp_block[:,:(N-1)*ndims] *= -1 
    mat[substitute_block*ndims:(substitute_block+1)*ndims,:] = comp_block
    return mat    
    
    
def composite_block(
    groundstate: np.ndarray
):
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
    
    # print('\n\n')
    # print(comp_block)
    # print('\n\n')
    # print(test_compblock(groundstate))
    # print(comp_block - test_compblock(groundstate))
    # print(np.abs(np.sum(comp_block - test_compblock(groundstate))))
    return comp_block
    
    
def test_compblock(groundstate: np.ndarray) -> np.ndarray:
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


def test_combblock(groundstate: np.ndarray) -> np.ndarray:
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


def alterate_summation(groundstate: np.ndarray) -> np.ndarray:
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


        
def get_Saccu(rots: np.ndarray,i,j) -> np.ndarray:
    saccu = np.eye(3)
    for k in range(i,j+1):
        saccu = saccu @ so3.euler2rotmat(rots[k])
    return saccu
    