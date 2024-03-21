import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .SO3 import so3
from .Transforms.transform_SO3 import euler2rotmat_so3
from .Transforms.transform_marginals import send_to_back_permutation

def nucleosome_free_energy(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],
    midstep_triads: np.ndarray
) -> np.ndarray:
    
    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(
        groundstate,
        midstep_constraint_locations
    )
    
    # transform stiffness matrix
    inv_transform = np.linalg.inv(transform)
    stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
    
    # find contraint excess values
    excess_vals = midstep_excess_vals(
        groundstate,
        midstep_constraint_locations,
        midstep_triads
    )
    C = excess_vals.flatten()
    
    # rearrange stiffness matrix
    full_replaced_ids = list()
    for i in range(len(replaced_ids)):
        full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
    
    P = send_to_back_permutation(len(stiffmat),full_replaced_ids)
    stiffmat_rearranged = P @ stiffmat_transformed @ P.T

    # select fluctuating, constraint and coupling part of matrix
    N  = len(stiffmat)
    NC = len(full_replaced_ids)
    NF = N-NC
    
    MF = stiffmat_rearranged[:NF,:NF]
    MC = stiffmat_rearranged[NF:,NF:]
    MM = stiffmat_rearranged[NF:,:NF]
    
    MFi = np.linalg.inv(MF)
    const_1 = 0.5 * C.T @ MC @ C
    
    MMTC = MM.T @ C
    const_2 = -0.5 * MMTC.T @ MFi @ MMTC
    
    
    
    
    
    
    
    # stiffmat_constrain = 
    
    
    




def midstep_excess_vals(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    midstep_triads: np.ndarray  
):
    
    num = len(midstep_constraint_locations)-1
    excess_vals = np.zeros((num,6))
    for i in range(num):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        triad1 = midstep_triads[i]
        triad2 = midstep_triads[i+1]
        partial_gs = groundstate[id1:id2+1] 
        excess_vals[i] = midstep_composition_excess(partial_gs,triad1,triad2) 
    return excess_vals
    
    


def midstep_composition_excess(
    groundstate: np.ndarray,
    triad1: np.ndarray,
    triad2: np.ndarray
) -> np.ndarray:
    
    g_ij = np.linalg.inv(triad1) @ triad2
    Smats = midstep_se3_groundstate(groundstate)
    s_ij = np.eye(4)
    for Smat in Smats:
        s_ij = s_ij @ Smat
    d_ij = np.linalg.inv(s_ij) @ g_ij
    X = so3.se3_rotmat2euler(d_ij)
    return X

    
def midstep_se3_groundstate(groundstate: np.ndarray):
    Phi0s = groundstate[:,:3]
    N = len(groundstate)
    # assign static rotation matrices
    srots = np.zeros((N,3,3))
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])   
    # assign translation vectors
    trans = np.copy(groundstate[:,3:])
    trans[0] = 0.5*trans[0]
    trans[-1] = 0.5* srots[-1].T @ trans[-1]
    
    Smats = np.zeros((N,4,4))
    for i in range(N):
        S = np.zeros((4,4))
        S[:3,:3] = srots[i]
        S[:3,3]  = trans[i]
        S[3,3]   = 1
        Smats[i] = S
    return Smats



def midstep_composition_transformation(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int]
) -> np.ndarray:
    N = len(groundstate)
    mat = np.eye(N*6)
    replaced_ids = []
    for i in range(len(midstep_constraint_locations)-1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        partial_gs = groundstate[id1:id2+1]
        midstep_comp_block = midstep_composition_block(partial_gs)
        mat[id2*6:id2*6+6,id1*6:id2*6+6] = midstep_comp_block
        replaced_ids.append(id2)
    return mat, replaced_ids


def midstep_composition_block(
    groundstate: np.ndarray
) -> np.ndarray:
    if len(groundstate) < 2:
        raise ValueError(f'midstep_composition_block: grounstate needs to contain at least two elements. {len(groundstate)} provided.')
    
    Phi0s = groundstate[:,:3]
    # ss    = groundstate[:,3:]
    
    N = len(groundstate)
    # assign static rotation matrices
    srots = np.zeros((N,3,3))
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])    
    
    # assign translation vectors
    trans = np.copy(groundstate[:,3:])
    trans[0] = 0.5*trans[0]
    trans[-1] = 0.5* srots[-1].T @ trans[-1]
    
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    comp_block  = np.zeros((ndims,N*ndims))
    
    ################################  
    # set middle blocks (i < k < j)
    for k in range(i,j+1):
        Saccu = midstep_Saccu(srots,k+1,j)
        comp_block[:3,k*6:k*6+3]   = Saccu.T
        comp_block[3:,k*6+3:k*6+6] = Saccu.T
        
        coup = np.zeros((3,3))
        for l in range(k+1,j+1):
            coup += so3.hat_map(-midstep_Saccu(srots,l,j).T @ trans[l])
        coup = coup @ Saccu.T
        comp_block[3:,k*6:k*6+3] = coup
    
    ################################  
    # set first block (i)
    Saccu = midstep_Saccu(srots,i+1,j)
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,:3] = 0.5 * Saccu.T @ Hprod
    comp_block[3:,3:] = 0.5 * Saccu.T
    
    coup = np.zeros((3,3))
    # first term
    for l in range(1,j+1):
        coup += so3.hat_map(-midstep_Saccu(srots,l,j).T @ trans[l])
    coup = coup @ Saccu.T
    # second term
    coup += Saccu.T @ srots[i].T @ so3.hat_map(trans[i])
    # multoply everything with 0.5 * Hprod
    coup = 0.5 * coup @ Hprod
    # assign coupling term
    comp_block[3:,:3] = coup
    
    ################################  
    # set last block (j)
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,j*6:j*6+3]   = 0.5 * Hprod
    comp_block[3:,j*6+3:j*6+6] = 0.5 * srots[-1]
    return comp_block


def midstep_Saccu(srots: np.ndarray,i,j) -> np.ndarray:
    saccu = np.eye(3)
    for k in range(i,j+1):
        saccu = saccu @ srots[k]
    return saccu


# def midstep_srots_and_trans(groundstate: np.ndarray) -> np.ndarray:
#     Phi0s = groundstate[:,:3]
#     N = len(Phi0s)
#     srots = np.zeros((N,3,3))
#     srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
#     srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
#     for l in range(1,len(srots)-1):
#         srots[l] = so3.euler2rotmat(Phi0s[l])  
    
#     trans = np.copy(groundstate[:,3:])
#     trans[0] = 0.5*trans[0]
#     trans[-1] = 0.5* srots[-1].T @ trans[-1]
#     return srots, trans

