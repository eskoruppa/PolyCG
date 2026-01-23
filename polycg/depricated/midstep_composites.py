from __future__ import annotations

import numpy as np

from ..SO3 import so3


def midstep_excess_vals(
    groundstate: np.ndarray,  # shape (N, 6): SE(3) parameters for N base-pair steps
    midstep_constraint_locations: list[int],  # M+1 indices marking segment boundaries
    midstep_triads: np.ndarray  # shape (M+1, 4, 4): SE(3) transformation matrices
) -> np.ndarray:  # shape (M, 6): excess values for M segments
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
    groundstate: np.ndarray,  # shape (N, 6): SE(3) parameters
    triad1: np.ndarray,  # shape (4, 4): first SE(3) transformation matrix
    triad2: np.ndarray  # shape (4, 4): second SE(3) transformation matrix
) -> np.ndarray:  # shape (6,): excess SE(3) coordinates
    
    g_ij = np.linalg.inv(triad1) @ triad2
    Smats = midstep_se3_groundstate(groundstate)
    s_ij = np.eye(4)
    for Smat in Smats:
        s_ij = s_ij @ Smat
    d_ij = np.linalg.inv(s_ij) @ g_ij
    X = so3.se3_rotmat2euler(d_ij)
    return X

    
def midstep_se3_groundstate(
    groundstate: np.ndarray  # shape (N, 6): SE(3) parameters
) -> np.ndarray:  # shape (N, 4, 4): SE(3) transformation matrices
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
    groundstate: np.ndarray,  # shape (N, 6): SE(3) parameters
    midstep_constraint_locations: list[int]  # M+1 indices marking segment boundaries
) -> tuple[np.ndarray, list[int]]:  # (transformation matrix shape (N*6, N*6), replaced indices)
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
    groundstate: np.ndarray  # shape (N, 6): SE(3) parameters for segment
) -> np.ndarray:  # shape (6, N*6): composition block matrix
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


def midstep_Saccu(
    srots: np.ndarray,  # shape (N, 3, 3): static rotation matrices
    i: int,  # start index
    j: int  # end index
) -> np.ndarray:  # shape (3, 3): accumulated rotation matrix
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

