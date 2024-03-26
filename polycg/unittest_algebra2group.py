import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp
import sys, time

from typing import List, Tuple, Callable, Any, Dict
from .cgnaplus import cgnaplus_bps_params

from .Transforms.transform_cayley2euler import *
from .Transforms.transform_marginals import *
from .Transforms.transform_statevec import *
from .Evals.kullbackleibler import *
from .Transforms.transform_algebra2group import *
from .Transforms.transform_midstep2triad import *

from numba import njit

    
def test_algebra2group(seq: str = 'ACGATC',num_confs = 10000):
    np.set_printoptions(linewidth=250)
    
    # generate stiffness
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq,translations_in_nm=False,euler_definition=False)
    print('stiff generated')
    
    
    translation_as_midstep = True
    
    # convert to eulers
    algebra_gs  = cayley2euler(cayley_gs)
    
    print('converted stiff to euler')
    algebra_stiff = cayley2euler_stiffmat(cayley_gs,cayley_stiff,rotation_first=True)
    
    ### CHECK FUNCTION DEFINITION
    # gs,sti = cgnaplus_bps_params(seq,translations_in_nm=False,group_split=False,euler_definition=True)
    # print(np.sum(algebra_stiff-sti))
    # sys.exit()
    ### CHECKS OUT
    
    print('converted gs to euler')
    group_gs = np.copy(algebra_gs)
    
    if translation_as_midstep:
        for i,vec in enumerate(group_gs):
            Phi_0 = vec[:3]
            zeta_0 = vec[3:]
            sqrtS = so3.euler2rotmat(0.5*Phi_0)
            s = sqrtS @ zeta_0
            group_gs[i,3:] = s
            
    # linear transformations
    HX = algebra2group_lintrans(algebra_gs,rotation_first=True,translation_as_midstep=translation_as_midstep)
    HX_inv = group2algebra_lintrans(group_gs,rotation_first=True,translation_as_midstep=translation_as_midstep)
    
    group_stiff = algebra2group_stiffmat(algebra_gs,algebra_stiff,rotation_first=True,translation_as_midstep=translation_as_midstep)
    
    
    # # HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # gs,sti = cgnaplus_bps_params(seq,translations_in_nm=False,group_split=True,euler_definition=True)
    # print(np.sum(group_stiff-sti))
    # sys.exit()
    
    test_algebra_stiff = group2algebra_stiffmat(group_gs,group_stiff,rotation_first=True,translation_as_midstep=translation_as_midstep)
    print(f'group_stiff transformation test: diff = {np.sum(algebra_stiff-test_algebra_stiff)}')
    
    ########################
    print('Sample algebra')
    algebra_cov = np.linalg.inv(algebra_stiff)
    algebra_dx = np.random.multivariate_normal(np.zeros(len(algebra_cov)), algebra_cov, num_confs)
    
    print('sampled')
    algebra_dx = statevec2vecs(algebra_dx,vdim=6)
    algebra = algebra_gs + algebra_dx
    
    algebra_triad = np.copy(algebra)
    if translation_as_midstep:
        print('Convert midstep to triad definition of translations')
        for i,conf in enumerate(algebra_triad):
            for j,vec in enumerate(conf):
                Phi_0 = vec[:3]
                zeta_0 = vec[3:]
                sqrtS = so3.euler2rotmat(0.5*Phi_0)
                s = sqrtS @ zeta_0
                algebra_triad[i,j,3:] = s
                
    print('transform to rotmats')
    gs_rotmats = euler2rotmat(group_gs)
    rotmats = euler2rotmat(algebra_triad)

    # test_algebra = rotmat2euler(rotmats)
    # print(f'euler2rotmat transformation test: diff = {np.sum(algebra_triad-test_algebra)}')

    print('calculate group_dx')
    group_dx = np.zeros(algebra_dx.shape)
    for i,g_set in enumerate(rotmats):
        for j,g in enumerate(g_set):
            s = gs_rotmats[j]
            d = np.linalg.inv(s) @ g
            # print(d)
            group_dx[i,j] = so3.se3_rotmat2euler(d,rotation_first=True)
            # print(group_dx[i,j])
    
    print('calculate covariance matrix')
    group_cov = covmat(group_dx)
    print('calculate stiffness matrix')
    group_stiff_sampled = np.linalg.inv(group_cov)
    
    print('\n\nSTEP 2')
    print('Group stiffness sampled')
    print(group_stiff_sampled[6:12,6:12])
    print('Group stiffness transformed')
    print(group_stiff[6:12,6:12])
    print('Difference')
    print((group_stiff-group_stiff_sampled)[6:12,6:12])
    print('Relative Difference')
    print(((group_stiff-group_stiff_sampled)/group_stiff_sampled)[6:12,6:12])
    
    
    
    print('\n\nSTEP 3')
    print('Group stiffness sampled')
    print(group_stiff_sampled[12:18,12:18])
    print('Group stiffness transformed')
    print(group_stiff[12:18,12:18])
    print('Difference')
    print((group_stiff-group_stiff_sampled)[12:18,12:18])
    print('Relative Difference')
    print(((group_stiff-group_stiff_sampled)/group_stiff_sampled)[12:18,12:18])
    
    
    print('\n\nGroup stiffness transformed rotmarginal')
    group_stiff_rotmarginal = matrix_rotmarginal(group_stiff)
    print(group_stiff_rotmarginal[3:9,3:9])
    
    
    
    ########################
    print('\n\nTest transformation only for rotations')
    # rotational marginals
    algebra_rot_gs    = algebra_gs[:,:3]
    algebra_rot_stiff = matrix_rotmarginal(algebra_stiff)
    
    group_rot_stiff = algebra2group_stiffmat(algebra_rot_gs,algebra_rot_stiff,rotation_first=True,translation_as_midstep=translation_as_midstep)
    

    print(group_rot_stiff.shape)
    print(algebra_rot_gs.shape)
    print(group_rot_stiff[3:9,3:9]*0.34)
    
    ########################
    print('Sample rot algebra')
    algebra_rot_cov = np.linalg.inv(algebra_rot_stiff)
    algebra_rot_dx = np.random.multivariate_normal(np.zeros(len(algebra_rot_cov)), algebra_rot_cov, num_confs)
    
    print('calculate full algebra vectors')
    algebra_rot_dx = statevec2vecs(algebra_rot_dx,vdim=3)
    algebra_rot = algebra_rot_gs + algebra_rot_dx
    
    print('transform to rotmats')
    gs_rotmats_rot = euler2rotmat(algebra_rot_gs)
    rotmats_rot = euler2rotmat(algebra_rot)

    test_algebra_rot = rotmat2euler(rotmats_rot)
    print(f'euler2rotmat rot transformation test: diff = {np.sum(algebra_rot-test_algebra_rot)}')

    print('calculate group_dx')
    group_rot_dx = np.zeros(algebra_rot_dx.shape)
    for i,g_set in enumerate(rotmats_rot):
        for j,g in enumerate(g_set):
            s = gs_rotmats_rot[j]
            
            d = s.T @ g
            # print(d)
            group_rot_dx[i,j] = so3.rotmat2euler(d)
            # print(group_dx[i,j])
    
    print('calculate covariance matrix')
    group_rot_cov = covmat(group_rot_dx)
    group_rot_stiff_sampled = np.linalg.inv(group_rot_cov)
    
    print('Group stiffness sampled')
    print(group_rot_stiff_sampled[3:9,3:9])
    
    print('Group stiffness transformed')
    print(group_rot_stiff[3:9,3:9])

    
    
    
@njit
def covmat(vecs):
    dim = vecs.shape[-1]*vecs.shape[-2]
    cov = np.zeros((dim,dim))
    for i in range(len(vecs)):
        cov += np.outer(vecs[i],vecs[i])
    cov /= len(vecs)
    return cov
    




if __name__ == "__main__":
    
    seq = 'ACGATC'
    num_confs = 1000000
    
    print(f'len = {len(seq)}')
    
    test_algebra2group(seq,num_confs=num_confs)