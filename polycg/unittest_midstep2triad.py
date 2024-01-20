import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp
import sys

from typing import List, Tuple, Callable, Any, Dict
from .cgnaplus import cgnaplus_bps_params

from .transform_cayley2euler import *
from .transform_marginals import *
from .transform_statevec import *
from .kullbackleibler import *
from .transform_translation_midstep2triad import *

from numba import njit

    
def test_midstep2triad(seq: str = 'ACGATC',num_confs = 10000):
    np.set_printoptions(linewidth=250)
    
    # generate stiffness
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq)
    
    # convert to eulers
    midstep_gs  = cayley2euler(cayley_gs)
    midstep_stiff = cayley2euler_stiffmat(cayley_gs,cayley_stiff,rotation_first=True)
    
    # Sample midstep ensemble
    midstep_cov = np.linalg.inv(midstep_stiff)
    midstep_dx = np.random.multivariate_normal(np.zeros(len(midstep_cov)), midstep_cov, num_confs)
    
    midstep_dx = statevec2vecs(midstep_dx,vdim=6)
    midstep = midstep_gs + midstep_dx
    
    triad_gs = translation_midstep2triad(midstep_gs,rotation_map='euler',rotation_first=True)
    triad    = translation_midstep2triad(midstep,rotation_map='euler',rotation_first=True)
    triad_dx = triad - triad_gs
    
    triad_cov = covmat(triad_dx)
    triad_stiff_sampled = np.linalg.inv(triad_cov)
    
    L_t2m = triad2midstep_lintrans(midstep_gs,rotation_first=True,split_fluctuations='vector',groundstate_definition='midstep')
    L_t2m_2 = triad2midstep_lintrans(triad_gs,rotation_first=True,split_fluctuations='vector',groundstate_definition='triad')
    
    print('#####################################')
    print('Check triad2midstep_lintrans and midstep2triad_lintrans groundstate_definition definitions')
    if np.abs(np.sum(L_t2m-L_t2m_2)) <= 1e-12:
        print('triad2midstep consistency checks out')
    else:
        print('triad2midstep inconsistency found!!!!!!')
        print(f'triad2midstep = {np.abs(np.sum(L_t2m-L_t2m_2))}')
        
    # L_m2t   = midstep2triad_lintrans(midstep_gs,rotation_first=True,split_fluctuations='vector',groundstate_definition='midstep')
    # L_m2t_2 = midstep2triad_lintrans(triad_gs,rotation_first=True,split_fluctuations='vector',groundstate_definition='triad')
    # if np.abs(np.sum(L_m2t-L_m2t_2)) <= 1e-12:
    #     print('midstep2triad consistency checks out')
    # else:
    #     print('midstep2triad inconsistency found!!!!!!')
    #     print(f'midstep2triad = {np.abs(np.sum(L_m2t-L_m2t_2))}')

    print(L_t2m[:6,:6])

    triad_stiff = L_t2m.T @ midstep_stiff @ L_t2m
    
    print('\nComparing triad stiffnesses')
    
    print('Triad stiffness sampled midstep ensemble')
    print(triad_stiff_sampled[3:6,3:6])
    
    print('Triad stiffness via transformation')
    print(triad_stiff[3:6,3:6])
    
    print('midstep stiffness')
    print(midstep_stiff[3:6,3:6])
    
    print('rotatonal components')
    print(triad_stiff[:3,:3])
    print(triad_stiff_sampled[:3,:3])
    print(midstep_stiff[:3,:3])
    
    
    
    
    
    

    




    # # full covariance
    # cayley_cov     = np.linalg.inv(cayley_stiff)
    
    # # rotational covariance
    # cayley_rot_cov = np.linalg.inv(cayley_rot_stiff)
    
    # # sample configs
    # cayleys_dx = np.random.multivariate_normal(np.zeros(len(cayley_cov)), cayley_cov, num_confs)
    # cayleys_rot_dx = np.random.multivariate_normal(np.zeros(len(cayley_rot_cov)), cayley_rot_cov, num_confs)
    
    # cayleys_dx     = statevec2vecs(cayleys_dx,vdim=6)
    # cayleys_rot_dx = statevec2vecs(cayleys_rot_dx,vdim=3)
    
    # cayleys     = cayley_gs     + cayleys_dx
    # cayleys_rot = cayley_rot_gs + cayleys_rot_dx
    
    # print(cayleys.shape)
    # print(cayleys_rot.shape)
    
    # cov    = covmat(cayleys_dx)
    # covrot = covmat(cayleys_rot_dx)
    
    
    
    
    
    
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
    num_confs = 100000
    
    test_midstep2triad(seq,num_confs=num_confs)
    

    
    


