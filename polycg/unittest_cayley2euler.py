import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp
import sys

from typing import List, Tuple, Callable, Any, Dict
from .cgnaplus import cgnaplus_bps_params

from .transforms.transform_cayley2euler import *
from .transforms.transform_marginals import *
from .transforms.transform_statevec import *
from .evals.kullbackleibler import *
from numba import njit

def test_order_of_rotation_marginals(seq: str = 'ACGATC'):
    np.set_printoptions(linewidth=250)
    
    # generate stiffness
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq)
    
    # rotational marginals
    cayley_rot_gs    = cayley_gs[:,:3]
    cayley_rot_stiff = matrix_rotmarginal(cayley_stiff)
    
    # full covariance
    cayley_cov     = np.linalg.inv(cayley_stiff)
    
    # rotational covariance
    cayley_rot_cov = np.linalg.inv(cayley_rot_stiff)
    
    # check consistency of rotational components
    if np.abs(np.sum(cayley_cov[:3,:3]-cayley_rot_cov[:3,:3])) > 1e-10:
        print(f'Rotational covariance inconsistent')
    else:
        print('Rotational covariance marginals checks out')
    
    
def test_cayley2euler(seq: str = 'ACGATC',num_confs = 10000):
    np.set_printoptions(linewidth=250)
    
    # generate stiffness
    # parameter_set_name = 'Prmset_cgRNA+_OL3_CGF_10mus_int_12mus_ends'
    # cayley_gs,cayley_stiff = cgnaplus_bps_params(seq,parameter_set_name=parameter_set_name)
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq)
    
    # rotational marginals
    cayley_rot_gs    = cayley_gs[:,:3]
    cayley_rot_stiff = matrix_rotmarginal(cayley_stiff)
    
    # full covariance
    cayley_cov     = np.linalg.inv(cayley_stiff)
    
    # rotational covariance
    cayley_rot_cov = np.linalg.inv(cayley_rot_stiff)
    
    # sample configs
    cayleys_dx = np.random.multivariate_normal(np.zeros(len(cayley_cov)), cayley_cov, num_confs)
    cayleys_rot_dx = np.random.multivariate_normal(np.zeros(len(cayley_rot_cov)), cayley_rot_cov, num_confs)
    
    cayleys_dx     = statevec2vecs(cayleys_dx,vdim=6)
    cayleys_rot_dx = statevec2vecs(cayleys_rot_dx,vdim=3)
    
    cayleys     = cayley_gs     + cayleys_dx
    cayleys_rot = cayley_rot_gs + cayleys_rot_dx
    
    print(cayleys.shape)
    print(cayleys_rot.shape)
    
    cov    = covmat(cayleys_dx)
    covrot = covmat(cayleys_rot_dx)
    
    print((cov[:3,:3]-covrot[:3,:3])/covrot[:3,:3])
    
    # transform to euler
    eulers     = cayley2euler(cayleys)
    eulers_rot = cayley2euler(cayleys_rot)
    eulers_gs  = cayley2euler(cayley_gs)
    eulers_rot_gs = cayley2euler(cayley_rot_gs)
    eulers_dx     = eulers - eulers_gs
    eulers_rot_dx = eulers_rot - eulers_rot_gs
    
    # test transformation
    if np.abs(np.sum(eulers[:,:,:3]-cayley2euler(cayleys[:,:,:3]))) > 1e-10:
        print('Caylay2euler inconsistent with and without translations')
    else:
        print('Caylay2euler consistency checks out')
        
    eulers_cov     = covmat(eulers_dx)
    eulers_rot_cov = covmat(eulers_rot_dx)
    
    eulers_stiff     = np.linalg.inv(eulers_cov)
    eulers_rot_stiff = np.linalg.inv(eulers_rot_cov)
    
    print('#####################################')
    # print('Sampled Euler stiffness with translations')
    # print(eulers_stiff[:6,:6])
    print('Sampled Euler stiffness marginalied after transformation')
    print(matrix_rotmarginal(eulers_stiff)[:3,:3])
    print('Sampled Euler stiffness marginalied before transformation')
    print(eulers_rot_stiff[:3,:3])
    
    
    euler_stiff_lintrans = cayley2euler_stiffmat(cayley_gs,cayley_stiff)
    euler_rot_stiff_lintrans = cayley2euler_stiffmat(cayley_rot_gs,cayley_rot_stiff)
    
    # print('Transformed Euler stiffness with translations')
    # print(euler_stiff_lintrans[:6,:6])
    print('Transformed Euler stiffness with translations')
    print(matrix_rotmarginal(euler_stiff_lintrans)[:3,:3])
    print('Transformed Euler stiffness marginalied before transformation')
    print(euler_rot_stiff_lintrans[:3,:3])
    
    # twist-stretch coupling    
    # k = 1
    # print(euler_stiff_lintrans[6*k:6*k+6,6*k:6*k+6])
    # twist_stretch = matrix_marginal(euler_stiff_lintrans[6*k:6*k+6,6*k:6*k+6],select_indices=np.array([0,0,1,1,1,1]))
    # print(twist_stretch)
    # sys.exit()
    
    print('#####################################')
    print(cayley_stiff[:3,:3])
    print(euler2cayley_stiffmat(eulers_gs,euler_stiff_lintrans)[:3,:3])
    print('#####################################')
    print(cayley_rot_stiff[:3,:3])
    print(euler2cayley_stiffmat(eulers_rot_gs,euler_rot_stiff_lintrans)[:3,:3])
    
    
    Hec = so3.euler2cayley_linearexpansion(eulers_rot_gs[0])
    Hce = so3.cayley2euler_linearexpansion(cayley_rot_gs[0])
    
    print(Hec @ Hce)
    print(Hce @ Hec)
    
    mu = vecs2statevec(eulers_rot_gs)
    kldiv1 = kl_divergence(mu,eulers_rot_stiff,mu,euler_rot_stiff_lintrans)
    kldiv2 = kl_divergence(mu,euler_rot_stiff_lintrans,mu,eulers_rot_stiff)
    kldiv3 = kl_divergence_sym(mu,euler_rot_stiff_lintrans,mu,eulers_rot_stiff)
    
    print(f'{kldiv1=}')
    print(f'{kldiv2=}')
    print(f'{kldiv3=}')

    kldiv1 = kl_divergence(vecs2statevec(cayley_rot_gs),cayley_rot_stiff,mu,euler_rot_stiff_lintrans)
    kldiv2 = kl_divergence(mu,euler_rot_stiff_lintrans,vecs2statevec(cayley_rot_gs),cayley_rot_stiff)
    kldiv3 = kl_divergence_sym(mu,euler_rot_stiff_lintrans,vecs2statevec(cayley_rot_gs),cayley_rot_stiff)
    
    print(f'{kldiv1=}')
    print(f'{kldiv2=}')
    print(f'{kldiv3=}')
    
    diff = eulers_rot_stiff - euler_rot_stiff_lintrans
    matdiff1 = np.sqrt(np.trace(diff @ diff)) / len(diff)
    diff = cayley_rot_stiff - euler_rot_stiff_lintrans
    matdiff2 = np.sqrt(np.trace(diff @ diff)) / len(diff)

    print(f'{matdiff1=}')
    print(f'{matdiff2=}')
    
    
@njit
def covmat(vecs):
    dim = vecs.shape[-1]*vecs.shape[-2]
    cov = np.zeros((dim,dim))
    for i in range(len(vecs)):
        cov += np.outer(vecs[i],vecs[i])
    cov /= len(vecs)
    return cov
    








if __name__ == "__main__":
    
    seq = 'ACGATC'*2
    num_confs = 100000
    
    test_order_of_rotation_marginals(seq)
    test_cayley2euler(seq,num_confs=num_confs)
    

    
    


