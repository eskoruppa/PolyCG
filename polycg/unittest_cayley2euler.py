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
from numba import njit

def test_order_of_rotation_marginals(seq: str = 'ACGATC',num_confs = 100000):
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
    eulers     = cayleys2eulers(cayleys)
    eulers_rot = cayleys2eulers(cayleys_rot)
    eulers_gs  = cayleys2eulers(cayley_gs)
    eulers_rot_gs = cayleys2eulers(cayley_rot_gs)
    eulers_dx     = eulers - eulers_gs
    eulers_rot_dx = eulers_rot - eulers_rot_gs
    
    # test transformation
    if np.abs(np.sum(eulers[:,:,:3]-cayleys2eulers(cayleys[:,:,:3]))) > 1e-10:
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
    
    test_order_of_rotation_marginals(seq)
    sys.exit()
    

    
    





    gs_euler    = cayleys2eulers(gs)
    stiff_euler = cayley2euler_stiffmat(gs,stiff)
    
    
    print(stiff.shape)
    
    rot_stiff_cayley = matrix_rotmarginal(stiff)
    rot_gs_cayley    = gs[:,:3]
    
    print(np.linalg.inv(rot_stiff_cayley)[:3,:3])
    print(np.linalg.inv(stiff)[:3,:3])
    

    sys.exit()
    
    
        
    rot_stiff_cayley = stiff
    rot_gs_cayley    = gs
    
    rot_stiff_euler = cayley2euler_stiffmat(rot_gs_cayley,rot_stiff_cayley)
    rot_gs_euler    = cayleys2eulers(rot_gs_cayley)
    

    num_confs = 100000
    rot_covmat_cayley = np.linalg.inv(rot_stiff_cayley)
    
    dx_cayley = np.random.multivariate_normal(np.zeros(len(rot_covmat_cayley)), rot_covmat_cayley, num_confs)
    
    rot_cayleys = rot_gs_cayley + statevec2vecs(dx_cayley,vdim=6)
    rot_eulers  = cayleys2eulers(rot_cayleys)
    
    print(rot_cayleys[1])
    print(rot_eulers[1])
    sys.exit()
    
    
    rot_euler_dx = rot_eulers - rot_gs_euler
    
    cov = np.zeros(rot_covmat_cayley.shape)
    for i in range(len(rot_euler_dx)):
        cov += np.outer(rot_euler_dx[i],rot_euler_dx[i])
    cov /= len(rot_euler_dx)
    
    rot_stiff_euler_sampled = np.linalg.inv(cov)
    


    print(rot_stiff_euler[:3,:3])
    print(rot_stiff_euler_sampled[:3,:3])
    
    # print(matrix_rotmarginal(rot_stiff_euler_sampled)[:3,:3])
    
    
    
    sys.exit()
    
    
    
    num_confs = 100000
    covmat_euler  = np.linalg.inv(stiff_euler)
    covmat_cayley = np.linalg.inv(stiff)
    
    dx_cayley = np.random.multivariate_normal(np.zeros(len(covmat_cayley)), covmat_cayley, num_confs)
    # dx_euler  = np.random.multivariate_normal(np.zeros(len(covmat_cayley)), covmat_cayley, num_confs)
    
    cayleys = gs + statevec2vecs(dx_cayley,vdim=6)
    eulers  = cayleys2eulers(cayleys)
    
    # mean euler
    eulers_state = vecs2statevec(eulers)
    mean_euler = np.mean(eulers_state,axis=0)
    dx_euler = eulers_state - vecs2statevec(gs_euler)
    # cov euler
    cov = np.zeros((len(eulers_state[-1]),)*2)
    for i in range(len(dx_euler)):
        cov += np.outer(dx_euler[i],dx_euler[i])
    cov /= len(dx_euler)
    
    print('diff in mean')
    print((statevec2vecs(mean_euler,vdim=6)-gs_euler) / gs_euler)
    
    euler_stiff_sampled = np.linalg.inv(cov)

    print(f'diff stiff = {np.sum(stiff_euler - euler_stiff_sampled)}')
    
    pos = 3
    print(stiff_euler[pos*6:pos*6+3,pos*6:pos*6+3]*0.34)
    print(euler_stiff_sampled[pos*6:pos*6+3,pos*6:pos*6+3]*0.34)
    
    
    sys.exit()
    
    
    
    
    
    
    
    
    
    
    dx = np.random.multivariate_normal(np.zeros(len(covmat)), covmat, num_confs)
    
    
    cov = np.zeros(covmat.shape)
    for i in range(len(dx)):
        cov += np.outer(dx[i],dx[i])
    cov /= len(dx)
    
    stiff_euler_test = np.linalg.inv(cov)
    
    print(f'diff = {np.sum(stiff_euler-stiff_euler_test)}')
    
    print(stiff_euler[:6,:6])
    print(stiff_euler_test[:6,:6])
    
    
    sys.exit()
    
    
    
    
    # print(stiff[:6,:6])
    # print(stiff_euler[:6,:6])
    
    from .transform_translation_midstep2triad import *
    
    gs_euler_triad = translation_midstep2triad(gs_euler,rotation_map = 'euler')
    
    H = midstep2triad_lintrans(gs_euler,split_fluctuations='vector')
    H2 = midstep2triad_lintrans(gs_euler,split_fluctuations='matrix',groundstate_definition='triad')
    
    euler_del = np.random.normal(0,5/180*np.pi,size=gs_euler.shape)
    # euler_del[:,:3] = 0
    
    euler = gs_euler + euler_del
    euler_triad = translation_midstep2triad(euler,rotation_map = 'euler')
    
    euler_triad_del = euler_triad - gs_euler_triad

    
    euler_triad_del_test = statevec2vecs(H @ vecs2statevec(euler_del),vdim=6)
    
    euler_triad_del_test2 = statevec2vecs(H2 @ vecs2statevec(euler_del),vdim=6)
    
    for i in range(len(euler_triad_del)):
        print('###########')
        print(euler_triad_del[i])
        print(euler_triad_del_test[i])
        print(euler_triad_del_test2[i])
        print(euler_del[i])
        
        print(f'del norm = {np.linalg.norm(euler_triad_del[i,3:]) - np.linalg.norm(euler_del[i,3:])}')
        
    

    sys.exit()
    
 
   
    print('\n#############################')
    rotstiff = matrix_rotmarginal(stiff)
    for i in range(len(rotstiff)//3):
        print(rotstiff[i*3:(i+1)*3,i*3:(i+1)*3]*0.34)
        
    retained_ids = [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26]
    stiff2 = marginal_schur_complement(stiff,retained_ids)
    
    for i in range(len(rotstiff)//3):
        print(stiff2[i*3:(i+1)*3,i*3:(i+1)*3]*0.34)
        
    gs_cayley = statevec2vecs(gs,vdim=6)
    gs_euler = cayleys2eulers(gs_cayley)
    
    # from .transform_SE3 import se3_eulers2rotmats, se3_vecs2rotmats, se3_rotmats2triads, se3_triads2rotmats, se3_rotmats2vecs, se3_transformations_normal2midsteptrans
    from .transform_SE3 import *
    
    from .transform_translation_midstep2triad import *
    
    gs_triad = translation_midstep2triad(gs_euler)
    
    rot_mid = se3_eulers2rotmats(gs_euler)
    rot_tri = se3_transformations_midstep2triad(rot_mid)
    
    gs_triad_test = se3_rotmats2vecs(rot_tri,rotation_map='euler')
    
    
    print('#########')
    print(np.sum(gs_euler-gs_triad_test))
    print(np.sum(gs_triad-gs_triad_test))
    
    
    
    
    
    
    
    