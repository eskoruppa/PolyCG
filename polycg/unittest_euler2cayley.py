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
from numba import njit

def test_order_of_rotation_marginals(seq: str = 'ACGATC',num_confs = 100000):
    np.set_printoptions(linewidth=250)
    
    # generate stiffness
    euler_gs,euler_stiff = cgnaplus_bps_params(seq)
    
    # rotational marginals
    euler_rot_gs    = euler_gs[:,:3]
    euler_rot_stiff = matrix_rotmarginal(euler_stiff)
    
    # full covariance
    euler_cov     = np.linalg.inv(euler_stiff)
    
    # rotational covariance
    euler_rot_cov = np.linalg.inv(euler_rot_stiff)
    
    # check consistency of rotational components
    if np.abs(np.sum(euler_cov[:3,:3]-euler_rot_cov[:3,:3])) > 1e-10:
        print(f'Rotational covariance inconsistent')
    else:
        print('Rotational covariance marginals checks out')
    
    # sample configs
    eulers_dx = np.random.multivariate_normal(np.zeros(len(euler_cov)), euler_cov, num_confs)
    eulers_rot_dx = np.random.multivariate_normal(np.zeros(len(euler_rot_cov)), euler_rot_cov, num_confs)
    
    eulers_dx     = statevec2vecs(eulers_dx,vdim=6)
    eulers_rot_dx = statevec2vecs(eulers_rot_dx,vdim=3)
    
    eulers     = euler_gs     + eulers_dx
    eulers_rot = euler_rot_gs + eulers_rot_dx
    
    print(eulers.shape)
    print(eulers_rot.shape)
    
    cov    = covmat(eulers_dx)
    covrot = covmat(eulers_rot_dx)
    
    print((cov[:3,:3]-covrot[:3,:3])/covrot[:3,:3])
    
    # transform to cayley
    cayleys     = euler2cayley(eulers)
    cayleys_rot = euler2cayley(eulers_rot)
    cayleys_gs  = euler2cayley(euler_gs)
    cayleys_rot_gs = euler2cayley(euler_rot_gs)
    cayleys_dx     = cayleys - cayleys_gs
    cayleys_rot_dx = cayleys_rot - cayleys_rot_gs
    
    # test transformation
    if np.abs(np.sum(cayleys[:,:,:3]-euler2cayley(eulers[:,:,:3]))) > 1e-10:
        print('Caylay2cayley inconsistent with and without translations')
    else:
        print('Caylay2cayley consistency checks out')
        
    cayleys_cov     = covmat(cayleys_dx)
    cayleys_rot_cov = covmat(cayleys_rot_dx)
    
    cayleys_stiff     = np.linalg.inv(cayleys_cov)
    cayleys_rot_stiff = np.linalg.inv(cayleys_rot_cov)
    
    print('#####################################')
    # print('Sampled cayley stiffness with translations')
    # print(cayleys_stiff[:6,:6])
    print('Sampled cayley stiffness marginalied after transformation')
    print(matrix_rotmarginal(cayleys_stiff)[:3,:3])
    print('Sampled cayley stiffness marginalied before transformation')
    print(cayleys_rot_stiff[:3,:3])
    
    
    cayley_stiff_lintrans = euler2cayley_stiffmat(euler_gs,euler_stiff)
    cayley_rot_stiff_lintrans = euler2cayley_stiffmat(euler_rot_gs,euler_rot_stiff)
    
    # print('Transformed cayley stiffness with translations')
    # print(cayley_stiff_lintrans[:6,:6])
    print('Transformed cayley stiffness with translations')
    print(matrix_rotmarginal(cayley_stiff_lintrans)[:3,:3])
    print('Transformed cayley stiffness marginalied before transformation')
    print(cayley_rot_stiff_lintrans[:3,:3])
    
    print('#####################################')
    print(euler_stiff[:3,:3])
    print(cayley2euler_stiffmat(cayleys_gs,cayley_stiff_lintrans)[:3,:3])
    print('#####################################')
    print(euler_rot_stiff[:3,:3])
    print(cayley2euler_stiffmat(cayleys_rot_gs,cayley_rot_stiff_lintrans)[:3,:3])
    
    print('#####################################')
    print('#####################################')
    print('#####################################')
    print(f'full diff = {np.abs(np.sum(eulers-cayley2euler(cayleys)))}')
    print(f'rot  diff = {np.abs(np.sum(eulers_rot-cayley2euler(cayleys_rot)))}')
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
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
    

    
    





    gs_cayley    = euler2cayley(gs)
    stiff_cayley = euler2cayley_stiffmat(gs,stiff)
    
    
    print(stiff.shape)
    
    rot_stiff_euler = matrix_rotmarginal(stiff)
    rot_gs_euler    = gs[:,:3]
    
    print(np.linalg.inv(rot_stiff_euler)[:3,:3])
    print(np.linalg.inv(stiff)[:3,:3])
    

    sys.exit()
    
    
        
    rot_stiff_euler = stiff
    rot_gs_euler    = gs
    
    rot_stiff_cayley = euler2cayley_stiffmat(rot_gs_euler,rot_stiff_euler)
    rot_gs_cayley    = euler2cayley(rot_gs_euler)
    

    num_confs = 100000
    rot_covmat_euler = np.linalg.inv(rot_stiff_euler)
    
    dx_euler = np.random.multivariate_normal(np.zeros(len(rot_covmat_euler)), rot_covmat_euler, num_confs)
    
    rot_eulers = rot_gs_euler + statevec2vecs(dx_euler,vdim=6)
    rot_cayleys  = euler2cayley(rot_eulers)
    
    print(rot_eulers[1])
    print(rot_cayleys[1])
    sys.exit()
    
    
    rot_cayley_dx = rot_cayleys - rot_gs_cayley
    
    cov = np.zeros(rot_covmat_euler.shape)
    for i in range(len(rot_cayley_dx)):
        cov += np.outer(rot_cayley_dx[i],rot_cayley_dx[i])
    cov /= len(rot_cayley_dx)
    
    rot_stiff_cayley_sampled = np.linalg.inv(cov)
    


    print(rot_stiff_cayley[:3,:3])
    print(rot_stiff_cayley_sampled[:3,:3])
    
    # print(matrix_rotmarginal(rot_stiff_cayley_sampled)[:3,:3])
    
    
    
    sys.exit()
    
    
    
    num_confs = 100000
    covmat_cayley  = np.linalg.inv(stiff_cayley)
    covmat_euler = np.linalg.inv(stiff)
    
    dx_euler = np.random.multivariate_normal(np.zeros(len(covmat_euler)), covmat_euler, num_confs)
    # dx_cayley  = np.random.multivariate_normal(np.zeros(len(covmat_euler)), covmat_euler, num_confs)
    
    eulers = gs + statevec2vecs(dx_euler,vdim=6)
    cayleys  = euler2cayley(eulers)
    
    # mean cayley
    cayleys_state = vecs2statevec(cayleys)
    mean_cayley = np.mean(cayleys_state,axis=0)
    dx_cayley = cayleys_state - vecs2statevec(gs_cayley)
    # cov cayley
    cov = np.zeros((len(cayleys_state[-1]),)*2)
    for i in range(len(dx_cayley)):
        cov += np.outer(dx_cayley[i],dx_cayley[i])
    cov /= len(dx_cayley)
    
    print('diff in mean')
    print((statevec2vecs(mean_cayley,vdim=6)-gs_cayley) / gs_cayley)
    
    cayley_stiff_sampled = np.linalg.inv(cov)

    print(f'diff stiff = {np.sum(stiff_cayley - cayley_stiff_sampled)}')
    
    pos = 3
    print(stiff_cayley[pos*6:pos*6+3,pos*6:pos*6+3]*0.34)
    print(cayley_stiff_sampled[pos*6:pos*6+3,pos*6:pos*6+3]*0.34)
    
    
    sys.exit()
    
    
    
    
    
    
    
    
    
    
    dx = np.random.multivariate_normal(np.zeros(len(covmat)), covmat, num_confs)
    
    
    cov = np.zeros(covmat.shape)
    for i in range(len(dx)):
        cov += np.outer(dx[i],dx[i])
    cov /= len(dx)
    
    stiff_cayley_test = np.linalg.inv(cov)
    
    print(f'diff = {np.sum(stiff_cayley-stiff_cayley_test)}')
    
    print(stiff_cayley[:6,:6])
    print(stiff_cayley_test[:6,:6])
    
    
    sys.exit()
    
    
    
    
    # print(stiff[:6,:6])
    # print(stiff_cayley[:6,:6])
    
    from .transforms.transform_midstep2triad import *
    
    gs_cayley_triad = translation_midstep2triad(gs_cayley,rotation_map = 'cayley')
    
    H = midstep2triad_lintrans(gs_cayley,split_fluctuations='vector')
    H2 = midstep2triad_lintrans(gs_cayley,split_fluctuations='matrix',groundstate_definition='triad')
    
    cayley_del = np.random.normal(0,5/180*np.pi,size=gs_cayley.shape)
    # cayley_del[:,:3] = 0
    
    cayley = gs_cayley + cayley_del
    cayley_triad = translation_midstep2triad(cayley,rotation_map = 'cayley')
    
    cayley_triad_del = cayley_triad - gs_cayley_triad

    
    cayley_triad_del_test = statevec2vecs(H @ vecs2statevec(cayley_del),vdim=6)
    
    cayley_triad_del_test2 = statevec2vecs(H2 @ vecs2statevec(cayley_del),vdim=6)
    
    for i in range(len(cayley_triad_del)):
        print('###########')
        print(cayley_triad_del[i])
        print(cayley_triad_del_test[i])
        print(cayley_triad_del_test2[i])
        print(cayley_del[i])
        
        print(f'del norm = {np.linalg.norm(cayley_triad_del[i,3:]) - np.linalg.norm(cayley_del[i,3:])}')
        
    

    sys.exit()
    
 
   
    print('\n#############################')
    rotstiff = matrix_rotmarginal(stiff)
    for i in range(len(rotstiff)//3):
        print(rotstiff[i*3:(i+1)*3,i*3:(i+1)*3]*0.34)
        
    retained_ids = [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26]
    stiff2 = marginal_schur_complement(stiff,retained_ids)
    
    for i in range(len(rotstiff)//3):
        print(stiff2[i*3:(i+1)*3,i*3:(i+1)*3]*0.34)
        
    gs_euler = statevec2vecs(gs,vdim=6)
    gs_cayley = euler2cayley(gs_euler)
    
    # from .transform_SE3 import se3_cayleys2rotmats, se3_vecs2rotmats, se3_rotmats2triads, se3_triads2rotmats, se3_rotmats2vecs, se3_transformations_normal2midsteptrans
    from .transforms.transform_SE3 import *
    
    from .transforms.transform_midstep2triad import *
    
    gs_triad = translation_midstep2triad(gs_cayley)
    
    rot_mid = se3_cayleys2rotmats(gs_cayley)
    rot_tri = se3_transformations_midstep2triad(rot_mid)
    
    gs_triad_test = se3_rotmats2vecs(rot_tri,rotation_map='cayley')
    
    
    print('#########')
    print(np.sum(gs_cayley-gs_triad_test))
    print(np.sum(gs_triad-gs_triad_test))
    
    
    
    
    
    
    
    