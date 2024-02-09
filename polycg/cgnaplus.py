import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp
import sys

from typing import List, Tuple, Callable, Any, Dict
from .cgNA_plus.modules.cgDNAUtils import constructSeqParms

from .transform_marginals import vector_marginal, matrix_marginal, unwrap_wildtypes, matrix_marginal_assignment, vector_marginal_assignment
from .transform_units import conversion
from .transform_statevec import statevec2vecs, vecs2statevec

CURVES_PLUS_DATASET_NAME = "cgDNA+_Curves_BSTJ_10mus_FS"

def cgnaplus_bps_params(
    sequence: str, 
    remove_factor_five: bool = True, 
    parameter_set_name: str = 'curves_plus',
    separate_vectors: bool = True,
    translations_in_nm: bool = True
    ) -> Tuple[np.ndarray,np.ndarray]:
    
    if parameter_set_name == 'curves_plus':
        parameter_set_name = CURVES_PLUS_DATASET_NAME
    
    gs,stiff = constructSeqParms(sequence,parameter_set_name)
    names = cgnaplus_name_assign(sequence)
    select_names = ["y*"]
    stiff = matrix_marginal_assignment(stiff,select_names,names,block_dim=6)
    gs    = vector_marginal_assignment(gs,select_names,names,block_dim=6)
    stiff = stiff.toarray()
    if remove_factor_five:
        factor = 5
        gs   = conversion(gs,1./factor,block_dim=6,dofs=[0,1,2])
        stiff = conversion(stiff,factor,block_dim=6,dofs=[0,1,2])
    
    if translations_in_nm:
        factor = 10
        gs   = conversion(gs,1./factor,block_dim=6,dofs=[3,4,5])
        stiff = conversion(stiff,factor,block_dim=6,dofs=[3,4,5])
    
    if separate_vectors:
        gs = statevec2vecs(gs,vdim=6)  
    return gs, stiff


def cgnaplus_name_assign(seq: str, dof_names=["W", "x", "C", "y"]) -> List[str]:
    """
    Generates the sequence of contained degrees of freedom for the specified sequence.
    The default names follow the convention introduced on the cgNA+ website
    """
    if len(dof_names) != 4:
        raise ValueError(
            f"Requires 4 names for the degrees of freedom. {len(dof_names)} given."
        )
    N = len(seq)
    if N == 0:
        return []
    vars = list()
    for i in range(1, N + 1):
        vars += [f"{dofn}{i}" for dofn in dof_names]
    return vars[1:-2]


if __name__ == "__main__":
    
    np.set_printoptions(linewidth=250)
    
    seq = 'ACGATCGAAGCGATATCGCGGCGGGGGGGGCATTT'
    seq = 'ACGATC'

    print(len(seq))
    gs,stiff = cgnaplus_bps_params(seq)
    

    # for i in range(len(stiff)//3):
    #     print(stiff[i*3:(i+1)*3,i*3:(i+1)*3]*0.34)
    
    from .transform_cayley2euler import *
    from .transform_marginals import *
    from .transform_statevec import *

    gs_euler    = cayley2euler(gs)
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
    rot_gs_euler    = cayley2euler(rot_gs_cayley)
    

    num_confs = 100000
    rot_covmat_cayley = np.linalg.inv(rot_stiff_cayley)
    
    dx_cayley = np.random.multivariate_normal(np.zeros(len(rot_covmat_cayley)), rot_covmat_cayley, num_confs)
    
    rot_cayleys = rot_gs_cayley + statevec2vecs(dx_cayley,vdim=6)
    rot_eulers  = cayley2euler(rot_cayleys)
    
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
    eulers  = cayley2euler(cayleys)
    
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
    
    from .transform_midstep2triad import *
    
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
    gs_euler = cayley2euler(gs_cayley)
    
    # from .transform_SE3 import se3_eulers2rotmats, se3_vecs2rotmats, se3_rotmats2triads, se3_triads2rotmats, se3_rotmats2vecs, se3_transformations_normal2midsteptrans
    from .transform_SE3 import *
    
    from .transform_midstep2triad import *
    
    gs_triad = translation_midstep2triad(gs_euler)
    
    rot_mid = se3_eulers2rotmats(gs_euler)
    rot_tri = se3_transformations_midstep2triad(rot_mid)
    
    gs_triad_test = se3_rotmats2vecs(rot_tri,rotation_map='euler')
    
    
    print('#########')
    print(np.sum(gs_euler-gs_triad_test))
    print(np.sum(gs_triad-gs_triad_test))
    
    
    
    
    
    
    
    