import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp
import sys, time
from matplotlib import pyplot as plt

from typing import List, Tuple, Callable, Any, Dict
from .cgnaplus import cgnaplus_bps_params

from .transform_SE3 import *
from .transform_cayley2euler import *
from .transform_marginals import *
from .transform_statevec import *
from .transform_algebra2group import *
from .composites import *
from .Evals.tangent_correlation import TangentCorr
from .aux import load_sequence
from .partials import partial_stiff

from .pyConDec.pycondec import cond_jit


def sample_lb(gs: np.ndarray, stiff: Any, num_confs: int, mmax: int, cov: Any = None) -> np.ndarray:
        
    if cov is None:
        if sp.sparse.isspmatrix(stiff):
            cov = sp.sparse.linalg.inv(stiff).toarray()
        else:
            cov = np.linalg.inv(stiff)
    
    nbatch = 10
    batches = num_confs // nbatch
    batchnums = [nbatch for i in range(batches)]
    if num_confs % nbatch != 0:
        batchnums.append(num_confs % nbatch)

    gs = statevec2vecs(gs,vdim=6)
    nbp = len(gs)+1
    Ss = euler2rotmat(gs)

    tancor = TangentCorr(mmax)

    for b,num in enumerate(batchnums):
        print(f'batch {b+1}/{len(batchnums)}')
        
        
        dX = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov.toarray(), num)
        dX = statevec2vecs(dX,vdim=6)
        
        # Ds1 = euler2rotmat(dX)
        Ds = e2r(dX)
        confs = gen_conf(Ss,Ds)
     
        tancor.add_conf(confs[:,:,:3,3])
    
        lbdata = tancor.lb
        iter = mmax // 5
        if iter < 1:
            iter = 1
        print(tancor.disc_len)
        print(lbdata[:,::iter])
    
    
    lbfn = 'Data/JanSeq/Lipfert_2kb.lb'
    lbdata = tancor.lb  
    np.save(lbfn,lbdata)

            
@cond_jit
def gen_conf(Ss,Ds):
    num = len(Ds)
    nbp = len(Ds[0])+1
    confs = np.zeros((num,nbp,4,4))
    for c in range(num):
        confs[c,0] = np.eye(4)
        for i in range(nbp-1):
            confs[c,i+1] = confs[c,i] @ Ss[i] @ Ds[c,i]
    return confs

@cond_jit
def e2r(dX):
    rotmats = np.zeros((len(dX),len(dX[0]),4,4))
    for c,eulers in enumerate(dX):
        for i, euler in enumerate(eulers):
            rotmats[c,i] = so3.se3_euler2rotmat(euler)
    return rotmats
    
    

if __name__ == "__main__":
    
    # fn_gs    = sys.argv[1]
    # fn_stiff = sys.argv[2]
    # N        = int(sys.argv[3])
    
    # fn_gs    = 'Data/JanSeq/Lipfert_2kb_params_gs.npy'
    # fn_stiff = 'Data/JanSeq/Lipfert_2kb_params_stiff.npz'
    # N        = 10000
    # # stiffmat and groundstate
    # stiff = sp.sparse.load_npz(fn_stiff)
    # gs    = np.load(fn_gs)
    
    mmax = 160
    
    N        = 10000
    fn_seq = 'Data/JanSeq/Lipfert_2kb'
    seq = load_sequence(fn_seq)
    closed = False
    
    seq = seq[:201]
    
    method = cgnaplus_bps_params
    stiffgen_args = {
        'translations_in_nm': True, 
        'euler_definition': True, 
        'group_split' : True,
        'parameter_set_name' : 'curves_plus',
        'remove_factor_five' : True,
        }
    
    block_size = 120
    overlap_size = 20
    tail_size = 20
    nbps = len(seq)-1
    
    if overlap_size > nbps:
        overlap_size = nbps-1
    if block_size > nbps:
        block_size = nbps
    
    print('Generating partial stiffness matrix with')    
    print(f'block_size:   {block_size}')
    print(f'overlap_size: {overlap_size}')
    print(f'tail_size:    {tail_size}')

    gs,stiff = partial_stiff(seq,method,stiffgen_args,block_size=block_size,overlap_size=overlap_size,tail_size=tail_size,closed=closed,ndims=6)
    # gs,stiff = cgnaplus_bps_params(seq,translations_in_nm=True,euler_definition=True,group_split=True)
    
    cov = stiff.invert().to_sparse()
    stiff = stiff.to_sparse()
    
    sample_lb(gs,stiff,N,mmax,cov=cov)