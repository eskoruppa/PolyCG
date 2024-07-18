import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp
import sys, time
from matplotlib import pyplot as plt

from typing import List, Tuple, Callable, Any, Dict
from .cgnaplus import cgnaplus_bps_params

from .transforms.transform_cayley2euler import *
from .transforms.transform_marginals import *
from .transforms.transform_statevec import *
from .transforms.transform_algebra2group import *
from numba import njit

from .composites import *




 
def test_lb(seq: str = 'ACGATC'):
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    # generate stiffness
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq,translations_in_nm=True)
    print('stiff generated')
    
    translation_as_midstep = True
    
    # convert to eulers
    algebra_gs  = cayley2euler(cayley_gs)
    algebra_stiff = cayley2euler_stiffmat(cayley_gs,cayley_stiff,rotation_first=True)
    
    group_gs = np.copy(algebra_gs)
    if translation_as_midstep:
        for i,vec in enumerate(group_gs):
            Phi_0 = vec[:3]
            zeta_0 = vec[3:]
            sqrtS = so3.euler2rotmat(0.5*Phi_0)
            s = sqrtS @ zeta_0
            group_gs[i,3:] = s
    
    from .transforms.transform_midstep2triad import midstep2triad
    test_group_gs = midstep2triad(algebra_gs)
    
    # for i in range(len(group_gs)):
    #     print(np.abs(np.sum(group_gs[i]-test_group_gs[i])))
    
    # sys.exit()
    
    group_stiff = algebra2group_stiffmat(algebra_gs,algebra_stiff,rotation_first=True,translation_as_midstep=translation_as_midstep)
    
    idfrom = 50
    idto   = int(50+10.5*10)+2
    
    N = idto-idfrom
    
    nsteps = idto - idfrom
    disc_len = 0.34
    
    group_gs = group_gs[idfrom:idto]
    group_stiff = group_stiff[idfrom*6:idto*6,idfrom*6:idto*6]
    
    switched_gs = np.copy(group_gs)

    Tv = composite_matrix(switched_gs)
    Ti = inv_composite_matrix(switched_gs)
    
    Mcomp = Ti.T @ group_stiff @ Ti    
    Msum = marginal_schur_complement(Mcomp,retained_ids=[i+len(Mcomp)-6 for i in range(6)])
    
    Mrot = marginal_schur_complement(Msum,[0,1,2])
    
    print('calculated stiffness matrix')
    print(Msum)
    print('rot')
    print(Mrot * N * disc_len )
     
    
    

    
    
    
    
    
    
    




if __name__ == "__main__":
    
    seq = 'ACGATCGAGATCCTATGCTAGGTCGGAATCCGATCATACTGGC'*5
    num_confs = 5000000
    
    print(f'len = {len(seq)}')
    
    # test_block(seq,num_confs=num_confs)
    
    test_lb(seq)
    
