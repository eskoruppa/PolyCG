import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp

from typing import List, Tuple, Callable, Any, Dict
from .cgNA_plus.modules.cgDNAUtils import constructSeqParms

from .transform_marginals import vector_marginal, matrix_marginal, unwrap_wildtypes, matrix_marginal_assignment, vector_marginal_assignment
from .transform_units import conversion
from .transform_state import statevec2vecs, vecs2statevec

CURVES_PLUS_DATASET_NAME = "cgDNA+_Curves_BSTJ_10mus_FS"

def cgnaplus_bps_params(
    sequence: str, 
    remove_factor_five: bool = True, 
    parameter_set_name: str = 'curves_plus',
    separate_vectors: bool = True
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
        gs   = conversion(gs,factor,block_dim=6,dofs=[0,1,2])
        stiff = conversion(stiff,factor,block_dim=6,dofs=[0,1,2])
    
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
    
    seq = 'ACGATCGAAGCGATATCGCGGCGGGGGGGGCATTT'
    seq = 'ACGATC'

    print(len(seq))
    gs,stiff = cgnaplus_bps_params(seq)
    
    for i in range(len(stiff)//3):
        print(stiff[i*3:(i+1)*3,i*3:(i+1)*3]*0.34)
    
    from .transform_cayley2euler import *
    from .transform_marginals import *
    
    
    # # print(stiff*0.34)
    
    
    # print(gs.shape)
    # gs_euler = cayleys2eulers(gs)
    
    # gs_cayley = eulers2cayleys(gs_euler)   
    
    # print(f'euler_cayley diff = {np.sum(gs-gs_cayley)}')
    
    
    print('\n#############################')
    rotstiff = matrix_rotmarginal(stiff,6,[0,1,2])
    for i in range(len(rotstiff)//3):
        print(rotstiff[i*3:(i+1)*3,i*3:(i+1)*3]*0.34)