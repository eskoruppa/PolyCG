import numpy as np
from scipy.sparse import csr_matrix, spmatrix, coo_matrix
from scipy import sparse

from typing import List, Tuple, Callable, Any, Dict
from .cgNA_plus.modules.cgDNAUtils import constructSeqParms

CURVES_PLUS_DATASET_NAME = "cgDNA+_Curves_BSTJ_10mus_FS"



def cgnaplus_bps_params(sequence: str, factor_five: bool = False, parameter_set_name: str = 'curves_plus'):
    
    if parameter_set_name == 'curves_plus':
        parameter_set_name = CURVES_PLUS_DATASET_NAME
    
    gs,stiff = constructSeqParms(sequence,parameter_set_name)
    
    print(gs.shape)
    print(stiff.shape)
    
    
    
    




if __name__ == "__main__":
    
    seq = 'ACGATCGAAGCGATCGATCGATCGGAGGGGAGAGCCCTATATTCHCTAAA'

    
    cgnaplus_bps_params(seq)
    