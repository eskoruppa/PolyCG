import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
import sys
import argparse

from typing import List, Tuple, Callable, Any, Dict
from .cgNA_plus.modules.cgDNAUtils import constructSeqParms

from .Transforms.transform_marginals import vector_marginal, matrix_marginal, unwrap_wildtypes, matrix_marginal_assignment, vector_marginal_assignment
from .Transforms.transform_units import conversion
from .Transforms.transform_statevec import statevec2vecs, vecs2statevec

from .Transforms.transform_cayley2euler import cayley2euler, cayley2euler_stiffmat
from .Transforms.transform_algebra2group import algebra2group_stiffmat
from .Transforms.transform_midstep2triad import midstep2triad

from .partials import partial_stiff
from .Aux.aux import load_sequence

CURVES_PLUS_DATASET_NAME = "cgDNA+_Curves_BSTJ_10mus_FS"
# CURVES_PLUS_DATASET_NAME = "cgDNA+ps1"

def cgnaplus_bps_params(
    sequence: str, 
    translations_in_nm: bool = True,
    euler_definition: bool = True,
    group_split: bool = False,
    parameter_set_name: str = 'curves_plus',
    remove_factor_five: bool = True 
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
    
    gs = statevec2vecs(gs,vdim=6) 

    if euler_definition:
        # cayley2euler_stiffmat requires gs in cayley definition
        stiff = cayley2euler_stiffmat(gs,stiff,rotation_first=True)
        gs = cayley2euler(gs)
    
    if group_split:
        if not euler_definition:
            raise ValueError('The group_split option requires euler_definition to be set!')
        
        stiff = algebra2group_stiffmat(gs,stiff,rotation_first=True,translation_as_midstep=True)  
        gs    = midstep2triad(gs)
         
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



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Generate PolyMC input files")
    parser.add_argument('-seqfn', '--sequence_filename', type=str, required=True) 
    # parser.add_argument('-out', '--out_filename', type=str, required=True) 
    parser.add_argument('-nm',    '--translations_in_nm', type=bool, default=True)
    parser.add_argument('-euler', '--euler_definition', type=bool, default=True)
    parser.add_argument('-group', '--group_split', type=bool, default=True)
    parser.add_argument('-fac5',  '--keep_factor_five', type=bool, default=False)
    parser.add_argument('-set',   '--parameter_set_name', type=str, default='curves_plus')
    
    parser.add_argument('-closed','--closed', type=bool, default=False)
    parser.add_argument('-bs',    '--block_size', type=int, default=120)
    parser.add_argument('-os',    '--overlap_size', type=int, default=20)
    parser.add_argument('-ts',    '--tail_size', type=int, default=20)
    
    args = parser.parse_args()
    
    seq = load_sequence(args.sequence_filename)
    # seq = seq[:10]
    
    if len(seq) == 0:
        raise IOError('Empty sequence found')

    nbps = len(seq)-1
    if args.closed:
        nbps += 1
    print(f'Sequence contains {len(seq)} base pairs.')

    method = cgnaplus_bps_params
    stiffgen_args = {
        'translations_in_nm': args.translations_in_nm, 
        'euler_definition': args.euler_definition, 
        'group_split' : args.group_split,
        'parameter_set_name' : args.parameter_set_name,
        'remove_factor_five' : not args.keep_factor_five,
        }
    
    block_size = args.block_size
    overlap_size = args.overlap_size
    tail_size = args.tail_size
    
    if overlap_size > nbps:
        overlap_size = nbps-1
    if block_size > nbps:
        block_size = nbps
    
    print('Generating partial stiffness matrix with')    
    print(f'block_size:   {block_size}')
    print(f'overlap_size: {overlap_size}')
    print(f'tail_size:    {tail_size}')

    if len(seq) - 1 <= overlap_size:
        gs, stiffar = cgnaplus_bps_params(seq, **stiffgen_args)
        stiff = sp.sparse.lil_matrix(stiffar.shape)
        stiff[:,:] = stiffar
        stiff = stiff.tocsc()
    else:
        gs,stiff = partial_stiff(seq,method,stiffgen_args,block_size=block_size,overlap_size=overlap_size,tail_size=tail_size,closed=args.closed,ndims=6)

    basefn = args.sequence_filename + '_params'
    if args.closed:
        basefn += '_closed'

    fn_gs = basefn + '_gs.npy'
    fn_stiff = basefn + '_stiff.npz'
    
    if sp.sparse.issparse(stiff):
        spstiff = stiff
    else:
        spstiff = stiff.to_sparse()
    sparse.save_npz(fn_stiff,spstiff)
    np.save(fn_gs,gs)