import sys, os
import numpy as np
import scipy as sp
from typing import List, Tuple, Callable, Any, Dict

from .SO3 import so3
from .transforms.transform_SO3 import euler2rotmat_so3
from .composites import composite_matrix, inv_composite_matrix, composite_groundstate
from .transforms.transform_marginals import marginal_schur_complement, matrix_marginal
from .utils.bmat import BlockOverlapMatrix
from .transforms.transform_statevec import vecs2statevec, statevec2vecs

import time


############################################################################################
####################### Coarse Grain stiffness matrix and groundstate ######################
############################################################################################

CG_PARTIAL_CUTOFF = 240

def coarse_grain(
    groundstate: np.ndarray,
    stiffmat: np.ndarray | sp.sparse.spmatrix,
    composite_size: int,
    start_id: int = 0,
    end_id: int = None,
    closed: bool = False,
    allow_partial: bool = False,
    block_ncomp: int = 16,
    overlap_ncomp: int = 4,
    tail_ncomp: int = 2,
    allow_crop: bool = True,
    substitute_block: int = -1,
    use_sparse: bool = True
):   
     
    if len(groundstate.shape) != 2:
        raise ValueError(f'Shape of groundstate should be (N,ndims), but encountered {groundstate.shape}.')
    
    if closed:
        raise NotImplementedError('Coarse graining of closed chains is not yet implemented')
    
    if start_id is None:
        start_id = 0
    
    groundstate,stiffmat = _crop_gs_and_stiff(
        groundstate,
        stiffmat,
        composite_size,
        start_id,
        end_id,
        match_crop = allow_crop
    )
    cg_gs = cg_groundstate(
        groundstate,
        composite_size,
        )
    
    nbps = len(groundstate)
    
    if nbps > (block_ncomp+overlap_ncomp+tail_ncomp)*composite_size and allow_partial:
        cg_stiff = cg_stiff_partial(
            groundstate,
            stiffmat,
            composite_size,
            block_ncomp,
            overlap_ncomp,
            tail_ncomp,
            closed = closed,
            substitute_block = substitute_block, 
            use_sparse=use_sparse
        )
    else:
        cg_stiff = cg_stiffmat(
            groundstate,
            stiffmat,
            composite_size,
            closed = closed,
            substitute_block = substitute_block,
            use_sparse=use_sparse
        )
    return cg_gs, cg_stiff

############################################################################################
####################### Coarse Grain  groundstate ##########################################
############################################################################################

def cg_groundstate(
    groundstate: np.ndarray,
    composite_size: int,
) -> np.ndarray:

    if len(groundstate.shape) != 2:
        raise ValueError(f'Shape of groundstate should be (N,ndims), but encountered {groundstate.shape}.')

    ncg  = len(groundstate) // composite_size
    cg_gs = np.zeros((ncg,groundstate.shape[-1]))
    for i in range(ncg):
        cg_gs[i] = composite_groundstate(groundstate[i*composite_size:(i+1)*composite_size])
    return cg_gs

############################################################################################
####################### Coarse Grain stiffness matrix ######################################
############################################################################################
def cg_stiffmat(
    groundstate: np.ndarray,
    stiffmat: np.ndarray | sp.sparse.spmatrix,
    composite_size: int,
    closed: bool = False,
    substitute_block: int = -1,
    use_sparse: bool = True
) -> np.ndarray | sp.sparse.spmatrix:   

    if closed:
        raise NotImplementedError('Partial coarse graining is not yet implemented for closed chains')
    
    if len(groundstate.shape) != 2:
        raise ValueError(f'Shape of groundstate should be (N,ndims), but encountered {groundstate.shape}.')

    ndims = groundstate.shape[-1]
    substitute_block = substitute_block % composite_size
    ncg  = len(groundstate) // composite_size
     
    com_blocks = []
    retained_ids = []
    retained_bps = np.zeros(len(groundstate))
    for i in range(ncg):
        gs = groundstate[i*composite_size:(i+1)*composite_size]
        inv_comp_block = inv_composite_matrix(gs,substitute_block=substitute_block)
        com_blocks.append(inv_comp_block)
        
        start_id = i*composite_size*ndims
        retained_ids += [start_id + substitute_block*ndims + j for j in range(ndims)]
        retained_bps[i*composite_size + substitute_block] = 1
    Pinv = sp.sparse.block_diag(com_blocks)
        
    if not use_sparse:
        # dense version
        Pi = Pinv.toarray()
        Mp = Pi.T @ stiffmat @ Pi
        cg_stiff = marginal_schur_complement(Mp,retained_ids=retained_ids)
    
    else:
        if not sp.sparse.issparse(stiffmat):
            stiffmat = sp.sparse.block_diag((stiffmat,))
        
        Mp = Pinv.transpose() @ stiffmat @ Pinv
        cg_stiff = matrix_marginal(Mp, retained_bps, block_dim=ndims)
    return cg_stiff

############################################################################################
####################### Cropping methods ###################################################
############################################################################################

def _crop_gs(
    gs: np.ndarray,
    composite_size: int,
    start_id: int,
    end_id: int,
    match_crop: bool = True
    ) -> np.ndarray:
    ndims = gs.shape[-1]
    if end_id is None:
        end_id = len(gs)
    if start_id < 0:
        start_id = start_id % len(gs)
    num = end_id - start_id
    diff = num % composite_size
    if diff != 0 and not match_crop:
        raise ValueError(f'Parsed index range needs to be a multiple of composite_step. For automatic cropping set match_crop to True.')
    end_id = end_id - diff
    gs = gs[start_id:end_id]
    return gs

def _crop_gs_and_stiff(
    gs: np.ndarray,
    stiff: np.ndarray | sp.sparse.spmatrix,
    composite_size: int,
    start_id: int,
    end_id: int,
    match_crop: bool = True,
    ) -> Tuple[np.ndarray,np.ndarray | sp.sparse.spmatrix]:
    ndims = gs.shape[-1]
    if end_id is None:
        end_id = len(gs)
    if start_id < 0:
        start_id = start_id % len(gs)
    num = end_id - start_id
    diff = num % composite_size
    if diff != 0 and not match_crop:
        raise ValueError(f'Parsed index range needs to be a multiple of composite_step. For automatic cropping set match_crop to True.')
    end_id = end_id - diff
    gs = gs[start_id:end_id]
    stiff  = stiff[start_id*ndims:end_id*ndims,start_id*ndims:end_id*ndims]  
    return gs,stiff

############################################################################################
####################### Partial coarse graining ############################################
############################################################################################


CG_PARTIALS_MIN_BLOCK = 4

def cg_stiff_partial(
    groundstate: np.ndarray,
    stiffmat: np.ndarray | sp.sparse.spmatrix,
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    closed: bool = False,
    substitute_block: int = -1,
    use_sparse: bool = True
) -> np.ndarray | sp.sparse.spmatrix:

    if len(groundstate.shape) != 2:
        raise ValueError(f'Shape of groundstate should be (N,ndims), but encountered {groundstate.shape}.')

    Nbps = len(groundstate)
    Ncg = Nbps // composite_size

    if overlap_ncomp > Ncg:
        raise ValueError(
            f"Overlap ({overlap_ncomp}) should not exceed the number of cg steps ({Ncg})!"
        )
    if block_ncomp <= overlap_ncomp:
        raise ValueError(
            f"block_size ({block_ncomp}) needs to be larger than overlap_size ({overlap_ncomp})."
        )
    if block_ncomp + tail_ncomp < CG_PARTIALS_MIN_BLOCK:
        raise ValueError(
            f"Number of blocks too small. block_ncomp+tail_ncomp={block_ncomp+tail_ncomp}. Needs to be at least {CG_PARTIALS_MIN_BLOCK}."
        )

    if closed:
        raise NotImplementedError('Partial coarse graining is not yet implemented for closed chains')
    
    return _cg_stiff_partial_linear(
        groundstate,
        stiffmat,
        composite_size,
        block_ncomp,
        overlap_ncomp,
        tail_ncomp,
        substitute_block=substitute_block,
        use_sparse=use_sparse
    )
    
    
def _cg_stiff_partial_linear(
    gs: np.ndarray,
    stiff: np.ndarray | sp.sparse.spmatrix,
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    substitute_block: int = -1,
    use_sparse: bool = True
) -> sp.sparse.spmatrix:
    
    if len(gs.shape) != 2:
        raise ValueError(f'Shape of groundstate should be (N,ndims), but encountered {gs.shape}.')

    Nbps = gs.shape[0]
    ndims = gs.shape[1]
    gs = vecs2statevec(gs)
    assert (
        Nbps % composite_size == 0
    ), f"Nbps ({Nbps}) is not a multiple of composite_size ({composite_size})."
    Ncg = Nbps // composite_size

    block_incr = block_ncomp - overlap_ncomp
    Nsegs = int(np.floor((Ncg - overlap_ncomp) / block_incr))
    lastseg_id = Nsegs - 1

    cgstiff = BlockOverlapMatrix(
        ndims,
        average=True,
        periodic=False,
        fixed_size=True,
        xlo=0,
        xhi=Ncg * ndims,
        ylo=0,
        yhi=Ncg * ndims,
    )

    for i in range(Nsegs):
        # block range
        id1 = i * block_incr
        id2 = id1 + block_ncomp

        if i == lastseg_id:
            id2 = Ncg
        assert (
            id2 <= Ncg
        ), f"id2 ({id2}) should never exceed the number of cg steps, Ncg ({Ncg})."

        print(
            f"Coarse-graining from bps {id1*composite_size} to {id2*composite_size} ({Ncg*composite_size} in total)."
        )

        lid = id1 - tail_ncomp
        uid = id2 + tail_ncomp

        if lid < 0:
            lid = 0
        if uid > Ncg:
            uid = Ncg

        al = lid * composite_size * ndims
        au = uid * composite_size * ndims

        pgs = gs[al:au]
        pstiff = stiff[al:au, al:au]
        
        # coarse-grain block
        block_cgstiff = cg_stiffmat(
            statevec2vecs(pgs,ndims),
            pstiff,
            composite_size,
            substitute_block=substitute_block,
            use_sparse=use_sparse,
        )

        cl = (id1 - lid) * ndims
        cu = block_cgstiff.shape[0] - (uid - id2) * ndims
        pcgstiff = block_cgstiff[cl:cu, cl:cu]

        mid1 = id1 * ndims
        mid2 = id2 * ndims
        cgstiff.add_block(pcgstiff, mid1, mid2, y1=mid1, y2=mid2)
      
    return cgstiff.to_sparse()      
    
    
    
if __name__ == '__main__':
        
    from .cgnaplus import cgnaplus_bps_params
    from .utils.aux import load_sequence
    from .partials import partial_stiff
    
    np.set_printoptions(linewidth=250, precision=3, suppress=True,edgeitems=12)
    
    fn_seq = 'Data/JanSeq/Lipfert_2kb'
    fn_seq = 'Data/JanSeq/Lipfert_1kb'
    # fn_seq = 'Data/JanSeq/Lipfert_7p9kb'
    
    base_fn = fn_seq
    # base_fn = 'Data/JanSeq/Lipfert_2kb'
    
    fn_gs = base_fn + '_gs.npy'
    fn_stiff = base_fn + '_stiff.npz'
    
    seq = load_sequence(fn_seq)
    closed = False
    
    # seq = seq[:2001]
    
    composite_size = 10
    start_id  = 0
    end_id    = None
    
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
    
    t1 = time.time()
    cg_gs,cg_stiff = coarse_grain(gs,stiff,composite_size,start_id=start_id,end_id=end_id,allow_partial=True)
    t2 = time.time()
    print(f'dt = {(t2-t1)}')
    
    base_fn = base_fn + f'_cg{composite_size}_params'
    fn_gs = base_fn + '_gs.npy'
    fn_stiff = base_fn + '_stiff.npz'
    sp.sparse.save_npz(fn_stiff,cg_stiff)
    np.save(fn_gs,cg_gs)
    