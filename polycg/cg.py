from __future__ import annotations
import sys, os
import numpy as np
import scipy as sp

from .SO3 import so3
from .transforms.transform_SO3 import euler2rotmat_so3
from .composites import composite_matrix, inv_composite_matrix, composite_groundstate
from .transforms.transform_marginals import marginal_schur_complement, matrix_marginal
from .utils.bmat import BlockOverlapMatrix
from .transforms.transform_statevec import vecs2statevec, statevec2vecs
from .utils.console_output import print_progress

import time


############################################################################################
####################### Coarse Grain stiffness matrix and groundstate ######################
############################################################################################

CG_PARTIAL_CUTOFF = 240

def coarse_grain(
    groundstate: np.ndarray,  # shape (N, 3) or (N, 6): SO(3) or SE(3) parameters
    stiffmat: np.ndarray | sp.sparse.spmatrix,  # shape (N*ndims, N*ndims): stiffness matrix
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
    use_sparse: bool = True,
    verbose: bool = False,
    print_info: bool = False,
) -> tuple[np.ndarray, np.ndarray | sp.sparse.spmatrix]:  # (cg_groundstate, cg_stiffmat)   
     
    if len(groundstate.shape) != 2:
        raise ValueError(f'Shape of groundstate should be (N,ndims), but encountered {groundstate.shape}.')
    
    if closed:
        raise NotImplementedError('Coarse graining of closed chains is not yet implemented')
    
    if start_id is None:
        start_id = 0
        
    if print_info and verbose:
        print('Convert to sparse matrix')
    stiffmat = stiffmat.to_sparse() if isinstance(stiffmat, BlockOverlapMatrix) else stiffmat
    
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
            use_sparse=use_sparse,
            verbose=verbose,
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
    groundstate: np.ndarray,  # shape (N, 3) or (N, 6): SO(3) or SE(3) parameters
    composite_size: int,
) -> np.ndarray:  # shape (N//composite_size, 3) or (N//composite_size, 6): coarse-grained groundstate

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
    groundstate: np.ndarray,  # shape (N, 3) or (N, 6): SO(3) or SE(3) parameters
    stiffmat: np.ndarray | sp.sparse.spmatrix,  # shape (N*ndims, N*ndims): stiffness matrix
    composite_size: int,
    closed: bool = False,
    substitute_block: int = -1,
    use_sparse: bool = True
) -> np.ndarray | sp.sparse.spmatrix:  # shape (Ncg*ndims, Ncg*ndims): coarse-grained stiffness   

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
    gs: np.ndarray,  # shape (N, ndims): groundstate parameters
    composite_size: int,
    start_id: int,
    end_id: int,
    match_crop: bool = True
    ) -> np.ndarray:  # shape (M, ndims): cropped groundstate
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
    gs: np.ndarray,  # shape (N, ndims): groundstate parameters
    stiff: np.ndarray | sp.sparse.spmatrix,  # shape (N*ndims, N*ndims): stiffness matrix
    composite_size: int,
    start_id: int,
    end_id: int,
    match_crop: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | sp.sparse.spmatrix]:  # (cropped_gs, cropped_stiff)
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
    groundstate: np.ndarray,  # shape (N, 3) or (N, 6): SO(3) or SE(3) parameters
    stiffmat: np.ndarray | sp.sparse.spmatrix,  # shape (N*ndims, N*ndims): stiffness matrix
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    closed: bool = False,
    substitute_block: int = -1,
    use_sparse: bool = True,
    verbose: bool = False,
) -> np.ndarray | sp.sparse.spmatrix:  # shape (Ncg*ndims, Ncg*ndims): coarse-grained stiffness

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
            f"Number of blocks too small. block_ncomp+tail_ncomp={block_ncomp+tail_ncomp}. Needs to be at least {CG_PARTIALS_MIN_BLOCK}.\n \
            The sequence is too small for the chosen composite_size!"
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
        use_sparse=use_sparse,
        verbose=verbose,
    )
    
    
def _cg_stiff_partial_linear(
    gs: np.ndarray,  # shape (N, 3) or (N, 6): SO(3) or SE(3) parameters
    stiff: np.ndarray | sp.sparse.spmatrix,  # shape (N*ndims, N*ndims): stiffness matrix
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    substitute_block: int = -1,
    use_sparse: bool = True,
    verbose: bool = False,
) -> sp.sparse.spmatrix:  # shape (Ncg*ndims, Ncg*ndims): coarse-grained stiffness
    
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
        average=True,
        periodic=False,
        fixed_size=True,
        xlo=0,
        xhi=Ncg * ndims,
        ylo=0,
        yhi=Ncg * ndims,
    )

    if verbose:
        print(f"Coarse-graining {Nsegs} blocks ({Ncg*composite_size} total bps, composite_size={composite_size})")
    for i in range(Nsegs):
        # block range
        id1 = i * block_incr
        id2 = id1 + block_ncomp

        if i == lastseg_id:
            id2 = Ncg
        assert (
            id2 <= Ncg
        ), f"id2 ({id2}) should never exceed the number of cg steps, Ncg ({Ncg})."

        if verbose:
            print_progress(i + 1, Nsegs, prefix='Progress:', suffix=f'Block {i+1}/{Nsegs} (bps {id1*composite_size}-{id2*composite_size})')

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
      
    return cgstiff     
    
    
    
if __name__ == '__main__':
        
    from .cgnaplus import cgnaplus_bps_params
    
    # from polycg import cgnaplus_bps_params
    # from polycg import coarse_grain
    
    nbp = 101
    composite_size = 10
    
    seq = ''.join(['ATCG'[np.random.randint(0,4)] for _ in range(nbp)])
    shape,stiff = cgnaplus_bps_params(
        seq,
        translations_in_nm = True, 
        euler_definition = True, 
        group_split = True,
        parameter_set_name = 'curves_plus',
        remove_factor_five = True
    )
    
    cg_shape,cg_stiff = coarse_grain(shape,stiff,composite_size)

    print(cg_shape.shape)