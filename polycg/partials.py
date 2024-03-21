from __future__ import annotations
import sys
import numpy as np
from typing import Tuple, List, Callable, Any, Dict
from .Aux.bmat import BlockOverlapMatrix
from .Transforms.transform_statevec import statevec2vecs

PARTIALS_MIN_BLOCK = 4

#######################################################################################
#######################################################################################


def partial_stiff(
    seq: str,
    stiffgen_method: Callable,
    stiffgen_args: Dict[str, Any],
    block_size: int,
    overlap_size: int,
    tail_size: int,
    closed: bool = False,
    ndims: int = 6
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    nbps = len(seq)
    if not closed:
        nbps -= 1
    if overlap_size > nbps:
        raise ValueError(f"Overlap size should not exceed the number of bps!")

    if block_size <= overlap_size:
        raise ValueError(
            f"block_size ({block_size}) needs to be larger than overlap_size ({overlap_size})."
        )

    if block_size + tail_size < PARTIALS_MIN_BLOCK:
        raise ValueError(
            f"blocks too small. block_size+tail_size={block_size+tail_size}. Needs to be at least {PARTIALS_MIN_BLOCK}."
        )
        
    if closed:
        gs,stiff = _partial_stiff_closed(
            seq,
            stiffgen_method,
            stiffgen_args,
            block_size,
            overlap_size,
            tail_size,
            ndims=ndims,
        )
    else:
        gs, stiff = _partial_stiff_linear(
            seq,
            stiffgen_method,
            stiffgen_args,
            block_size,
            overlap_size,
            tail_size,
            ndims=ndims,
        )
    if isinstance(stiff,BlockOverlapMatrix):
        print('convert to sparse')
        stiff = stiff.to_sparse()
    return gs, stiff


#######################################################################################
#######################################################################################


def _partial_stiff_linear(
    seq: str,
    stiffgen_method: Callable,
    stiffgen_args: Dict[str, Any],
    block_size: int,
    overlap_size: int,
    tail_size: int,
    ndims: int = 6,
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    Nseq = len(seq)
    N = Nseq - 1
    block_incr = block_size - overlap_size
    Nsegs = int(np.floor((N - overlap_size) / block_incr))
    lastseg_id = Nsegs - 1

    stiff = BlockOverlapMatrix(
        ndims,
        average=True,
        periodic=False,
        fixed_size=True,
        xlo=0,
        xhi=N * ndims,
        ylo=0,
        yhi=N * ndims,
    )
    gs = np.zeros(ndims * N)
    cnts = np.zeros(ndims * N)

    for i in range(Nsegs):
        # block range
        id1 = i * block_incr
        id2 = id1 + block_size
        if id2 > N:
            id2 = N

        if i == lastseg_id:
            id2 = N
        assert id2 <= N, "bu exceeds bounds for seg other than last."

        print(f"Generating stiffness from bps {id1} to {id2} ({N} in total).")

        pgs, pstiff = _extract_bps_stiff(
            seq,
            id1,
            id2,
            tail_size,
            stiffgen_method,
            stiffgen_args,
            ndims,
            periodic=False,
        )
        mid1 = id1 * ndims
        mid2 = id2 * ndims
        stiff.add_block(pstiff, mid1, mid2, y1=mid1, y2=mid2)

        if id2 <= N:
            gs[mid1:mid2] += pgs
            cnts[mid1:mid2] += 1
        else:
            mid2 = N * ndims
            gs[mid1:mid2] += pgs[: mid2 - mid1]
            cnts[mid1:mid2] += 1
    gs /= cnts
    gs = statevec2vecs(gs,ndims)
    return gs, stiff


#######################################################################################
#######################################################################################


def _partial_stiff_closed(
    seq: str,
    stiffgen_method: Callable,
    stiffgen_args: Dict[str, Any],
    block_size: int,
    overlap_size: int,
    tail_size: int,
    ndims: int = 3,
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    # the main sequence includes the step connecting the last and first bp
    # the full sequence includes the overlap region on top of that

    N_main = len(seq)
    N_full = N_main + overlap_size

    block_incr = block_size - overlap_size
    Nsegs = int(np.floor(N_full / block_incr))
    lastseg_id = Nsegs - 1

    stiff = BlockOverlapMatrix(
        ndims,
        average=True,
        periodic=True,
        xlo=0,
        xhi=N_main * ndims,
        ylo=0,
        yhi=N_main * ndims,
    )
    gs = np.zeros(ndims * N_main)
    cnts = np.zeros(ndims * N_main)

    for i in range(Nsegs):
        # block range
        id1 = i * block_incr
        id2 = id1 + block_size

        if i == lastseg_id and id2 < N_full:
            id2 = N_full

        print(f"Generating stiffness from bps {id1} to {id2} ({N_full} in total).")

        pgs, pstiff = _extract_bps_stiff(
            seq,
            id1,
            id2,
            tail_size,
            stiffgen_method,
            stiffgen_args,
            ndims,
            periodic=True,
        )

        mid1 = id1 * ndims
        mid2 = id2 * ndims

        stiff.add_block(pstiff, mid1, mid2, y1=mid1, y2=mid2)

        if id2 <= N_main:
            gs[mid1:mid2] += pgs
            cnts[mid1:mid2] += 1
        else:
            mid2 = N_main * ndims
            gs[mid1:mid2] += pgs[: mid2 - mid1]
            cnts[mid1:mid2] += 1
    gs /= cnts
    gs = statevec2vecs(gs,ndims)
    return gs, stiff


#######################################################################################
#######################################################################################


def _extract_bps_stiff(
    seq: str,
    id1: int,
    id2: int,
    tail_size: int,
    stiffgen_method: Callable,
    stiffgen_args: Dict[str, Any],
    ndims: int = 3,
    periodic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    sid1 = id1
    sid2 = id2 + 1

    if not periodic:
        if sid1 < 0:
            raise IndexError(
                f"Lower bound ({id1}) cannot be negative for non-periodic boundary."
            )
        if sid2 > len(seq):
            raise IndexError(
                f"Upper bound ({id2}) out of bounds for sequence of length {len(seq)}."
            )

        ladd_id = sid1 - tail_size
        uadd_id = sid2 + tail_size
        if ladd_id < 0:
            ladd_id = 0
        if uadd_id > len(seq):
            uadd_id = len(seq)
        genseq = seq[ladd_id:uadd_id]
        
        gs, stiff = stiffgen_method(genseq, **stiffgen_args)
        if len(gs.shape) > 1:
            gs = gs.flatten()
        
        cl = (sid1 - ladd_id) * ndims
        cu = (uadd_id - sid2) * ndims

        lgs = len(gs)
        cgs = gs[cl : lgs - cu]
        cstiff = stiff[cl : lgs - cu, cl : lgs - cu]
                
        assert (
            len(cgs) == (id2 - id1) * ndims
        ), f"invalid length of gs ({len(cgs)=}). Should be {(id2-id1)*ndims=}"
        return cgs, cstiff

    # if periodic
    lseq = len(seq)
    extseq = str(seq)

    lid = sid1 - tail_size
    ladds = 0
    if lid < 0:
        ladds = int(np.ceil(-lid / lseq))
        extseq += ladds * seq
        lid += ladds * lseq

    uid = sid2 + tail_size
    uadds = 0
    if uid > lseq:
        uadds = int(np.ceil(uid / lseq))
        extseq += uadds * seq
    uid += ladds * lseq

    fseq = extseq[lid:uid]
    gs, stiff = stiffgen_method(fseq, **stiffgen_args)
    cut = tail_size * ndims
    gs = gs[cut:-cut]
    stiff = stiff[cut:-cut, cut:-cut]

    assert (
        len(gs) == (id2 - id1) * ndims
    ), f"invalid length of gs ({len(gs)}), should be {(id2-id1)*ndims}"
    return gs, stiff


#######################################################################################
#######################################################################################
#######################################################################################
