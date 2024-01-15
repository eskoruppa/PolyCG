from __future__ import annotations
import sys
import numpy as np
from typing import Tuple, List, Callable, Any, Dict

from .bmat import BlockOverlapMatrix
from .partials import partial_stiff
from .cg import composite_groundstate


CG_PARTIALS_MIN_BLOCK = 4

#######################################################################################
#######################################################################################


def partial_cg(
    seq: str,
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    stiffgen_method: Callable,
    stiffgen_args: Dict[str, Any],
    cg_method: Callable,
    cg_method_args: Dict[str, Any] = {},
    rotation_map: str = "euler",
    split_fluctuations: str = "matrix",
    closed: bool = False,
    ndims: int = 3,
    first_entry: int = 0,
    static_left: bool = True,
    rescale: bool = True,
    allow_crop: bool = False,
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    seq = seq[first_entry:]
    Nbps = _get_nbps(seq, closed)

    if Nbps % composite_size != 0:
        if closed:
            raise ValueError(
                f"Coarse graining of closed molecules require the number of bps ({Nbps}) to be a multiple of the composite size ({composite_size})."
            )
        if not allow_crop:
            raise ValueError(
                f"The number of base pair steps (#bp-1={Nbps}) needs to be a multiple of the composite size ({composite_size}). Set allow_crop to True for auto-cropping."
            )

        # crop sequence to correct length
        tailcut = (len(seq) - 1) % composite_size
        seq = seq[: len(seq) - tailcut]
        Nbps = _get_nbps(seq, closed)

        print(f"Cropped Sequence to length {len(seq)}.")

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
        return _partial_cg_closed(
            seq,
            composite_size,
            block_ncomp,
            overlap_ncomp,
            tail_ncomp,
            stiffgen_method,
            stiffgen_args,
            cg_method,
            cg_method_args,
            rotation_map=rotation_map,
            ndims=ndims,
        )

    return _partial_cg_linear(
        seq,
        composite_size,
        block_ncomp,
        overlap_ncomp,
        tail_ncomp,
        stiffgen_method,
        stiffgen_args,
        cg_method,
        cg_method_args,
        rotation_map=rotation_map,
        ndims=ndims,
    )


#######################################################################################
#######################################################################################


def _partial_cg_closed(
    seq: str,
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    stiffgen_method: Callable,
    stiffgen_args: Dict[str, Any],
    cg_method: Callable,
    cg_method_args: Dict[str, Any] = {},
    rotation_map: str = "euler",
    ndims: int = 3,
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    Nbps = len(seq)
    assert (
        Nbps % composite_size == 0
    ), f"Nbps ({Nbps}) is not a multiple of composite_size ({composite_size})."
    Ncg = Nbps // composite_size
    Ncg_full = Ncg + overlap_ncomp

    block_size = block_ncomp * composite_size
    overlap_size = overlap_ncomp * composite_size
    tail_size = tail_ncomp * composite_size

    gs, stiff = partial_stiff(
        seq,
        stiffgen_method,
        stiffgen_args,
        block_size,
        overlap_size,
        tail_size,
        closed=True,
        ndims=ndims,
    )

    block_incr = block_ncomp - overlap_ncomp
    Nsegs = int(np.floor(Ncg_full / block_incr))
    lastseg_id = Nsegs - 1

    cgstiff = BlockOverlapMatrix(
        ndims,
        average=True,
        periodic=True,
        xlo=0,
        xhi=Ncg * ndims,
        ylo=0,
        yhi=Ncg * ndims,
    )

    for i in range(Nsegs):
        # block range
        id1 = i * block_incr
        id2 = id1 + block_ncomp

        if i == lastseg_id and id2 < Ncg_full:
            id2 = Ncg_full
            # assert (
            #     id2 >= Ncg_full
            # ), f"For last segment id2 ({id2}) should be larger or equal to Ncg ({Ncg})."

        print(
            f"Coarse-graining from bps {id1*composite_size} to {id2*composite_size} ({Ncg_full*composite_size} in total)."
        )

        lid = id1 - tail_ncomp
        uid = id2 + tail_ncomp

        al = lid * composite_size * ndims
        au = uid * composite_size * ndims

        # pgs    = gs[al:au]
        pgs = _periodic_gs(gs, al, au)
        pstiff = stiff[al:au, al:au]

        # coarse-grain block
        block_cgstiff = cg_method(pgs, pstiff, composite_size, **cg_method_args)

        cl = tail_ncomp * ndims
        cu = len(block_cgstiff) - cl
        pcgstiff = block_cgstiff[cl:cu, cl:cu]

        mid1 = id1 * ndims
        mid2 = id2 * ndims
        cgstiff.add_block(pcgstiff, mid1, mid2, y1=mid1, y2=mid2)

    cggs = composite_groundstate(
        gs, composite_size, first_entry=0, rotation_map=rotation_map
    )
    return cggs, cgstiff


#######################################################################################
#######################################################################################


def _periodic_gs(gs: np.ndarray, id1: int, id2: int):
    N = len(gs)
    ladd = 0
    if id1 < 0:
        ladd = int(np.ceil(-id1 / N))
    uadd = 0
    if id2 > N:
        uadd = int(np.floor(id2 / N))
    cgs = np.concatenate((gs,) * (1 + ladd + uadd))
    return cgs[ladd * N + id1 : ladd * N + id2]


#######################################################################################
#######################################################################################


def _partial_cg_linear(
    seq: str,
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    stiffgen_method: Callable,
    stiffgen_args: Dict[str, Any],
    cg_method: Callable,
    cg_method_args: Dict[str, Any] = {},
    rotation_map: str = "euler",
    ndims: int = 3,
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    Nbps = len(seq) - 1
    assert (
        Nbps % composite_size == 0
    ), f"Nbps ({Nbps}) is not a multiple of composite_size ({composite_size})."
    Ncg = Nbps // composite_size

    # Ncg_full = Ncg + overlap_ncomp

    block_size = block_ncomp * composite_size
    overlap_size = overlap_ncomp * composite_size
    tail_size = tail_ncomp * composite_size

    gs, stiff = partial_stiff(
        seq,
        stiffgen_method,
        stiffgen_args,
        block_size,
        overlap_size,
        tail_size,
        closed=False,
        ndims=ndims,
    )

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
        block_cgstiff = cg_method(pgs, pstiff, composite_size, **cg_method_args)

        cl = (id1 - lid) * ndims
        cu = len(block_cgstiff) - (uid - id2) * ndims
        pcgstiff = block_cgstiff[cl:cu, cl:cu]

        mid1 = id1 * ndims
        mid2 = id2 * ndims
        cgstiff.add_block(pcgstiff, mid1, mid2, y1=mid1, y2=mid2)

    cggs = composite_groundstate(
        gs, composite_size, first_entry=0, rotation_map=rotation_map
    )
    return cggs, cgstiff


#######################################################################################
#######################################################################################


def _get_nbps(seq, closed):
    nbps = len(seq)
    if not closed:
        nbps -= 1
    return nbps
