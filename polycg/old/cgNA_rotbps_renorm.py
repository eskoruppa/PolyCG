from __future__ import annotations
import sys
import numpy as np
from typing import Tuple

from .cgNA_rotbps import _cgna_rotbps
from .bmat import BlockOverlapMatrix

from .partials_cg import partial_cg
from .cg import composite_groundstate
from .cg import cg_stiff_euler_groupsplit

# CGNA_DEFAULT_DATASET = "cgDNA+_Curves_BSTJ_10mus_FS"
# CGNA_BLOCK_SIZE = 140
# CGNA_OVERLAP_SIZE = 20
# CGNA_TAIL_SIZE = 20

CGNA_MIN_FOR_PARTIAL = 200
CGNA_PARTIAL_BLOCK_SIZE   = 160
CGNA_PARTIAL_OVERLAP_SIZE = 20
CGNA_PARTIAL_TAIL_SIZE    = 20

CGNA_PARTIAL_MIN_BLOCK_NCOMP = 6
CGNA_PARTIAL_MIN_OVERLAP_NCOMP = 2
CGNA_PARTIAL_MIN_TAIL_NCOMP = 2


##########################################################################################################
############### Coarse-Grained basepair step groundstate and stiffness ###################################
##########################################################################################################

def cg_cgna_rotbps(
    seq: str,
    composite_size: int,
    first_entry: int = 0,
    closed: bool = False,
    rotation_map: str = "euler",
    split_fluctuations: str = "matrix",
    disc_len = 0.34,
    in_nm: bool = True,
    gs_units: str = "rad",
    static_left: bool = True,
    rescale: bool = True,
    ps_set: str = 'default',
    allow_partial: bool = True,
    allow_crop: bool = False,
    block_ncomp: int|None=None,
    overlap_ncomp: int|None=None,
    tail_ncomp: int|None=None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    _check_setting(
        rotation_map,
        split_fluctuations,
        in_nm,
        gs_units,
        static_left,
        rescale,
        ps_set 
    )

    if block_ncomp is None:
        block_ncomp = int(np.ceil(CGNA_PARTIAL_BLOCK_SIZE/composite_size)) 
        if block_ncomp < CGNA_PARTIAL_MIN_BLOCK_NCOMP:
            block_ncomp = CGNA_PARTIAL_MIN_BLOCK_NCOMP
    if overlap_ncomp is None:
        overlap_ncomp = int(np.ceil(CGNA_PARTIAL_OVERLAP_SIZE/composite_size))
        if overlap_ncomp < CGNA_PARTIAL_MIN_OVERLAP_NCOMP:
            overlap_ncomp = CGNA_PARTIAL_MIN_OVERLAP_NCOMP
    if tail_ncomp is None:
        tail_ncomp = int(np.ceil(CGNA_PARTIAL_TAIL_SIZE/composite_size))
        if tail_ncomp < CGNA_PARTIAL_MIN_TAIL_NCOMP:
            tail_ncomp = CGNA_PARTIAL_MIN_TAIL_NCOMP
    
    if allow_partial and len(seq) >= CGNA_MIN_FOR_PARTIAL:
        cggs,cgstiff = _cg_cgna_rotbps_partial(
            seq,
            composite_size,
            block_ncomp,
            overlap_ncomp,
            tail_ncomp,
            first_entry=first_entry,
            closed=closed,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            disc_len=disc_len,
            in_nm=in_nm,
            gs_units='rad',
            static_left=static_left,
            rescale=rescale,
            ps_set=ps_set,
            allow_crop=allow_crop        
        )

    elif closed:
        one_block = len(seq) // composite_size + overlap_ncomp
        cggs,cgstiff = _cg_cgna_rotbps_partial(
            seq,
            composite_size,
            one_block,
            overlap_ncomp,
            tail_ncomp,
            first_entry=0,
            closed=closed,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            disc_len=disc_len,
            in_nm=in_nm,
            gs_units='rad',
            static_left=static_left,
            rescale=rescale,
            ps_set=ps_set,
            allow_crop=allow_crop        
        )
    else:
        if len(seq) % composite_size != 1:
            if not allow_crop:
                raise ValueError(
                    f"The number of base pair steps (#bp-1={len(seq)-1}) needs to be a multiple of the composite size ({composite_size}). Set allow_crop to True for auto-cropping."
                )
            # crop sequence to correct length
            tailcut = (len(seq) - 1) % composite_size
            seq = seq[: len(seq) - tailcut]
        
        cggs, cgstiff = _cg_cgna_rotbps(
            seq,
            composite_size,
            first_entry = first_entry,
            rotation_map = rotation_map,
            split_fluctuations = split_fluctuations,
            disc_len = disc_len,
            in_nm = in_nm,
            gs_units = 'rad',
            static_left = static_left,
            rescale = rescale,
            ps_set = ps_set
        )
        bom_cgstiff = BlockOverlapMatrix(
            3,
            average=True,
            xlo=0,
            xhi=len(cggs),
            ylo=0,
            yhi=len(cggs),
            periodic=closed,
            fixed_size=True,
            check_bounds=True,
            check_bounds_on_read=False,
        )
        bom_cgstiff.add_block(cgstiff,x1=0,x2=len(cggs))
        cgstiff = bom_cgstiff
    
    if gs_units == 'deg':
        cggs *= 180./np.pi    
    return cggs, cgstiff
        
    
#######################################################################################
#######################################################################################

def _cg_cgna_rotbps_partial(
    seq: str,
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    tail_ncomp: int,
    first_entry: int = 0,
    closed: bool = False,
    rotation_map: str = "euler",
    split_fluctuations: str = "matrix",
    disc_len = 0.34,
    in_nm: bool = True,
    gs_units: str = "rad",
    static_left: bool = True,
    rescale: bool = True,
    ps_set: str = 'default',
    allow_crop = False,
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    
    _check_setting(
        rotation_map,
        split_fluctuations,
        in_nm,
        gs_units,
        static_left,
        rescale,
        ps_set 
    )
        
    stiffgen_method = _cgna_rotbps
    stiffgen_args = {
        "in_nm": in_nm,
        "disc_len": disc_len,
        "rotation_map": rotation_map,
        "split_fluctuations": split_fluctuations,
        "gs_units": 'rad',
        "ps_set": ps_set,
    }
    ndims = 3

    cg_method = cg_stiff_euler_groupsplit
    cg_method_args = {"static_left": static_left, "rescale": rescale}

    cggs, cgstiff = partial_cg(
        seq,
        composite_size,
        block_ncomp,
        overlap_ncomp,
        tail_ncomp,
        stiffgen_method,
        stiffgen_args,
        cg_method,
        cg_method_args=cg_method_args,
        rotation_map=stiffgen_args["rotation_map"],
        split_fluctuations=stiffgen_args["split_fluctuations"],                         
        closed=closed,
        ndims=ndims,
        first_entry=first_entry,
        static_left=static_left,
        rescale=rescale,
        allow_crop=allow_crop,
    )
    
    return cggs, cgstiff
    
    
    
#######################################################################################
#######################################################################################


def _cg_cgna_rotbps(
    seq: str,
    composite_size: int,
    first_entry: int = 0,
    rotation_map: str = "euler",
    split_fluctuations: str = "matrix",
    disc_len: float = 0.34,
    in_nm: bool = True,
    gs_units: str = "rad",
    static_left: bool = True,
    rescale: bool = True,
    ps_set: str = 'default'
) -> Tuple[np.ndarray, np.ndarray]:

    _check_setting(
        rotation_map,
        split_fluctuations,
        in_nm,
        gs_units,
        static_left,
        rescale,
        ps_set 
    )

    gs, stiff = _cgna_rotbps(
        seq,
        in_nm=in_nm,
        disc_len=disc_len,
        rotation_map=rotation_map,
        split_fluctuations=split_fluctuations,
        gs_units=gs_units,
        ps_set=ps_set
    )
    cggs = composite_groundstate(
        gs, composite_size, first_entry=first_entry, rotation_map=rotation_map
    )
    if rotation_map == "euler":
        if split_fluctuations == "matrix":
            cgM = cg_stiff_euler_groupsplit(
                gs,
                stiff,
                composite_size,
                first_entry=first_entry,
                static_left=static_left,
                rescale=rescale,
            )
        else:
            raise Exception("Not yet implemented")

    elif rotation_map == "cayley":
        if split_fluctuations == "matrix":
            raise ValueError(
                f'Option split_fluctuations="matrix" only implemented for Euler rotation map.'
            )
        else:
            raise Exception("Not yet implemented")
    return cggs, cgM

#######################################################################################
#######################################################################################


def _check_setting(
    rotation_map,
    split_fluctuations,
    in_nm,
    gs_units,
    static_left,
    rescale,
    ps_set 
):
    if rotation_map != "euler":
        raise ValueError(f'At this point only euler vectors are supported for coarse-graining. Please set rotation_map to "euler".')
    if split_fluctuations != "matrix":
        raise ValueError(f'At this point only matrix splitting of static components is supported for coarse-graining. Please set split_fluctuations to "matrix".')
    

#######################################################################################
#######################################################################################
#######################################################################################


if __name__ == "__main__":


    reps = 500
    seq = "atcgttagcgatatcgtacc" * reps #+ 'a'
    print(len(seq))
    
    composite_size = 10
    
    closed        = True
    allow_partial = True
    allow_crop    = False
    
    block_ncomp=None
    overlap_ncomp=None
    tail_ncomp=None
    
    import time
    t1 = time.time()

    gs, stiff = cg_cgna_rotbps(
        seq,
        composite_size,
        first_entry = 0,
        closed = closed,
        allow_partial=allow_partial,
        allow_crop = allow_crop,
        block_ncomp=block_ncomp,
        overlap_ncomp=overlap_ncomp,
        tail_ncomp=tail_ncomp,
    )
    
    t2 = time.time()
    print(f'dt = {(t2-t1)}')
    
    stiff.check_bounds_on_read = False
    
    nbps = len(seq)
    if not closed:
        nbps -= 1
    
    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    
    print(nbps/composite_size*3)
    print(stiff.shape)

    
    print(stiff[-3:3,-3:3])
    
    
    print(stiff[len(gs)-3:len(gs)+3,len(gs)-3:len(gs)+3])
    
    # print(stiff[len(gs)-2:len(gs),len(gs)-2:len(gs)])