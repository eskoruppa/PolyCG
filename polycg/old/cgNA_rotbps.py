from __future__ import annotations
import sys
import numpy as np
from typing import Tuple

from .cgNA_plus.modules.cgDNAUtils import constructSeqParms

from .conversions import stiff2rad, gs2deg, gs2rad
from .conversions import rotbps_cayley2euler, rotbps_algebra2group
from .marginals import marginal, rot_marginal, marginal_sparse

from .bmat import BlockOverlapMatrix
from .partials import partial_stiff

CGNA_DEFAULT_DATASET = "cgDNA+_Curves_BSTJ_10mus_FS"
CGNA_BLOCK_SIZE = 140
CGNA_OVERLAP_SIZE = 20
CGNA_TAIL_SIZE = 20

CGNA_MIN_FOR_PARTIAL = 250

##########################################################################################################
############### Rotational Base pair-step degrees of freedom from cgNA+ ##################################
##########################################################################################################


def cgna_rotbps(
    seq: str,
    closed: bool = False,
    in_nm: bool = True,
    disc_len: bool = 0.34,
    rotation_map: str = "cayley",
    split_fluctuations: str = "vector",
    gs_units: str = "rad",
    ps_set: str = "default",
    allow_partial: bool = True,
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    """Returns the groundstate (gs) and stiffness matrix of the rotational component of the base pair step degrees of freedom
        for the provided sequence.

    Args:
        seq (str): sequence of reference strand nucleotides
        in_nm (bool): express stiffness matrix in nanometers
        disc_len (float): discretization length. Only important if stiffness matrix is expressed in nanometers
        rotation_map (str): selected map between rotation rotation coordinates and rotation matrix.
                        Options:    - cayley: default cnDNA map (Cayley map)
                                    - euler:  Axis angle representation.
        split_fluctuations (str):
        gs_units (str): express groundstate in units of radians (rad), degrees (deg)
        ps_set (str): parameter set

    Returns:
        Tuple[np.ndarray,np.ndarray]: Groundstate and stiffness matrix.
    """  
    
    if allow_partial and len(seq) > CGNA_MIN_FOR_PARTIAL:
        print('run partial')
        return _cgna_rotbps_partial(
            seq,
            block_size=CGNA_BLOCK_SIZE,
            overlap_size=CGNA_OVERLAP_SIZE,
            tail_size=CGNA_TAIL_SIZE,
            closed=closed,
            in_nm=in_nm,
            disc_len=disc_len,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            gs_units=gs_units,
            ps_set=ps_set,
        )
        
    if closed:
        block_size=len(seq)+CGNA_OVERLAP_SIZE
        overlap_size=CGNA_OVERLAP_SIZE
        tail_size=CGNA_TAIL_SIZE
        
        if overlap_size > len(seq):
            overlap_size = len(seq)-1
        
        print(f'{block_size=}')
        print(f'{overlap_size=}')
        
        return _cgna_rotbps_partial(
            seq,
            block_size=block_size,
            overlap_size=overlap_size,
            tail_size=tail_size,
            closed=closed,
            in_nm=in_nm,
            disc_len=disc_len,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            gs_units=gs_units,
            ps_set=ps_set,
        )
    else:
        gs, stiff = _cgna_rotbps(
            seq,
            in_nm=in_nm,
            disc_len=disc_len,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            gs_units=gs_units,
            ps_set=ps_set,
        )
        bom_stiff = BlockOverlapMatrix(
            3,
            average=True,
            xlo=0,
            xhi=len(gs),
            ylo=0,
            yhi=len(gs),
            periodic=closed,
            fixed_size=True,
            check_bounds=True,
            check_bounds_on_read=False,
        )
        bom_stiff.add_block(stiff,x1=0,x2=len(gs))
        return gs, bom_stiff

#######################################################################################
#######################################################################################

def _cgna_rotbps_partial(
    seq: str,
    block_size: int = CGNA_BLOCK_SIZE,
    overlap_size: int = CGNA_OVERLAP_SIZE,
    tail_size: int = CGNA_TAIL_SIZE,
    closed: bool = False,
    in_nm: bool = True,
    disc_len: bool = 0.34,
    rotation_map: str = "cayley",
    split_fluctuations: str = "vector",
    gs_units: str = "rad",
    ps_set: str = "default",
) -> Tuple[np.ndarray, BlockOverlapMatrix]:
    stiffgen_method = _cgna_rotbps
    stiffgen_args = {
        "in_nm": in_nm,
        "disc_len": disc_len,
        "rotation_map": rotation_map,
        "split_fluctuations": split_fluctuations,
        "gs_units": gs_units,
        "ps_set": ps_set,
    }
    return partial_stiff(
        seq,
        stiffgen_method,
        stiffgen_args,
        block_size,
        overlap_size,
        tail_size,
        closed=closed,
        ndims=3,
    )


#######################################################################################
#######################################################################################

def _cgna_rotbps(
    seq: str,
    in_nm: bool = True,
    disc_len: bool = 0.34,
    rotation_map: str = "cayley",
    split_fluctuations: str = "vector",
    gs_units: str = "rad",
    ps_set: str = "default",
) -> Tuple[np.ndarray, np.ndarray]:
    if ps_set == "default":
        ps_set = CGNA_DEFAULT_DATASET

    # set whether dynamic component is split in so3 or SO3
    if (
        split_fluctuations.lower()
        in ["rotmat", "mat", "matrix", "rotationmatrix", "group", "liegroup"]
        or split_fluctuations == "SO3"
    ):
        if rotation_map.lower() != "euler":
            raise ValueError(
                f"Option for matrix decomposition of static and dynamic components is only available for Euler rotation map."
            )
        vecsplit = False
    elif (
        split_fluctuations.lower() in ["vec", "vector", "algebra", "liealgebra"]
        or split_fluctuations == "so3"
    ):
        vecsplit = True
    else:
        raise ValueError(
            f'Unknown value "{split_fluctuations}" for split_fluctuations.'
        )
        
    # Generate static component and stiffness matrix
    gs, stiff_sp = constructSeqParms(seq, ps_set)
    # stiff = stiff_sp.toarray()

    # # take marginals to rotational base pair step components and convert to radians
    # pgs,mstiff = marginal(gs,stiff,seq,['y*'],raise_invalid=True)

    pgs, mstiff = marginal_sparse(gs, stiff_sp, seq, ["y*"], raise_invalid=True)
    rbps_gs, rbps_stiff = rot_marginal(pgs, mstiff)
    rbps_stiff_rad = stiff2rad(rbps_stiff, only_rot=True) * disc_len

    # Cayley map option
    if rotation_map.lower() == "cayley":
        if gs_units.lower() in ["deg", "degree", "degrees"]:
            return gs2deg(rbps_gs, only_rot=True), rbps_stiff_rad
        elif gs_units.lower() in ["rad", "radians", "rads"]:
            return gs2rad(rbps_gs, only_rot=True), rbps_stiff_rad
        else:
            ValueError(f'Unknown gs_units "{gs_units}"')

    # Euler map option
    elif rotation_map.lower() == "euler":
        gs_rad = gs2rad(rbps_gs, only_rot=True)
        gs_euler, stiff_euler = rotbps_cayley2euler(gs_rad, rbps_stiff_rad)

        if not vecsplit:
            stiff_euler = rotbps_algebra2group(gs_euler, stiff_euler)

        if gs_units.lower() in ["deg", "degree", "degrees"]:
            return gs_euler * 180 / np.pi, stiff_euler
        elif gs_units.lower() in ["rad", "radians", "rads"]:
            return gs_euler, stiff_euler
        else:
            ValueError(f'Unknown gs_units "{gs_units}"')
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


##########################################################################################################
##########################################################################################################
##########################################################################################################

if __name__ == "__main__":
    
    reps = 1
    seq = "atcgttagcgatatcgtacc" * reps  # + 'atcgttagcg'
    print(len(seq))
    
    gs,stiff = cgna_rotbps(
        seq,
        closed = False,
        allow_partial = False,
    )
    
    print(stiff[-1:2,-1:2])
    print(stiff[len(gs)-2:len(gs)+1,len(gs)-2:len(gs)+1])
    
    
    
    check_linear = 1
    check_closed = 1
    
    # CHECK LINEAR
    if check_linear:
        reps = 10
        seq = "atcgttagcgatatcgtacc" * reps + "a"  # + 'atcgttagcgaaaaaaaaaa'
        print(len(seq))

        stiffgen_method = _cgna_rotbps
        stiffgen_args = {
            "in_nm": True,
            "disc_len": 0.34,
            "rotation_map": "euler",
            "split_fluctuations": "matrix",
            "gs_units": "rad",
            "ps_set": "default",
        }

        block_size = 40
        overlap_size = 20
        tail_size = 20
        closed = False

        gs, stiff = partial_stiff(
            seq,
            stiffgen_method,
            stiffgen_args,
            block_size,
            overlap_size,
            tail_size,
            closed=closed,
        )

        fgs, fstiff = stiffgen_method(seq, **stiffgen_args)

        print("###################")
        print("Checking difference")
        ndims = 3
        for i in range(len(seq) - 1):
            diff = (
                (gs[i * ndims : (i + 1) * ndims] - fgs[i * ndims : (i + 1) * ndims])
                * 180
                / np.pi
            )
            diff = np.abs(diff)
            if np.max(diff) > 0.0001:
                print(f"gs - {i}: {diff}")

        sdiff = stiff[:, :] - fstiff
        print(np.sum(np.abs(sdiff)))

    # CHECK CLOSED
    if check_closed:
        reps = 10
        seq = "atcgttagcgatatcgtacc" * reps  # + 'atcgttagcg'

        block_size = 40
        overlap_size = 20
        tail_size = 20
        closed = True

        ndims = 3

        stiffgen_method = _cgna_rotbps
        stiffgen_args = {
            "in_nm": True,
            "disc_len": 0.34,
            "rotation_map": "euler",
            "split_fluctuations": "matrix",
            "gs_units": "rad",
            "ps_set": "default",
        }

        import time

        t1 = time.time()

        gs, stiff = partial_stiff(
            seq,
            stiffgen_method,
            stiffgen_args,
            block_size,
            overlap_size,
            tail_size,
            closed=closed,
            ndims=ndims,
        )
        t2 = time.time()
        print(f"dt = {t2-t1}")
        t1 = time.time()

        ext = 50

        mainseq = seq + seq[0]
        addseq = seq[1:]

        extseq = addseq[-ext:] + mainseq + addseq[:ext]
        ffgs, ffstiff = stiffgen_method(extseq, **stiffgen_args)

        t2 = time.time()
        print(f"dt = {t2-t1}")
        t1 = time.time()

        fgs = ffgs[3 * ext : -3 * ext]
        fstiff = ffstiff[3 * ext : -3 * ext, 3 * ext : -3 * ext]

        lfstiff = ffstiff[3 * ext :, 3 * ext :]

        print("\n\n##############")

        print(f"{stiff.shape=}")
        print(f"{fstiff.shape=}")

        print(f"{len(gs)=}")
        print(f"{len(fgs)=}")

        np.set_printoptions(linewidth=200, precision=6, suppress=True)

        gsdiff = gs - fgs
        print(f"del gs = {np.sum(np.abs(gsdiff))}")

        cstiff = stiff[:, :]
        for i in range(len(stiff)):
            for j in range(len(stiff)):
                if np.abs(i - j) > ndims * overlap_size:
                    cstiff[i, j] = 0

        sdiff = cstiff - fstiff
        print(f"del stiff sum = {np.sum(np.abs(sdiff))}")
        print(f"del stiff max = {np.max(np.abs(sdiff))}")
        print(f"del stiff max = {np.max(np.abs(sdiff))}")

        print("###################")
        print("Checking gs difference")
        ndims = 3
        for i in range(len(mainseq) - 1):
            diff = (
                (gs[i * ndims : (i + 1) * ndims] - fgs[i * ndims : (i + 1) * ndims])
                * 180
                / np.pi
            )
            if np.max(diff) > 0.0001:
                print(f"gs - {i}:")
                print(f" diff: {diff}")
                print(f" gs:   {gs[i * ndims : (i + 1) * ndims]*180/np.pi}")
                print(f" fgs:  {fgs[i * ndims : (i + 1) * ndims]*180/np.pi}")

        # add = 3
        # for i in range(len(mainseq) - 1):
        #     print(i)
        #     print(sdiff[(i-add)*ndims:(i+1+add)*ndims,(i-add)*ndims:(i+1+add)*ndims])
