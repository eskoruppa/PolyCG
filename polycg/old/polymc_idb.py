import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .bmat import BlockOverlapMatrix

from .IOPolyMC.iopolymc import write_idb
from .sequence import (
    sequence_file,
    unique_oli_seq,
    unique_olis_in_seq,
    seq2oliseq,
    all_oligomers,
    randseq,
)

from .cgNA_rotbps import cgna_rotbps
from .cgNA_rotbps_renorm import cg_cgna_rotbps

"""
 TO DO:
    - For now the ground state vectors are always multiplied by 180/pi regardless 
      of whether gs_units was set to 'rad' or 'deg'. This needs to be conditional.
"""


##########################################################################################################
############### Generate IDB and seq files ###############################################################
##########################################################################################################


def stiff2idb(
    basefilename: str,
    gs: np.ndarray,
    stiff: BlockOverlapMatrix,
    couprange: int,
    closed: bool,
    seq: str | None = None,
    disc_len: float = 0.34,
    avg_inconsist: bool = True,
    generate_missing: bool = True,
    unique_sequence: bool = True,
    boundary_char: str = "x",
    exclude_chars: str = "y",
) -> None:
    ndims = 3
    Nbps = len(gs) // ndims
    if closed:
        Nbp = Nbps
    else:
        Nbp = Nbps + 1
    olisize = couprange2olisize(couprange)

    if seq is None or (unique_sequence and not unique_olis_in_seq(seq, olisize)):
        assignseq, chars = unique_oli_seq(
            Nbp, olisize, closed=closed, boundary=boundary_char, exclude=exclude_chars
        )
    else:
        assignseq = str(seq)
        chars = "".join(sorted(set(assignseq)))

    # allow cross boundary assignment
    stiff.check_bounds_on_read = False
    params = dict()
    for i in range(Nbps):
        oliseq = seq2oliseq(assignseq, i, couprange, closed)
        T0 = gs[i * ndims : (i + 1) * ndims]

        cl = (i - couprange) * ndims
        cu = (i + couprange + 1) * ndims
        M = stiff[cl:cu, cl:cu]
        coups = _mat2idbcoups(M)

        seqparams = {"seq": oliseq, "vec": T0, "interaction": coups}
        params[oliseq] = seqparams

    if generate_missing:
        params = _add_missing_params(params, couprange, chars, ndims=ndims)

    idbdict = dict()
    idbdict["interaction_range"] = couprange
    idbdict["monomer_types"] = chars
    idbdict["disc_len"] = disc_len
    idbdict["avg_inconsist"] = avg_inconsist
    idbdict["params"] = params

    idbfn = str(basefilename)
    if ".idb" not in idbfn.lower():
        idbfn += ".idb"
    write_idb(idbfn, idbdict, decimals=3)
    sequence_file(basefilename, assignseq, add_extension=True)
    sequence_file(basefilename + ".origseq", seq, add_extension=False)


def _mat2idbcoups(M: np.ndarray):
    ndims = 3
    N = len(M) // ndims
    couprange = (N - 1) // 2
    mats = [
        M[couprange * ndims : (couprange + 1) * ndims, i * ndims : (i + 1) * ndims]
        for i in range(N)
    ]
    for i in range(couprange):
        mats[i] = mats[i].T
    return [_mat2idb_entry(mat) for mat in mats]


def _mat2idb_entry(mat: np.ndarray):
    entry = ["stiffmat"]
    for i in range(3):
        for j in range(3):
            entry.append(mat[i, j])
    return entry


def couprange2olisize(couprange: int) -> int:
    return (1 + couprange) * 2


def olisize2couprange(olisize: int) -> int:
    return olisize // 2 - 1


def _add_missing_params(
    params: dict[str, Any], coup_range: int, chars: str, ndims: int = 3
):
    num_coup = 1 + 2 * coup_range
    oli_size = (coup_range + 1) * 2
    contained_olis = [key for key in params.keys()]
    for oli in all_oligomers(oli_size, chars):
        if oli not in contained_olis:
            seqparams = {
                "seq": oli,
                "vec": np.zeros(3),
                "interaction": [
                    _mat2idb_entry(np.zeros((3, 3))) for i in range(num_coup)
                ],
            }
            params[oli] = seqparams
    return params


if __name__ == "__main__":
    fn = "idbs/test"
    closed = True
    couprange = 2

    reps = 20
    seq = "atcgttagcgatatcgtacc" * reps
    if not closed:
        seq += "a"
    print(len(seq))

    gs, stiff = cgna_rotbps(
        seq,
        closed=closed,
        allow_partial=True,
        gs_units="deg",
        rotation_map="euler",
        split_fluctuations="matrix",
    )

    stiff2idb(fn, gs, stiff, couprange, closed, seq)

    ########################################################################
    ########################################################################
    ########################################################################

    fn = "idbs/test_cg2"
    closed = True
    couprange = 2
    composite_size = 20

    disc_len = 0.34

    block_ncomp = 14
    overlap_ncomp = 2
    tail_ncomp = 4

    reps = 40
    seq = "atcgttagcgatatcgtacc" * reps
    if not closed:
        seq += "a"
    print(len(seq))
    seq = randseq(len(seq))

    gs, stiff = cgna_rotbps(
        seq,
        closed=closed,
        allow_partial=False,
        gs_units="deg",
        rotation_map="euler",
        split_fluctuations="matrix",
    )

    gs, stiff = cg_cgna_rotbps(
        seq,
        composite_size,
        first_entry=0,
        closed=closed,
        allow_partial=True,
        allow_crop=False,
        block_ncomp=block_ncomp,
        overlap_ncomp=overlap_ncomp,
        tail_ncomp=tail_ncomp,
        gs_units="deg",
        rotation_map="euler",
        split_fluctuations="matrix",
    )

    cgNbps = len(gs)
    if closed:
        cgNbps += 1
    assignseq, chars = unique_oli_seq(
        cgNbps, couprange2olisize(couprange), closed=closed, boundary="x", exclude="y"
    )

    stiff2idb(
        fn, gs, stiff, couprange, closed, assignseq, disc_len=disc_len * composite_size
    )
