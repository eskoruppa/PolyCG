import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .bps_params import seq2rotbps, seq2rotbps_partials
from .cgNA_rotbps import cgna_rotbps
from .IOPolyMC.iopolymc import dna_oligomers, write_idb
from .sequence import (
    all_oligomers,
    randseq,
    sequence_file,
    unique_oli_seq,
    unique_olis_in_seq,
)

# from cgstiff import randseq,seq2rotbps
# from cgstiff import dna_oligomers
# from cgstiff import write_idb
# from cgstiff import seq_edit


"""
 TO DO:
    - For now the ground state vectors are always multiplied by 180/pi regardless 
      of whether gs_units was set to 'rad' or 'deg'. This needs to be conditional.
"""


GENERATOR_DEFAULT_ARGS = {
    "in_nm": True,
    "disc_len": 0.34,
    "rotation_map": "euler",
    "split_fluctuations": "vector",
    "gs_units": "rad",
    "ps_set": "default",
    "partial_enabled": True,
}

##########################################################################################################
############### Generate IDB and seq files ###############################################################
##########################################################################################################


def polymc_full(
    filename: str,
    coup_range: int,
    disc_len: float = 0.34,
    avg_inconsist: bool = True,
    avg_flank_size: int = 2,
    termini_seg_size: int = 8,
    num_termini_segs: int = 1,
    param_generator: Callable = seq2rotbps,
    genargs: dict = GENERATOR_DEFAULT_ARGS,
) -> None:
    """
    Generates a full idb file covering all sequence segments. Both the ground state and the stiffness blocks are averaged
    over all flanking segments of size avg_flank_size.
    """
    olidicts = polymc_couplings(
        coup_range,
        avg_flank_size,
        termini_seg_size,
        num_termini_segs=num_termini_segs,
        param_generator=param_generator,
        genargs=genargs,
    )
    olidicts2idb(
        filename,
        olidicts,
        coup_range=-1,
        disc_len=disc_len,
        avg_inconsist=avg_inconsist,
    )


def polymc_seq(
    basefilename: str,
    seq: str,
    coup_range: int,
    param_generator: Callable = seq2rotbps,
    genargs: dict = GENERATOR_DEFAULT_ARGS,
    ndims: int = 3,
    closed: bool = False,
    closure_copy_size: int = 25,
    lower_terminus_seq: str = "",
    upper_terminus_seq: str = "",
    generate_missing: bool = True,
    disc_len: float = 0.34,
    avg_inconsist=True,
) -> None:
    """Creates IDB file for specified sequence

    Args:
        basefilename (str): basefilename for outputfiles
        seq (str): sequence
        coup_range (int): coupling range ( oligomer size = (coup_range+1)*2 )
        param_generator (callable): Method for generating the sequence dependent parameters (defaults to seq2rotbps)
        param_generator_additional_args (dict): Additional parameters needed for param_generator. Assumes param_generator takes seq,*param_generator_additional_args as arguments. Defaults to [].
        ndims (int): number of degrees of freedom per base pair-step (defaults to 3 - rotational dofs)
        closed (bool): closed molecule (defaults to False)
        closure_copy_size (int): size of the copied segments at the terminus. Only relevant for closed=True (defaults to 25)
        lower_terminus_seq (str): additional sequence used to append lower terminus to obtain non-openended paramters. Only relevant for closed=False (defaults to '')
        upper_terminus_seq (str): additional sequence used to append upper terminus to obtain non-openended paramters. Only relevant for closed=False (defaults to '')
        generate_missing (bool): generate all unique oligomers (defaults to True)
        disc_len (float): Discretization length

    Returns:
        None
    """
    if closed:
        olidicts, idbseq, monomertypes = seq_couplings_closed(
            seq,
            coup_range,
            param_generator=param_generator,
            genargs=genargs,
            ndims=ndims,
            closure_copy_size=closure_copy_size,
            generate_missing=generate_missing,
        )
    else:
        olidicts, idbseq, monomertypes = seq_couplings_open(
            seq,
            coup_range,
            param_generator=param_generator,
            genargs=genargs,
            ndims=ndims,
            lower_terminus_seq=lower_terminus_seq,
            upper_terminus_seq=upper_terminus_seq,
            generate_missing=generate_missing,
        )

    olidicts2idb(
        basefilename,
        olidicts,
        monomertypes=monomertypes,
        coup_range=coup_range,
        disc_len=disc_len,
        avg_inconsist=avg_inconsist,
    )
    sequence_file(basefilename, idbseq, add_extension=True)
    sequence_file(basefilename + ".origseq", seq, add_extension=False)


# def params2idb(filename: str, groundstate: np.ndarray, stiff: np.ndarray, coup_range: int, disc_len: float=0.34, seq: str=None, closed: bool=False):

# olidicts = coup2olidicts(gs,
#                             stiff,
#                             coup_range,
#                             assignseq,
#                             chars,
#                             ndims=ndims,
#                             allow_boundary_assignment=True,
#                             generate_missing=generate_missing)


def polymc_cg(
    filename: str,
    seq: str,
    composite_size: int,
    coup_range: int,
    closed_topol: bool,
    rotation_map="euler",
    split_fluctuations="matrix",
    static_left=True,
    disc_len=0.34,
    avg_inconsist=True,
) -> None:
    """
    Generates the idb and sequence files for the provided sequence coarse grained to the specified composite step level.
    """

    raise Exception("IN PROGRESS")

    gs, M = cgna_rotbps(
        seq,
        composite_size,
        first_entry=0,
        rescale=True,
        rotation_map=rotation_map,
        split_fluctuations=split_fluctuations,
        in_nm=True,
        gs_units="rad",
        static_left=static_left,
    )

    cgna_rotbps()

    olidict = {"oli": oliseq, "T0": T0 * 180.0 / np.pi, "M": M}

    olidicts = seq_couplings(
        seq,
        coup_range,
        left_terminus_seq=left_terminus_seq,
        right_terminus_seq=right_terminus_seq,
        disc_len=disc_len,
    )
    olidicts2idb(
        filename,
        olidicts,
        coup_range=-1,
        disc_len=disc_len,
        avg_inconsist=avg_inconsist,
    )
    sequence_file(filename, seq)


###################################################################################################################
###################################################################################################################
###################################################################################################################


def cg_polymc_seq_linear(
    basefilename: str,
    seq: str,
    coup_range: int,
    composite_size: int,
    first_entry: int = 0,
    rotation_map: str = "euler",
    split_fluctuations: str = "matrix",
    in_nm: bool = True,
    gs_units: str = "rad",
    static_left: bool = True,
    rescale: bool = True,
    ndims: int = 3,
    generate_missing: bool = True,
    disc_len: float = 0.34,
) -> None:
    """ """

    cggs, cgstiff = cgna_rotbps(
        seq,
        composite_size,
        first_entry=first_entry,
        rotation_map=rotation_map,
        split_fluctuations=split_fluctuations,
        in_nm=in_nm,
        gs_units=gs_units,
        static_left=static_left,
        rescale=rescale,
    )

    N = len(cggs) // ndims
    oli_size = coup_range2oli_size(coup_range)
    idbseq, monomertypes = unique_oli_seq(N - 2 * coup_range, oli_size, closed=False)

    olidicts = coup2olidicts_linear(
        cggs,
        cgstiff,
        coup_range,
        idbseq,
        monomertypes,
        ndims=ndims,
        generate_missing=generate_missing,
    )

    olidicts2idb(
        basefilename,
        olidicts,
        monomertypes=monomertypes,
        coup_range=-1,
        disc_len=disc_len * composite_size,
        avg_inconsist=True,
    )
    sequence_file(basefilename, idbseq, add_extension=True)
    sequence_file(basefilename + ".origseq", seq, add_extension=False)


###################################################################################################################
###################################################################################################################
###################################################################################################################


def polymc_couplings(
    coup_range: int,
    avg_flank_size: int,
    termini_seg_size: int,
    num_termini_segs: int = 1,
    param_generator: Callable = seq2rotbps,
    genargs: dict = GENERATOR_DEFAULT_ARGS,
) -> List[Dict]:
    """
    Generates ground state and stiffness matrix for all oligomers covering the specified coupling range. Both the ground
    state and the stiffness blocks are averaged over all flanking segments of size avg_flank_size. Both termini are covered
    by a random sequence of termini_seg_size bp. Setting num_termini_segs to a value larger than 1 will average over that
    number of terminus sequences.
    """
    olidicts = list()
    olis = dna_oligomers((coup_range + 1) * 2, omit_equiv=False)

    for i, oli in enumerate(olis):
        t1 = time.time()
        print(f"{oli} {i+1}/{len(olis)}")
        T0, M = oli_coupling(
            oli,
            avg_flank_size,
            termini_seg_size,
            num_termini_segs=num_termini_segs,
            param_generator=param_generator,
            genargs=genargs,
        )
        olidict = {"oli": oli, "T0": T0 * 180.0 / np.pi, "M": M}
        olidicts.append(olidict)
        t2 = time.time()
        print(f"dt = {t2-t1}")
    return olidicts


def oli_coupling(
    oli: str,
    avg_flank_size: int,
    termini_seg_size: int,
    num_termini_segs: int = 1,
    param_generator: Callable = seq2rotbps,
    genargs: dict = GENERATOR_DEFAULT_ARGS,
) -> Tuple[np.ndarray]:
    """
    Generates ground state and stiffness matrix for the given oligomer sequence. Both the ground state and the stiffness
    blocks are averaged over all flanking segments of size avg_flank_size. Both termini are covered by a random sequence
    of termini_seg_size bp. Setting num_termini_segs to a value larger than 1 will average over that number of terminus
    sequences.
    """
    coup_range = len(oli) // 2 - 1
    if avg_flank_size > 0:
        flanks = dna_oligomers(avg_flank_size, omit_equiv=False)
    else:
        flanks = [""]

    ltermini = ["CG" + randseq(termini_seg_size - 2) for i in range(num_termini_segs)]
    rtermini = ["CG" + randseq(termini_seg_size - 2) for i in range(num_termini_segs)]

    M_list = list()
    T0_list = list()

    Cov_acc = np.zeros((3 * (len(oli) - 1), 3 * (len(oli) - 1)))
    T0_acc = np.zeros(3)
    T0sq_acc = np.zeros(3)
    count = 0

    T0max = np.array([-100.0, -100.0, -100.0])
    T0min = np.array([100.0, 100.0, 100.0])

    for i, lflank in enumerate(flanks):
        # print(f'{i}/{len(flanks)}')
        for rflank in flanks:
            for lterminus in ltermini:
                for rterminus in rtermini:
                    seq = lterminus + lflank + oli + rflank + rterminus
                    gs, stiff = param_generator(seq, **genargs)
                    M = _select_mid_block(stiff, coup_range)
                    T0 = _select_mid_gs(gs)

                    for d in range(3):
                        if T0[d] > T0max[d]:
                            T0max[d] = T0[d]
                        if T0[d] < T0min[d]:
                            T0min[d] = T0[d]

                    Cov_acc += np.linalg.inv(M)
                    T0_acc += T0
                    T0sq_acc += T0**2
                    count += 1

    M_mean = np.linalg.inv(Cov_acc / count)
    T0_mean = T0_acc / count
    # T0_sm   = T0sq_acc/count
    # T0_std  = np.sqrt(T0_sm - T0_mean**2)

    return T0_mean, M_mean


###################################################################################################################
###################################################################################################################
###################################################################################################################


def olidicts2idb(
    filename: str,
    olidicts: List[Dict],
    monomertypes: str = "atcg",
    coup_range: int = -1,
    disc_len: float = 0.34,
    avg_inconsist: bool = True,
    add_extension: bool = True,
):
    """
    Writes the oligomer ground state and stiffness data to a PolyMC interaction database file (IDB).
    """
    params = dict()
    for oli in olidicts:
        seq = oli["oli"]
        vec = oli["T0"]
        coups = get_oli_coups(oli, coup_range)
        seqparams = {"seq": seq.lower(), "vec": vec, "interaction": coups}
        params[seq] = seqparams

    idbdict = dict()
    idbdict["interaction_range"] = coup_range
    idbdict["monomer_types"] = monomertypes
    idbdict["disc_len"] = disc_len
    idbdict["avg_inconsist"] = avg_inconsist
    idbdict["params"] = params

    if add_extension:
        if ".idb" not in filename.lower():
            filename += ".idb"
    write_idb(filename, idbdict, decimals=3)


def get_oli_coups(oligomer: dict, coup_range: int):
    M = oligomer["M"]
    max_range = (len(M) // 3 - 1) // 2
    if coup_range > max_range or coup_range == -1:
        coup_range = max_range
    coups = list()
    # left coups
    for i in range(coup_range):
        Ml = M[i * 3 : (i + 1) * 3, max_range * 3 : (max_range + 1) * 3]
        coups.append(_mat2idb_entry(Ml))
    # mid coup
    M0 = M[max_range * 3 : (max_range + 1) * 3, max_range * 3 : (max_range + 1) * 3]
    coups.append(_mat2idb_entry(M0))
    # right coups
    for i in range(coup_range):
        Mr = M[
            max_range * 3 : (max_range + 1) * 3,
            (max_range + 1 + i) * 3 : (max_range + 2 + i) * 3,
        ]
        coups.append(_mat2idb_entry(Mr))
    return coups


def _mat2idb_entry(mat: np.ndarray):
    entry = ["stiffmat"]
    for i in range(3):
        for j in range(3):
            entry.append(mat[i, j])
    return entry


def _select_mid_block(stiff: np.ndarray, num_coup: float) -> np.ndarray:
    N = len(stiff) // 3
    if N % 2 != 1:
        raise ValueError("Number of bp needs to be even")
    range = 1 + 2 * num_coup
    lr = (N - range) // 2
    block = stiff[lr * 3 : -lr * 3, lr * 3 : -lr * 3]
    return block


def _select_mid_gs(gs: np.ndarray) -> np.ndarray:
    N = len(gs) // 3
    if N % 2 != 1:
        raise ValueError("Number of bp needs to be even")
    Nh = N // 2
    return gs[3 * Nh : 3 * Nh + 3]


def _harmonic_mean(mats: List[np.ndarray]) -> np.ndarray:
    if len(mats) == 0:
        print("Warning: _harmonic_mean encountered an empty list of matrices.")
        return None
    invsum = np.zeros(mats[0].shape)
    for mat in mats:
        invsum += np.linalg.inv(mat)
    return np.linalg.inv(invsum / len(mats))


def coup_range2oli_size(coup_range: int) -> int:
    return (1 + coup_range) * 2


def oli_size2coup_range(oli_size: int) -> int:
    return oli_size // 2 - 1


###################################################################################################################
###################################################################################################################
###################################################################################################################


def seq_couplings(
    seq: str,
    coup_range: int,
    left_terminus_seq: str = "",
    right_terminus_seq: str = "",
    disc_len: float = 0.34,
    generate_missing: bool = True,
) -> List[dict]:
    """ """
    print(
        "Warning: seq_couplings is depricated and will soon be removed. Use seq_couplings_open instead"
    )

    olidicts, idbseq, chars = seq_couplings_open(
        seq,
        coup_range,
        lower_terminus_seq=left_terminus_seq,
        upper_terminus_seq=right_terminus_seq,
        disc_len=disc_len,
        generate_missing=generate_missing,
    )
    return olidicts

    # oli_size = (1+coup_range)*2
    # N = len(seq)
    # full_seq = left_terminus_seq + seq + right_terminus_seq
    # gs,stiff = seq2rotbps(full_seq)

    # Nl = len(left_terminus_seq)
    # Nr = len(right_terminus_seq)
    # Nf = len(full_seq)
    # N  = len(seq)
    # id_mid = coup_range
    # Mentries = (2*coup_range+1)*3

    # olidicts = list()
    # contained_olis = list()
    # for s in range(Nl,Nl+N-1):
    #     nleft = coup_range
    #     if s < coup_range:
    #         nleft = s
    #     nright = coup_range
    #     if Nf-s-2<coup_range:
    #         nright = Nf-s-2
    #     # oligomer sequencce
    #     oliseq = 'x'*(coup_range-nleft) + full_seq[s-nleft:s+nright+2] + 'x'*(coup_range-nright)
    #     # select ground state
    #     T0 = gs[s*3:(s+1)*3]
    #     # stiffness matrix of segment
    #     M  = np.zeros((Mentries,Mentries))
    #     M[(id_mid-nleft)*3:(id_mid+1+nright)*3,(id_mid-nleft)*3:(id_mid+1+nright)*3] = stiff[(s-nleft)*3:(s+1+nright)*3,(s-nleft)*3:(s+1+nright)*3]

    #     olidict = {'oli':oliseq,'T0':T0*180./np.pi,'M':M}
    #     olidicts.append(olidict)
    #     contained_olis.append(oliseq)

    # if generate_missing:
    #     olis = dna_oligomers((coup_range+1)*2,omit_equiv=False)
    #     for oli in olis:
    #         oli = oli.upper()
    #         if oli not in contained_olis:
    #             olidict = {'oli':oli,'T0':np.zeros(3),'M':np.zeros((Mentries,Mentries))}
    #             olidicts.append(olidict)
    # return olidicts


###################################################################################################################
###################################################################################################################
###################################################################################################################


def seq_couplings_open(
    seq: str,
    coup_range: int,
    param_generator: Callable = seq2rotbps,
    genargs: dict = GENERATOR_DEFAULT_ARGS,
    ndims: int = 3,
    lower_terminus_seq: str = "",
    upper_terminus_seq: str = "",
    generate_missing: bool = True,
) -> Tuple[List[dict], str, str]:
    """Generates oligomer interactions for an open molecule

    Args:
        seq (str): sequence
        coup_range (int): coupling range ( oligomer size = (coup_range+1)*2 )
        param_generator (callable): Method for generating the sequence dependent parameters (defaults to seq2rotbps)
        param_generator_additional_args (dict): Additional parameters needed for param_generator. Assumes param_generator takes seq,*param_generator_additional_args as arguments. Defaults to [].
        ndims (int): number of degrees of freedom per base pair-step (defaults to 3 - rotational dofs)
        lower_terminus_seq (str): additional sequence used to append lower terminus to obtain non-openended paramters
        upper_terminus_seq (str): additional sequence used to append upper terminus to obtain non-openended paramters
        generate_missing (bool): generate all unique oligomers (defaults to True)

    Returns:
        List(dict),str,str: List of oligomers and idb sequence and used sequence characters. That sequence may differ from the specified sequence to assure unique assignment of all oligomers
    """
    # seq2rotbps

    # args

    oli_size = (1 + coup_range) * 2
    N = len(seq)

    # assign unique sequence
    if unique_olis_in_seq(seq, oli_size):
        idbseq = str(seq)
        chars = "".join(set(seq))
    else:
        idbseq, chars = unique_oli_seq(N, oli_size, closed=False)

    assignseq = "x" * coup_range + idbseq + "x" * coup_range

    # generate couplings
    full_seq = lower_terminus_seq + seq + upper_terminus_seq
    gs, stiff = param_generator(full_seq, **genargs)

    # restrict to relevant range
    lw = len(lower_terminus_seq) * ndims
    up = ((len(gs) // 3) - len(upper_terminus_seq)) * ndims
    gs = gs[lw:up]
    stiff = stiff[lw:up, lw:up]

    # assign oligomers
    olidicts = list()
    for i in range(N - 1):
        T0, M = extract_oligomer_params(
            gs, stiff, coup_range, i, ndims=ndims, allow_boundary_assignment=True
        )
        oliseq = assignseq[i : i + oli_size]
        olidict = {"oli": oliseq, "T0": T0 * 180.0 / np.pi, "M": M}
        olidicts.append(olidict)

    if generate_missing:
        olidicts = _gen_missing_params(olidicts, coup_range, chars, ndims=ndims)

    return olidicts, idbseq, chars


###################################################################################################################
###################################################################################################################
###################################################################################################################


def seq_couplings_closed(
    seq: str,
    coup_range: int,
    param_generator: Callable = seq2rotbps,
    genargs: dict = GENERATOR_DEFAULT_ARGS,
    ndims: int = 3,
    closure_copy_size: int = 25,
    generate_missing: bool = True,
) -> Tuple[List[dict], str, str]:
    """Generates oligomer interactions for a closed molecule

    Args:
        seq (str): sequence
        coup_range (int): coupling range ( oligomer size = (coup_range+1)*2 )
        param_generator (callable): Method for generating the sequence dependent parameters (defaults to seq2rotbps)
        param_generator_additional_args (dict): Additional parameters needed for param_generator. Assumes param_generator takes seq,*param_generator_additional_args as arguments. Defaults to [].
        ndims (int): number of degrees of freedom per base pair-step (defaults to 3 - rotational dofs)
        closure_copy_size (int): size of the copied segments at the terminus (defaults to 25)
        generate_missing (bool): generate all unique oligomers (defaults to True)

    Returns:
        List(dict),str,str: List of oligomers and idb sequence and used sequence characters. That sequence may differ from the specified sequence to assure unique assignment of all oligomers
    """

    # general variables
    nbp = len(seq)
    oli_size = (1 + coup_range) * 2

    if coup_range >= nbp:
        raise ValueError(
            f"The coupling range should not exceed the size of the oligomer (or for implementational reasons equal)"
        )
    if closure_copy_size < coup_range:
        raise ValueError(
            f"Closure size needs to be at least as large as the coupling range."
        )

    # assign closure_copy sequences
    # the right one is one longer than the left one to account for the coupling spanning over
    # boundary (connecting the last and the first bp)
    clseq = seq * int(np.ceil((closure_copy_size + 1) / nbp))
    left_closure_seq = clseq[-closure_copy_size:]
    right_closure_seq = clseq[: closure_copy_size + 1]

    # full sequence
    full_seq = left_closure_seq + seq + right_closure_seq

    # generate couplings
    gs, stiff = param_generator(full_seq, **genargs)

    # seqlist = [full_seq[i:i+2] for i in range(len(full_seq)-1)]
    # print(len(gs)//ndims)
    # print(seqlist)
    # print(full_seq)

    # crop couplings
    lo = (closure_copy_size - coup_range) * ndims
    hi = len(gs) + (-closure_copy_size + coup_range) * ndims
    gs = gs[lo:hi]
    stiff = stiff[lo:hi, lo:hi]
    assert (
        len(gs) // ndims - 2 * coup_range == nbp
    ), "Invalid dimension of cropped couplings"

    # print(nbp)
    # print(seqlist[lo//ndims:hi//ndims])
    # print(len(gs)//ndims)

    # assign unique sequence
    assignseq = seq[nbp - coup_range :] + seq + seq[: coup_range + 1]
    if unique_olis_in_seq(assignseq, oli_size):
        idbseq = str(seq)
        chars = "".join(set(idbseq))
    else:
        idbseq, chars = unique_oli_seq(nbp, oli_size, closed=True)
        assignseq = idbseq[nbp - coup_range :] + idbseq + idbseq[: coup_range + 1]

    # print(oli_size)
    # print(seq)
    # print(f'seq:       {" "*coup_range+seq}')
    # print(f'idbseq:    {" "*coup_range+idbseq}')
    # print(f'assignseq: {assignseq}')
    

    # assign oligomers
    olidicts = list()
    for i in range(nbp):
        T0, M = extract_oligomer_params(
            gs,
            stiff,
            coup_range,
            i + coup_range,
            ndims=ndims,
            allow_boundary_assignment=False,
        )
        oliseq = assignseq[i : i + oli_size]
        olidict = {"oli": oliseq, "T0": T0 * 180.0 / np.pi, "M": M}
        olidicts.append(olidict)

        # print(i)
        # print(oliseq)

    ###########################
    # match coupling
    # lower terminus
    for l in range(coup_range):
        olidicts[l]["M"][
            : coup_range * ndims, coup_range * ndims : (coup_range + 1) * ndims
        ] = olidicts[-1]["M"][
            (l + 1) * ndims : (l + 1 + coup_range) * ndims,
            (coup_range + 1 + l) * ndims : (coup_range + 2 + l) * ndims,
        ]

    # upper terminus
    for l in range(coup_range):
        olidicts[len(olidicts) - 1 - l]["M"][
            coup_range * ndims : (coup_range + 1) * ndims, (coup_range + 1) * ndims :
        ] = olidicts[0]["M"][
            (coup_range - 1 - l) * ndims : (coup_range - l) * ndims,
            (coup_range - l) * ndims : (2 * coup_range - l) * ndims,
        ]

    # coups0  = get_oli_coups(olidicts[0],coup_range)
    # coups1  = get_oli_coups(olidicts[1],coup_range)
    # coupsm1 = get_oli_coups(olidicts[-1],coup_range)
    # coupsm2 = get_oli_coups(olidicts[-2],coup_range)

    # np.set_printoptions(precision=4, linewidth=200, suppress=True)

    # print('########')
    # for c in coupsm2:
    #     print(c[1:])
    # print('########')
    # for c in coupsm1:
    #     print(c[1:])
    # print('########')
    # for c in coups0:
    #     print(c[1:])
    # print('########')
    # for c in coups1:
    #     print(c[1:])

    # sys.exit()

    if generate_missing:
        olidicts = _gen_missing_params(olidicts, coup_range, chars, ndims=ndims)

    return olidicts, idbseq, chars


def seq_couplings_closed__(
    seq: str,
    coup_range: int,
    param_generator: Callable = seq2rotbps,
    genargs: dict = GENERATOR_DEFAULT_ARGS,
    ndims: int = 3,
    closure_copy_size: int = 25,
    generate_missing: bool = True,
) -> Tuple[List[dict], str, str]:
    """Generates oligomer interactions for a closed molecule

    Args:
        seq (str): sequence
        coup_range (int): coupling range ( oligomer size = (coup_range+1)*2 )
        param_generator (callable): Method for generating the sequence dependent parameters (defaults to seq2rotbps)
        param_generator_additional_args (dict): Additional parameters needed for param_generator. Assumes param_generator takes seq,*param_generator_additional_args as arguments. Defaults to [].
        ndims (int): number of degrees of freedom per base pair-step (defaults to 3 - rotational dofs)
        closure_copy_size (int): size of the copied segments at the terminus (defaults to 25)
        generate_missing (bool): generate all unique oligomers (defaults to True)

    Returns:
        List(dict),str,str: List of oligomers and idb sequence and used sequence characters. That sequence may differ from the specified sequence to assure unique assignment of all oligomers
    """
    if closure_copy_size < coup_range:
        raise ValueError(
            f"Closure size needs to be at least as large as the coupling range."
        )

    oli_size = (1 + coup_range) * 2
    N = len(seq)

    # assign unique sequence
    extended_seq = seq + seq[: coup_range + 1]
    if unique_olis_in_seq(extended_seq, oli_size):
        idbseq = str(seq)
        assignseq = seq[len(seq) - coup_range :] + str(seq + seq[: oli_size // 2])
        chars = "".join(set(seq))
    else:
        # assignseq, chars = unique_oli_seq(N-1,oli_size,closed=True)
        # idbseq = assignseq[:-(coup_range+1)]
        # assignseq = idbseq[len(idbseq)-coup_range:] + assignseq

        assignseq, chars = unique_oli_seq(N, oli_size, closed=True)
        idbseq = assignseq[: -(coup_range + 1)]
        assignseq = idbseq[len(idbseq) - coup_range :] + assignseq

    # generate couplings
    full_seq = seq[-closure_copy_size:] + seq + seq[:closure_copy_size]
    gs, stiff = param_generator(full_seq, **genargs)

    # assign oligomers
    olidicts = list()
    for i in range(N):
        T0, M = extract_oligomer_params(
            gs,
            stiff,
            coup_range,
            i + closure_copy_size,
            ndims=3,
            allow_boundary_assignment=False,
        )
        oliseq = assignseq[i : i + oli_size]
        olidict = {"oli": oliseq, "T0": T0 * 180.0 / np.pi, "M": M}
        olidicts.append(olidict)

    ###########################
    # match coupling
    # lower terminus
    for l in range(coup_range):
        olidicts[l]["M"][
            : coup_range * ndims, coup_range * ndims : (coup_range + 1) * ndims
        ] = olidicts[-1]["M"][
            (l + 1) * ndims : (l + 1 + coup_range) * ndims,
            (coup_range + 1 + l) * ndims : (coup_range + 2 + l) * ndims,
        ]

    # upper terminus
    for l in range(coup_range):
        olidicts[len(olidicts) - 1 - l]["M"][
            coup_range * ndims : (coup_range + 1) * ndims, (coup_range + 1) * ndims :
        ] = olidicts[0]["M"][
            (coup_range - 1 - l) * ndims : (coup_range - l) * ndims,
            (coup_range - l) * ndims : (2 * coup_range - l) * ndims,
        ]

    # coups0  = get_oli_coups(olidicts[0],coup_range)
    # coups1  = get_oli_coups(olidicts[1],coup_range)
    # coupsm1 = get_oli_coups(olidicts[-1],coup_range)
    # coupsm2 = get_oli_coups(olidicts[-2],coup_range)

    # print('########')
    # for c in coupsm2:
    #     print(c[1:])
    # print('########')
    # for c in coupsm1:
    #     print(c[1:])
    # print('########')
    # for c in coups0:
    #     print(c[1:])
    # print('########')
    # for c in coups1:
    #     print(c[1:])

    if generate_missing:
        olidicts = _gen_missing_params(olidicts, coup_range, chars, ndims=ndims)

    return olidicts, idbseq, chars


###################################################################################################################
###################################################################################################################
###################################################################################################################


def coup2olidicts_linear(
    gs: np.ndarray,
    stiff: np.ndarray,
    coup_range: int,
    seq: str,
    chars: str,
    ndims: int = 3,
    generate_missing: bool = True,
) -> dict:
    """Generates the olidicts for a particular sequence. Boundary steps are explicitely assigned
    as such.

    """
    assert (
        len(gs) // ndims == len(seq) - 1
    ), "The length of the specified sequence does not match the dimension of the provided couplings."

    print(len(gs) // ndims)
    print(len(seq))

    N = len(gs) // ndims
    oli_size = oli_size2coup_range(coup_range)

    print(seq)

    olidicts = list()
    for i in range(N):
        T0, M = extract_oligomer_params(
            gs, stiff, coup_range, i, ndims=ndims, allow_boundary_assignment=True
        )
        slo = i - coup_range
        shi = i + 2 + coup_range

        addlo = ""
        addhi = ""
        if slo < 0:
            addlo += "x" * (-slo)
            slo = 0
        if shi > N + 1:
            addhi += "x" * (shi - N - 1)
            shi = N + 1

        oliseq = addlo + seq[slo:shi] + addhi
        print(f"{addlo} - {seq[slo:shi]} - {addhi}")

        olidict = {"oli": oliseq, "T0": T0 * 180.0 / np.pi, "M": M}
        olidicts.append(olidict)

    if generate_missing:
        olidicts = _gen_missing_params(olidicts, coup_range, chars, ndims=ndims)
    return olidicts


###################################################################################################################
###################################################################################################################
###################################################################################################################


def extract_oligomer_params(
    groundstate: np.ndarray,
    stiff: np.ndarray,
    coup_range: int,
    id: int,
    ndims: int = 3,
    allow_boundary_assignment=False,
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(groundstate)
    if len(groundstate) != len(groundstate):
        raise ValueError(f"Size of groundstate and stiffness matrix are inconsistent")
    if len(groundstate) % ndims != 0:
        raise ValueError(
            f"Dimension of groundstate is inconsistent with the specified number of degrees of freedom pair base pair-step"
        )
    Nbps = N // ndims
    ncoup = coup_range * 2 + 1
    if Nbps < ncoup:
        raise ValueError(f"Not enough base pair-steps for single coupling range")

    sgs = groundstate[id * ndims : (id + 1) * ndims]

    # close to lower terminus
    if id < coup_range:
        if not allow_boundary_assignment:
            raise ValueError(
                f"Assignment id out of range: Cannot assign boundary oligomers (lower terminus)"
            )
        M = np.zeros((ncoup * ndims,) * 2)
        nmissing = coup_range - id
        nassign = ncoup - nmissing
        M[nmissing * ndims :, nmissing * ndims :] = stiff[
            : nassign * ndims, : nassign * ndims
        ]
        return sgs, M

    # close to upper terminus
    if id >= Nbps - coup_range:
        if not allow_boundary_assignment:
            raise ValueError(
                f"Assignment id out of range: Cannot assign boundary oligomers (upper terminus)"
            )
        M = np.zeros((ncoup * ndims,) * 2)
        nmissing = id - (Nbps - coup_range - 1)
        nassign = ncoup - nmissing
        M[: nassign * ndims, : nassign * ndims] = stiff[
            -nassign * ndims :, -nassign * ndims :
        ]
        return sgs, M

    M = stiff[
        (id - coup_range) * ndims : (id + coup_range + 1) * ndims,
        (id - coup_range) * ndims : (id + coup_range + 1) * ndims,
    ]
    return sgs, M


def _gen_missing_params(
    olidicts: List[dict], coup_range: int, chars: str, ndims: int = 3
):
    oli_size = (coup_range + 1) * 2
    Mentries = (2 * coup_range + 1) * ndims
    contained_olis = [olidict["oli"] for olidict in olidicts]
    olis = all_oligomers(oli_size, chars)
    for oli in olis:
        if oli not in contained_olis:
            olidict = {
                "oli": oli,
                "T0": np.zeros(ndims),
                "M": np.zeros((Mentries, Mentries)),
            }
            olidicts.append(olidict)
    return olidicts
