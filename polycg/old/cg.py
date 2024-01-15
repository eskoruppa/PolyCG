import sys
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .conversions import (
    statevec2vecs,
    eulers2rotmats,
    rotmats2triads,
    rotmats2eulers,
    triads2rotmats,
    vecs2statevec,
    vecs2rotmats,
    rotmats2vecs,
    fluctrotmats2rotmats,
    rotmats2fluctrotmats,
)
from .mc_sampling import sample_free
from .SO3 import so3


def sample_cgrot_euler_algebra(
    covmat: np.ndarray,
    groundstate: np.ndarray,
    num_confs: int,
    composite_size: int,
    first_entry: int = 0,
    omit_mismatching_tail: bool = True,
) -> np.ndarray:
    """Generates an ensemble of rotational variables (euler vector components) for a given level of coarse graining.

    Args:
        covmat (np.ndarray): Covariance matrix
        groundstate (np.ndarray): Groundstate configuration, that is added to the Gaussian sampled fluctuations
        num_confs (int): Number of configurations
        composite_size (int): Level of coarse-graining. Every composite_size triad is retained and on the basis of these the new set of varibles is calculated.
        first_entry (int): First considered entry, ignoring all preceeding entries. Defaults to 0.
        omit_mismatching_tail (bool, optional): If set to True, the sequence is shortened to match the provided level of coarse-graining. Defaults to False.

    Raises:
        ValueError: If omit_mismatching_tail is False and the length of the sequence is not a multiple of composite_size

    Returns:
        np.ndarray: Ensemble of Euler vectors corresponding to the coarse-grained chain
    """

    if not omit_mismatching_tail and (len(covmat) // 3) % composite_size != 0:
        raise ValueError(
            f"Number of bp steps needs to be a multiple of the coarse-graining step (composite_size). To force coarse graining by discarding the mismatching tail set omit_mismatching_tail to True."
        )

    # sample according to covariance matrix
    statevecs = sample_free(covmat, num_confs, groundstate=groundstate)

    # calculate triads
    eulers = statevec2vecs(statevecs)
    rotmats = eulers2rotmats(eulers)

    # rotmats to triads
    triads = rotmats2triads(rotmats)
    # subsample triads
    cgtriads = triads[:, first_entry::composite_size]
    # calculate state vectors of coarse-grained configuration
    cgrotmats = triads2rotmats(cgtriads)

    cgeulers = rotmats2eulers(cgrotmats)
    cgstatevecs = vecs2statevec(cgeulers)
    return cgstatevecs


def sample_cgrot_euler_group(
    covmat: np.ndarray,
    groundstate: np.ndarray,
    num_confs: int,
    composite_size: int,
    first_entry: int = 0,
    static_left: bool = True,
    omit_mismatching_tail: bool = True,
) -> np.ndarray:
    ndims = 3
    if not omit_mismatching_tail and (len(covmat) // ndims) % composite_size != 0:
        raise ValueError(
            f"Number of bp steps needs to be a multiple of the coarse-graining step (composite_size). To force coarse graining by discarding the mismatching tail set omit_mismatching_tail to True."
        )

    # sample according to covariance matrix
    statevecs = sample_free(covmat, num_confs, groundstate=None)

    # calculate triads
    eulers_fluct = statevec2vecs(statevecs)
    eulers_gs = statevec2vecs(groundstate)

    drotmats = eulers2rotmats(eulers_fluct)
    rotmats = fluctrotmats2rotmats(eulers_gs, drotmats, static_left=static_left)

    # rotmats to triads
    triads = rotmats2triads(rotmats)

    # subsample triads
    cgtriads = triads[:, first_entry::composite_size]

    # calculate state vectors of coarse-grained configuration
    cgrotmats = triads2rotmats(cgtriads)

    cggs = composite_groundstate(
        groundstate, composite_size, first_entry=first_entry, rotation_map="euler"
    )
    eulers_cggs = statevec2vecs(cggs)
    drotmats = rotmats2fluctrotmats(eulers_cggs, cgrotmats)

    cgeulers_fluct = rotmats2eulers(drotmats)
    cgstatevecs = vecs2statevec(cgeulers_fluct)

    return cgstatevecs


##########################################################################################################
############### Composite Groundstate ####################################################################
##########################################################################################################


def cg_groundstate(
    groundstate: np.ndarray,
    composite_size: int,
    first_entry: int = 0,
    rotation_map: str = "euler",
) -> np.ndarray:
    return composite_groundstate(
        groundstate, composite_size, first_entry=first_entry, rotation_map=rotation_map
    )


def composite_groundstate(
    groundstate: np.ndarray, composite_size: int, first_entry=0, rotation_map="euler"
) -> np.ndarray:
    """Composite groundstate according to level of coarse-graining

    Args:
        groundstate (np.ndarray): Rotational groundstate configuration
        composite_size (int): Level of coarse-graining. Every composite_size triad is retained and on the basis of these the new set of varibles is calculated.
        first_entry (int): First considered entry, ignoring all preceeding entries. Defaults to 0.
        rotation_map (str): selected map between rotation rotation coordinates and rotation matrix.
                        Options:    - cayley: default cnDNA map (Cayley map)
                                    - euler:  Axis angle representation.

    Returns:
        np.ndarray: composite groundstate vector
    """

    _check_cg_dimensions(
        composite_size, groundstate, None, first_entry=first_entry, ndims=3
    )

    # calculate triads
    vecs = statevec2vecs(groundstate)
    rotmats = vecs2rotmats(vecs, rotation_map=rotation_map)
    triads = rotmats2triads(rotmats)

    # subsample triads
    cgtriads = triads[first_entry::composite_size]

    # calculate state vectors of coarse-grained configuration
    cgrotmats = triads2rotmats(cgtriads)
    cgvecs = rotmats2vecs(cgrotmats, rotation_map=rotation_map)
    cggs = vecs2statevec(cgvecs)
    return cggs


##########################################################################################################
############### Coarse-grain by patching together partials ###############################################
##########################################################################################################


def cg_bpsrot_partial(
    gs: np.ndarray,
    stiff: np.ndarray,
    composite_size: int,
    block_ncomp: int,
    overlap_ncomp: int,
    cg_stiff_method: Callable,
    cg_stiff_method_args: dict = {},
    rotation_map: str = "euler",
    split_fluctuations: str = "matrix",
    termini_ncomp: int = 2,
    first_entry: int = 0,
    static_left: bool = True,
    rescale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # crop groundstate and stiffness to multiple of composite size

    # a = )
    raise DeprecationWarning("This function is depricated. Please use cg_cgna_rotbps with the keyword argument allow_partial=True for coarse graining via partial generation.")

    gs, stiff = _cg_composite_crop(
        composite_size, gs=gs, stiff=stiff, first_entry=first_entry
    )

    # calculate composite groudnstate
    cggs = composite_groundstate(
        gs, composite_size, first_entry=0, rotation_map=rotation_map
    )

    # check if overlap exceeds block size
    if overlap_ncomp >= block_ncomp:
        overlap_ncomp = block_ncomp - 1
        print(
            f"Warning: number of overlap composites needs to be smaller than block size. Set to {overlap_ncomp}."
        )

    # if system size is smaller than one block directly calculate coarse grained stiffness
    ndims = 3
    num_comp = len(gs) // ndims // composite_size
    if num_comp <= block_ncomp:
        cgstiff = cg_stiff_method(gs, stiff, composite_size, **cg_stiff_method_args)
        return cggs, cgstiff

    # calculate number of blocks
    blockstep = block_ncomp - overlap_ncomp
    num_blocks = 1 + int(np.ceil((num_comp - block_ncomp) / (blockstep)))

    # initialize stiff
    cgstiff = np.zeros((num_comp * ndims,) * 2)
    #########################
    # upper boundary of block
    hi_mainblock = block_ncomp
    if hi_mainblock > num_comp:
        hi_mainblock = num_comp
    if hi_mainblock + termini_ncomp <= num_comp:
        num_hi_terminus_ncomp = termini_ncomp
    else:
        num_hi_terminus_ncomp = num_comp - hi_mainblock
    hi_fullblock = hi_mainblock + num_hi_terminus_ncomp

    hi_bps = hi_fullblock * composite_size

    # select block
    partial_gs = gs[: hi_bps * ndims]
    partial_stiff = stiff[: hi_bps * ndims, : hi_bps * ndims]
    # coarse-grain partial
    fullblock_cgstiff = cg_stiff_method(
        partial_gs, partial_stiff, composite_size, **cg_stiff_method_args
    )

    # crop block
    hi = len(fullblock_cgstiff) - num_hi_terminus_ncomp * ndims
    mainblock_cgstiff = fullblock_cgstiff[:hi, :hi]

    # add to cgmatrix
    hi = hi_mainblock * ndims
    cgstiff[:hi, :hi] += mainblock_cgstiff

    for i in range(1, num_blocks):
        # lower boundary of block
        lo_mainblock = i * blockstep
        if lo_mainblock >= termini_ncomp:
            num_lo_terminus_ncomp = termini_ncomp
        else:
            num_lo_terminus_ncomp = lo_mainblock
        lo_fullblock = lo_mainblock - num_lo_terminus_ncomp

        # upper boundary of block
        hi_mainblock = lo_mainblock + block_ncomp
        if hi_mainblock > num_comp:
            hi_mainblock = num_comp
        if hi_mainblock + termini_ncomp <= num_comp:
            num_hi_terminus_ncomp = termini_ncomp
        else:
            num_hi_terminus_ncomp = num_comp - hi_mainblock
        hi_fullblock = hi_mainblock + num_hi_terminus_ncomp

        # bounds of block in terms of non-reduced matrix
        lo_bps = lo_fullblock * composite_size
        hi_bps = hi_fullblock * composite_size
        # select block
        partial_gs = gs[lo_bps * ndims : hi_bps * ndims]
        partial_stiff = stiff[
            lo_bps * ndims : hi_bps * ndims, lo_bps * ndims : hi_bps * ndims
        ]
        # coarse-grain block
        fullblock_cgstiff = cg_stiff_method(
            partial_gs, partial_stiff, composite_size, **cg_stiff_method_args
        )
        # crop block
        lo = num_lo_terminus_ncomp * ndims
        hi = len(fullblock_cgstiff) - num_hi_terminus_ncomp * ndims
        mainblock_cgstiff = fullblock_cgstiff[lo:hi, lo:hi]
        # add to cgmatrix
        lo = lo_mainblock * ndims
        hi = hi_mainblock * ndims
        cgstiff[lo:hi, lo:hi] += mainblock_cgstiff
        # divide overlap region by two
        size_mainblock = hi_mainblock - lo_mainblock
        lo_div = lo_mainblock * ndims
        if size_mainblock >= overlap_ncomp:
            hi_div = (lo_mainblock + overlap_ncomp) * ndims
        else:
            hi_div = size_mainblock * ndims
        cgstiff[lo_div:hi_div, lo_div:hi_div] /= 2

    # full_stiff = cg_stiff_method(gs,stiff,composite_size,**cg_stiff_method_args)
    # diff = np.abs(cgstiff - full_stiff)

    # print(f'diff = {np.linalg.norm(diff)}')
    # print(f'max = {np.max(diff)}')
    # print(f'cgstiff = {np.max(cgstiff)}')
    # print(f'full_stiff = {np.max(full_stiff)}')

    # for i in range(len(diff)//3):
    #     print('################')
    #     print(i)
    #     # print(full_stiff[i*3:(i+1)*3,i*3:(i+1)*3])
    #     # print(cgstiff[i*3:(i+1)*3,i*3:(i+1)*3])
    #     print(diff[i*3:(i+1)*3,i*3:(i+1)*3]*2)

    return cggs, cgstiff


def _cg_composite_crop(
    composite_size: int,
    gs: np.ndarray = None,
    stiff: np.ndarray = None,
    first_entry: int = 0,
    ndims: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """crops stiffness matrix and ground state to approritate size, such that the number of base pair steps
    is a multiple of composite size.
    """
    # limit matrices to relevant range
    tailcut = (len(gs) // ndims - first_entry) % composite_size
    if first_entry > 0:
        if stiff is not None:
            stiff = stiff[first_entry * ndims :, first_entry * ndims :]
        if gs is not None:
            gs = gs[first_entry * ndims :]
    if tailcut > 0:
        if stiff is not None:
            stiff = stiff[: -tailcut * ndims, : -tailcut * ndims]
        if gs is not None:
            gs = gs[: -tailcut * ndims]
    return gs, stiff


def _check_cg_dimensions(
    composite_size: int,
    groundstate: np.ndarray | None = None,
    stiff: np.ndarray | None = None,
    first_entry: int = 0,
    ndims: int = 3,
) -> None:
    if stiff is not None and (len(stiff) // ndims - first_entry) % composite_size != 0:
        raise ValueError(
            f"The number of bps ({(len(stiff) // ndims - first_entry)}) in the provided stiffness matrix minutes the first_entry index needs to be a multiple of the composite size ({composite_size})."
        )
    if (
        groundstate is not None
        and (len(groundstate) // ndims - first_entry) % composite_size != 0
    ):
        raise ValueError(
            f"The number of bps ({(len(groundstate) // ndims - first_entry)}) in the provided groundstate vector minutes the first_entry index needs to be a multiple of the composite size ({composite_size})."
        )


##########################################################################################################
############### Euler Composite for group decomposition ##################################################
##########################################################################################################


def cg_stiff_euler_groupsplit(
    groundstate: np.ndarray,
    stiff: np.ndarray,
    composite_size: int,
    first_entry: int = 0,
    static_left: bool = True,
    rescale: bool = True,
) -> np.ndarray:
    """

    Args:
        groundstate (np.ndarray): Rotational groundstate configuration (dim: 3N)
        stiff (np.ndarray): stiffness matrix (dim: 3Nx3N)
        composite_size (int): Number of consecutive segments to be combined into composite step
        first_entry (int): First considered entry, ignoring all preceeding entries. Defaults to 0.
        static_left (bool): Specifies whether static component of the rotation matrix is defined to be on the left or right.
                            Defaults to True (left definition).

    Returns:
        np.ndarray: Stiffness matrix for composite degrees of freedom
    """
    ndims = 3

    _check_cg_dimensions(
        composite_size, groundstate, stiff, first_entry=first_entry, ndims=ndims
    )

    T, composite_ids = cg_euler_groupsplit_transform(
        groundstate, composite_size, first_entry=first_entry, static_left=static_left
    )

    # limit matrices to relevant range
    tailcut = (len(groundstate) // ndims - first_entry) % composite_size
    if first_entry > 0:
        T = T[first_entry * ndims :, first_entry * ndims :]
        stiff = stiff[first_entry * ndims :, first_entry * ndims :]
        composite_ids = [i - first_entry for i in composite_ids]
    if tailcut > 0:
        T = T[: -tailcut * ndims, : -tailcut * ndims]
        stiff = stiff[: -tailcut * ndims, : -tailcut * ndims]

    Tinv = np.linalg.inv(T)
    M_comp = Tinv.T @ stiff @ Tinv
    Mcg = schur_complement(M_comp, composite_ids)
    if rescale:
        Mcg *= composite_size
    return Mcg


def cg_euler_groupsplit_transform(
    groundstate: np.ndarray,
    composite_size: int,
    first_entry: int = 0,
    static_left: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """

    Args:
        groundstate (np.ndarray): Rotational groundstate configuration (dim: 3N)
        composite_size (int): Number of consecutive segments to be combined into composite step
        first_entry (int): First considered entry, ignoring all preceeding entries. Defaults to 0.
        static_left (bool): Specifies whether static component of the rotation matrix is defined to be on the left or right.
                            Defaults to True (left definition).

    Returns:
        Tuple[np.ndarray, List[int]]: Returns transformation matrix (dim: 3Nx3N) and list of degrees of freedom part of the generated composites
    """
    ndims = 3

    T = np.zeros((len(groundstate),) * 2)
    composite_ids = list()

    N = len(groundstate) // ndims
    M = (N - first_entry) // composite_size

    for i in range(M):
        id_i = first_entry + i * composite_size
        id_j = first_entry + (i + 1) * composite_size - 1

        Tij = composite_rotationstep_euler_groupsplit(
            groundstate,
            id_i,
            id_j,
            replace_id=-1,
            static_left=static_left,
            block_eye=True,
        )
        comp = [id_j * ndims + i for i in range(ndims)]

        T += Tij
        composite_ids += comp

    return T, composite_ids


def composite_rotationstep_euler_groupsplit(
    groundstate: np.ndarray,
    id_i: int,
    id_j: int,
    replace_id: int = -1,
    static_left: bool = True,
    remaining_to_zero: bool = True,
    block_eye: bool = False,
) -> np.ndarray:
    """Rotation matrix that generates composite between specified steps. Based on the Euler rotation map definition with the static component split on the level of the Group (SO(3)), rather than the algebra (so(3)).
       R = SD (for static_left=True) and R=DS (otherwise), with S the rotation matrix containing the static components and D the rotation matrix containing the fluctuation components.

    Args:
        groundstate (np.ndarray): Rotational groundstate configuration (dim: 3N)
        id_i (int): first id in sequence (i and j reference the index of the triad junctions (all three contained dofs) rather than the individual dofs in the groundstate)
        id_j (int): last id_ in sequence. Is included following the mathematical definition rather than python convention.
        replace_id (int): entry that is replaced by the composite. Defaults to -1, which relaces the last entry in the sequence
        static_left (bool): Specifies whether static component of the rotation matrix is defined to be on the left or right.
                            Defaults to True (left definition).
        remaining_to_zero (bool): If True sets all entries expect for the replaced composite to zero. Otherwise these are trained (identity returned on diagonal) Defaults to True.
        block_eye (bool): identity map for non-replaced elements within composite block. Defaults to False

    Returns:
        np.ndarray: Returns matrix (dim: 3Nx3N) that replaces a single vector by the specified composite
    """

    # init transformation matrix
    ndims = 3

    if remaining_to_zero:
        T = np.zeros((len(groundstate),) * 2)
        if block_eye:
            T[
                id_i * ndims : (id_j + 1) * ndims, id_i * ndims : (id_j + 1) * ndims
            ] = np.eye((id_j - id_i + 1) * ndims)
    else:
        T = np.eye(len(groundstate))

    # define index to be replaced
    if replace_id == -1:
        replace_id = id_j

    # generate static component matrices
    gsvecs = statevec2vecs(groundstate[ndims * id_i : ndims * (id_j + 1)])
    Smats = vecs2rotmats(gsvecs, rotation_map="euler")

    # calculate composite transformation
    if static_left:
        Saccu = np.eye(ndims)
        T[
            replace_id * ndims : (replace_id + 1) * ndims,
            ndims * id_j : ndims * (id_j + 1),
        ] = Saccu
        for i in range(id_j - id_i):
            Saccu = np.dot(Smats[-1 - i].T, Saccu)
            T[
                replace_id * ndims : (replace_id + 1) * ndims,
                ndims * (id_j - 1 - i) : ndims * (id_j - i),
            ] = Saccu
    else:
        Saccu = np.eye(ndims)
        T[
            replace_id * ndims : (replace_id + 1) * ndims,
            ndims * id_i : ndims * (id_i + 1),
        ] = Saccu
        for i in range(id_j - id_i):
            Saccu = np.dot(Saccu, Smats[i])
            T[
                replace_id * ndims : (replace_id + 1) * ndims,
                ndims * (id_i + i + 1) : ndims * (id_i + i + 2),
            ] = Saccu
    return T


##########################################################################################################
############### Schur Complement and Permutation matrices ################################################
##########################################################################################################


def schur_complement(mat: np.ndarray, retained_ids: List[int]) -> np.ndarray:
    """Schur complement of matrix to retain the specified degrees of freedom

    Args:
        mat (np.ndarray): Given matrix
        retained_ids (List[int]): List of dof that are to be retained

    Returns:
        np.ndarray: reduced matrix
    """
    # calculate permutation matrix
    P = send_to_back_permutation(len(mat), retained_ids)
    # rearrange matrix
    pmat = so3.dots([P, mat, P.T])
    # select partial matrix
    ND = len(retained_ids)
    NA = len(mat) - ND
    A = pmat[
        :NA,
        :NA,
    ]
    B = pmat[:NA, NA:]
    D = pmat[NA:, NA:]
    # calculate schur complement
    schur = D - so3.dots([B.T, np.linalg.inv(A), B])
    return schur


def permutation_matrix(order: List[int]) -> np.ndarray:
    """
    Generates permutation matrix that rearranges elements according to the order specified in idlist
    """
    # check if all entries are contained
    missing = list()
    for i in range(np.max(order) + 1):
        if i not in order:
            missing.append(i)
    if len(missing) > 0:
        raise ValueError(
            f"Not all indices contained in order list: Missing indices: {missing}"
        )

    P = np.zeros((len(order),) * 2)
    for new, old in enumerate(order):
        P[new, old] = 1
    return P


def send_to_back_permutation(
    N: int, move_back_ids: List[int], ordered: bool = False
) -> np.ndarray:
    """Matrix that rearranges the terms to move the specified elements to the back

    Args:
        N (int): number of degrees of freedom
        move_back_ids (List[int]): List of dof that are to be moved to the back
        ordered (bool): Switch to preserve the natural order of the elements given in move_back_ids.
                        If set to True, the list is ordered. (Defaults to False)

    Returns:
        np.ndarray: transformation matrix ((dim: NxN))
    """
    T = np.zeros((N,) * 2)
    if ordered:
        move_back_ids = sorted(move_back_ids)
    leading = [i for i in range(N) if i not in move_back_ids]
    order = leading + move_back_ids
    return permutation_matrix(order)
