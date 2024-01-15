from __future__ import annotations
import numpy as np
from typing import Tuple

from .marginals import marginal, rot_marginal, marginal_sparse
from .conversions import stiff2rad, gs2deg, gs2rad
from .cgNA_plus.modules.cgDNAUtils import constructSeqParms

from .bmat import BlockOverlapMatrix

from .conversions import cayleys2eulers_lintrans, cayleys2eulers

# from .conversions import eulers2cayleys_lintrans, eulers2cayleys
from .conversions import statevec2vecs, vecs2statevec
from .conversions import splittransform_algebra2group, splittransform_group2algebra

from warnings import warn


_DEFAULT_DATASET = "cgDNA+_Curves_BSTJ_10mus_FS"


def seq2rotbps(
    seq: str,
    in_nm=True,
    disc_len=0.34,
    rotation_map="cayley",
    split_fluctuations="vector",
    gs_units="rad",
    ps_set="default",
    partial_enabled=True,
) -> Tuple[np.ndarray, np.ndarray]:

    raise DeprecationWarning('The method seq2rotbps is deprecated. Please use cgna_rotbps instead.')    

    if partial_enabled and len(seq) > 250:
        return seq2rotbps_partials(
            seq,
            partial_size=160,
            avg_size=14,
            overlap_size=14,
            in_nm=in_nm,
            disc_len=disc_len,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            gs_units=gs_units,
            ps_set=ps_set,
            pout=False,
        )

    return _seq2rotbps(
        seq,
        in_nm=in_nm,
        disc_len=disc_len,
        rotation_map=rotation_map,
        split_fluctuations=split_fluctuations,
        gs_units=gs_units,
        ps_set=ps_set,
    )


def _seq2rotbps(
    seq: str,
    in_nm=True,
    disc_len=0.34,
    rotation_map="cayley",
    split_fluctuations="vector",
    gs_units="rad",
    ps_set="default",
) -> Tuple[np.ndarray, np.ndarray]:
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

    if ps_set == "default":
        ps_set = _DEFAULT_DATASET

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
            stiff_euler = rotbps_stiff_algebra2group(gs_euler, stiff_euler)

        if gs_units.lower() in ["deg", "degree", "degrees"]:
            return gs_euler * 180 / np.pi, stiff_euler
        elif gs_units.lower() in ["rad", "radians", "rads"]:
            return gs_euler, stiff_euler
        else:
            ValueError(f'Unknown gs_units "{gs_units}"')
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')


##########################################################################################################
############### Sample Paritials and combine #############################################################
##########################################################################################################

        
def seq2rotbps_partials(
    seq: str,
    partial_size: int,
    avg_size: int,
    overlap_size: int,
    in_nm=True,
    disc_len=0.34,
    rotation_map="cayley",
    split_fluctuations="vector",
    gs_units="rad",
    ps_set="default",
    pout=False,
) -> Tuple[np.ndarray, np.ndarray]:
    
    warn('The method seq2rotbps_partials is soon to be deprecated.', DeprecationWarning, stacklevel=2)

    if len(seq) < partial_size * 1.5:
        return seq2rotbps(
            seq,
            in_nm=in_nm,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            gs_units=gs_units,
        )

    N = len(seq)
    tgs = np.zeros(3 * (N - 1))
    cnts = np.zeros(3 * (len(seq) - 1))
    tstiff = np.zeros((3 * (N - 1),) * 2)
    cntsst = np.zeros((3 * (N - 1),) * 2)

    Nsegs = int(np.ceil((N - 1) / partial_size))

    if pout:
        print(f"Nsegs = {Nsegs}")
        print(seq)

    for i in range(Nsegs):
        l = i * partial_size
        u = (i + 1) * partial_size

        sl = l - avg_size - overlap_size
        su = u + avg_size + overlap_size

        pl = sl
        pu = su - 1

        cl = overlap_size
        cu = overlap_size
        if sl < 0:
            sl = 0
            pl = 0
            cl = l - avg_size
            if cl < 0:
                cl = 0
        if su > N - 1:
            su = N - 1
            pu = N - 2
            cu = pu - u - avg_size + 1
            if cu < 0:
                cu = 0

        pseq = seq[sl : su + 1]

        if pout:
            print(f"l,u:   {l},{u}")
            print(f"sl,su: {sl},{su}")

            print(f"pl,pu: {pl},{pu}")
            print(f"cl,cu: {cl},{cu}")
            print(sl * "_" + pseq + (N - su - 1) * "_")

        gs, stiff = seq2rotbps(
            pseq,
            in_nm=in_nm,
            rotation_map=rotation_map,
            split_fluctuations=split_fluctuations,
            gs_units=gs_units,
        )

        x, y = (pl + cl) * 3, (pu - cu + 1) * 3
        xa, ya = 3 * cl, len(gs) - 3 * cu

        tgs[x:y] += gs[xa:ya]
        cnts[x:y] += 1
        tstiff[x:y, x:y] += stiff[xa:ya, xa:ya]
        cntsst[x:y, x:y] += 1

    cntsst[cntsst == 0] = 1

    tgs /= cnts
    tstiff /= cntsst

    return tgs, tstiff




##########################################################################################################
############### Conversions ##############################################################################
##########################################################################################################


def rotbps_cayley2euler(gs: np.ndarray, stiff: np.ndarray) -> np.ndarray:
    """Converts groundstate and stiffness matrix from Cayley map representation to Euler map representation. Transformation of stiffness matrix assumes the magnitude of the rotation vector to be dominated by the groundstate.

    Args:
        gs (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix expressed in arbitrary units

    Returns:
        Tuple[np.ndarray,np.ndarray]: Groundstate and stiffness matrix.
    """
    gs_euler = vecs2statevec(cayleys2eulers(statevec2vecs(gs)))
    Tc2e = cayleys2eulers_lintrans(gs)
    Tc2e_inv = np.linalg.inv(Tc2e)
    # stiff_euler = np.matmul(Tc2e_inv.T,np.matmul(stiff,Tc2e_inv))
    stiff_euler = Tc2e_inv.T @ stiff @ Tc2e_inv
    return gs_euler, stiff_euler


def rotbps_stiff_algebra2group(
    gs_euler: np.ndarray, stiff_euler: np.ndarray
) -> np.ndarray:
    """Converts stiffness matrix for fluctuations split in so3 to fluctuations split in SO3.

    Args:
        gs (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix for fluctuations split in so3 using the Euler map definition for vector components. Expressed in arbitrary units but e

    Returns:
        Tuple[np.ndarray,np.ndarray]: Stiffness matrix for fluctuations split in SO3.
    """
    Ta2g_full = splittransform_algebra2group(gs_euler)
    Ta2g_full_inv = np.linalg.inv(Ta2g_full)
    # stiff_euler_group = np.matmul(Ta2g_full_inv.T,np.matmul( stiff_euler, Ta2g_full_inv))
    stiff_euler_group = Ta2g_full_inv.T @ stiff_euler @ Ta2g_full_inv
    return stiff_euler_group


def rotbps_stiff_group2algebra(
    gs_euler: np.ndarray, stiff_euler: np.ndarray
) -> np.ndarray:
    """Converts stiffness matrix for fluctuations split in SO3 to fluctuations split in so3.

    Args:
        gs (np.ndarray): groundstate expressed in radians
        stiff (np.ndarray): stiffness matrix for fluctuations split in SO3 using the Euler map definition for vector components. Expressed in arbitrary units but e

    Returns:
        Tuple[np.ndarray,np.ndarray]: Stiffness matrix for fluctuations split in so3.
    """
    Ta2g_full = splittransform_group2algebra(gs_euler)
    Ta2g_full_inv = np.linalg.inv(Ta2g_full)
    # stiff_euler_algebra = np.matmul(Ta2g_full_inv.T,np.matmul( stiff_euler, Ta2g_full_inv))
    stiff_euler_algebra = Ta2g_full_inv.T @ stiff_euler @ Ta2g_full_inv
    return stiff_euler_algebra


def seq2bps(
    seq: str, in_nm=True, ps_set=_DEFAULT_DATASET, disc_len=0.34
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the ground state and stiffness matrix of the base pair step degrees of freedom
    for the provided sequence.

    ToDo: Implement conversion of only rotational degrees of freedom to degrees (groundstate) and radians (stiffness matrix)
          Requires the selection of the appropriate dofs
    """

    raise Exception("Implementation not finished ")
    gs, stiff = constructSeqParms(seq, ps_set)
    stiff = stiff.toarray()

    pgs, mstiff = marginal(gs, stiff, seq, ["y*"], raise_invalid=True)
    rbps_gs, rbps_stiff = rot_marginal(pgs, mstiff)
    rbps_stiff_rad = stiff2rad(rbps_stiff, only_rot=True) * disc_len
    rbps_gs_rad = gs2deg(rbps_gs, only_rot=True)
    return rbps_gs_rad, rbps_stiff_rad
