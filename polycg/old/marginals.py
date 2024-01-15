import numpy as np
from scipy.sparse import csr_matrix, spmatrix, coo_matrix
from scipy import sparse

from typing import List, Tuple, Callable, Any, Dict


def var_assign(seq: str, dof_names=["W", "x", "C", "y"]) -> List[str]:
    """
    Generates the sequence of contained degrees of freedom for the specified sequence.
    The default names follow the convention introduced on the cgNA+ website
    """
    if len(dof_names) != 4:
        raise ValueError(
            f"Requires 4 names for the degrees of freedom. {len(dof_names)} given."
        )
    N = len(seq)
    if N == 0:
        return []
    vars = list()
    for i in range(1, N + 1):
        vars += [f"{dofn}{i}" for dofn in dof_names]
    return vars[1:-2]


def gen_select(seq: str, select: List[str], raise_invalid=True) -> List[str]:
    """
    Generated sequence of selected degrees of freedom with wildtype assignment via asterisk.
    'x*' will for example insert all not yet contained 'xi' degrees of freedom in ascending order
    in the location x* was inserted in the provided select List.
    """
    vars = var_assign(seq)
    full_select = list()
    for sel in select:
        if "*" in sel:
            sel_bare = sel.replace("*", "")
            # adds = [var for var in vars if sel_bare in var]
            adds = [var for var in vars if var.startswith(sel_bare)]
            if len(adds) == 0 and raise_invalid:
                raise ValueError(f"Invalid select argument {sel}")
            for add in adds:
                if add not in select:
                    full_select.append(add)
            continue
        if sel in vars:
            full_select.append(sel)
            continue
        if raise_invalid:
            raise ValueError(f"Invalid select argument {sel}")
    return full_select


def marginal(
    gs: np.ndarray, 
    stiff: np.ndarray, 
    seq: str, 
    select: List[str], 
    raise_invalid=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the marginal of ground state and stiffness for the selected degrees of freedom (in blocks of 6)
    """
    select = gen_select(seq, select, raise_invalid=raise_invalid)
    vars = var_assign(seq)
    ndims = len(stiff) // len(vars)

    #########################################
    # calculate stiffness matrix marginal ###
    # reorder matrix
    def permuation_map(vars, select):
        pm = list()
        for sel in select:
            pm.append(vars.index(sel))
        for var in vars:
            if var not in select:
                pm.append(vars.index(var))
        return pm

    pm = permuation_map(vars, select)
    # init matrix of basis change
    CB = np.zeros(stiff.shape)
    eye = np.eye(ndims)
    for i, j in enumerate(pm):
        CB[ndims * i : ndims * (i + 1), ndims * j : ndims * (j + 1)] = eye
    # reordering matrix
    Mro = np.matmul(CB, np.matmul(stiff, CB.T))
    # select partial matrices
    NA = ndims * len(select)
    NB = len(Mro) - NA
    A = Mro[:NA, :NA]
    D = Mro[NA:, NA:]
    B = Mro[:NA, NA:]
    C = Mro[NA:, :NA]
    # calculate Schur complement
    MA = A - np.dot(B, np.dot(np.linalg.inv(D), C))

    #########################################
    # extract partial ground state        ###
    partial_gs = np.concatenate(
        [gs[vars.index(sel) * 6 : (vars.index(sel) + 1) * 6] for sel in select]
        # [gs[vars.index(sel) * ndims : (vars.index(sel) + 1) * ndims] for sel in select]
    ).flatten()
    return partial_gs, MA


def marginal_gs(
    gs: np.ndarray, 
    seq: str, 
    select: List[str], 
    raise_invalid=True
) -> Tuple[np.ndarray, np.ndarray]:
    select = gen_select(seq, select, raise_invalid=raise_invalid)
    vars = var_assign(seq)
    ndims = len(gs) // len(vars)
    partial_gs = np.concatenate(
        # [gs[vars.index(sel) * 6 : (vars.index(sel) + 1) * 6] for sel in select]
        [gs[vars.index(sel) * ndims : (vars.index(sel) + 1) * ndims] for sel in select]
    ).flatten()
    return partial_gs
    

def marginal_sparse(
    gs: np.ndarray, 
    stiff: spmatrix, 
    seq: str, 
    select: List[str], 
    raise_invalid=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the marginal of ground state and stiffness for the selected degrees of freedom (in blocks of 6)
    """
    select = gen_select(seq, select, raise_invalid=raise_invalid)
    vars = var_assign(seq)
    ndims = len(gs) // len(vars)

    #########################################
    # calculate stiffness matrix marginal ###
    # reorder matrix
    def permuation_map(vars, select):
        pm = list()
        for sel in select:
            pm.append(vars.index(sel))
        for var in vars:
            if var not in select:
                pm.append(vars.index(var))
        return pm

    pm = permuation_map(vars, select)

    rows = list()
    cols = list()
    vals = list()
    for i, j in enumerate(pm):
        for d in range(ndims):
            rows.append(ndims * i + d)
            cols.append(ndims * j + d)
            vals.append(1)

    CB = coo_matrix((vals, (rows, cols)), dtype=float, shape=stiff.shape)
    Mro_coo = CB.dot(stiff.dot(CB.transpose()))
    Mro = Mro_coo.tocsc()

    # select partial matrices
    nr, nc = Mro.shape
    NA = ndims * len(select)
    NB = nr - NA
    A = Mro[:NA, :NA]
    D = Mro[NA:, NA:]
    B = Mro[:NA, NA:]
    C = Mro[NA:, :NA]

    # calculate Schur complement
    MA = A - B.dot(sparse.linalg.spsolve(D, C))
    MA = MA.toarray()

    #########################################
    # extract partial ground state ##########
    partial_gs = np.concatenate(
        [gs[vars.index(sel) * 6 : (vars.index(sel) + 1) * 6] for sel in select]
    ).flatten()
    return partial_gs, MA


def rot_marginal(gs: np.ndarray, stiff: np.ndarray):
    """
    Extract marginal for rotational degrees of freedom
    """
    N = len(gs) // 6
    #########################################
    # calculate stiffness matrix marginal ###
    # reorder matrix
    eye3 = np.eye(3)
    # init matrix of basis change
    CB = np.zeros(stiff.shape)
    for i in range(N):
        rfx = i
        rfy = 2 * i
        tfx = N + i
        tfy = 2 * i + 1
        CB[3 * rfx : 3 * (rfx + 1), 3 * rfy : 3 * (rfy + 1)] = eye3
        CB[3 * tfx : 3 * (tfx + 1), 3 * tfy : 3 * (tfy + 1)] = eye3
    # reordering matrix
    Mro = np.matmul(CB, np.matmul(stiff, CB.T))
    # select partial matrices
    Nh = len(Mro) // 2
    A = Mro[:Nh, :Nh]
    D = Mro[Nh:, Nh:]
    B = Mro[:Nh, Nh:]
    C = Mro[Nh:, :Nh]
    # calculate Schur complement
    MA = A - np.dot(B, np.dot(np.linalg.inv(D), C))
    #########################################
    # extract partial ground state ##########
    partial_gs = np.concatenate([gs[6 * i : 6 * i + 3] for i in range(N)]).flatten()
    return partial_gs, MA


if __name__ == "__main__":
    
    seq = 'ACGATCGA'
    vars = var_assign(seq, dof_names=["W", "x", "C", "y"])
    select = ['y*']
    select = gen_select(seq, select, raise_invalid=True)
    
    print(vars)
    print(select)
    
    pm = list()
    for sel in select:
        pm.append(vars.index(sel))
        print(vars.index(sel))
    