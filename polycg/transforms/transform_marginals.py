from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
from ..SO3 import so3
from warnings import warn


def matrix_marginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix,  # shape (N, N): square matrix
    select_indices: np.ndarray,  # shape (M,): boolean selection array
    block_dim: int = 1
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:  # shape (K, K): marginal matrix
    """
    Extract the marginal of a square matrix using Schur complement.
    
    Computes the marginal distribution by integrating out (marginalizing) unselected degrees
    of freedom using the Schur complement. The matrix is treated as blocks of size block_dim,
    and the selection is applied at the block level.
    
    Args:
        matrix: Square matrix to marginalize (dense or sparse).
        select_indices: Boolean array indicating which blocks to retain.
        block_dim: Size of each block. Matrix size must equal len(select_indices) * block_dim.
    
    Returns:
        Marginalized matrix containing only the selected degrees of freedom.
        Returns same type (dense/sparse) as input.
    
    Raises:
        ValueError: If matrix is not square or dimensions are incompatible.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'Provided matrix is not a square matrix. Has shape {matrix.shape}.')
    
    select_indices = proper_select_indices(select_indices)
    perm_map = permuation_map(select_indices)
    
    if not sparse.issparse(matrix):
        # numpy matrix
        if block_dim*len(select_indices) != len(matrix):
            raise ValueError(f'Size of matrix ({matrix.size}) is incompatible with length of select_indices ({len(select_indices)}) for specified block dimension ({block_dim}).')
        CB       = permutation_matrix(perm_map, block_dim=block_dim)
        Mro = CB @ matrix @ CB.T
        # Mro = np.matmul(CB, np.matmul(matrix, CB.T))
        # select partial matrices
        NA = block_dim * np.sum(select_indices)
        A = Mro[:NA, :NA]
        D = Mro[NA:, NA:]
        B = Mro[:NA, NA:]
        C = Mro[NA:, :NA]
        # calculate Schur complement
        # MA = A - np.dot(B, np.dot(np.linalg.inv(D), C)) 
        MA = A - B @ np.linalg.inv(D) @ C 
    else:
        rows = list()
        cols = list()
        vals = list()
        for i, j in enumerate(perm_map):
            for d in range(block_dim):
                rows.append(block_dim * i + d)
                cols.append(block_dim * j + d)
                vals.append(1)
        CB = coo_matrix((vals, (rows, cols)), dtype=float, shape=matrix.shape)
        # Mro_coo = CB.dot(matrix.dot(CB.transpose()))
        Mro_coo = CB @ matrix @ CB.transpose()
        Mro = Mro_coo.tocsc()
        # select partial matrices
        # nr, nc = Mro.shape
        NA = block_dim * np.sum(select_indices)
        A = Mro[:NA, :NA]
        D = Mro[NA:, NA:]
        B = Mro[:NA, NA:]
        C = Mro[NA:, :NA]
                
        # calculate Schur complement
        # MA = A - B.dot(sparse.linalg.spsolve(D, C))
        MA = A - B @ sparse.linalg.spsolve(D, C)
    return MA

def vector_marginal(
    vector: np.ndarray,  # shape (N,): input vector
    select_indices: np.ndarray,  # shape (M,): boolean selection array
    block_dim: int = 1
) -> np.ndarray:  # shape (K,): marginal vector
    """
    Extract selected components from a vector based on block selection.
    
    Selects elements from the vector according to select_indices, treating the vector
    as composed of blocks of size block_dim.
    
    Args:
        vector: Input vector to extract from.
        select_indices: Boolean array indicating which blocks to retain.
        block_dim: Size of each block.
    
    Returns:
        Vector containing only the selected components.
    """
    select_indices = proper_select_indices(select_indices)
    if block_dim > 1:
        sel_ind = np.outer(select_indices,np.ones(block_dim))
        select_indices = sel_ind.flatten()                
    select_indices = select_indices.astype(dtype=bool)
    return vector[select_indices]
        

def proper_select_indices(
    select_indices: np.ndarray  # shape (N,): selection array
) -> np.ndarray:  # shape (N,): normalized selection array
    """
    Normalize selection indices to binary (0 or 1) integer array.
    
    Converts any non-zero values to 1, effectively creating a boolean mask
    represented as integers.
    
    Args:
        select_indices: Array with numeric values.
    
    Returns:
        Integer array with only 0 and 1 values.
    """
    sel_indices = np.copy(select_indices)
    sel_indices[sel_indices != 0] = 1
    sel_indices = sel_indices.astype(dtype=int)
    return sel_indices

def permuation_map(
    select_indices: np.ndarray  # shape (N,): selection array
) -> np.ndarray:  # shape (N,): permutation map
    """
    Generate permutation map that moves selected indices to the front.
    
    Creates a permutation array where selected (non-zero) indices appear first,
    followed by unselected (zero) indices, preserving the original order within
    each group.
    
    Args:
        select_indices: Binary array (0 or 1) indicating selection.
    
    Returns:
        Permutation map array where selected indices come first.
    """
    retain = list()
    discard = list()
    for i in range(len(select_indices)):
        if select_indices[i] == 1:
            retain.append(i)
        else:
            discard.append(i)
    return np.array(retain+discard)
    
def permutation_matrix(
    perm_map: np.ndarray,  # shape (N,): permutation map
    block_dim: int = 1
) -> np.ndarray:  # shape (N*block_dim, N*block_dim): permutation matrix
    """
    Construct permutation matrix from permutation map for block structures.
    
    Creates a matrix that permutes blocks of size block_dim according to the
    given permutation map. Each entry in perm_map corresponds to a block.
    
    Args:
        perm_map: Array specifying the permutation order.
        block_dim: Size of each block to permute together.
    
    Returns:
        Permutation matrix that rearranges blocks according to perm_map.
    """
    N = len(perm_map)*block_dim
    # init matrix of basis change
    CB = np.zeros((N,N))
    eye = np.eye(block_dim)    
    for i, j in enumerate(perm_map):
        CB[block_dim * i : block_dim * (i + 1), block_dim * j : block_dim * (j + 1)] = eye
    return CB

############################################################################################
####################### Marginalization via name assignment ################################
############################################################################################

def matrix_marginal_assignment(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix,  # shape (N, N): square matrix
    select_names: list[str],
    names: list[str],
    block_dim: int = 1
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:  # shape (K, K): marginal matrix
    """
    Extract matrix marginal using named selection instead of indices.
    
    Convenience wrapper around matrix_marginal that allows selection by names
    rather than boolean indices. Supports wildcard patterns with '*'.
    
    Args:
        matrix: Square matrix to marginalize.
        select_names: List of names to retain (supports wildcards like 'rot*').
        names: Full list of names corresponding to matrix blocks.
        block_dim: Size of each named block.
    
    Returns:
        Marginalized matrix for selected names.
    """
    select_indices = select_names2indices(select_names,names)
    return matrix_marginal(matrix,select_indices,block_dim=block_dim)

def vector_marginal_assignment(
    vector: np.ndarray,  # shape (N,): input vector
    select_names: list[str],
    names: list[str],
    block_dim: int = 1
) -> np.ndarray:  # shape (K,): marginal vector
    """
    Extract vector marginal using named selection instead of indices.
    
    Convenience wrapper around vector_marginal that allows selection by names
    rather than boolean indices. Supports wildcard patterns with '*'.
    
    Args:
        vector: Input vector to extract from.
        select_names: List of names to retain (supports wildcards like 'rot*').
        names: Full list of names corresponding to vector blocks.
        block_dim: Size of each named block.
    
    Returns:
        Vector containing only selected named components.
    """
    select_indices = select_names2indices(select_names,names)
    return vector_marginal(vector,select_indices,block_dim=block_dim)


def select_names2indices(
    select_names: list[str],
    names: list[str]
) -> np.ndarray:  # shape (N,): boolean selection indices
    """
    Convert list of selected names to boolean index array.
    
    Matches select_names against names list to create a boolean selection array.
    Supports wildcard patterns (names ending in '*').
    
    Args:
        select_names: Names to select (can include wildcards like 'rot*').
        names: Complete list of available names.
    
    Returns:
        Boolean array indicating which names are selected.
    """
    select_names = unwrap_wildtypes(select_names,names)
    select_indices = np.zeros(len(names),dtype=bool)
    for i,name in enumerate(names):
        if name in select_names:
            select_indices[i] = 1
    return select_indices 
    
def unwrap_wildtypes(
    select_names: list[str],
    names: list[str]
) -> list[str]:
    """
    Expand wildcard patterns in select_names to matching names.
    
    Processes wildcards (names ending in '*') by matching the prefix against
    all available names. For example, 'rot*' matches 'rot1', 'rot2', etc.
    
    Args:
        select_names: Names to select, potentially with wildcards.
        names: Complete list of available names to match against.
    
    Returns:
        Expanded list of names with wildcards resolved.
    """
    wildtypes = [name.replace('*','') for name in select_names if '*' in name]
    return [name for name in names if name in select_names or name[0] in wildtypes]    



############################################################################################
####################### Marginalization degrees of freedom within blocks ###################
############################################################################################


def _blockmarginal_select_indices(
    target_size: int, 
    block_size: int, 
    block_index_list: np.ndarray | list[int]
    ) -> np.ndarray:  # shape (target_size,): selection indices
    block_index_list = sorted(list(set(block_index_list)))
    if block_index_list[0] < 0 or block_index_list[-1] >= block_size:
        raise ValueError(f'Invalid index encountered in block_index_list: Out of bounds!')

    partial = np.zeros(block_size)
    partial[block_index_list] = 1
    nblocks = target_size // block_size
    select_indices = np.outer(np.ones(nblocks),partial).flatten()
    return select_indices 

def matrix_blockmarginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix,  # shape (N, N): square matrix
    block_size: int,
    block_index_list: np.ndarray | list[int]
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:  # shape (K, K): marginal matrix
    """
    Extract marginal by retaining specific indices within each block.
    
    Divides the matrix into blocks of size block_size and retains only the components
    specified in block_index_list within each block. This is useful for extracting
    subsets of degrees of freedom that repeat across multiple blocks.
    
    Args:
        matrix: Square matrix to marginalize.
        block_size: Size of each repeating block.
        block_index_list: Indices within each block to retain (e.g., [0,1,2] for first 3).
    
    Returns:
        Marginalized matrix with only selected indices from each block.
    
    Raises:
        ValueError: If matrix size is not a multiple of block_size or matrix is not square.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'Provided matrix is not a square matrix. Has shape {matrix.shape}.')
    
    if matrix.shape[0]%block_size != 0:
        raise ValueError(f'Matrix size is not a multiple of the specified block size.')

    select_indices = _blockmarginal_select_indices(matrix.shape[0], block_size, block_index_list)
    return matrix_marginal(matrix,select_indices,block_dim=1)


def vector_blockmarginal(
    vector: np.ndarray,  # shape (N,): input vector
    block_size: int,
    block_index_list: np.ndarray | list[int]
    ) -> np.ndarray:  # shape (K,): marginal vector
    """
    Extract vector marginal by retaining specific indices within each block.
    
    Divides the vector into blocks of size block_size and retains only the components
    specified in block_index_list within each block.
    
    Args:
        vector: Input vector to extract from.
        block_size: Size of each repeating block.
        block_index_list: Indices within each block to retain.
    
    Returns:
        Vector containing only selected indices from each block.
    
    Raises:
        ValueError: If vector size is not a multiple of block_size.
    """
    if vector.shape[0]%block_size != 0:
        raise ValueError(f'Matrix size is not a multiple of the specified block size.')
    
    select_indices = _blockmarginal_select_indices(vector.shape[0], block_size, block_index_list)
    return vector_marginal(vector,select_indices,block_dim=1)


############################################################################################
####################### Rotation and Translation marginalizer ##############################
############################################################################################

def matrix_rotmarginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix,  # shape (N*6, N*6): SE(3) matrix
    rotation_first: bool = True
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:  # shape (N*3, N*3): rotation marginal
    """
    Extract rotational marginal from SE(3) matrix.
    
    Marginalizes out translation degrees of freedom, retaining only rotational
    components from 6-DOF SE(3) blocks.
    
    Args:
        matrix: SE(3) stiffness or covariance matrix with 6-DOF blocks.
        rotation_first: If True, rotations are first 3 DOFs; if False, last 3 DOFs.
    
    Returns:
        3-DOF rotational marginal matrix.
    """
    if rotation_first:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[0,1,2])
    else:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[3,4,5])
    
def matrix_transmarginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix,  # shape (N*6, N*6): SE(3) matrix
    rotation_first: bool = True
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:  # shape (N*3, N*3): translation marginal
    """
    Extract translational marginal from SE(3) matrix.
    
    Marginalizes out rotation degrees of freedom, retaining only translational
    components from 6-DOF SE(3) blocks.
    
    Args:
        matrix: SE(3) stiffness or covariance matrix with 6-DOF blocks.
        rotation_first: If True, rotations are first 3 DOFs; if False, last 3 DOFs.
    
    Returns:
        3-DOF translational marginal matrix.
    """
    if rotation_first:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[3,4,5])
    else:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[0,1,2])
    
def vector_rotmarginal(
    vector: np.ndarray,  # shape (N*6,): SE(3) vector
    rotation_first: bool = True
) -> np.ndarray:  # shape (N*3,): rotation marginal
    """
    Extract rotational components from SE(3) vector.
    
    Selects only the rotational degrees of freedom from 6-DOF SE(3) blocks.
    
    Args:
        vector: SE(3) vector with 6-DOF blocks (rotation + translation).
        rotation_first: If True, rotations are first 3 DOFs; if False, last 3 DOFs.
    
    Returns:
        3-DOF rotational vector.
    """
    if rotation_first:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[0,1,2])
    else:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[3,4,5])
    
def vector_transmarginal(
    vector: np.ndarray,  # shape (N*6,): SE(3) vector
    rotation_first: bool = True
) -> np.ndarray:  # shape (N*3,): translation marginal
    """
    Extract translational components from SE(3) vector.
    
    Selects only the translational degrees of freedom from 6-DOF SE(3) blocks.
    
    Args:
        vector: SE(3) vector with 6-DOF blocks (rotation + translation).
        rotation_first: If True, rotations are first 3 DOFs; if False, last 3 DOFs.
    
    Returns:
        3-DOF translational vector.
    """
    if rotation_first:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[3,4,5])
    else:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[0,1,2])
    
    

##########################################################################################################
############### Schur Complement and Permutation matrices ################################################
##########################################################################################################


def marginal_schur_complement(
    mat: np.ndarray,  # shape (N, N): square matrix
    retained_ids: list[int]
) -> np.ndarray:  # shape (K, K): reduced matrix via Schur complement
    """
    Compute Schur complement to retain specified degrees of freedom.
    
    Uses the Schur complement formula to marginalize out degrees of freedom,
    retaining only those specified in retained_ids. This is equivalent to
    computing the conditional distribution in Gaussian systems.
    
    Note:
        For sparse matrices, use matrix_marginal instead for better efficiency.
        This function converts sparse matrices to dense with a deprecation warning.
    
    Args:
        mat: Square matrix (preferably dense).
        retained_ids: List of DOF indices to retain.
    
    Returns:
        Reduced matrix containing only retained degrees of freedom.
    
    Warns:
        DeprecationWarning: If mat is sparse (inefficient conversion to dense).
    """
    if sp.sparse.issparse(mat):
        warn('marginal_schu_complement currently does not support sparse matrices. Matrix converted to dense. For more efficient handling for sparse matrices use matrix_marginal..', DeprecationWarning, stacklevel=2)
        mat = mat.toarray()
    
    # calculate permutation matrix
    P = send_to_back_permutation(mat.shape[0], retained_ids)
    # rearrange matrix
    pmat = P @ mat @ P.T
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
    schur = D - B.T @ np.linalg.inv(A) @ B
    return schur


def permutation_matrix_by_indices(
    order: list[int]  # length N: permutation order
) -> np.ndarray:  # shape (N, N): permutation matrix
    """
    Generate permutation matrix from explicit index ordering.
    
    Creates a permutation matrix that rearranges elements according to the
    specified order. The order list must contain all indices from 0 to max(order)
    exactly once.
    
    Args:
        order: List specifying the new order of indices (e.g., [2,0,1] to shift right).
    
    Returns:
        Permutation matrix P where P @ v reorders vector v according to order.
    
    Raises:
        ValueError: If order list is missing any indices.
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
    N: int, 
    move_back_ids: list[int], 
    ordered: bool = False
) -> np.ndarray:  # shape (N, N): transformation matrix
    """
    Generate permutation matrix that moves specified elements to the end.
    
    Creates a permutation matrix that moves the specified indices to the back
    while preserving the relative order of remaining elements. Useful for
    preparing matrices for Schur complement operations.
    
    Args:
        N: Total number of degrees of freedom.
        move_back_ids: Indices of DOFs to move to the back.
        ordered: If True, sort move_back_ids before moving them.
    
    Returns:
        Permutation matrix that sends specified DOFs to the end.
    """
    if isinstance(move_back_ids,np.ndarray):
        move_back_ids = move_back_ids.tolist()
    T = np.zeros((N,) * 2)
    if ordered:
        move_back_ids = sorted(move_back_ids)
    leading = [i for i in range(N) if i not in move_back_ids]
    order = leading + move_back_ids
    return permutation_matrix_by_indices(order)