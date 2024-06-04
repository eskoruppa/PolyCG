import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
from typing import List, Tuple, Callable, Any, Dict, Optional
from ..SO3 import so3
from warnings import warn

def matrix_marginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix, 
    select_indices: np.ndarray,
    block_dim: int = 1
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:
    """
    Extracts the marginal of a square matrix matrix for blocks of size block_dim for the provided select_indices (boolean array). The  
    The dimension of the matrix needs to match the size of select_indices times block_dim.
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
    vector: np.ndarray,
    select_indices: np.ndarray,
    block_dim: int = 1
) -> np.ndarray:
    select_indices = proper_select_indices(select_indices)
    if block_dim > 1:
        sel_ind = np.outer(select_indices,np.ones(block_dim))
        select_indices = sel_ind.flatten()                
    select_indices = select_indices.astype(dtype=bool)
    return vector[select_indices]
        

def proper_select_indices(select_indices: np.ndarray) -> np.ndarray:
    sel_indices = np.copy(select_indices)
    sel_indices[sel_indices != 0] = 1
    sel_indices = sel_indices.astype(dtype=int)
    return sel_indices

def permuation_map(select_indices: np.ndarray) -> np.ndarray:
    """Permutation map. 

    Args:
        select_indices (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    retain = list()
    discard = list()
    for i in range(len(select_indices)):
        if select_indices[i] == 1:
            retain.append(i)
        else:
            discard.append(i)
    return np.array(retain+discard)
    
def permutation_matrix(perm_map: np.ndarray, block_dim: int = 1) -> np.ndarray:
    
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
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix, 
    select_names: List[str],
    names: List[str],
    block_dim: int = 1
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:
    select_indices = select_names2indices(select_names,names)
    return matrix_marginal(matrix,select_indices,block_dim=block_dim)

def vector_marginal_assignment(
    vector: np.ndarray, 
    select_names: List[str],
    names: List[str],
    block_dim: int = 1
) -> np.ndarray:
    select_indices = select_names2indices(select_names,names)
    return vector_marginal(vector,select_indices,block_dim=block_dim)


def select_names2indices(
    select_names: List[str],
    names: List[str]
) -> np.ndarray:
    select_names = unwrap_wildtypes(select_names,names)
    select_indices = np.zeros(len(names),dtype=bool)
    for i,name in enumerate(names):
        if name in select_names:
            select_indices[i] = 1
    return select_indices 
    
def unwrap_wildtypes(
    select_names: List[str],
    names: List[str]
) -> List[str]:
    wildtypes = [name.replace('*','') for name in select_names if '*' in name]
    return [name for name in names if name in select_names or name[0] in wildtypes]    



############################################################################################
####################### Marginalization degrees of freedom within blocks ###################
############################################################################################


def _blockmarginal_select_indices(
    target_size: int, 
    block_size: int, 
    block_index_list: np.ndarray | List[int]
    ) -> np.ndarray:
    block_index_list = sorted(list(set(block_index_list)))
    if block_index_list[0] < 0 or block_index_list[-1] >= block_size:
        raise ValueError(f'Invalid index encountered in block_index_list: Out of bounds!')

    partial = np.zeros(block_size)
    partial[block_index_list] = 1
    nblocks = target_size // block_size
    select_indices = np.outer(np.ones(nblocks),partial).flatten()
    return select_indices 

def matrix_blockmarginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix, 
    block_size: int,
    block_index_list: np.ndarray | List[int]
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:
    """
    Splits the matrix into blocks of size block_size and retains within each block only the components specified in block_index_list
    """ 
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'Provided matrix is not a square matrix. Has shape {matrix.shape}.')
    
    if matrix.shape[0]%block_size != 0:
        raise ValueError(f'Matrix size is not a multiple of the specified block size.')

    select_indices = _blockmarginal_select_indices(matrix.shape[0], block_size, block_index_list)
    return matrix_marginal(matrix,select_indices,block_dim=1)


def vector_blockmarginal(
    vector: np.ndarray,
    block_size: int,
    block_index_list: np.ndarray | List[int]
    ) -> np.ndarray:
    
    if vector.shape[0]%block_size != 0:
        raise ValueError(f'Matrix size is not a multiple of the specified block size.')
    
    select_indices = _blockmarginal_select_indices(vector.shape[0], block_size, block_index_list)
    return vector_marginal(vector,select_indices,block_dim=1)


############################################################################################
####################### Rotation and Translation marginalizer ##############################
############################################################################################

def matrix_rotmarginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix, 
    rotation_first: bool = True
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:
    if rotation_first:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[0,1,2])
    else:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[3,4,5])
    
def matrix_transmarginal(
    matrix: np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix, 
    rotation_first: bool = True
) -> np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix | sp.sparse.coo_matrix:
    if rotation_first:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[3,4,5])
    else:
        return matrix_blockmarginal(matrix,block_size=6,block_index_list=[0,1,2])
    
def vector_rotmarginal(
    vector: np.ndarray,
    rotation_first: bool = True
) -> np.ndarray:
    if rotation_first:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[0,1,2])
    else:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[3,4,5])
    
def vector_transmarginal(
    vector: np.ndarray,
    rotation_first: bool = True
) -> np.ndarray:
    if rotation_first:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[3,4,5])
    else:
        return vector_blockmarginal(vector,block_size=6,block_index_list=[0,1,2])
    
    

##########################################################################################################
############### Schur Complement and Permutation matrices ################################################
##########################################################################################################


def marginal_schur_complement(mat: np.ndarray, retained_ids: List[int]) -> np.ndarray:
    """Schur complement of matrix to retain the specified degrees of freedom

    Args:
        mat (np.ndarray): Given matrix
        retained_ids (List[int]): List of dof that are to be retained

    Returns:
        np.ndarray: reduced matrix
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


def permutation_matrix_by_indices(order: List[int]) -> np.ndarray:
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
    if isinstance(move_back_ids,np.ndarray):
        move_back_ids = move_back_ids.tolist()
    T = np.zeros((N,) * 2)
    if ordered:
        move_back_ids = sorted(move_back_ids)
    leading = [i for i in range(N) if i not in move_back_ids]
    order = leading + move_back_ids
    return permutation_matrix_by_indices(order)