from __future__ import annotations

import numpy as np
import scipy as sp

##########################################################################################################
##########################################################################################################
############### Unit Conversions #########################################################################
##########################################################################################################
##########################################################################################################

def conversion(
    array: np.ndarray | sp.sparse.spmatrix, 
    factor: float, 
    block_dim: int = 6, 
    dofs: list[int] = None
    ) -> np.ndarray | sp.sparse.spmatrix:  # shape (N,) or (N, N)
    """Apply unit conversion factor to specific degrees of freedom in vectors or matrices.
    
    Converts units by multiplying specified degrees of freedom by a conversion factor.
    For vectors, multiplies selected elements. For matrices, multiplies both rows and
    columns corresponding to selected degrees of freedom.
    
    Args:
        array: Vector or matrix to convert. Shape (N,) for vectors or (N, N) for matrices.
               Can be np.ndarray or scipy sparse matrix.
        factor: Conversion factor to multiply selected degrees of freedom.
        block_dim: Size of blocks in the array (e.g., 6 for SE(3) vectors).
        dofs: List of degree of freedom indices to convert. If None, converts all elements.
              Indices are taken modulo block_dim.

    Returns:
        Converted array with same shape as input and same type (np.ndarray or sparse matrix).
        
    Raises:
        ValueError: If array has more than 2 dimensions.
    """

    if sp.sparse.issparse(array):
        # if everything is to be converted
        if dofs is None:
            return array.multiply(factor)
        
        # reduce to be converted degrees of freedom
        dofs = sorted(list(set([dof % block_dim for dof in dofs])))
        indices = np.concatenate([np.arange(dof, array.shape[0], block_dim) for dof in dofs])
        
        # Create diagonal scaling matrix to scale rows and columns efficiently
        scale_diag = np.ones(array.shape[0])
        scale_diag[indices] = factor
        scale_matrix = sp.sparse.diags(scale_diag)
        
        # Scale rows (left multiplication) and columns (right multiplication)
        return scale_matrix @ array @ scale_matrix
    else:
        # Original numpy implementation
        dims = len(array.shape)
        # check if valid shape
        if dims > 2:
            raise ValueError(f'Conversion only supports vectors and matrices. Encountered array of shape {array.shape}.')
        # if everything is to be converted
        if dofs is None:
            return array * factor
        
        # reduce to be converted degrees of freedom    
        carray = np.copy(array)
        dofs = sorted(list(set([dof % block_dim for dof in dofs])))
        indices = np.concatenate([np.arange(dof, array.shape[0], block_dim) for dof in dofs])
        
        carray[indices] *= factor
        if dims == 2:
            carray[:, indices] *= factor
        return carray



def conversion_old(array: np.ndarray, factor: float, block_dim: int = 6, dofs: list[int] = None):  # shape (N,) or (N, N)
    """Apply unit conversion factor to specific degrees of freedom in vectors or matrices.
    
    Converts units by multiplying specified degrees of freedom by a conversion factor.
    For vectors, multiplies selected elements. For matrices, multiplies both rows and
    columns corresponding to selected degrees of freedom.
    
    Args:
        array: Vector or matrix to convert. Shape (N,) for vectors or (N, N) for matrices.
        factor: Conversion factor to multiply selected degrees of freedom.
        block_dim: Size of blocks in the array (e.g., 6 for SE(3) vectors).
        dofs: List of degree of freedom indices to convert. If None, converts all elements.
              Indices are taken modulo block_dim.

    Returns:
        Converted array with same shape as input.
        
    Raises:
        ValueError: If array has more than 2 dimensions.
    """
    dims = len(array.shape)
    # check if valid shape
    if dims > 2:
        raise ValueError(f'Conversion only supports vectors and matrices. Encountered array of shape {array.shape}.')
    # if everthing is to be converted
    carray = np.copy(array)
    if dofs is None:
        for i in range(dims):
            carray *= factor
        return carray
    # reduce to be converted degrees of freedom    
    dofs = sorted(list(set([dof%block_dim for dof in dofs])))
    for dof in dofs:
        carray[dof::block_dim] *= factor
        if dims == 2:
            carray[:,dof::block_dim] *= factor
    return carray


if __name__ == '__main__':
    
    stiff = np.ones((12,12))
    factor = 5
    nmat = conversion(stiff,factor,block_dim=6,dofs=[0,1,2])
    print(nmat)
    
            
    
# def fifth2rad(val: Any) -> float:
#     """
#     convert single value from fifth radians into radians
#     """
#     return val / 5


# def fifth2deg(val: Any) -> float:
#     """
#     convert single value from fifth radians into degrees
#     """
#     return val * 180 / np.pi / 5


# def gs2rad(gs: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert ground state vector from fifth radians into radians. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _gsconf(fifth2rad, gs, only_rot=only_rot)


# def gs2deg(gs: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert ground state vector from fifth radians into degrees. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _gsconf(fifth2deg, gs, only_rot=only_rot)


# def _gsconf(convfunc: Callable, gs: np.ndarray, only_rot=False):
#     gs = np.copy(gs)
#     if not only_rot:
#         if len(gs) % 6 != 0:
#             raise ValueError(
#                 f"Unexpected dimension of gs. Expecting multiple of 6, but received {len(gs)} (len(gs)%6={len(gs)%6})."
#             )
#         N = len(gs) // 6
#         for i in range(N):
#             gs[6 * i : 6 * i + 3] = convfunc(gs[6 * i : 6 * i + 3])
#         return gs
#     return convfunc(gs)


# def stiff2rad(stiff: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert stiffness matrix from fifth radians into radians. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _stiffconf(fifth2rad(1), stiff, only_rot=only_rot)


# def stiff2deg(stiff: np.ndarray, only_rot=False) -> np.ndarray:
#     """
#     convert stiffness matrix from fifth radians into degrees. If only_rot is True it will assume that all values are rotations.
#     Otherwise it is assumed that each block has 6 entries
#     """
#     return _stiffconf(fifth2deg(1), stiff, only_rot=only_rot)


# def _stiffconf(fac: float, stiff: np.ndarray, only_rot=False):
#     stiff = np.copy(stiff)
#     if not only_rot:
#         if len(stiff) % 6 != 0:
#             raise ValueError(
#                 f"Unexpected dimension of gs. Expecting multiple of 6, but received {len(stiff)} (len(gs)%6={len(stiff)%6})."
#             )
#         N = len(stiff) // 6
#         for i in range(N):
#             stiff[6 * i : 6 * i + 3, :] /= fac
#             stiff[:, 6 * i : 6 * i + 3] /= fac
#         return stiff
#     return stiff / fac**2