import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
from typing import List, Tuple, Callable, Any, Dict

def kl_divergence(
    mu1: np.ndarray, 
    M1: np.ndarray, 
    mu2: np.ndarray, 
    M2: np.ndarray,
    symmetrize: bool = False
    ) -> np.ndarray:
    
    if symmetrize:
        return kl_divergence_sym(mu1,M1,mu2,M2)    
    dv = mu1-mu2
    return 0.5 * ( np.linalg.det(M2)/np.linalg.det(M1) - len(M1) +np.trace(M2 @ np.linalg.inv(M1)) + dv.T @ M2 @ dv )

def kl_divergence_sym(
    mu1: np.ndarray, 
    M1: np.ndarray, 
    mu2: np.ndarray, 
    M2: np.ndarray
    ) -> np.ndarray:
    
    dv = mu1-mu2
    return 0.5 * ( np.trace(M2 @ np.linalg.inv(M1)) + np.trace(M1 @ np.linalg.inv(M2)) - 2* len(M1) + dv.T @ (M1+M2) @ dv )