import numpy as np
from typing import List, Tuple, Callable, Any, Dict


##########################################################################################################
############### Conversions between state vectors (3N) and 3-vectors (N,3) ###############################
##########################################################################################################

def statevec2vecs(statevec: np.ndarray, vdim: int) -> np.ndarray:
    """reshapes configuration of full state (vdim*N) vectors into vdim-vectors. Turns last dimensions from (vdim*N) to (N,vdim)

    Args:
        vecs (np.ndarray): collection of state vectors

    Returns:
        np.ndarray: collection of vdim-vectors
    """
    if statevec.shape[-1] == vdim:
        return statevec
    
    if statevec.shape[-1] % vdim != 0:
        raise ValueError(
            f"statevec is inconsistent with list of euler vectors. THe number of entries needs to be a multiple of vdim. len(statevec)%vdim = {len(statevec)%vdim}"
        )
    
    shape = list(statevec.shape)
    newshape = shape[:-1] + [shape[-1] // vdim, vdim]
    return np.reshape(statevec, tuple(newshape))


def vecs2statevec(vecs: np.ndarray) -> np.ndarray:
    """reshapes configuration of vdim-vectors into full state (vdim*N) vectors. Turns last dimensions from (N,vdim) to (vdim*N,)

    Args:
        vecs (np.ndarray): collection of vdim-vectors

    Returns:
        np.ndarray: collection of state vectors
    """
    shape = list(vecs.shape)
    newshape = shape[:-1]
    newshape[-1] *= shape[-1]
    return np.reshape(vecs, tuple(newshape))