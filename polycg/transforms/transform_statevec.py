from __future__ import annotations

import numpy as np


##########################################################################################################
############### Conversions between state vectors (3N) and 3-vectors (N,3) ###############################
##########################################################################################################

def statevec2vecs(statevec: np.ndarray, vdim: int) -> np.ndarray:  # shape (..., vdim*N) -> (..., N, vdim)
    """Reshape state vectors from flat format (vdim*N) to vector format (N, vdim).
    
    Converts the last dimension from flat state vector representation to structured
    vector format. For example, converts (3*N,) to (N, 3) for 3D vectors.

    Args:
        statevec: State vector array with last dimension of size (vdim*N). If already
                 in shape (..., vdim), returns unchanged.
        vdim: Dimension of individual vectors (e.g., 3 for 3D, 6 for SE(3)).

    Returns:
        Reshaped array with last dimension split into (N, vdim).
        
    Raises:
        ValueError: If last dimension size is not a multiple of vdim.
    """
    if statevec.shape[-1] == vdim:
        return statevec
    
    if statevec.shape[-1] % vdim != 0:
        raise ValueError(
            f"statevec2vecs: statevec is inconsistent with list of euler vectors. The number of entries needs to be a multiple of vdim. len(statevec)%vdim = {len(statevec)%vdim}"
        )
    
    shape = list(statevec.shape)
    newshape = shape[:-1] + [shape[-1] // vdim, vdim]
    return np.reshape(statevec, tuple(newshape))


def vecs2statevec(vecs: np.ndarray) -> np.ndarray:  # shape (..., N, vdim) -> (..., vdim*N)
    """Reshape vector format (N, vdim) to flat state vector format (vdim*N).
    
    Converts structured vector representation to flat state vector format. Merges the
    last two dimensions (N, vdim) into a single flat dimension (vdim*N).

    Args:
        vecs: Vector array with shape (..., N, vdim), where N is number of vectors
             and vdim is dimension of each vector.

    Returns:
        Flattened array with last two dimensions merged into (vdim*N,).
    """
    shape = list(vecs.shape)
    newshape = shape[:-1]
    newshape[-1] *= shape[-1]
    return np.reshape(vecs, tuple(newshape))