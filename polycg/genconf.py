import numpy as np
from .SO3 import so3
from .transforms.transform_SE3 import euler2rotmat_se3

def gen_config(params: np.ndarray, disc_len: float=0.34):
    if len(params.shape) == 1:
        raise ValueError(f'params should be at least twodimensional')
        
    n = params.shape[-2]
    dims = params.shape[-1]
    # if multiple snapshots
    if len(params.shape) > 2:
        taus = np.zeros(params.shape[:-2]+(n+1,4,4))
        for i in range(len(params)):
            taus[i] = gen_config(params[i],disc_len=disc_len)
        return taus
    
    if dims == 3:
        ext_params = np.zeros((n,6))
        ext_params[:,:3] = params
        ext_params[:,5]  = disc_len
    else:
        ext_params = params
    se3trans = euler2rotmat_se3(ext_params)
    taus = np.zeros((n+1,4,4))
    taus[0] = np.eye(4)
    for i in range(n):
        taus[i+1] = taus[i] @ se3trans[i]
    return taus