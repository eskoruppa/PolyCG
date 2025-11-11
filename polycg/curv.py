import sys, os
import argparse
import numpy as np

from .gen_params import gen_params, gen_config
from .cgnaplus import cgnaplus_bps_params

def dnacurv(
    seq: str,
    disc_len: int, 
    model: str = 'cgna+', 
    angles: bool = True, 
    degrees: bool = True
    ) -> dict:
    """_summary_

    Args:
        seq (str): DNA sequence 
        disc_len (int): size of segments over which bend angles are calculated
        model (str, optional): DNA structure and elasticity model. Defaults to 'cgna+'.
        angles (bool, optional): express curvature as an angle. Defaults to True. If False, values are expressed as cos(theta).
        degrees (bool, optional): Express angles in degrees. Defaults to True. Automatically False, if angles is False.

    Returns:
        numpy array 
    """
    prms = gen_params(model,seq,composite_size=1)
    taus = gen_config(prms['gs'])
    N = len(seq)-1
    M = N - disc_len + 1
    vals = np.zeros(M)
    for i in range(M):
        vals[i] = np.dot(taus[i,:3,2],taus[i+disc_len,:3,2])
    if angles:
        vals = np.arccos(vals)
    if degrees and angles:
        vals = np.rad2deg(vals) 
    curvdict = {
        'curvature': vals, 
        'sequence': seq, 
        'disc_len': disc_len,
        'model': model,
        'angles': angles,
        'degrees': degrees}
    return curvdict
    
    
if __name__ == "__main__":
    
    seq = sys.argv[1]
    disc = int(sys.argv[2])
    # seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(N)])
    # seq = 'ATCGATCGATCGATCG'
    # disc = 10
    crv = dnacurv(seq,disc,degrees=True)
    print(crv)