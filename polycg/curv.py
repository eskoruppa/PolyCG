import sys, os
import argparse
import numpy as np

from ._gen_params import gen_params
from .genconf import gen_config

def dnacurv(
    seq: str,
    span: int, 
    model: str = 'cgna+', 
    angles: bool = True, 
    degrees: bool = True
    ) -> dict:
    """Generates the angles due to intrinsic curvature spanning the next span base pairs in a sliding window through the provided sequence.

    Args:
        seq (str): DNA sequence 
        span (int): number of segments over which bend angles are calculated
        model (str, optional): DNA structure and elasticity model. Defaults to 'cgna+'.
        angles (bool, optional): express curvature as an angle. Defaults to True. If False, values are expressed as cos(theta).
        degrees (bool, optional): Express angles in degrees. Defaults to True. Automatically False, if angles is False.

    Returns:
        np.ndarray
    """
    prms = gen_params(model,seq,composite_size=1)
    taus = gen_config(prms['gs'])
    N = len(seq)-1
    M = N - span + 1
    vals = np.zeros(M)
    for i in range(M):
        vals[i] = np.dot(taus[i,:3,2],taus[i+span,:3,2])
    if angles:
        vals = np.arccos(vals)
    if degrees and angles:
        vals = np.rad2deg(vals) 
    curvdict = {
        'curvature': vals, 
        'sequence': seq, 
        'span': span,
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