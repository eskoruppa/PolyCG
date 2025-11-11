import sys, os
import argparse
import numpy as np

from .gen_params import gen_params, gen_config
from .cgnaplus import cgnaplus_bps_params

def dnacurv(seq: str, disc: int, model: str = 'cgna+', angles: bool = True, degrees: bool = False) -> dict:
    prms = gen_params(model,seq,composite_size=1)
    taus = gen_config(prms['gs'])
    N = len(seq)-1
    M = N - disc + 1
    vals = np.zeros(M)
    for i in range(M):
        vals[i] = np.dot(taus[i,:3,2],taus[i+disc,:3,2])
    if angles:
        vals = np.arccos(vals)
    if degrees:
        vals = np.rad2deg(vals) 
    return vals
    
    
if __name__ == "__main__":
    
    seq = sys.argv[1]
    disc = int(sys.argv[2])
    # seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(N)])
    # seq = 'ATCGATCGATCGATCG'
    # disc = 10
    crv = dnacurv(seq,disc,degrees=True)
    print(crv)