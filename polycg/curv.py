from __future__ import annotations

import sys, os
import argparse
import numpy as np
from typing import Sequence

from ._gen_params import gen_params
from .genconf import gen_config

def dnacurv(
    seq: str,
    span: int | Sequence[int],
    model: str = 'cgna+',
    angles: bool = True,
    degrees: bool = True
    ) -> dict:
    """Generates the angles due to intrinsic curvature spanning the next span base pairs in a sliding window through the provided sequence.

    Args:
        seq (str): DNA sequence
        span (int | Sequence[int]): number of segments over which bend angles are calculated. May also be a list or array of spans, in which case the configuration is generated once and evaluated for every requested span.
        model (str, optional): DNA structure and elasticity model. Defaults to 'cgna+'.
        angles (bool, optional): express curvature as an angle. Defaults to True. If False, values are expressed as cos(theta).
        degrees (bool, optional): Express angles in degrees. Defaults to True. Automatically False, if angles is False.

    Returns:
        dict: 'curvature' contains a single array of len(seq)-span values if span is a scalar, and a list of such arrays, one per requested span, if span is a sequence. Since the arrays differ in length they are not merged into a single 2D array. 'span' mirrors the argument in the same manner.
    """
    single = np.ndim(span) == 0
    spans = [int(span)] if single else [int(s) for s in span]
    if len(spans) == 0:
        raise ValueError('no span provided')
    for s in spans:
        if s < 1:
            raise ValueError(f'span needs to be at least 1, encountered {s}')
        if s > len(seq)-1:
            raise ValueError(f'span {s} exceeds the {len(seq)-1} steps contained in the provided sequence')

    prms = gen_params(model,seq,composite_size=1,print_info=True,verbose=True)
    taus = gen_config(prms.shape_params)
    # tangents of all len(seq) base pair triads
    tans = taus[:,:3,2]

    vals = []
    for s in spans:
        # val[i] = tans[i].tans[i+s], for i in [0,len(seq)-s)
        val = np.einsum('ij,ij->i',tans[:-s],tans[s:])
        if angles:
            # dot products of numerically normalized tangents may marginally exceed 1
            val = np.arccos(np.clip(val,-1,1))
            if degrees:
                val = np.rad2deg(val)
        vals.append(val)

    curvdict = {
        'curvature': vals[0] if single else vals,
        'sequence': seq,
        'span': spans[0] if single else spans,
        'model': model,
        'angles': angles,
        'degrees': degrees}
    return curvdict
    
    
if __name__ == "__main__":
    
    seq = sys.argv[1]
    # single span, or several as a comma separated list
    disc = [int(s) for s in sys.argv[2].split(',')]
    if len(disc) == 1:
        disc = disc[0]
    # seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(N)])
    # seq = 'ATCGATCGATCGATCG'
    # disc = 10
    crv = dnacurv(seq,disc,degrees=True)
    print(crv)