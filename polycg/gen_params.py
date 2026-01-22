from __future__ import annotations

import sys, os
import time
import argparse
import numpy as np
import scipy as sp
from typing import Any, Callable, Dict, List, Tuple

# load models
from .cgnaplus import cgnaplus_bps_params
from .models.RBPStiff.read_params import GenStiffness
# load partial stiffness generation
from .partials import partial_stiff
# load coarse graining methods
from .cg import coarse_grain
# load sequence from sequence file
from .utils.load_seq import load_sequence
# write sequence file
from .utils.seq import write_seqfile
# load so3 
from .SO3 import so3
# load visualization methods
from .out.visualization import cgvisual
# load gen_params
from ._gen_params import gen_params

# # load iopolymc output methods
# from .IOPolyMC.iopolymc import write_xyz, gen_pdb

    
# ##################################################################################################################
# ##################################################################################################################
# ##################################################################################################################

# def gen_config(params: np.ndarray):
#     if len(params.shape) == 1:
#         pms = params.reshape(len(params)//6,6)
#     else:
#         pms = params
#     taus = np.zeros((len(pms)+1,4,4))
#     taus[0] = np.eye(4)
#     for i,pm in enumerate(pms):
#         g = so3.se3_euler2rotmat(pm)
#         taus[i+1] = taus[i] @ g
#     return taus

##################################################################################################################
##################################################################################################################
##################################################################################################################


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate ground state and stiffness matrix")
    parser.add_argument('-m',       '--model',              type=str, default = 'cgnaplus', choices=['cgnaplus','lankas','olson'])
    parser.add_argument('-cg',      '--composite_size',     type=int, default = 1)
    parser.add_argument('-seqfn',   '--sequence_file',      type=str, default = None)
    parser.add_argument('-seq',     '--sequence',           type=str, default = None)
    parser.add_argument('-closed',  '--closed',             action='store_true') 
    parser.add_argument('-nc',      '--no_crop',            action='store_true') 
    parser.add_argument('-np',      '--no_partial',         action='store_true') 
    parser.add_argument('-sid',     '--start_id',           type=int, default=0) 
    parser.add_argument('-eid',     '--end_id',             type=int, default=None) 
    parser.add_argument('-o',       '--output_basename',    type=str, default = None, required=False)
    parser.add_argument('-nv',      '--no_visualization',   action='store_true') 
    parser.add_argument('-bpst',   '--include_bps_triads', action='store_true') 
    
    # parser.add_argument('-xyz',     '--gen_xyz',            action='store_true') 
    # parser.add_argument('-pdb',     '--gen_pdb',            action='store_true') 
     
    args = parser.parse_args()
    
    model           = args.model
    composite_size  = args.composite_size
    seqfn           = args.sequence_file
    seq             = args.sequence
    closed          = args.closed
    allow_crop      = not args.no_crop
        
    allow_partial   = not args.no_partial
    start_id        = args.start_id
    end_id          = args.end_id
    
    ##################################################################################################################
    
    if seq is None:
        if seqfn is None:
            raise ValueError(f'Requires either a sequence (-seq) or a sequence file (-seqfn)')
        seq = load_sequence(seqfn)
            
    cgnap_setname = 'curves_plus'
    
    params = gen_params(
        model , 
        seq,
        composite_size,
        closed=closed,
        start_id=start_id,
        end_id=end_id,
        allow_partial=allow_partial,
        allow_crop=allow_crop,
        cgnap_setname = cgnap_setname
    )
    
    if args.output_basename is None:
        if args.sequence_file is None:
            raise ValueError(f'Either output filename or sequence filename have to be specified.')
        base_fn = os.path.splitext(seqfn)[0]
    else:
        base_fn = args.output_basename
    cg_fn = base_fn + f'_cg{composite_size}'
    fn_gs = cg_fn + '_gs.npy'
    fn_stiff = cg_fn + '_stiff.npz'
    
    print(f'writing stiffness to "{fn_stiff}"')
    print(f'writing groundstate to "{fn_gs}"')
    add = ''
    if composite_size > 1:
        add = 'cg_'
    sp.sparse.save_npz(fn_stiff,params[add+'stiff'])
    np.save(fn_gs,params[add+'gs'])
    
    # write sequence file
    seqfn = base_fn + '.seq'
    write_seqfile(seqfn,seq,add_extension=True)
    
    # visualization
    if not args.no_visualization:
        visdir = base_fn
        vis_seq = seq
        if closed:
            vis_seq += seq[0]
        cgvisual(visdir,params['gs'],vis_seq,composite_size,start_id,bead_radius=composite_size*0.34*0.5,include_bps_triads=args.include_bps_triads)
    