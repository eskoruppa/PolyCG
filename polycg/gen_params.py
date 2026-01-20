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


# def gen_params(
#     model: str, 
#     sequence: str,
#     composite_size: int = 1,
#     closed: bool = False,
#     sparse: bool = True,
#     start_id: int = 0,
#     end_id:   int = None,
#     allow_partial: bool = True,
#     block_size: int = 120,
#     overlap_size: int = 20,
#     tail_size: int = 20,
#     allow_crop: bool = True,
#     cgnap_setname: str = 'curves_plus'
#     ) -> Dict:

#     ##################################################################################################################
#     # Generate Stiffness and groundstate

#     #########################################################
#     # Crystal structure data from Olson et al. 1998
#     # https://www.pnas.org/doi/full/10.1073/pnas.95.19.11163

#     if model.lower() in ['crystal','cry','olson']:
#         genstiff = GenStiffness(method='crystal')
#         stiff,gs = genstiff.gen_params(sequence,use_group=False,sparse=True)
    
#     #########################################################
#     # MD data from Lankas et al. 2003
#     # https://doi.org/10.1016/S0006-3495(03)74710-9
    
#     if model.lower() in ['md','lankas']:
#         genstiff = GenStiffness(method='md')
#         stiff,gs = genstiff.gen_params(sequence,use_group=False,sparse=True)
    
#     #########################################################
#     # cgNA+, Sharma et al.
#     # https://doi.org/10.1016/j.jmb.2023.167978
     
#     if model.lower() in ['cgnaplus','cgna+','cgnap']:
        
#         if allow_partial:
#             method = cgnaplus_bps_params
#             stiffgen_args = {
#                 'translations_in_nm': True, 
#                 'euler_definition': True, 
#                 'group_split' : True,
#                 'parameter_set_name' : cgnap_setname,
#                 'remove_factor_five' : True,
#                 'rotations_only': False
#                 }
        
#             nbps = len(sequence)
#             if not closed:
#                 nbps -= 1
            
#             if overlap_size > nbps:
#                 overlap_size = nbps-1
#             if block_size > nbps:
#                 block_size = nbps
            
#             print('Generating partial stiffness matrix with')    
#             print(f'block_size:   {block_size}')
#             print(f'overlap_size: {overlap_size}')
#             print(f'tail_size:    {tail_size}')

#             gs,stiff = partial_stiff(
#                 sequence,
#                 method,
#                 stiffgen_args,
#                 block_size=block_size,
#                 overlap_size=overlap_size,
#                 tail_size=tail_size,
#                 closed=closed,
#                 ndims=6
#             )
        
#         else:
#             gs,stiff = cgnaplus_bps_params(
#                 sequence,
#                 parameter_set_name=cgnap_setname,
#                 )
    
#     params = {
#         'seq' : sequence,
#         'gs': gs,
#         'stiff' : stiff
#     }
    
#     if composite_size <= 1:
#         return params
    
#     ##################################################################################################################
#     # Coarse-grain parameters

#     block_ncomp     = int(np.ceil(block_size/composite_size))
#     overlap_ncomp   = int(np.ceil(overlap_size/composite_size)) 
#     tail_ncomp      = int(np.ceil(tail_size/composite_size)) 

#     cg_gs, cg_stiff = coarse_grain(
#         gs,
#         stiff,
#         composite_size,
#         start_id=start_id,
#         end_id=end_id,
#         closed=closed,
#         allow_partial=allow_partial,
#         block_ncomp=block_ncomp,
#         overlap_ncomp=overlap_ncomp,
#         tail_ncomp=tail_ncomp,
#         allow_crop=allow_crop,
#         use_sparse=sparse,
#         )
    
#     params['cg_gs'] = cg_gs
#     params['cg_stiff'] = cg_stiff
#     return params
        
    
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
            
    sparse = True
    cgnap_setname = 'curves_plus'
    
    params = gen_params(
        model , 
        seq,
        composite_size,
        closed=closed,
        sparse=sparse,
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
        cgvisual(visdir,params['gs'],seq,composite_size,start_id,bead_radius=composite_size*0.34*0.5,include_bps_triads=args.include_bps_triads)
    