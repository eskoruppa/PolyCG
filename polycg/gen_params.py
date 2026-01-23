from __future__ import annotations

import sys, os
import argparse
import numpy as np
import scipy as sp
from pathlib import Path

# load sequence from sequence file
from .utils.load_seq import load_sequence
# write sequence file
from .utils.seq import write_seqfile
# load gen_params
from ._gen_params import gen_params
# load visualization methods
from .out.visualization import visualize_chimerax, visualize_pdb, visualize_xyz

    
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
    parser.add_argument('-xyz',     '--gen_xyz',      action='store_true')
    parser.add_argument('-pdb',     '--gen_pdb',      action='store_true')
    parser.add_argument('-vis',     '--visualize_cgrbp',    action='store_true')
    parser.add_argument('-bpst',    '--include_bps_triads', action='store_true') 
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
        model, 
        seq,
        composite_size,
        closed=closed,
        start_id=start_id,
        end_id=end_id,
        allow_partial=allow_partial,
        allow_crop=allow_crop,
        cgnap_setname = cgnap_setname,
        verbose=True,
        print_info=False,
    )
    
    # print(f"\nParameter shapes:")
    # print(f"  shape_params: {params.shape_params.shape}")
    # print(f"  stiffmat: {params.stiffmat.shape}")
    # if params.cg_shape_params is not None:
    #     print(f"  cg_shape_params: {params.cg_shape_params.shape}")
    # if params.cg_stiffmat is not None:
    #     print(f"  cg_stiffmat: {params.cg_stiffmat.shape}")
    
    if args.output_basename is None:
        base_fn = Path(seqfn)
    else:
        base_fn = Path(args.output_basename)
    
    if composite_size > 1:
        params.save_cg_coeffs(base_fn.with_name(base_fn.stem + '_closed') if closed else base_fn)
    else:
        params.save_coeffs(base_fn.with_name(base_fn.stem + '_closed') if closed else base_fn)
    
    # write sequence file
    seqfn = base_fn.with_suffix('.seq')
    write_seqfile(seqfn,params.sequence,add_extension=True)
    
    # visualization cgrbp
    if args.visualize_cgrbp:
        vis_seq = params.sequence
        if closed:
            vis_seq += seq[0]
        if composite_size > 1:
            bead_radius = composite_size*0.34*0.5
        else:
            bead_radius = 0
        visualize_chimerax(base_fn, vis_seq, composite_size, shape_params=params.shape_params, start_id=start_id, bead_radius=bead_radius,include_bps_triads=args.include_bps_triads) 
        
    if args.gen_pdb and not args.visualize_cgrbp:
        vis_seq = params.sequence
        if closed:
            vis_seq += seq[0]
        visualize_pdb(base_fn, vis_seq, shape_params=params.shape_params)
        
    if args.gen_xyz:
        visualize_xyz(base_fn, composite_size, shape_params=params.shape_params, start_id=start_id)