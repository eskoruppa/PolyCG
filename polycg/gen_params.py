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
    
    parser = argparse.ArgumentParser(
        description="Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA. "
                    "Generate ground state configuration and stiffness matrix for rigid base pair (RBP) models at any resolution. "
                    "Supported base pair step stiffness libraries: cgNA+ (Sharma et al. 2023), MD (Lankas et al. 2003), and Crystal (Olson et al. 1998). "
                    "Implementation of the method described in Skoruppa & Schiessel, Phys. Rev. Research 7, 013044 (2025).",
        epilog="Example usage (using cgNA+ model by default):\n"
               "  Basic generation:         python -m polycg.gen_params -seqfn Examples/1kbp\n"
               "  Coarse-grained:           python -m polycg.gen_params -seqfn Examples/200bp -cg 5\n"
               "  Closed (circular):        python -m polycg.gen_params -seqfn Examples/40bp -cg 10 -closed\n"
               "  With visualization:       python -m polycg.gen_params -seqfn Examples/40bp -cg 5 -pdb -vis -bpst\n"
               "  MD parameters:            python -m polycg.gen_params -seqfn Examples/1kbp -m md\n"
               "  Crystal parameters:       python -m polycg.gen_params -seqfn Examples/1kbp -m crystall\n"
               "  Direct sequence:          python -m polycg.gen_params -seq ATCGATCG -cg 1\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-m',       '--model',              type=str, default = 'cgnaplus', choices=['cgnaplus','md','crystall'],
                        help='DNA model for parameter generation (default: cgnaplus)')
    parser.add_argument('-cg',      '--composite_size',     type=int, default = 1,
                        help='Number of base pairs per coarse-grained bead (default: 1 for all-atom)')
    parser.add_argument('-seqfn',   '--sequence_file',      type=str, default = None,
                        help='Path to DNA sequence file (.seq extension)')
    parser.add_argument('-seq',     '--sequence',           type=str, default = None,
                        help='DNA sequence as string (alternative to -seqfn)')
    parser.add_argument('-closed',  '--closed',             action='store_true',
                        help='Generate closed (circular) DNA configuration') 
    parser.add_argument('-nc',      '--no_crop',            action='store_true',
                        help='Disable automatic sequence cropping for boundary conditions') 
    parser.add_argument('-np',      '--no_partial',         action='store_true',
                        help='Disable partial block assembly for stiffness matrix') 
    parser.add_argument('-sid',     '--start_id',           type=int, default=0,
                        help='Starting base pair index for subsection generation (default: 0)') 
    parser.add_argument('-eid',     '--end_id',             type=int, default=None,
                        help='Ending base pair index for subsection generation (default: full sequence)') 
    parser.add_argument('-o',       '--output_basename',    type=str, default = None, required=False,
                        help='Base filename for output files (default: derived from sequence file)')
    parser.add_argument('-xyz',     '--gen_xyz',            action='store_true',
                        help='Generate XYZ coordinate file')
    parser.add_argument('-pdb',     '--gen_pdb',            action='store_true',
                        help='Generate PDB structure file')
    parser.add_argument('-vis',     '--visualize_cgrbp',    action='store_true',
                        help='Generate ChimeraX visualization script (.cxc)')
    parser.add_argument('-bpst',    '--include_bps_triads', action='store_true',
                        help='Include base pair step triads in visualization (requires -vis)') 
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