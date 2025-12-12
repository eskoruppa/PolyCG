import sys, os
import time
import argparse
import numpy as np
import scipy as sp
import hashlib
from typing import Any, Callable, Dict, List, Tuple

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from functools import cached_property

from .cgrbp_methods.lmp_topol import CGRBPTopology

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

from .gen_params import gen_params





def gen_datafile(
    filename: str,
    topology : CGRBPTopology,
    config):
    
    with open(filename,'w') as f:
        
        # atoms
        f.write(f'{len(topology.bonds)} bonds\n')
        f.write(f'{len(topology.angles)} angles\n')
        f.write(f'{len(topology.dihedrals)} dihedrals\n')
        # atom types
        f.write(f'{len(topology.bondtypes)} bond types\n')
        f.write(f'{len(topology.angletypes)} angle types\n')
        f.write(f'{len(topology.dihedraltypes)} dihedral types\n')
        # ellipsoids
        
        f.write('\n')
        
        # xlo
        # ylo
        # zlo
        
        # masses
        
        # atoms
        
        
        if len(topology.bondtypes) > 0:
            f.write(f'\nBond Coeffs\n\n')
            for bondtype in topology.bondtypes:
                f.write(f'{bondtype.to_str()}\n')
            f.write('\n')

        if len(topology.angletypes) > 0:
            f.write(f'\nAngle Coeffs\n\n')
            for angletype in topology.angletypes:
                f.write(f'{angletype.to_str()}\n')
            f.write('\n')

        if len(topology.dihedraltypes) > 0:
            f.write(f'\nDihedral Coeffs\n\n')
            for dihedraltype in topology.dihedraltypes:
                f.write(f'{dihedraltype.to_str()}\n')
            f.write('\n')

        if len(topology.bonds) > 0:
            f.write(f'\nBonds\n\n')
            for bond in topology.bonds:
                f.write(f'{bond.to_str()}\n')
            f.write('\n')

        if len(topology.angles) > 0:
            f.write(f'\nAngles\n\n')
            for angle in topology.angles:
                f.write(f'{angle.to_str()}\n')
            f.write('\n')
            
        if len(topology.dihedrals) > 0:
            f.write(f'\nDihedrals\n\n')
            for dihedral in topology.dihedrals:
                f.write(f'{dihedral.to_str()}\n')
            f.write('\n')
            

        
    
    





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate PolyMC input files")
    parser.add_argument('-m',       '--model',              type=str, default = 'cgnaplus', choices=['cgnaplus','lankas','olson'])
    parser.add_argument('-cg',      '--composite_size',     type=int, default = 1)
    parser.add_argument('-seqfn',   '--sequence_file',      type=str, default = None)
    parser.add_argument('-seq',     '--sequence',           type=str, default = None)
    parser.add_argument('-cr',      '--coupling_range',     type=int, default = 4)
    parser.add_argument('-dec',     '--decimals',           type=int, default = 2)
    parser.add_argument('-closed',  '--closed',             action='store_true') 
    parser.add_argument('-nc',      '--no_crop',            action='store_true') 
    parser.add_argument('-np',      '--no_partial',         action='store_true') 
    parser.add_argument('-sid',     '--start_id',           type=int, default=0) 
    parser.add_argument('-eid',     '--end_id',             type=int, default=None) 
    parser.add_argument('-o',       '--output_basename',    type=str, default = None, required=False)
    parser.add_argument('-nv',      '--no_visualization',   action='store_true') 
    # parser.add_argument('-xyz',     '--gen_xyz',            action='store_true') 
    # parser.add_argument('-pdb',     '--gen_pdb',            action='store_true') 
     
    args = parser.parse_args()
    
    model           = args.model
    composite_size  = args.composite_size
    coupling_range  = args.coupling_range
    decimals        = args.decimals
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
    
    for key in params:
        print(key)
    
    print(params['gs'].shape)
    print(params['stiff'].shape)
    print(params['seq'])
    
    if 'cg_stiff' in params:
        stiff   = params['cg_stiff']
        gs      = params['cg_gs']
    else: 
        stiff   = params['stiff']
        gs      = params['gs']
        
    topol = CGRBPTopology(gs,stiff,coupling_range=coupling_range,decimals=decimals)
    
    filename = 'test.data'
    gen_datafile(filename,topol,None)