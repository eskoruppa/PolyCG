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
# load so3 
from .SO3 import so3


def gen_params(
    model: str, 
    sequence: str,
    composite_size: int = 1,
    closed: bool = False,
    sparse: bool = True,
    start_id: int = 0,
    end_id:   int = None,
    allow_partial: bool = True,
    block_size: int = 120,
    overlap_size: int = 20,
    tail_size: int = 20,
    allow_crop: bool = True,
    cgnap_setname: str = 'curves_plus'
    ) -> Dict:

    ##################################################################################################################
    # Generate Stiffness and groundstate

    #########################################################
    # Crystal structure data from Olson et al. 1998
    # https://www.pnas.org/doi/full/10.1073/pnas.95.19.11163

    if model.lower() in ['crystal','cry','olson']:
        genstiff = GenStiffness(method='crystal')
        params = genstiff.gen_params(sequence,use_group=False,sparse=True)
        stiff,gs = params['stiffness'], params['groundstate']
    
    #########################################################
    # MD data from Lankas et al. 2003
    # https://doi.org/10.1016/S0006-3495(03)74710-9
    
    if model.lower() in ['md','lankas']:
        genstiff = GenStiffness(method='md')
        params = genstiff.gen_params(sequence,use_group=False,sparse=True)
        stiff,gs = params['stiffness'], params['groundstate']
    
    #########################################################
    # cgNA+, Sharma et al.
    # https://doi.org/10.1016/j.jmb.2023.167978
     
    if model.lower() in ['cgnaplus','cgna+','cgnap']:
        
        if allow_partial:
            method = cgnaplus_bps_params
            stiffgen_args = {
                'translations_in_nm': True, 
                'euler_definition': True, 
                'group_split' : True,
                'parameter_set_name' : cgnap_setname,
                'remove_factor_five' : True,
                'rotations_only': False
                }
        
            nbps = len(sequence)
            if not closed:
                nbps -= 1
            
            if overlap_size > nbps:
                overlap_size = nbps-1
            if block_size > nbps:
                block_size = nbps
            
            print('Generating partial stiffness matrix with')    
            print(f'block_size:   {block_size}')
            print(f'overlap_size: {overlap_size}')
            print(f'tail_size:    {tail_size}')

            gs,stiff = partial_stiff(
                sequence,
                method,
                stiffgen_args,
                block_size=block_size,
                overlap_size=overlap_size,
                tail_size=tail_size,
                closed=closed,
                ndims=6
            )
        
        else:
            gs,stiff = cgnaplus_bps_params(
                sequence,
                parameter_set_name=cgnap_setname,
                translations_in_nm=True,
                euler_definition=True,
                group_split=True,
                remove_factor_five=True,
                rotations_only=False
                )
    
    params = {
        'seq' : sequence,
        'gs': gs,
        'stiff' : stiff
    }
    
    if composite_size <= 1:
        return params
    
    ##################################################################################################################
    # Coarse-grain parameters

    block_ncomp     = int(np.ceil(block_size/composite_size))
    overlap_ncomp   = int(np.ceil(overlap_size/composite_size)) 
    tail_ncomp      = int(np.ceil(tail_size/composite_size)) 

    cg_gs, cg_stiff = coarse_grain(
        gs,
        stiff,
        composite_size,
        start_id=start_id,
        end_id=end_id,
        closed=closed,
        allow_partial=allow_partial,
        block_ncomp=block_ncomp,
        overlap_ncomp=overlap_ncomp,
        tail_ncomp=tail_ncomp,
        allow_crop=allow_crop,
        use_sparse=sparse,
        )
    
    params['cg_gs'] = cg_gs
    params['cg_stiff'] = cg_stiff
    return params
        
    
##################################################################################################################
##################################################################################################################
##################################################################################################################

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
