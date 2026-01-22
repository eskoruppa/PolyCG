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
# load BlockOverlapMatrix
from .utils.bmat import BlockOverlapMatrix, crop_periodic_fold_fill_zeros


_GEN_PARAMS_CGNAPLUS_ROT_IN_NM    = True
_GEN_PARAMS_CGNAPLUS_EULER_DEF    = True
_GEN_PARAMS_CGNAPLUS_GROUP_SPLIT  = True
_GEN_PARAMS_CGNAPLUS_REMOVE_FAC_5 = True
_GEN_PARAMS_CGNAPLUS_ROT_ONLY     = False
_GEN_PARAMS_NDIMS = 6

_GEN_PARAMS_OLSON_IDENTIFIERS =     ['crystal','cry','olson']
_GEN_PARAMS_LANKAS_IDENTIFIERS =    ['md','lankas']
_GEN_PARAMS_CGNAP_IDENTIFIERS =     ['cgnaplus','cgna+','cgnap']

_GEN_PARAMS_ACCEPTED_IDENTIFIERS = _GEN_PARAMS_OLSON_IDENTIFIERS + \
    _GEN_PARAMS_LANKAS_IDENTIFIERS + _GEN_PARAMS_CGNAP_IDENTIFIERS

_GEN_PARAMS_MAX_COMPOSITE_SIZE = 40


def gen_params(
    model: str, 
    sequence: str,
    composite_size: int = 1,
    closed: bool = False,
    start_id: int = 0,
    end_id:   int = None,
    allow_partial: bool = True,
    block_size: int = 120,
    overlap_size: int = 20,
    tail_size: int = 20,
    allow_crop: bool = False,
    cgnap_setname: str = 'curves_plus',
    ) -> dict[str]:
    """
    Generate sequence-dependent ground-state and stiffness parameters, with optional coarse-graining.

    This function generates a base-pair(-step) ground-state vector and stiffness matrix for the
    provided DNA sequence using one of the supported models. For `composite_size > 1`, it also
    computes coarse-grained parameters by aggregating consecutive base-pair steps into composite
    units.

    Model handling
    --------------
    The `model` argument selects the underlying parameter source:
    - Olson et al. (crystal) and Lankas et al. (MD) are treated as local models and are generated
      directly through `GenStiffness(...).gen_params(..., sparse=True)`.
    - cgNA+ is treated as a non-local model. When `allow_partial=True`, parameters are generated
      via `partial_stiff(...)` (segment-wise assembly with overlaps and tails) and then assembled
      into a sparse matrix.

    Topology handling
    -----------------
    - If `closed=False`, parameters are generated for a linear chain (N = len(sequence) - 1 steps).
      Coarse-graining supports selecting a subrange of composites via `start_id` and `end_id`.
    - If `closed=True`, parameters are generated for a closed topology (cyclic chain). Closed
      topology requires `len(sequence)` to be a multiple of `composite_size`, requires `start_id=0`,
      and does not support `end_id`. For cgNA+ with `composite_size > 1`, the sequence is extended,
      parameters are generated in open mode, coarse-grained, cropped to the target ring segment, and
      the stiffness is folded back into the fundamental domain using periodic wrapping.

    Overlap and tail parameters
    ---------------------------
    `block_size`, `overlap_size`, and `tail_size` control segment-wise generation in `partial_stiff`.
    The segment stride is `block_size - overlap_size`. `tail_size` adds extra bases to each segment
    and is removed after generation to reduce boundary artifacts.

    Parameters
    ----------
    model : str
        Identifier selecting the parameter model. Must be one of `_GEN_PARAMS_ACCEPTED_IDENTIFIERS`.
    sequence : str
        DNA sequence used for parameter generation.
    composite_size : int, default 1
        Number of consecutive steps grouped into one composite unit during coarse-graining. A value
        of 1 disables coarse-graining.
    closed : bool, default False
        If True, generate parameters for a closed (cyclic) topology. If False, generate parameters
        for a linear topology.
    start_id : int, default 0
        Start composite index for coarse-graining selection in linear topology.
    end_id : int, optional
        End composite index (exclusive) for coarse-graining selection in linear topology. Not
        supported for closed topology.
    allow_partial : bool, default True
        If True, allows segment-wise generation via `partial_stiff` for models that support it
        (notably cgNA+). Required for cgNA+ in closed topology.
    block_size : int, default 120
        Segment size (in base-pair steps) used by `partial_stiff`.
    overlap_size : int, default 20
        Segment overlap (in base-pair steps) used by `partial_stiff`. The effective overlap is
        increased to at least `composite_size` when coarse-graining is requested.
    tail_size : int, default 20
        Number of additional bases included on each side of a segment in `partial_stiff`.
    allow_crop : bool, default False
        Forwarded to `coarse_grain` to control whether cropped coarse-grained matrices are allowed.
    cgnap_setname : str, default "curves_plus"
        cgNA+ parameter set name forwarded to `cgnaplus_bps_params`.

    Returns
    -------
    dict
        Dictionary containing generated parameters. Keys depend on `composite_size` and `closed`:

        Always present:
        - 'seq'    : input sequence
        - 'closed' : topology flag
        - 'gs'     : ground-state (shape depends on model; typically (N, ndims))
        - 'stiff'  : stiffness matrix (SciPy sparse matrix)

        Present when `composite_size > 1`:
        - 'cg_gs'    : coarse-grained ground-state
        - 'cg_stiff' : coarse-grained stiffness (SciPy sparse matrix)

        For closed cgNA+ with coarse-graining, the dictionary includes 'cg_gs' and 'cg_stiff' and
        may omit 'stiff' depending on the selected path.

    Raises
    ------
    ValueError
        If `model` is not recognized, if `block_size <= overlap_size`, if `start_id > end_id`,
        if `composite_size` exceeds `_GEN_PARAMS_MAX_COMPOSITE_SIZE`, or if closed-topology
        constraints are violated (length compatibility, start/end restrictions, or missing
        `allow_partial` for cgNA+).
    """    

    
    if model.lower() not in _GEN_PARAMS_ACCEPTED_IDENTIFIERS:
        ident_str =  '",\n "'.join(_GEN_PARAMS_ACCEPTED_IDENTIFIERS)
        raise ValueError(f'Unknown model identifier. Needs to be one of "{ident_str}"')
    
    if composite_size > _GEN_PARAMS_MAX_COMPOSITE_SIZE:
        raise ValueError(f'Composites larger than {_GEN_PARAMS_MAX_COMPOSITE_SIZE} are currently no supported. Recommended size <= 10!')
    
    if overlap_size < composite_size:
        overlap_size = composite_size
    
    if block_size <= overlap_size:
        raise ValueError(f'block_size ({block_size}) needs to be larger than overlap_size ({overlap_size}).')
    
    if end_id is not None:
        if start_id > end_id:
            raise ValueError(f'Invalid id range start_id > end_id ({start_id} > {end_id})')
    
    if composite_size <= 1:
        print('RAISE EXCEPTION!')
        
    if not closed:

        ##################################################################################################################
        # Generate Stiffness and groundstate

        #########################################################
        # Crystal structure data from Olson et al. 1998
        # https://www.pnas.org/doi/full/10.1073/pnas.95.19.11163

        if model.lower() in _GEN_PARAMS_OLSON_IDENTIFIERS:
            genstiff = GenStiffness(method='crystal')
            params = genstiff.gen_params(sequence,use_group=False,sparse=True)
            stiff,gs = params['stiffness'], params['groundstate']
        
        #########################################################
        # MD data from Lankas et al. 2003
        # https://doi.org/10.1016/S0006-3495(03)74710-9
        
        if model.lower() in _GEN_PARAMS_LANKAS_IDENTIFIERS:
            genstiff = GenStiffness(method='md')
            params = genstiff.gen_params(sequence,use_group=False,sparse=True)
            stiff,gs = params['stiffness'], params['groundstate']
        
        #########################################################
        # cgNA+, Sharma et al.
        # https://doi.org/10.1016/j.jmb.2023.167978
        
        if model.lower() in _GEN_PARAMS_CGNAP_IDENTIFIERS:
            
            if allow_partial:
                method = cgnaplus_bps_params
                stiffgen_args = {
                    'translations_in_nm':   _GEN_PARAMS_CGNAPLUS_ROT_IN_NM, 
                    'euler_definition':     _GEN_PARAMS_CGNAPLUS_EULER_DEF, 
                    'group_split' :         _GEN_PARAMS_CGNAPLUS_GROUP_SPLIT,
                    'parameter_set_name' :  cgnap_setname,
                    'remove_factor_five' :  _GEN_PARAMS_CGNAPLUS_REMOVE_FAC_5,
                    'rotations_only':       _GEN_PARAMS_CGNAPLUS_ROT_ONLY
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

                gs,bmat_stiff = partial_stiff(
                    sequence,
                    method,
                    stiffgen_args,
                    block_size=block_size,
                    overlap_size=overlap_size,
                    tail_size=tail_size,
                    closed=closed,
                    ndims=_GEN_PARAMS_NDIMS
                )
                stiff = bmat_stiff.to_sparse()
            
            else:
                gs,stiff = cgnaplus_bps_params(
                    sequence,
                    parameter_set_name=cgnap_setname,
                    translations_in_nm= _GEN_PARAMS_CGNAPLUS_ROT_IN_NM,
                    euler_definition=   _GEN_PARAMS_CGNAPLUS_EULER_DEF,
                    group_split=        _GEN_PARAMS_CGNAPLUS_GROUP_SPLIT,
                    remove_factor_five= _GEN_PARAMS_CGNAPLUS_REMOVE_FAC_5,
                    rotations_only=     _GEN_PARAMS_CGNAPLUS_ROT_ONLY
                    )
        
        params = {
            'seq' : sequence,
            'closed' : closed,
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
            allow_partial=allow_partial,
            block_ncomp=block_ncomp,
            overlap_ncomp=overlap_ncomp,
            tail_ncomp=tail_ncomp,
            allow_crop=allow_crop,
            use_sparse=True,
            )
        
        if isinstance(cg_stiff, BlockOverlapMatrix):
            cg_stiff = cg_stiff.to_sparse()   
        params['cg_gs'] = cg_gs
        params['cg_stiff'] = cg_stiff
        return params
    
    else:
        
        nbps = len(sequence)
        if nbps % composite_size != 0:
            raise ValueError(f'For closed topology the length of the sequence ({nbps}) must be a multiple of the composite size ({composite_size}).')
        if start_id != 0:
            raise ValueError(f'Closed topology requires start_id to be set to 0.')
        if end_id is not None:
            raise ValueError(f'Closed topology does not support setting and end_id.')
        
        ##################################################################################################################
        ##################################################################################################################
        ##################################################################################################################
        # Local models
        if model.lower() in _GEN_PARAMS_OLSON_IDENTIFIERS + _GEN_PARAMS_LANKAS_IDENTIFIERS: 

            ##################################################################################################################
            # Generate Stiffness and groundstate

            #########################################################
            # Crystal structure data from Olson et al. 1998
            # https://www.pnas.org/doi/full/10.1073/pnas.95.19.11163

            if model.lower() in _GEN_PARAMS_OLSON_IDENTIFIERS:
                genstiff = GenStiffness(method='crystal')
                ext_seq = sequence + sequence[0]
                params = genstiff.gen_params(ext_seq,use_group=False,sparse=True)
                stiff,gs = params['stiffness'], params['groundstate']
            
            #########################################################
            # MD data from Lankas et al. 2003
            # https://doi.org/10.1016/S0006-3495(03)74710-9
            
            if model.lower() in _GEN_PARAMS_LANKAS_IDENTIFIERS:
                genstiff = GenStiffness(method='md')
                ext_seq = sequence + sequence[0]
                params = genstiff.gen_params(ext_seq,use_group=False,sparse=True)
                stiff,gs = params['stiffness'], params['groundstate']
                
            params = {
                'closed': True,
                'seq' : sequence,
                'gs': gs,
                'stiff' : stiff
            }
        
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
                allow_partial=allow_partial,
                block_ncomp=block_ncomp,
                overlap_ncomp=overlap_ncomp,
                tail_ncomp=tail_ncomp,
                allow_crop=False,
                use_sparse=True,
                )
        
            if isinstance(cg_stiff, BlockOverlapMatrix):
                cg_stiff = cg_stiff.to_sparse()
            params['cg_gs'] = cg_gs
            params['cg_stiff'] = cg_stiff
            return params
                

        
        ##################################################################################################################
        ##################################################################################################################
        ##################################################################################################################
        # Non-local models

        #########################################################
        # cgNA+, Sharma et al.
        # https://doi.org/10.1016/j.jmb.2023.167978
        
        if model.lower() in _GEN_PARAMS_CGNAP_IDENTIFIERS:
            
            if not allow_partial: 
                raise ValueError('Generation of closed topology for cgNA+ requires allow_partial to account for wrapped couplings')
            
            method = cgnaplus_bps_params
            stiffgen_args = {
                'translations_in_nm':   _GEN_PARAMS_CGNAPLUS_ROT_IN_NM, 
                'euler_definition':     _GEN_PARAMS_CGNAPLUS_EULER_DEF, 
                'group_split' :         _GEN_PARAMS_CGNAPLUS_GROUP_SPLIT,
                'parameter_set_name' :  cgnap_setname,
                'remove_factor_five' :  _GEN_PARAMS_CGNAPLUS_REMOVE_FAC_5,
                'rotations_only':       _GEN_PARAMS_CGNAPLUS_ROT_ONLY
                }
            
            if composite_size <= 1:
    
                if overlap_size > nbps:
                    overlap_size = nbps-1
                if block_size > nbps:
                    block_size = nbps
                    
                print('Generating partial stiffness matrix with')    
                print(f'block_size:   {block_size}')
                print(f'overlap_size: {overlap_size}')
                print(f'tail_size:    {tail_size}')

                gs,bmat_stiff = partial_stiff(
                    sequence,
                    method,
                    stiffgen_args,
                    block_size=block_size,
                    overlap_size=overlap_size,
                    tail_size=tail_size,
                    closed=closed,
                    ndims=_GEN_PARAMS_NDIMS
                )
                stiff = bmat_stiff.to_sparse()
                
                params = {
                    'closed': True,
                    'seq' : sequence,
                    'gs': gs,
                    'stiff' : stiff
                }
                return params
            
            
            ##################################################################
            # extend the sequence in both directions, 
            # generate parameters, 
            # coarse grain
            # crop left to start
            # crop right to 1 overlap_size // composite_size beyond target
            # resize by wrapping periodically
            
            overlap_size = int(np.ceil(overlap_size / composite_size)) * composite_size
            if overlap_size > nbps:
                overlap_size = nbps-composite_size
            if block_size > nbps:
                block_size = nbps
            
            # extend sequence
            n_right_overlap_extends = 2
            extended_seq = sequence[-overlap_size:] + \
                sequence + (''.join(sequence for i in range(n_right_overlap_extends)))[:n_right_overlap_extends*overlap_size+1]
            
            print('Generating partial stiffness matrix with')    
            print(f'block_size:   {block_size}')
            print(f'overlap_size: {overlap_size}')
            print(f'tail_size:    {tail_size}')

            ext_gs,bmat_ext_stiff = partial_stiff(
                extended_seq,
                method,
                stiffgen_args,
                block_size=block_size,
                overlap_size=overlap_size,
                tail_size=tail_size,
                closed=False,
                ndims=_GEN_PARAMS_NDIMS
            )
            

            ######################################
            # Coarse-grain parameters

            block_ncomp     = int(np.ceil(block_size/composite_size))
            overlap_ncomp   = int(np.ceil(overlap_size/composite_size)) 
            tail_ncomp      = int(np.ceil(tail_size/composite_size)) 

            if bmat_ext_stiff.shape[0] % composite_size != 0:
                raise ValueError('Incompatible matrix size. Should be multiple of composite size.')
            
            allow_partial = True
            cg_gs_ext, bmat_cg_stiff_ext = coarse_grain(
                ext_gs,
                bmat_ext_stiff,
                composite_size,
                start_id=start_id,
                end_id=end_id,
                allow_partial=allow_partial,
                block_ncomp=block_ncomp,
                overlap_ncomp=overlap_ncomp,
                tail_ncomp=tail_ncomp,
                allow_crop=False,
                use_sparse=True,
                )
            
            cg_gs = cg_gs_ext[overlap_ncomp:-n_right_overlap_extends*overlap_ncomp]
            cg_nbps = nbps // composite_size
            
            if isinstance(bmat_cg_stiff_ext,BlockOverlapMatrix):
                cg_stiff_ext = bmat_cg_stiff_ext.to_sparse(xlo=overlap_ncomp*_GEN_PARAMS_NDIMS, xhi = (2*overlap_ncomp+cg_nbps)*_GEN_PARAMS_NDIMS,
                                                           ylo=overlap_ncomp*_GEN_PARAMS_NDIMS, yhi = (2*overlap_ncomp+cg_nbps)*_GEN_PARAMS_NDIMS,)  
            else:
                cg_stiff_ext = bmat_cg_stiff_ext[overlap_ncomp*_GEN_PARAMS_NDIMS:(2*overlap_ncomp+cg_nbps)*_GEN_PARAMS_NDIMS,
                                                 overlap_ncomp*_GEN_PARAMS_NDIMS:(2*overlap_ncomp+cg_nbps)*_GEN_PARAMS_NDIMS]
            
            cg_stiff = crop_periodic_fold_fill_zeros(cg_stiff_ext,cg_nbps*_GEN_PARAMS_NDIMS,cg_nbps*_GEN_PARAMS_NDIMS)
            params = {
                'closed': True,
                'seq' : sequence,
                'gs' : ext_gs[overlap_size:-n_right_overlap_extends*overlap_size],
                'cg_gs': cg_gs,
                'cg_stiff' : cg_stiff
            }
            return params

    
if __name__ == '__main__':
    
    N = 100
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(N)])
    print(seq)
    
    model = 'cgna+'
    # model = 'MD'
    cg = 10
    closed = True
    allow_crop = False
    gen_params(model,seq,composite_size=cg,closed=closed,allow_crop=allow_crop)
     