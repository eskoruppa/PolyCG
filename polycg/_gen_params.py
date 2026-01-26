from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import numpy as np
import scipy as sp

from .cg import coarse_grain
from .cgnaplus import cgnaplus_bps_params
from .models.RBPStiff.read_params import GenStiffness
from .partials import partial_stiff
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



@dataclass
class DNAParameters:
    """Container for DNA model parameters with full metadata."""
    
    # Core data - always present
    sequence: str
    model: str
    shape_params: np.ndarray  # ground state
    stiffmat: sp.sparse.spmatrix  # stiffness matrix
    
    # Topology
    closed: bool = False
    composite_size: int = 1
    
    # Coarse-grained data - optional
    cg_shape_params: np.ndarray | None = None
    cg_stiffmat: sp.sparse.spmatrix | None = None
    
    # Metadata for reconstruction/visualization
    start_id: int = 0
    end_id: int | None = None
    disc_len: float = 0.34  # nm, for visualization
    
    # Model-specific parameters
    metadata: dict = field(default_factory=dict)
    
    @property
    def n_bps(self) -> int:
        """Number of base pairs."""
        return len(self.sequence)
    
    @property
    def n_steps(self) -> int:
        """Number of base-pair steps."""
        return self.n_bps - (0 if self.closed else 1)
    
    @property
    def is_coarse_grained(self) -> bool:
        """Check if coarse-grained parameters are available."""
        return self.composite_size > 1 and self.cg_shape_params is not None
    
    def __post_init__(self):
        """Validate dataclass fields after initialization."""
        # Validate shape_params dimensions
        if self.shape_params.ndim != 2:
            raise ValueError(
                f"shape_params must be 2D array, got shape {self.shape_params.shape}"
            )
        
        n_steps, ndims = self.shape_params.shape
        if ndims != _GEN_PARAMS_NDIMS:
            raise ValueError(
                f"shape_params must have {_GEN_PARAMS_NDIMS} columns, got {ndims}"
            )
        
        # Validate stiffmat dimensions match shape_params
        expected_size = n_steps * _GEN_PARAMS_NDIMS
        if self.stiffmat.shape != (expected_size, expected_size):
            raise ValueError(
                f"stiffmat shape {self.stiffmat.shape} incompatible with "
                f"shape_params {self.shape_params.shape}. Expected ({expected_size}, {expected_size})"
            )
        
        # Validate coarse-grained parameters if present
        if self.cg_shape_params is not None:
            if self.cg_shape_params.ndim != 2:
                raise ValueError(
                    f"cg_shape_params must be 2D array, got shape {self.cg_shape_params.shape}"
                )
            
            cg_n_steps, cg_ndims = self.cg_shape_params.shape
            if cg_ndims != _GEN_PARAMS_NDIMS:
                raise ValueError(
                    f"cg_shape_params must have {_GEN_PARAMS_NDIMS} columns, got {cg_ndims}"
                )
            
            if self.cg_stiffmat is not None:
                expected_cg_size = cg_n_steps * _GEN_PARAMS_NDIMS
                if self.cg_stiffmat.shape != (expected_cg_size, expected_cg_size):
                    raise ValueError(
                        f"cg_stiffmat shape {self.cg_stiffmat.shape} incompatible with "
                        f"cg_shape_params {self.cg_shape_params.shape}. "
                        f"Expected ({expected_cg_size}, {expected_cg_size})"
                    )
    
    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        cg_info = ""
        if self.is_coarse_grained:
            cg_info = f", CG: {self.cg_shape_params.shape[0]} composites"
        
        topology = "closed" if self.closed else "open"
        
        return (
            f"DNAParameters(model='{self.model}', "
            f"seq_len={len(self.sequence)}, "
            f"n_steps={self.n_steps}, "
            f"topology={topology}, "
            f"composite_size={self.composite_size}"
            f"{cg_info})"
        )
    
    def get_params(self, coarse_grained: bool = False) -> tuple[np.ndarray, sp.sparse.spmatrix]:
        """
        Get shape parameters and stiffness matrix.
        
        Parameters
        ----------
        coarse_grained : bool, default=False
            If True, return coarse-grained parameters. If False, return base-level parameters.
        
        Returns
        -------
        shape_params : np.ndarray
            Shape parameters array.
        stiffmat : sp.sparse.spmatrix
            Stiffness matrix.
        
        Raises
        ------
        ValueError
            If coarse_grained=True but coarse-grained parameters are not available.
        """
        if coarse_grained:
            if not self.is_coarse_grained:
                raise ValueError(
                    f"Coarse-grained parameters not available. "
                    f"composite_size={self.composite_size}, "
                    f"cg_shape_params={'available' if self.cg_shape_params is not None else 'None'}"
                )
            return self.cg_shape_params, self.cg_stiffmat
        else:
            return self.shape_params, self.stiffmat
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result = {
            'seq': self.sequence,
            'closed': self.closed,
            'shape_params': self.shape_params,
            'stiffmat': self.stiffmat
        }
        if self.is_coarse_grained:
            result['cg_shape_params'] = self.cg_shape_params
            result['cg_stiffmat'] = self.cg_stiffmat
        return result
    
    def save_cg_coeffs(self, base_fn: str | Path) -> None:
        """Save shape parameters to a .npy file."""
        base_fn = Path(base_fn)
        cg_fn = base_fn.with_name(base_fn.stem + f'_cg{self.composite_size}')
        fn_gs = cg_fn.with_name(cg_fn.stem + '_gs.npy')
        fn_stiff = cg_fn.with_name(cg_fn.stem + '_stiff.npz')
        
        if self.cg_stiffmat is not None:
            sp.sparse.save_npz(fn_stiff,self.cg_stiffmat)
        if self.cg_shape_params is not None:
            np.save(fn_gs,self.cg_shape_params)
        else:
            raise ValueError("No coarse-grained shape parameters to save.")
    
    def save_coeffs(self, base_fn: str | Path) -> None:
        """Save shape parameters to a .npy file."""
        base_fn = Path(base_fn)
        fn_gs = base_fn.with_name(base_fn.stem + '_gs.npy')
        fn_stiff = base_fn.with_name(base_fn.stem + '_stiff.npz')
        
        if self.stiffmat is not None:
            sp.sparse.save_npz(fn_stiff,self.stiffmat)
        if self.shape_params is not None:
            np.save(fn_gs,self.shape_params)
        else:
            raise ValueError("No shape parameters to save.")



    cg_shape_params: np.ndarray | None = None
    cg_stiffmat: sp.sparse.spmatrix | None = None



def _build_cgnaplus_args(cgnap_setname: str) -> dict[str, Any]:
    """
    Build cgNA+ stiffness generation arguments dictionary.
    
    Parameters
    ----------
    cgnap_setname : str
        Parameter set name for cgNA+ model.
    
    Returns
    -------
    dict
        Dictionary of arguments for cgNA+ parameter generation.
    """
    return {
        'translations_in_nm': _GEN_PARAMS_CGNAPLUS_ROT_IN_NM,
        'euler_definition': _GEN_PARAMS_CGNAPLUS_EULER_DEF,
        'group_split': _GEN_PARAMS_CGNAPLUS_GROUP_SPLIT,
        'parameter_set_name': cgnap_setname,
        'remove_factor_five': _GEN_PARAMS_CGNAPLUS_REMOVE_FAC_5,
        'rotations_only': _GEN_PARAMS_CGNAPLUS_ROT_ONLY
    }


def _calculate_coarse_grain_params(
    block_size: int,
    overlap_size: int,
    tail_size: int,
    composite_size: int
) -> tuple[int, int, int]:
    """
    Calculate coarse-graining parameters for block-overlap assembly.
    
    Parameters
    ----------
    block_size : int
        Block size in base-pair steps.
    overlap_size : int
        Overlap size in base-pair steps.
    tail_size : int
        Tail size in base-pair steps.
    composite_size : int
        Composite size for coarse-graining.
    
    Returns
    -------
    block_ncomp : int
        Number of composites per block.
    overlap_ncomp : int
        Number of composites in overlap region.
    tail_ncomp : int
        Number of composites in tail region.
    """
    block_ncomp = int(np.ceil(block_size / composite_size))
    overlap_ncomp = int(np.ceil(overlap_size / composite_size))
    tail_ncomp = int(np.ceil(tail_size / composite_size))
    return block_ncomp, overlap_ncomp, tail_ncomp


def _apply_sequence_range(
    gs: np.ndarray,
    stiff: sp.sparse.spmatrix,
    sequence: str,
    start_id: int,
    end_id: int | None
) -> tuple[np.ndarray, sp.sparse.spmatrix, str]:
    """
    Apply start_id and end_id range selection to parameters and sequence.
    
    Parameters
    ----------
    gs : np.ndarray
        Shape parameters array.
    stiff : sp.sparse.spmatrix
        Stiffness matrix.
    sequence : str
        DNA sequence.
    start_id : int
        Starting index.
    end_id : int or None
        Ending index (exclusive).
    
    Returns
    -------
    gs : np.ndarray
        Cropped shape parameters.
    stiff : sp.sparse.spmatrix
        Cropped stiffness matrix.
    sequence : str
        Cropped sequence.
    """
    if end_id is not None:
        gs = gs[:end_id]
        stiff = stiff[:end_id * _GEN_PARAMS_NDIMS, :end_id * _GEN_PARAMS_NDIMS]
        sequence = sequence[:end_id + 1]
    if start_id != 0:
        gs = gs[start_id:]
        stiff = stiff[start_id * _GEN_PARAMS_NDIMS:, start_id * _GEN_PARAMS_NDIMS:]
        sequence = sequence[start_id:]
    return gs, stiff, sequence


def _log_partial_stiff_params(block_size: int, overlap_size: int, tail_size: int, print_info: bool = True, verbose: bool = True) -> None:
    """
    Log parameters for partial stiffness matrix generation.
    
    Parameters
    ----------
    block_size : int
        Block size in base-pair steps.
    overlap_size : int
        Overlap size in base-pair steps.
    tail_size : int
        Tail size in base-pair steps.
    print_info : bool, default=True
        If True, print parameter information in this file.
    verbose : bool, default=True
        Master verbose flag. If False, nothing is printed.
    """
    if verbose and print_info:
        print('Generating partial stiffness matrix with')
        print(f'block_size:   {block_size}')
        print(f'overlap_size: {overlap_size}')
        print(f'tail_size:    {tail_size}')


def _generate_local_model_params(
    model: str,
    sequence: str,
    closed: bool = False
) -> tuple[np.ndarray, sp.sparse.spmatrix]:
    """
    Generate parameters for local models (Olson or Lankas).
    
    Parameters
    ----------
    model : str
        Model identifier ('olson', 'crystal', 'cry', 'lankas', 'md').
    sequence : str
        DNA sequence. For closed topology, first base is appended automatically.
    closed : bool, default=False
        If True, extends sequence for closed topology by appending first base.
    
    Returns
    -------
    gs : np.ndarray
        Ground state parameters.
    stiff : sp.sparse.spmatrix
        Stiffness matrix.
    """
    # Determine method based on model identifier
    if model.lower() in _GEN_PARAMS_OLSON_IDENTIFIERS:
        method = 'crystal'
    elif model.lower() in _GEN_PARAMS_LANKAS_IDENTIFIERS:
        method = 'md'
    else:
        raise ValueError(f'Model {model} is not a local model.')
    
    # Extend sequence for closed topology
    seq = sequence + sequence[0] if closed else sequence
    
    # Generate parameters
    genstiff = GenStiffness(method=method)
    params = genstiff.gen_params(seq, use_group=True, sparse=True)
    
    return params['groundstate'], params['stiffness']


def _gen_params_open(
    model: str,
    sequence: str,
    composite_size: int,
    start_id: int,
    end_id: int | None,
    allow_partial: bool,
    block_size: int,
    overlap_size: int,
    tail_size: int,
    allow_crop: bool,
    cgnap_setname: str,
    print_info: bool = False,
    verbose: bool = False,
) -> DNAParameters:
    """
    Generate parameters for open (linear) topology.
    
    Internal helper function for gen_params.
    """
    closed = False
    
    # Generate base-level stiffness and ground state
    if model.lower() in _GEN_PARAMS_OLSON_IDENTIFIERS + _GEN_PARAMS_LANKAS_IDENTIFIERS:
        gs, stiff = _generate_local_model_params(model, sequence, closed=False)
    
    elif model.lower() in _GEN_PARAMS_CGNAP_IDENTIFIERS:
        if allow_partial:
            method = cgnaplus_bps_params
            stiffgen_args = _build_cgnaplus_args(cgnap_setname)
            
            nbps = len(sequence) - 1
            
            if overlap_size > nbps:
                overlap_size = nbps - 1
            if block_size > nbps:
                block_size = nbps
            
            _log_partial_stiff_params(block_size, overlap_size, tail_size, print_info=print_info, verbose=verbose)
            
            gs, bmat_stiff = partial_stiff(
                sequence,
                method,
                stiffgen_args,
                block_size=block_size,
                overlap_size=overlap_size,
                tail_size=tail_size,
                closed=closed,
                ndims=_GEN_PARAMS_NDIMS,
                verbose=verbose,
            )
            stiff = bmat_stiff.to_sparse()
        else:
            gs, stiff = cgnaplus_bps_params(
                sequence,
                parameter_set_name=cgnap_setname,
                translations_in_nm=_GEN_PARAMS_CGNAPLUS_ROT_IN_NM,
                euler_definition=_GEN_PARAMS_CGNAPLUS_EULER_DEF,
                group_split=_GEN_PARAMS_CGNAPLUS_GROUP_SPLIT,
                remove_factor_five=_GEN_PARAMS_CGNAPLUS_REMOVE_FAC_5,
                rotations_only=_GEN_PARAMS_CGNAPLUS_ROT_ONLY
            )
    
    # Handle no coarse-graining case
    if composite_size <= 1:
        gs, stiff, sequence = _apply_sequence_range(gs, stiff, sequence, start_id, end_id)
        
        return DNAParameters(
            sequence=sequence,
            model=model,
            shape_params=gs,
            stiffmat=stiff,
            closed=False,
            composite_size=composite_size,
            start_id=start_id,
            end_id=end_id
        )
    
    # Coarse-grain parameters
    block_ncomp, overlap_ncomp, tail_ncomp = _calculate_coarse_grain_params(
        block_size, overlap_size, tail_size, composite_size
    )
    
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
        verbose=verbose,
        print_info=print_info
    )
    
    if isinstance(cg_stiff, BlockOverlapMatrix):
        if print_info and verbose:
            print('Convert to sparse matrix')
        cg_stiff = cg_stiff.to_sparse()
    
    gs, stiff, sequence = _apply_sequence_range(gs, stiff, sequence, start_id, end_id)
    
    return DNAParameters(
        sequence=sequence,
        model=model,
        shape_params=gs,
        stiffmat=stiff,
        closed=False,
        composite_size=composite_size,
        cg_shape_params=cg_gs,
        cg_stiffmat=cg_stiff,
        start_id=start_id,
        end_id=end_id
    )


def _gen_params_closed(
    model: str,
    sequence: str,
    composite_size: int,
    start_id: int,
    end_id: int | None,
    allow_partial: bool,
    block_size: int,
    overlap_size: int,
    tail_size: int,
    cgnap_setname: str,
    print_info: bool = True,
    verbose: bool = False,
) -> DNAParameters:
    """
    Generate parameters for closed (circular) topology.
    
    Internal helper function for gen_params.
    """
    closed = True
    nbps = len(sequence)
    
    # Validate closed topology constraints
    if nbps % composite_size != 0:
        raise ValueError(
            f'For closed topology the length of the sequence ({nbps}) must be '
            f'a multiple of the composite size ({composite_size}).'
        )
    if start_id != 0:
        raise ValueError(f'Closed topology requires start_id=0, got start_id={start_id}')
    if end_id is not None:
        raise ValueError(f'Closed topology does not support end_id, got end_id={end_id}')
    
    # Local models (Olson, Lankas)
    if model.lower() in _GEN_PARAMS_OLSON_IDENTIFIERS + _GEN_PARAMS_LANKAS_IDENTIFIERS:
        gs, stiff = _generate_local_model_params(model, sequence, closed=True)
        
        # Coarse-grain parameters
        block_ncomp, overlap_ncomp, tail_ncomp = _calculate_coarse_grain_params(
            block_size, overlap_size, tail_size, composite_size
        )
        
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
            verbose=verbose,
            print_info=print_info
        )
        
        if isinstance(cg_stiff, BlockOverlapMatrix):
            if print_info and verbose:
                print('Convert to sparse matrix')
            cg_stiff = cg_stiff.to_sparse()
        
        return DNAParameters(
            sequence=sequence,
            model=model,
            shape_params=gs,
            stiffmat=stiff,
            closed=True,
            composite_size=composite_size,
            cg_shape_params=cg_gs,
            cg_stiffmat=cg_stiff,
            start_id=start_id,
            end_id=end_id
        )
    
    # Non-local models (cgNA+)
    if model.lower() in _GEN_PARAMS_CGNAP_IDENTIFIERS:
        if not allow_partial:
            raise ValueError(
                f'Generation of closed topology for cgNA+ requires allow_partial=True '
                f'to account for wrapped couplings (got allow_partial={allow_partial})'
            )
        
        method = cgnaplus_bps_params
        stiffgen_args = _build_cgnaplus_args(cgnap_setname)
        
        # Handle no coarse-graining case
        if composite_size <= 1:
            if overlap_size > nbps:
                overlap_size = nbps - 1
            if block_size > nbps:
                block_size = nbps
            
            _log_partial_stiff_params(block_size, overlap_size, tail_size, print_info=print_info, verbose=verbose)
            
            gs, bmat_stiff = partial_stiff(
                sequence,
                method,
                stiffgen_args,
                block_size=block_size,
                overlap_size=overlap_size,
                tail_size=tail_size,
                closed=closed,
                ndims=_GEN_PARAMS_NDIMS,
                verbose=verbose,
            )
            if print_info and verbose:
                print('Convert to sparse matrix')
            stiff = bmat_stiff.to_sparse()
            
            return DNAParameters(
                sequence=sequence,
                model=model,
                shape_params=gs,
                stiffmat=stiff,
                closed=True,
                composite_size=composite_size,
                start_id=start_id,
                end_id=end_id
            )
        
        # Coarse-graining case: extend sequence, generate, crop, and fold
        overlap_size = int(np.ceil(overlap_size / composite_size)) * composite_size
        if overlap_size > nbps:
            overlap_size = nbps - composite_size
        if block_size > nbps:
            block_size = nbps
        
        # Extend sequence for periodic boundary conditions
        n_right_overlap_extends = 2
        extended_seq = (
            sequence[-overlap_size:] +
            sequence +
            (''.join(sequence for i in range(n_right_overlap_extends)))[:n_right_overlap_extends * overlap_size + 1]
        )
        
        _log_partial_stiff_params(block_size, overlap_size, tail_size, print_info=print_info, verbose=verbose)
        
        ext_gs, bmat_ext_stiff = partial_stiff(
            extended_seq,
            method,
            stiffgen_args,
            block_size=block_size,
            overlap_size=overlap_size,
            tail_size=tail_size,
            closed=False,
            ndims=_GEN_PARAMS_NDIMS,
            verbose=verbose,
        )
        
        # Coarse-grain extended parameters
        block_ncomp, overlap_ncomp, tail_ncomp = _calculate_coarse_grain_params(
            block_size, overlap_size, tail_size, composite_size
        )
        
        if bmat_ext_stiff.shape[0] % composite_size != 0:
            raise ValueError(
                f'Incompatible matrix size: {bmat_ext_stiff.shape[0]} is not a multiple '
                f'of composite_size={composite_size}'
            )
        
        cg_gs_ext, bmat_cg_stiff_ext = coarse_grain(
            ext_gs,
            bmat_ext_stiff,
            composite_size,
            start_id=start_id,
            end_id=end_id,
            allow_partial=True,
            block_ncomp=block_ncomp,
            overlap_ncomp=overlap_ncomp,
            tail_ncomp=tail_ncomp,
            allow_crop=False,
            use_sparse=True,
            verbose=verbose,
            print_info=print_info,
        )
        
        # Crop coarse-grained parameters
        cg_gs = cg_gs_ext[overlap_ncomp:-n_right_overlap_extends * overlap_ncomp]
        cg_nbps = nbps // composite_size
        
        if isinstance(bmat_cg_stiff_ext, BlockOverlapMatrix):
            if print_info and verbose:
                print('Convert to sparse matrix')
            cg_stiff_ext = bmat_cg_stiff_ext.to_sparse(
                xlo=overlap_ncomp * _GEN_PARAMS_NDIMS,
                xhi=(2 * overlap_ncomp + cg_nbps) * _GEN_PARAMS_NDIMS,
                ylo=overlap_ncomp * _GEN_PARAMS_NDIMS,
                yhi=(2 * overlap_ncomp + cg_nbps) * _GEN_PARAMS_NDIMS,
            )
        else:
            cg_stiff_ext = bmat_cg_stiff_ext[
                overlap_ncomp * _GEN_PARAMS_NDIMS:(2 * overlap_ncomp + cg_nbps) * _GEN_PARAMS_NDIMS,
                overlap_ncomp * _GEN_PARAMS_NDIMS:(2 * overlap_ncomp + cg_nbps) * _GEN_PARAMS_NDIMS
            ]
        
        cg_stiff = crop_periodic_fold_fill_zeros(
            cg_stiff_ext,
            cg_nbps * _GEN_PARAMS_NDIMS,
            cg_nbps * _GEN_PARAMS_NDIMS
        )
        
        # Create base-level stiffness from extended version
        if print_info and verbose:
            print('Convert to sparse matrix')
        base_stiff = bmat_ext_stiff.to_sparse() if isinstance(bmat_ext_stiff, BlockOverlapMatrix) else bmat_ext_stiff
        base_stiff = base_stiff[
            overlap_size * _GEN_PARAMS_NDIMS:(overlap_size + nbps) * _GEN_PARAMS_NDIMS,
            overlap_size * _GEN_PARAMS_NDIMS:(overlap_size + nbps) * _GEN_PARAMS_NDIMS
        ]
        
        return DNAParameters(
            sequence=sequence,
            model=model,
            shape_params=ext_gs[overlap_size:-n_right_overlap_extends * overlap_size],
            stiffmat=base_stiff,
            closed=True,
            composite_size=composite_size,
            cg_shape_params=cg_gs,
            cg_stiffmat=cg_stiff,
            start_id=start_id,
            end_id=end_id
        )


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
    print_info: bool = True,
    verbose: bool = False,
    ) -> DNAParameters:
    """
    Generate sequence-dependent DNA shape and stiffness parameters with optional coarse-graining.

    This function computes base-pair step shape parameters (equilibrium configurations in SE(3)) and 
    their associated stiffness matrices for a given DNA sequence using established mechanical models. 
    Shape parameters describe the equilibrium geometry of each base-pair step as six degrees of 
    freedom (3 rotations, 3 translations) in the local reference frame. The stiffness matrix 
    characterizes the harmonic energy landscape around these equilibria. When `composite_size > 1`, 
    the function additionally performs coarse-graining by aggregating consecutive base-pair steps 
    into effective composite units with renormalized shape and stiffness parameters.

    Available Models
    ----------------
    The `model` parameter selects one of three sequence-dependent parameter sets:
    
    - **'olson'** (aliases: 'crystal', 'cry'): Crystal structure data from Olson et al. (1998) 
      based on analysis of high-resolution X-ray structures. Local dinucleotide model.
      
    - **'lankas'** (alias: 'md'): Molecular dynamics-derived parameters from Lankas et al. (2003) 
      based on microsecond-scale atomistic simulations. Local dinucleotide model.
      
    - **'cgnaplus'** (aliases: 'cgna+', 'cgnap'): The cgNA+ model from Sharma et al. (2023), 
      incorporating sequence-dependent non-local couplings beyond nearest neighbors. Non-local 
      model requiring special handling via `partial_stiff` for efficient assembly.

    Topology
    --------
    Linear topology (`closed=False`, default):
        Parameters are generated for a linear DNA chain with N base pairs and N-1 steps. 
        The functions supports extracting subranges via `start_id` and `end_id` when coarse-graining.
        
    Circular topology (`closed=True`):
        Parameters are generated for a closed DNA ring with periodic boundary conditions. 
        Requires: (1) `len(sequence)` must be divisible by `composite_size`, (2) `start_id=0`, 
        and (3) `end_id=None`. For cgNA+ with coarse-graining, the sequence is extended to 
        capture wrapped couplings, then cropped and folded back into the fundamental domain.

    Coarse-Graining Strategy
    -------------------------
    For large sequences or non-local models, the function employs a block-overlap assembly strategy 
    controlled by `block_size`, `overlap_size`, and `tail_size`. The sequence is divided into 
    overlapping segments (blocks) of length `block_size` with overlap `overlap_size`. Additional 
    `tail_size` bases are included on each segment boundary to minimize edge artifacts. After 
    generating parameters for each block, overlapping regions are averaged and the full matrices 
    are assembled. This approach enables memory-efficient handling of sequences with thousands of 
    base pairs.

    Parameters
    ----------
    model : str
        Model identifier. Accepted values: 'olson', 'crystal', 'cry' (Olson et al. 1998); 
        'lankas', 'md' (Lankas et al. 2003); 'cgnaplus', 'cgna+', 'cgnap' (Sharma et al. 2023).
    sequence : str
        DNA sequence string using standard nucleotide codes (A, T, C, G). For linear topology, 
        N bases yield N-1 base-pair steps. For circular topology, N bases yield N steps.
    composite_size : int, default=1
        Number of consecutive base-pair steps to aggregate into each coarse-grained composite unit. 
        A value of 1 disables coarse-graining and returns only base-level parameters.
    closed : bool, default=False
        Topology flag. If True, generate parameters for a closed circular DNA molecule with 
        periodic boundary conditions. If False, generate parameters for a linear molecule.
    start_id : int, default=0
        Starting composite index for selecting a subrange when coarse-graining linear molecules. 
        Must be 0 for circular topology.
    end_id : int or None, default=None
        Ending composite index (exclusive) for selecting a subrange when coarse-graining linear 
        molecules. If None, coarse-grains through the entire sequence. Not supported for circular 
        topology.
    allow_partial : bool, default=True
        Enable block-overlap assembly strategy via `partial_stiff` for memory-efficient generation 
        of large sequences. Required for cgNA+ model with circular topology. For local models 
        (Olson, Lankas), parameters can be generated directly without blocking.
    block_size : int, default=120
        Block size (in base-pair steps) for segment-wise assembly when `allow_partial=True`. 
        Larger blocks reduce edge effects but increase memory usage. Must exceed `overlap_size`.
    overlap_size : int, default=20
        Overlap size (in base-pair steps) between consecutive blocks. Overlapping regions are 
        averaged to ensure smooth assembly. Automatically increased to at least `composite_size` 
        when coarse-graining. Must be less than `block_size`.
    tail_size : int, default=20
        Number of extra bases included on each side of a block to reduce boundary artifacts. 
        These tail regions are discarded after parameter generation for each block.
    allow_crop : bool, default=False
        If True, allows returning cropped coarse-grained matrices when full assembly is not possible. 
        Passed to the `coarse_grain` function. Generally should remain False for production use.
    cgnap_setname : str, default='curves_plus'
        Parameter set name for cgNA+ model. Options include 'curves_plus' (default) and other 
        parameter sets provided by the cgNA+ framework. Only used when model is 'cgnaplus'.
    print_info : bool, default=True
        If True, print parameter information (block sizes, etc.) in this file. Only effective 
        when verbose=True.
    verbose : bool, default=False
        Master verbose flag controlling all print output. If False, no output is printed regardless 
        of show_params_info. If True, enables printing in this function and called functions 
        (partial_stiff, coarse_grain).

    Returns
    -------
    DNAParameters
        A dataclass instance containing the generated parameters with the following attributes:
        
        **Core attributes (always present):**
        
        - `sequence` : str
            The DNA sequence used for parameter generation.
        - `model` : str
            The model identifier used.
        - `shape_params` : np.ndarray, shape (N_steps, 6)
            Base-pair step shape parameters (equilibrium configurations). Each row contains 
            [tilt, roll, twist, shift, slide, rise] or equivalent coordinates depending on model.
        - `stiffmat` : scipy.sparse.spmatrix
            Stiffness matrix in sparse format. Shape (N_steps*6, N_steps*6) encoding the harmonic 
            force constants between all degrees of freedom.
        - `closed` : bool
            Topology flag indicating whether parameters are for a closed ring.
        - `composite_size` : int
            Coarse-graining level used (1 for no coarse-graining).
        - `start_id` : int
            Starting index of the range (0 if full sequence).
        - `end_id` : int or None
            Ending index of the range (None if full sequence).
        
        **Coarse-grained attributes (present when composite_size > 1):**
        
        - `cg_shape_params` : np.ndarray, shape (N_composites, 6)
            Coarse-grained shape parameters for composite units.
        - `cg_stiffmat` : scipy.sparse.spmatrix
            Coarse-grained stiffness matrix for composite units. Shape (N_composites*6, N_composites*6).
        
        **Utility attributes:**
        
        - `disc_len` : float
            Discretization length (0.34 nm by default) for visualization purposes.
        - `metadata` : dict
            Dictionary for storing additional model-specific information.
        
        **Properties:**
        
        - `n_bps` : int
            Number of base pairs in the sequence.
        - `n_steps` : int
            Number of base-pair steps (N-1 for linear, N for circular).
        - `is_coarse_grained` : bool
            Whether coarse-grained parameters are available.
        
        **Methods:**
        
        - `to_dict()` : dict
            Convert to dictionary format for backward compatibility with legacy code.

    Raises
    ------
    ValueError
        - If `model` is not one of the accepted identifiers.
        - If `composite_size < 1` or exceeds `_GEN_PARAMS_MAX_COMPOSITE_SIZE` (currently 40).
        - If `block_size <= overlap_size` (blocks must be larger than overlaps).
        - If `start_id > end_id` (invalid range).
        - If `end_id` exceeds sequence bounds.
        - For circular topology: if sequence length is not divisible by `composite_size`, if 
          `start_id != 0`, if `end_id` is specified, or if using cgNA+ without `allow_partial=True`.

    Notes
    -----
    - Shape parameters represent equilibrium configurations in SE(3) (Special Euclidean group) 
      characterizing the relative position and orientation of consecutive base-pair reference frames.
    - Stiffness matrices are symmetric positive semi-definite and stored in sparse format for 
      memory efficiency. They represent the second derivatives of the mechanical energy.
    - For visualization and downstream analysis, shape parameters can be converted to 3D triad 
      configurations using the `gen_config` function from the `genconf` module.
    - The default `disc_len=0.34` nm corresponds to the average rise per base pair in B-form DNA.

    References
    ----------
    - Olson et al. (1998). "DNA Sequence-Dependent Deformability Deduced from Protein-DNA Crystal 
      Complexes." PNAS 95(19): 11163-11168.
    - Lankas et al. (2003). "Sequence-Dependent Elastic Properties of DNA." Biophys. J. 85(5): 2872-2883.
    - Sharma et al. (2023). "cgNA+: A sequence-dependent coarse-grain model of double-stranded nucleic 
      acids." J. Mol. Biol. 435(2): 167978.

    Examples
    --------
    Generate base-level parameters for a short linear DNA sequence:
    
    >>> params = gen_params('cgnaplus', 'ATCGATCG', composite_size=1)
    >>> params.shape_params.shape
    (7, 6)
    >>> params.stiffmat.shape
    (42, 42)
    
    Generate coarse-grained parameters at 10 bp resolution:
    
    >>> params = gen_params('cgnaplus', 'A'*200, composite_size=10)
    >>> params.is_coarse_grained
    True
    >>> params.cg_shape_params.shape
    (19, 6)
    
    Generate parameters for a closed circular DNA molecule:
    
    >>> params = gen_params('cgnaplus', 'ATCG'*25, composite_size=10, closed=True)
    >>> params.closed
    True
    >>> params.n_steps
    100
    """    

    
    if model.lower() not in _GEN_PARAMS_ACCEPTED_IDENTIFIERS:
        ident_str = '", "'.join(_GEN_PARAMS_ACCEPTED_IDENTIFIERS)
        raise ValueError(
            f'Unknown model identifier: "{model}". '
            f'Must be one of: "{ident_str}"'
        )
    
    if composite_size > _GEN_PARAMS_MAX_COMPOSITE_SIZE:
        raise ValueError(
            f'composite_size ({composite_size}) exceeds maximum allowed '
            f'({_GEN_PARAMS_MAX_COMPOSITE_SIZE}). Recommended size <= 10'
        )
    
    if overlap_size < composite_size:
        overlap_size = composite_size
    
    if block_size <= overlap_size:
        raise ValueError(
            f'block_size ({block_size}) must be larger than overlap_size ({overlap_size})'
        )
    
    if end_id is not None:
        if start_id > end_id:
            raise ValueError(f'Invalid id range: start_id ({start_id}) > end_id ({end_id})')
        if end_id > len(sequence) - 1: 
            if closed:
                raise ValueError(
                    f'end_id ({end_id}) exceeds sequence bounds (max: {len(sequence) - 1}). '
                    f'Setting end_id is not allowed for closed topologies.'
                )
            else:
                raise ValueError(
                    f'end_id ({end_id}) exceeds sequence bounds (max: {len(sequence) - 1})'
                )

    
    if composite_size < 1:
        raise ValueError(f'composite_size must be >= 1, got {composite_size}')
    
    # Dispatch to topology-specific helper functions
    if not closed:
        return _gen_params_open(
            sequence=sequence,
            model=model,
            composite_size=composite_size,
            allow_partial=allow_partial,
            block_size=block_size,
            overlap_size=overlap_size,
            tail_size=tail_size,
            allow_crop=allow_crop,
            start_id=start_id,
            end_id=end_id,
            cgnap_setname=cgnap_setname,
            print_info=print_info,
            verbose=verbose,
        )
    else:
        return _gen_params_closed(
            sequence=sequence,
            model=model,
            composite_size=composite_size,
            allow_partial=allow_partial,
            block_size=block_size,
            overlap_size=overlap_size,
            tail_size=tail_size,
            start_id=start_id,
            end_id=end_id,
            cgnap_setname=cgnap_setname,
            print_info=print_info,
            verbose=verbose,
        )

    
if __name__ == '__main__':
    
    N = 100
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(N)])
    model = 'cgna+'
    cg = 10
    closed = True
    gen_params(model,seq,composite_size=cg,closed=closed)
     