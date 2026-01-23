from __future__ import annotations

from pathlib import Path
import os
import warnings
import numpy as np
from ..IOPolyMC import iopolymc as iopmc
from ..genconf import gen_config
from .. utils.path_methods import create_relative_path


def visualize_chimerax(
    base_fn: str | Path,
    seq: str,
    cg: int,
    *,
    shape_params: np.ndarray | None = None,
    taus: np.ndarray | None = None,
    start_id: int = 0,
    bead_radius: float | None = None,
    disc_len: float = 0.34,
    include_bps_triads: bool = False
) -> None:
    # Validate exactly one is provided
    if shape_params is None and taus is None:
        raise ValueError("Either 'shape_params' or 'taus' must be provided")
    if shape_params is not None and taus is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'taus', choose one")
    if cg < 1:
        raise ValueError(f"cg must be >= 1, got {cg}")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    if taus is None:
        taus = gen_config(shape_params,disc_len=disc_len)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    taus2pdb(pdbfn, taus, seq)
    
    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgtaus = taus[start_id::cg]
    _triads2bild(bildfn, cgtaus, alpha=1., scale=1, nm2aa=True, decimals=2)
    
    # create base pair step triad bild file
    bps_triads_bildfn = None
    if include_bps_triads and cg > 1:
        bps_triads_bildfn = base_fn.with_name(base_fn.name + '_bps_triads.bild')
        _triads2bild(bps_triads_bildfn, taus, alpha=1., scale=1, nm2aa=True, decimals=2)
        
    # create chimera cxc file
    cxcfn = base_fn.with_suffix('.cxc')
    if bead_radius > 0:
        spheres = np.zeros((len(cgtaus),4))
        spheres[:,:3] = cgtaus[:,:3,3]
        spheres[:,3] = bead_radius
    else:
       spheres = None
    _chimeracxc(cxcfn, pdbfn, triadfn=bildfn, spheres=spheres, nm2aa= True, decimals=2, additional_triadfn=bps_triads_bildfn)
    
    
def visualize_pdb(
    base_fn: str | Path,
    seq: str,
    *,
    shape_params: np.ndarray | None = None,
    taus: np.ndarray | None = None,
    disc_len: float = 0.34,
) -> None:
    # Validate exactly one is provided
    if shape_params is None and taus is None:
        raise ValueError("Either 'shape_params' or 'taus' must be provided")
    if shape_params is not None and taus is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'taus', choose one")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    if taus is None:
        taus = gen_config(shape_params,disc_len=disc_len)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    taus2pdb(pdbfn, taus, seq)
    
    
def visualize_xyz(
    base_fn: str | Path,
    cg: int,
    *,
    shape_params: np.ndarray | None = None,
    taus: np.ndarray | None = None,
    start_id: int = 0,
    disc_len: float = 0.34,
) -> None:
    # Validate exactly one is provided
    if shape_params is None and taus is None:
        raise ValueError("Either 'shape_params' or 'taus' must be provided")
    if shape_params is not None and taus is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'taus', choose one")
    if cg < 1:
        raise ValueError(f"cg must be >= 1, got {cg}")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    if taus is None:
        taus = gen_config(shape_params,disc_len=disc_len)
    
    cgtaus = taus[start_id::cg]
    
    # create 
    cgxyzfn = base_fn.with_name(base_fn.name + '_cg.xyz')
    xyz = {
        'types': ['C']*(len(cgtaus)),
        'pos'  : [cgtaus[:,:3,3]]
        }
    iopmc.write_xyz(cgxyzfn,xyz)
    


def cgvisual(
    base_fn: str | Path,
    shape_params: np.ndarray,
    seq: str,
    cg: int,
    start_id: int = 0,
    bead_radius: float = None,
    disc_len: float = 0.34,
    include_bps_triads: bool = False
):

    warnings.warn(
        "cgvisual is deprecated and will be removed in a future release. "
        "Please use visualize_pdb or visualize_xyz instead.",
        DeprecationWarning
    )

    if cg < 1:
        raise ValueError(f"cg must be >= 1, got {cg}")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    # calculate triads
    taus = gen_config(shape_params,disc_len=disc_len)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    taus2pdb(pdbfn, taus, seq)
    
    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgtaus = taus[start_id::cg]
    _triads2bild(bildfn, cgtaus, alpha=1., scale=1, nm2aa=True, decimals=2)
    
    # create base pair step triad bild file
    bps_triads_bildfn = None
    if include_bps_triads and cg > 1:
        bps_triads_bildfn = base_fn.with_name(base_fn.name + '_bps_triads.bild')
        _triads2bild(bps_triads_bildfn, taus, alpha=1., scale=1, nm2aa=True, decimals=2)
        
    # create chimera cxc file
    cxcfn = base_fn.with_suffix('.cxc')
    if bead_radius > 0:
        spheres = np.zeros((len(cgtaus),4))
        spheres[:,:3] = cgtaus[:,:3,3]
        spheres[:,3] = bead_radius
    else:
       spheres = None
    _chimeracxc(cxcfn, pdbfn, triadfn=bildfn, spheres=spheres, nm2aa= True, decimals=2, additional_triadfn=bps_triads_bildfn)
    
    cgxyzfn = base_fn.with_name(base_fn.name + '_cg.xyz')
    xyz = {
        'types': ['C']*(len(cgtaus)),
        'pos'  : [cgtaus[:,:3,3]]
        }
    iopmc.write_xyz(cgxyzfn,xyz)

   
def _chimeracxc(
    fn: Path | str,
    pdbfn: Path | str,
    triadfn: Path | str | None = None,
    spheres: np.ndarray | None = None,
    nm2aa: bool = True,
    decimals: int = 2,
    additional_triadfn: Path | str | None = None
):
    
    fn = Path(fn)
    if fn.suffix.lower() != '.cxc':
        fn = fn.with_suffix('.cxc')
    
    pdbfn = Path(pdbfn)
    modelnum = 0
    sphereids = []
    with open(fn,'w') as f:
        
        f.write(f'# scene settings\n')
        # white background
        f.write(f'set bgColor white\n')
        # simple lighting
        f.write(f'lighting simple\n')    
        # set silhouettes
        f.write(f'graphics silhouettes true color black width 1.5\n') 
        # open pdb twice
        f.write(f'\n# load pdb\n')
        f.write(f'open {pdbfn.name}\n') 
        f.write(f'open {pdbfn.name}\n')
        modelnum += 2 
        # dna visuals
        f.write(f'\n# set DNA visuals\n')
        f.write(f'style ball\n')
        f.write(f'nucleotides atoms\n')
        f.write(f'color white target a\n')
        f.write(f'color light gray target c\n')
        f.write(f'cartoon style nucleic xsect oval width 3.0 thick 1.2\n')
        f.write(f'hide #2 atoms\n')
        f.write(f'hide #1 cartoons\n')
        
        # open triads
        if triadfn is not None:
            triadfn = Path(triadfn)
            modelnum += 1 
            f.write(f'\n# load triads BILD\n')
            f.write(f'open {triadfn.name}\n') 
            
        # open triads
        if additional_triadfn is not None:
            additional_triadfn = Path(additional_triadfn)
            modelnum += 1 
            f.write(f'\n# load additional triads BILD\n')
            f.write(f'open {additional_triadfn.name}\n') 
        
        if spheres is not None:
            nm2aafac = 1
            if nm2aa:
                nm2aafac = 10
            def pt2str(pt):
                return ','.join([f'{np.round(p*nm2aafac,decimals=decimals)}' for p in pt[:3]])

            f.write(f'\n# Genergate spheres\n')
            for shid,sphere in enumerate(spheres):
                modelnum += 1
                sphereids.append(modelnum)
                f.write(f'shape sphere radius {np.round(sphere[3]*nm2aafac,decimals=decimals)} center {pt2str(sphere)} name sph{shid+1}\n')
                f.write(f'transparency #{modelnum} 75\n')

              
def _triads2bild(
    fn: Path | str,
    taus: np.ndarray,
    alpha: float = 1.,
    ucolor: str = 'default',
    vcolor: str = 'default',
    tcolor: str = 'default',
    scale: float = 1,
    nm2aa: bool = True,
    decimals: int = 2
): 
    
    if ucolor == 'default':
        ucolor = [64/255,91/255,4/255]
        # ucolor = [0.15294118, 0.47843137, 0.17647059]
    if vcolor == 'default':
        vcolor = [61/255,88/255,117/255]
        # vcolor = [0.17647059, 0.15294118, 0.47843137]
    if tcolor == 'default':
        tcolor = [153/255,30/255,46/255]
        # tcolor = [0.47843137, 0.17647059, 0.15294118]
        
    fn = Path(fn)
    if fn.suffix.lower() != '.bild':
        fn = fn.with_suffix('.bild')
        
    dist = np.mean(np.linalg.norm(taus[1:,:3,3]-taus[:-1,:3,3],axis=1))
    size = dist * 0.66 * scale
    nm2aafac = 1
    if nm2aa:
        nm2aafac = 10
    
    def _color2str(color):
        if isinstance(color,str):
            return color
        if hasattr(color, '__iter__') and len(color) == 3:
            return ' '.join([f'{c}' for c in color])
        raise ValueError(f'Invalid color {color}')
    
    def pt2str(pt):
        return ' '.join([f'{np.round(p*nm2aafac,decimals=decimals)}' for p in pt])
    
    shapestr = f'{np.round(size*nm2aafac/20,decimals=decimals)} {np.round(size*nm2aafac/20*2,decimals=decimals)} 0.70'
    with open(fn,'w') as f:
        if alpha < 1.0:
            f.write(f'.transparency {1-alpha}\n')
        for i,tau in enumerate(taus):
            tau = tau[:3]
            f.write(f'# triad {i+1}\n')
            f.write(f'.color {_color2str(ucolor)}\n')
            f.write(f'.arrow {pt2str(tau[:,3])} {pt2str(tau[:,3]+tau[:,0]*size)} {shapestr}\n')
            f.write(f'.color {_color2str(vcolor)}\n')
            f.write(f'.arrow {pt2str(tau[:,3])} {pt2str(tau[:,3]+tau[:,1]*size)} {shapestr}\n')
            f.write(f'.color {_color2str(tcolor)}\n')
            f.write(f'.arrow {pt2str(tau[:,3])} {pt2str(tau[:,3]+tau[:,2]*size)} {shapestr}\n')
    return fn
    
def params2pdb(fn: Path | str, params: np.ndarray, seq: str) -> None:
    taus2pdb(fn,gen_config(params),seq)
    
def taus2pdb(
    fn: Path | str, 
    taus: np.ndarray, 
    seq: str
    ) -> None:
    fn = Path(fn)
    if fn.suffix.lower() != '.pdb':
        fn = fn.with_suffix('.pdb')
    if len(taus) != len(seq):
        raise ValueError(f'Dimension of taus ({taus.shape}) and seq ({len(seq)}) do not match.')
    iopmc.gen_pdb(fn, taus[:,:3,3], taus[:,:3,:3], sequence=seq, center=False)
    
    