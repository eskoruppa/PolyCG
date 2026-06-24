from __future__ import annotations

from pathlib import Path
import os
from turtle import color
import warnings
import numpy as np
from ..IOPolyMC import iopolymc as iopmc
from ..genconf import gen_config
from .. utils.path_methods import create_relative_path


# Accepted (case-insensitive) values for the `spheres_as` argument:
#   'cmm'/'atoms' -> ChimeraX marker set (instanced atom spheres, the efficient default)
#   'bild'        -> per-sphere BILD surface meshes
#   'shape'       -> per-sphere `shape sphere` commands in the .cxc
_SPHERE_STYLES = ('cmm', 'atoms', 'bild', 'shape')


def visualize_chimerax(
    base_fn: str | Path,
    seq: str,
    cg: int,
    *,
    shape_params: np.ndarray | None = None,
    poses: np.ndarray | None = None,
    first_cg: int = 0,
    bead_radius: float | None = None,
    disc_len: float = 0.34,
    include_bps_triads: bool = False,
    config_filetype: str = 'pdb', # options: 'pdb', 'cif'
    additional_beads: dict[str, np.ndarray | float | str] | None = None,
    quality: float = 3.0,
    spheres_as: str = 'cmm',
    sphere_triangles: int = 1000
) -> None:
    # Validate exactly one is provided
    if shape_params is None and poses is None:
        raise ValueError("Either 'shape_params' or 'poses' must be provided")
    if shape_params is not None and poses is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'poses', choose one")
    if shape_params is not None:
        if shape_params.ndim !=2:
            raise ValueError(f"shape_params must be a 2D array, got shape {shape_params.shape}")
        if shape_params.shape[1] != 6:
            raise ValueError(f"shape_params must have shape (N,6), got shape {shape_params.shape}")
    if poses is not None:
        if poses.ndim !=3:
            raise ValueError(f"poses must be a 3D array, got shape {poses.shape}")
        if poses.shape[1:] != (4,4):
            raise ValueError(f"poses must have shape (N,4,4), got shape {poses.shape}")
    if cg < 1:
        raise ValueError(f"cg must be >= 1, got {cg}")
    if spheres_as.lower() not in _SPHERE_STYLES:
        raise ValueError(f"Invalid spheres_as {spheres_as!r}, expected one of {_SPHERE_STYLES}")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    if poses is None:
        poses = gen_config(shape_params,disc_len=disc_len)

    if len(poses) > 500:
        config_filetype = 'cif'

    if config_filetype == 'cif':
        if len(poses) > 2000:
            nchuncks = int(np.ceil(len(poses)/2000))
            pdbfn = []
            for i in range(nchuncks):
                chunk_poses = poses[i*2000:(i+1)*2000]
                chunk_fn = base_fn.with_name(base_fn.name + f'_part{i+1}.cif')
                poses2cif(chunk_fn, chunk_poses, seq[i*2000:(i+1)*2000])
                pdbfn.append(chunk_fn)
        else:
            # generate cif file
            pdbfn = base_fn.with_suffix('.cif')
            poses2cif(pdbfn, poses, seq)  
    elif config_filetype == 'pdb':
        # generate pdb file
        pdbfn = base_fn.with_suffix('.pdb')
        poses2pdb(pdbfn, poses, seq)
    else:
        raise ValueError(f"Invalid config_filetype {config_filetype}, expected 'pdb' or 'cif'")
    
    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgposes = poses[first_cg::cg]
    _triads2bild(bildfn, cgposes, alpha=1., scale=1, nm2aa=True, decimals=2)
    
    # create base pair step triad bild file
    bps_triads_bildfn = None
    if include_bps_triads and cg > 1:
        bps_triads_bildfn = base_fn.with_name(base_fn.name + '_bps_triads.bild')
        _triads2bild(bps_triads_bildfn, poses, alpha=1., scale=1, nm2aa=True, decimals=2)
        
    # create chimera cxc file
    cxcfn = base_fn.with_suffix('.cxc')
    if bead_radius > 0:
        spheres = np.zeros((len(cgposes),4))
        spheres[:,:3] = cgposes[:,:3,3]
        spheres[:,3] = bead_radius
    else:
       spheres = None
    _chimeracxc(cxcfn, pdbfn, triadfn=bildfn, spheres=spheres, nm2aa=True, decimals=2, additional_triadfn=bps_triads_bildfn, additional_beads=additional_beads, quality=quality, spheres_as=spheres_as, sphere_triangles=sphere_triangles)
    
    
def visualize_pdb(
    base_fn: str | Path,
    seq: str,
    *,
    shape_params: np.ndarray | None = None,
    poses: np.ndarray | None = None,
    disc_len: float = 0.34,
) -> None:
    # Validate exactly one is provided
    if shape_params is None and poses is None:
        raise ValueError("Either 'shape_params' or 'poses' must be provided")
    if shape_params is not None and poses is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'poses', choose one")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    if poses is None:
        poses = gen_config(shape_params,disc_len=disc_len)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    poses2pdb(pdbfn, poses, seq)
    
    
def visualize_xyz(
    base_fn: str | Path,
    cg: int,
    *,
    shape_params: np.ndarray | None = None,
    poses: np.ndarray | None = None,
    first_cg: int = 0,
    disc_len: float = 0.34,
) -> None:
    # Validate exactly one is provided
    if shape_params is None and poses is None:
        raise ValueError("Either 'shape_params' or 'poses' must be provided")
    if shape_params is not None and poses is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'poses', choose one")
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
        
    if poses is None:
        poses = gen_config(shape_params,disc_len=disc_len)
    
    cgposes = poses[first_cg::cg]
    
    # create 
    cgxyzfn = base_fn.with_name(base_fn.name + '_cg.xyz')
    xyz = {
        'types': ['C']*(len(cgposes)),
        'pos'  : [cgposes[:,:3,3]]
        }
    iopmc.write_xyz(str(cgxyzfn),xyz)
    


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
    poses = gen_config(shape_params,disc_len=disc_len)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    poses2pdb(pdbfn, poses, seq)
    
    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgposes = poses[start_id::cg]
    _triads2bild(bildfn, cgposes, alpha=1., scale=1, nm2aa=True, decimals=2)
    
    # create base pair step triad bild file
    bps_triads_bildfn = None
    if include_bps_triads and cg > 1:
        bps_triads_bildfn = base_fn.with_name(base_fn.name + '_bps_triads.bild')
        _triads2bild(bps_triads_bildfn, poses, alpha=1., scale=1, nm2aa=True, decimals=2)
        
    # create chimera cxc file
    cxcfn = base_fn.with_suffix('.cxc')
    if bead_radius > 0:
        spheres = np.zeros((len(cgposes),4))
        spheres[:,:3] = cgposes[:,:3,3]
        spheres[:,3] = bead_radius
    else:
       spheres = None
    _chimeracxc(cxcfn, pdbfn, triadfn=bildfn, spheres=spheres, nm2aa= True, decimals=2, additional_triadfn=bps_triads_bildfn)
    
    cgxyzfn = base_fn.with_name(base_fn.name + '_cg.xyz')
    xyz = {
        'types': ['C']*(len(cgposes)),
        'pos'  : [cgposes[:,:3,3]]
        }
    iopmc.write_xyz(cgxyzfn,xyz)

   
def _chimeracxc(
    fn: Path | str,
    pdbfns: Path | str | list[Path | str],
    triadfn: Path | str | None = None,
    spheres: np.ndarray | None = None,
    nm2aa: bool = True,
    decimals: int = 2,
    additional_triadfn: Path | str | None = None,
    additional_beads: dict[str, np.ndarray | float | str] | None = None,
    quality: float | None = 3.0,
    spheres_as: str = 'cmm',
    sphere_triangles: int = 1000
):
    
    fn = Path(fn)
    if fn.suffix.lower() != '.cxc':
        fn = fn.with_suffix('.cxc')
    
    if isinstance(pdbfns, (str, Path)):
        pdbfns = [pdbfns]

    modelnum = 0
    with open(fn,'w') as f:
        
        f.write(f'# scene settings\n')
        # white background
        f.write(f'set bgColor white\n')
        # simple lighting
        f.write(f'lighting simple\n')    
        # set silhouettes
        f.write(f'graphics silhouettes true color black width 1.5\n')
        # sphere/surface tessellation quality
        if quality is not None:
            f.write(f'graphics quality {quality}\n')

        for pdbfn in pdbfns:
            # open pdb twice
            f.write(f'\n# load pdb\n')
            f.write(f'open {pdbfn.name}\n') 
            f.write(f'open {pdbfn.name}\n')
            # dna visuals
            f.write(f'\n# set DNA visuals\n')
            f.write(f'style ball\n')
            f.write(f'nucleotides atoms\n')
            f.write(f'color white target a\n')
            f.write(f'color light gray target c\n')
            f.write(f'cartoon style nucleic xsect oval width 3.0 thick 1.2\n')
            modelnum += 1 
            f.write(f'hide #{modelnum} atoms\n')
            modelnum += 1 
            f.write(f'hide #{modelnum} cartoons\n')
        
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
            spheres_as = spheres_as.lower()
            if spheres_as in ('cmm', 'atoms'):
                # Represent the spheres as a marker set in their own file. Markers
                # are atoms, so ChimeraX draws them as GPU-instanced spheres rather
                # than N separate BILD surface meshes; per-bead radius lives in the
                # .cmm file. Atom spheres otherwise use an automatic level-of-detail
                # (10-2000 triangles each, fewer as the atom count grows) that looks
                # faceted for many beads, so pin a fixed triangle count and lift the
                # total-triangle ceiling (N*sphere_triangles) so it is not throttled.
                spheres_cmmfn = fn.with_name(fn.stem + '_spheres.cmm')
                _spheres2cmm(spheres_cmmfn, spheres, nm2aa=nm2aa, decimals=decimals)
                modelnum += 1
                f.write(f'\n# Load spheres as markers (instanced atom spheres)\n')
                f.write(f'open {spheres_cmmfn.name}\n')
                f.write(f'style #{modelnum} sphere\n')
                f.write(f'graphics quality atomTriangles {sphere_triangles} totalAtomTriangles {max(len(spheres), 1) * sphere_triangles}\n')
                f.write(f'transparency #{modelnum} 75 target a\n')
            elif spheres_as == 'bild':
                spheres_bildfn = fn.with_name(fn.stem + '_spheres.bild')
                _spheres2bild(spheres_bildfn, spheres, nm2aa=nm2aa, decimals=decimals)
                modelnum += 1
                f.write(f'\n# Load spheres BILD\n')
                f.write(f'open {spheres_bildfn.name}\n')
            elif spheres_as == 'shape':
                nm2aafac = 10 if nm2aa else 1
                def pt2str(pt):
                    return ','.join([f'{np.round(p*nm2aafac,decimals=decimals)}' for p in pt[:3]])
                f.write(f'\n# Generate spheres\n')
                for shid, sphere in enumerate(spheres):
                    modelnum += 1
                    f.write(f'shape sphere radius {np.round(sphere[3]*nm2aafac,decimals=decimals)} center {pt2str(sphere)} name sph{shid+1}\n')
                    f.write(f'transparency #{modelnum} 75\n')
            else:
                raise ValueError(f"Invalid spheres_as {spheres_as!r}, expected one of {_SPHERE_STYLES}")

        if additional_beads is not None:
            beads_bildfn = fn.with_name(fn.stem + '_beads.bild')
            _beads2bild(beads_bildfn, additional_beads, nm2aa=nm2aa, decimals=decimals)
            modelnum += 1
            f.write(f'\n# Load additional beads BILD\n')
            f.write(f'open {beads_bildfn.name}\n')
              
def _spheres2bild(
    fn: Path | str,
    spheres: np.ndarray,
    color: str | list = 'white',
    transparency: float = 0.75,
    nm2aa: bool = True,
    decimals: int = 2
) -> Path:
    fn = Path(fn)
    if fn.suffix.lower() != '.bild':
        fn = fn.with_suffix('.bild')

    nm2aafac = 10 if nm2aa else 1

    def pt2str(pt):
        return ' '.join([f'{np.round(p * nm2aafac, decimals=decimals)}' for p in pt[:3]])

    with open(fn, 'w') as f:
        f.write(f'.transparency {transparency}\n')
        if isinstance(color, str):
            f.write(f'.color {color}\n')
        elif hasattr(color, '__iter__') and len(color) == 3:
            f.write(f'.color {" ".join([str(c) for c in color])}\n')
        for i, sphere in enumerate(spheres):
            f.write(f'# sphere {i + 1}\n')
            f.write(f'.sphere {pt2str(sphere)} {np.round(sphere[3] * nm2aafac, decimals=decimals)}\n')

    return fn


def _spheres2cmm(
    fn: Path | str,
    spheres: np.ndarray,
    name: str = 'spheres',
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    nm2aa: bool = True,
    decimals: int = 2
) -> Path:
    """Write spheres as a ChimeraX marker set (.cmm).

    Each row of ``spheres`` is (x, y, z, radius). Markers are ``Atom`` instances,
    so ChimeraX renders them as GPU-instanced spheres whose smoothness follows the
    global ``graphics quality`` setting -- far cheaper and higher quality than the
    per-object surface meshes produced by ``_spheres2bild`` for large bead counts.
    The per-marker radius is stored in the file, so size scaling needs no commands.
    """
    fn = Path(fn)
    if fn.suffix.lower() != '.cmm':
        fn = fn.with_suffix('.cmm')

    nm2aafac = 10 if nm2aa else 1
    r, g, b = color

    with open(fn, 'w') as f:
        f.write(f'<marker_set name="{name}">\n')
        for i, sphere in enumerate(spheres):
            x, y, z = (np.round(c * nm2aafac, decimals=decimals) for c in sphere[:3])
            radius = np.round(sphere[3] * nm2aafac, decimals=decimals)
            f.write(
                f'<marker id="{i + 1}" x="{x}" y="{y}" z="{z}" '
                f'radius="{radius}" r="{r}" g="{g}" b="{b}"/>\n'
            )
        f.write('</marker_set>\n')

    return fn


def _beads2bild(
    fn: Path | str,
    beads: list[dict],
    nm2aa: bool = True,
    decimals: int = 2
) -> Path:
    fn = Path(fn)
    if fn.suffix.lower() != '.bild':
        fn = fn.with_suffix('.bild')

    nm2aafac = 10 if nm2aa else 1

    def pt2str(pt):
        return ' '.join([f'{np.round(p * nm2aafac, decimals=decimals)}' for p in pt[:3]])

    def color2str(color):
        if isinstance(color, str):
            return color
        if hasattr(color, '__iter__') and len(color) == 3:
            return ' '.join([f'{c}' for c in color])
        raise ValueError(f'Invalid color {color}')

    with open(fn, 'w') as f:
        for i, bead in enumerate(beads):
            pos = bead['position']
            bead_radius = bead.get('radius', 1.0)
            bead_color = bead.get('color', 'white')
            alpha = bead.get('alpha', 1.0)
            transparency = 1 - alpha
            f.write(f'# bead {i + 1}\n')
            f.write(f'.transparency {transparency}\n')
            f.write(f'.color {color2str(bead_color)}\n')
            f.write(f'.sphere {pt2str(pos)} {np.round(bead_radius * nm2aafac, decimals=decimals)}\n')

    return fn


def _triads2bild(
    fn: Path | str,
    poses: np.ndarray,
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
        
    dist = np.mean(np.linalg.norm(poses[1:,:3,3]-poses[:-1,:3,3],axis=1))
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
        for i,tau in enumerate(poses):
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
    poses2pdb(fn,gen_config(params),seq)
    
def poses2pdb(
    fn: Path | str, 
    poses: np.ndarray, 
    seq: str
    ) -> None:
    fn = Path(fn)
    if fn.suffix.lower() != '.pdb':
        fn = fn.with_suffix('.pdb')
    if len(poses) != len(seq):
        raise ValueError(f'Dimension of poses ({poses.shape}) and seq ({len(seq)}) do not match.')
    iopmc.gen_pdb(str(fn), poses[:,:3,3], poses[:,:3,:3], sequence=seq, center=False)
    
def poses2cif(
    fn: Path | str, 
    poses: np.ndarray, 
    seq: str
    ) -> None:
    fn = Path(fn)
    if fn.suffix.lower() != '.cif':
        fn = fn.with_suffix('.cif')
    if len(poses) != len(seq):
        raise ValueError(f'Dimension of poses ({poses.shape}) and seq ({len(seq)}) do not match.')
    iopmc.gen_cif(str(fn), poses[:,:3,3], poses[:,:3,:3], sequence=seq, center=False)
    
    