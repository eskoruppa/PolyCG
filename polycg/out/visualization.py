import os
import numpy as np
from ..IOPolyMC import iopolymc as iopmc
from ..genconf import gen_config

def cgvisual(basefn: str, params: np.ndarray, seq: str, cg: int, startid: int = 0, bead_radius: float = None, disc_len: float=0.34, include_bps_triads: bool = False):
    
    # generate configuration
    taus = gen_config(params,disc_len=disc_len)

    # # create folder
    # if not os.path.isdir(outdir):
    #     os.makedirs(outdir)
    
    # generate pdb file
    pdbfn = basefn + '.pdb'
    params2pdb(pdbfn, params, seq)
    
    # create bild file for triads
    bildfn = basefn + '_triads.bild'
    cgtaus = taus[startid::cg]
    triads2bild(bildfn, cgtaus, alpha=1., scale=1, nm2aa=True, decimals=2)
    
    bps_triads_bildfn = None
    if include_bps_triads and cg > 1:
        bps_triads_bildfn = basefn + '_bps_triads.bild'
        triads2bild(bps_triads_bildfn, taus, alpha=1., scale=1, nm2aa=True, decimals=2)
        
    
    # create chimera cxc file
    cxcfn = basefn + '.cxc'
    if bead_radius > 0:
        spheres = np.zeros((len(cgtaus),4))
        spheres[:,:3] = cgtaus[:,:3,3]
        spheres[:,3] = bead_radius
    else:
       spheres = None
    chimeracxc(cxcfn, pdbfn, triadfn=bildfn, spheres=spheres, nm2aa= True, decimals=2, additional_triadfn=bps_triads_bildfn)
    
    cgxyzfn = basefn + '_cg.xyz'
    xyz = {
        'types': ['C']*(len(cgtaus)),
        'pos'  : [cgtaus[:,:3,3]]
        }
    iopmc.write_xyz(cgxyzfn,xyz)

   
def chimeracxc(fn: str, pdbfn: str, triadfn: str = None, spheres: np.ndarray = None, nm2aa: bool = True, decimals=2, additional_triadfn: str = None):
    if os.path.splitext(fn)[-1].lower() != '.cxc':
        fn += '.cxc'
    
    modelnum = 0
    triadsid = 0
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
        f.write(f'open {os.path.basename(pdbfn)}\n') 
        f.write(f'open {os.path.basename(pdbfn)}\n')
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
            modelnum += 1 
            triadsid = modelnum
            f.write(f'\n# load triads BILD\n')
            f.write(f'open {os.path.basename(triadfn)}\n') 
            
        # open triads
        if additional_triadfn is not None:
            modelnum += 1 
            add_triadsid = modelnum
            f.write(f'\n# load additional triads BILD\n')
            f.write(f'open {os.path.basename(additional_triadfn)}\n') 
        
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

              
# def triads2bild(fn: str, taus: np.ndarray, alpha: float = 1., ucolor = 'green', vcolor = 'blue', tcolor = 'red', scale: float = 1, nm2aa: bool = True, decimals=2): 
def triads2bild(fn: str, taus: np.ndarray, alpha: float = 1., ucolor = 'default', vcolor = 'default', tcolor = 'default', scale: float = 1, nm2aa: bool = True, decimals=2): 
    
    if ucolor == 'default':
        ucolor = [64/255,91/255,4/255]
        # ucolor = [0.15294118, 0.47843137, 0.17647059]
    if vcolor == 'default':
        vcolor = [61/255,88/255,117/255]
        # vcolor = [0.17647059, 0.15294118, 0.47843137]
    if tcolor == 'default':
        tcolor = [153/255,30/255,46/255]
        # tcolor = [0.47843137, 0.17647059, 0.15294118]
        
    if os.path.splitext(fn)[-1].lower() != '.bild':
        fn += '.bild'
    
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
    
def params2pdb(fn: str, params: np.ndarray, seq: str):
     return taus2pdb(fn,gen_config(params),seq)
    
def taus2pdb(fn: str, taus: np.ndarray, seq: str):
    if os.path.splitext(fn)[-1].lower() != '.pdb':
        fn += '.pdb'
    if len(taus) != len(seq):
        raise ValueError(f'Dimension of taus ({taus.shape}) and seq ({len(seq)}) do not match.')
    iopmc.gen_pdb(fn, taus[:,:3,3], taus[:,:3,:3], sequence=seq, center=False)
    return fn
    
    