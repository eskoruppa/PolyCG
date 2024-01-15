import sys
from typing import List, Tuple, Callable, Any, Dict
import numpy as np
import matplotlib.pyplot as plt  

from .marginals import gen_select, var_assign
from .cgNA_plus.modules.cgDNAUtils import constructSeqParms
from .mc_sampling import sample_free
from .conversions import cayleys2rotmats, rotmats2triads, statevec2vecs, eulers2rotmats
from .SO3 import so3


def select_params(confs: np.ndarray, type='y'):
    
    if len(confs.shape) > 2:
        raise ValueError('Dimension higher than 2 is not supported')
      
    vars = var_assign(seq, dof_names=["W", "x", "C", "y"])
    select = gen_select(seq, [type+'*'], raise_invalid=True)
    
    print(vars)
    sys.exit()
    
    if len(confs.shape) == 2:
        params = np.zeros((len(confs),6*len(select)))
        for i,sel in enumerate(select):
            vid = vars.index(sel)
            params[:,6*i:6*(i+1)] = confs[:,6*vid:6*vid+6]
        return params
    params = np.zeros((6*len(select)))
    for i,sel in enumerate(select):
        vid = vars.index(sel)
        params[6*i:6*(i+1)] = confs[6*vid:6*vid+6]
    return params


def cgdna_bps2conf(
    bps_params: np.ndarray, 
    factor: float=1
) -> Tuple[np.ndarray,np.ndarray]:
    
    N = len(bps_params) // 6
    triads = np.zeros((N+1,3,3))
    pos    = np.zeros((N+1,3))
    triads[0] = np.eye(3)
    for i in range(N):
        # rotational
        cay   = bps_params[i*6:i*6+3]
        trans = bps_params[i*6+3:i*6+6]
        
        print(f'cay_bp = {cay}')
        print(f'trans = {trans}')
        
        R = so3.cayley2rotmat(cay/factor)
        triads[i+1] = np.dot(triads[i],R)
        # translational
        G = so3.euler2rotmat(so3.rotmat2euler(R)*0.5)
        mid = np.dot(triads[i],G)
        dr = np.dot(mid,trans)
        pos[i+1] = pos[i] + dr
    return pos,triads


def cgdna_bps2confs(
    bps_params: np.ndarray, 
    factor: float=1
) -> Tuple[np.ndarray,np.ndarray]:
    
    if len(bps_params.shape) == 1:
        return cgdna_bps2conf(bps_params,factor=factor)
    pos = list()
    triads = list()
    for i in range(len(bps_params)):
        p,t = cgdna_bps2confs(bps_params[i],factor=factor)
        pos.append(p)
        triads.append(t)
    return np.array(pos), np.array(triads)

    
def cgdna_base_conf(
    bps_params: np.ndarray, 
    bp_params: np.ndarray, 
    factor: float=1
):
    
    if len(bps_params) // 6 + 1  != len(bp_params) // 6 :
        raise ValueError(f'Invalid relative dimension of bps_params ({len(bps_params)}) and bp_params ({len(bp_params)}).')
    
    bp_pos, bp_triads = cgdna_bps2conf(bps_params,factor=factor)
    nbp = len(bp_pos)
    if nbp != len(bp_params) // 6:
        raise ValueError(f'Number of base pairs ({nbp}) is inconsistent with number of intra base pair parameter sets ({len(bp_params) // 6})')
    
    b1_pos = list()
    b2_pos = list()
    b1_triads = list()
    b2_triads = list()
    
    for i in range(nbp):
        cay   = bp_params[i*6:i*6+3]
        trans = bp_params[i*6+3:i*6+6]
    
        Lam = so3.cayley2rotmat(cay/factor)
        sqrtLam = so3.euler2rotmat(so3.rotmat2euler(Lam)*0.5)
        assert np.sum(np.abs(Lam-np.dot(sqrtLam,sqrtLam))) <= 1e-10, 'Error in calculation sqrt Lambda'

        print(f'cay_b = {cay}')
        print(f'trans = {trans}')
        
        b1_T = bp_triads[i].dot(sqrtLam)
        b2_T = bp_triads[i].dot(sqrtLam.T)
        
        b1_p = bp_pos[i] + 0.5 * bp_triads[i].dot(trans) * 10
        b2_p = bp_pos[i] - 0.5 * bp_triads[i].dot(trans) * 10
        
        b1_triads.append(b1_T)
        b2_triads.append(b2_T)
        b1_pos.append(b1_p)
        b2_pos.append(b2_p)
    
    b1_triads = np.array(b1_triads)
    b2_triads = np.array(b2_triads)
    b1_pos = np.array(b1_pos)
    b2_pos = np.array(b2_pos)
    return bp_pos, bp_triads, b1_pos, b1_triads, b2_pos, b2_triads
        

def plot_conf(
    bp_pos: np.ndarray, 
    bp_triads: np.ndarray, 
    b1_pos: np.ndarray = None,
    b1_triads: np.ndarray = None,
    b2_pos: np.ndarray = None,
    b2_triads: np.ndarray = None,
):
    
    # ploting
    fig = plt.figure(figsize=(10/2.54,10/2.54), dpi=300,facecolor='w',edgecolor='k')
    ax  = fig.add_subplot(111,projection='3d')

    alpha_bp = 0.7
    alpha_base = 1


    ax.scatter(*bp_pos.T,s=20,edgecolors='black',linewidth=1,color='red',alpha=alpha_bp)
    ax.plot(*bp_pos.T,linewidth=0.2,color='red',alpha=alpha_bp)
    
    ax.scatter(*b1_pos.T,s=20,edgecolors='black',linewidth=1,color='blue',alpha=alpha_base)
    ax.plot(*b1_pos.T,linewidth=0.2,color='blue',alpha=alpha_base)
    
    ax.scatter(*b2_pos.T,s=20,edgecolors='black',linewidth=1,color='green',alpha=alpha_base)
    ax.plot(*b2_pos.T,linewidth=0.2,color='green',alpha=alpha_base)


    x1 = np.min(bp_pos[:,0])
    x2 = np.max(bp_pos[:,0])
    y1 = np.min(bp_pos[:,1])
    y2 = np.max(bp_pos[:,1])
    z1 = np.min(bp_pos[:,2])
    z2 = np.max(bp_pos[:,2])

    xrge = x2-x1
    yrge = y2-y1
    zrge = z2-z1
    rge = np.max([xrge,yrge,zrge])

    mx = 0.5*(x1+x2)
    my = 0.5*(y1+y2)
    mz = 0.5*(z1+z2)
    xlim = [mx-0.55*rge,mx+0.55*rge]
    ylim = [my-0.55*rge,my+0.55*rge]
    zlim = [mz-0.55*rge,mz+0.55*rge]

    for x in range(2):
        for y in range(2):
            for z in range(2):
                ax.scatter([xlim[x]],[ylim[y]],[zlim[z]],alpha=0)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # X = p[:,0]
    # Y = p[:,1]
    # Z = p[:,2]
    # # Create cubic bounding box to simulate equal aspect ratio
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    # Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # # Comment or uncomment following both lines to test the fake bounding box:
    # for xb, yb, zb in zip(Xb, Yb, Zb):
    #     ax.plot([xb], [yb], [zb], 'w')


    # Make legend, set axes limits and labels
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    # ax.view_init(elev=20., azim=-35, roll=0)

    plt.show()
    
      




if __name__ == "__main__":
    
    np.set_printoptions(linewidth=300)
    print('####################################################') 

    CGNA_DEFAULT_DATASET = "cgDNA+_Curves_BSTJ_10mus_FS"

    num_confs = 1000
    disc_len = 0.34
    mmax     = 50
    
    reps = 10
    
    polys = list()
    polys.append('at')
    polys.append('ac')
    polys.append('ag')
    polys.append('tc')
    polys.append('tg')
    polys.append('cg')
    
    for poly in polys:
        
        seq = poly*reps
        print(len(seq))
        
        seq = 'ATCG'
        
        gs, stiff = constructSeqParms(seq, CGNA_DEFAULT_DATASET)
        
        stiff = stiff.toarray()
        covmat = np.linalg.inv(stiff)
        confs = sample_free(covmat, num_confs, groundstate=None)
                
        vars = var_assign(seq, dof_names=["W", "x", "C", "y"])
        select = gen_select(seq, ['y*'], raise_invalid=True)
        

        rbps_fl = np.zeros((len(confs),3*len(select)))
        rbps_gs = np.zeros((3*len(select)))
        
        bps_fl = np.zeros((len(confs),6*len(select)))
        bps_gs = np.zeros((6*len(select)))
        
        for i,sel in enumerate(select):
            vid = vars.index(sel)
            rbps_fl[:,3*i:3*(i+1)] = confs[:,6*vid:6*vid+3]
            rbps_gs[3*i:3*(i+1)]   = gs[6*vid:6*vid+3]
            
            bps_fl[:,6*i:6*(i+1)] = confs[:,6*vid:6*vid+6]
            bps_gs[6*i:6*(i+1)]   = gs[6*vid:6*vid+6]
            
        
        bps_fl = select_params(confs,'y')
        bps_gs = select_params(gs,'y')
        
        bp_fl = select_params(confs,'x')
        bp_gs = select_params(gs,'x')
        
        print(bps_gs.shape[0] // 6)
        print(bp_gs.shape[0] // 6)
        
        bps = bps_fl + bps_gs
        bp  = bp_fl  + bp_gs
        

        print(bps.shape)
        print(bp.shape)
        
        # bp_pos, bp_triads, b1_pos, b1_triads, b2_pos, b2_triads = cgdna_base_conf(bps[0],bp[0],factor=5)
        bp_pos, bp_triads, b1_pos, b1_triads, b2_pos, b2_triads = cgdna_base_conf(bps_gs,bp_gs,factor=5)
        
        plot_conf(bp_pos, bp_triads, b1_pos, b1_triads, b2_pos, b2_triads)
        
        sys.exit()
        

            
        rbps_fl *= 1./5 #* 180./np.pi
        rbps_gs *= 1./5 #* 180./np.pi

        rbps = rbps_fl + rbps_gs
        bps  = bps_fl  + bps_gs
        
        # print(bps_gs[5::6])
        # sys.exit()
        
        pos,triads = cgdna_bps2confs(bps,factor=5)
        
        print(pos.shape)
        print(triads.shape)
        
        
        vecs = statevec2vecs(rbps)
        rotmats = cayleys2rotmats(vecs)
        triads2  = rotmats2triads(rotmats)
        
        print(triads[0,1])
        print(triads2[0,1])
        
        print(np.sum(triads-triads2))
        
        
        ### Figure
        
        
     
        
        
        
        
        
        
        
        sys.exit()
        


                    
        # lb = persistence_length(triads,disc_len=disc_len,mmax=mmax)
        # print('################################################################################')
        # print(f'Poly {poly}')
        # print(seq)
        # print(lb[:,1].T)    