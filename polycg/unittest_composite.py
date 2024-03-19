import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix
from scipy import sparse
import scipy as sp
import sys, time
from matplotlib import pyplot as plt

from typing import List, Tuple, Callable, Any, Dict
from .cgnaplus import cgnaplus_bps_params

from .transform_cayley2euler import *
from .transform_marginals import *
from .transform_statevec import *
from .transform_algebra2group import *
from numba import njit

from .composites import *


def test_block(seq: str = 'ACGATC', num_confs = 10000):
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    # generate stiffness
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq,translations_in_nm=True)
    print('stiff generated')
    
    translation_as_midstep = True
    # convert to eulers
    algebra_gs  = cayley2euler(cayley_gs)
    algebra_stiff = cayley2euler_stiffmat(cayley_gs,cayley_stiff,rotation_first=True)    
    group_gs = np.copy(algebra_gs)
    if translation_as_midstep:
        for i,vec in enumerate(group_gs):
            Phi_0 = vec[:3]
            zeta_0 = vec[3:]
            sqrtS = so3.euler2rotmat(0.5*Phi_0)
            s = sqrtS @ zeta_0
            group_gs[i,3:] = s
    
    group_stiff = algebra2group_stiffmat(algebra_gs,algebra_stiff,rotation_first=True,translation_as_midstep=translation_as_midstep)
    
    idfrom = 30
    idto   = 33

    nsteps = idto - idfrom
    group_gs = group_gs[idfrom:idto]
    group_stiff = group_stiff[idfrom*6:idto*6,idfrom*6:idto*6]
    
    gs_rots  = group_gs[:,:3]
    gs_trans = group_gs[:,3:]
    
    Saccus = get_Saccu(gs_rots,0,2)
    
    
    print(so3.euler2rotmat(gs_rots[-1]))
    
    




    
def test_composite(seq: str = 'ACGATC',num_confs = 10000):
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    # generate stiffness
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq,translations_in_nm=True)
    print('stiff generated')
    
    translation_as_midstep = True
    
    # convert to eulers
    algebra_gs  = cayley2euler(cayley_gs)
    algebra_stiff = cayley2euler_stiffmat(cayley_gs,cayley_stiff,rotation_first=True)
    
    
    
    # rot = algebra_gs[0,:3]
    # trans = algebra_gs[0,3:]
    # S = so3.euler2rotmat(rot)
    # print(S.T @ so3.hat_map(trans) @ S)
    # print(so3.hat_map(S.T @ trans))
    # sys.exit()
    
    group_gs = np.copy(algebra_gs)
    if translation_as_midstep:
        for i,vec in enumerate(group_gs):
            Phi_0 = vec[:3]
            zeta_0 = vec[3:]
            sqrtS = so3.euler2rotmat(0.5*Phi_0)
            s = sqrtS @ zeta_0
            group_gs[i,3:] = s
    
    from .transform_midstep2triad import midstep2triad
    test_group_gs = midstep2triad(algebra_gs)
    
    # for i in range(len(group_gs)):
    #     print(np.abs(np.sum(group_gs[i]-test_group_gs[i])))
    # sys.exit()
    
    group_stiff = algebra2group_stiffmat(algebra_gs,algebra_stiff,rotation_first=True,translation_as_midstep=translation_as_midstep)
    
    idfrom = 30
    idto   = 40
    
    nsteps = idto - idfrom
    
    group_gs = group_gs[idfrom:idto]
    group_stiff = group_stiff[idfrom*6:idto*6,idfrom*6:idto*6]
    
    # for i in range(len(group_gs)):
    #     tmp = group_gs[i,5] 
    #     group_gs[i,5] = group_gs[i,4] 
    #     group_gs[i,4] = tmp
    
    switched_gs = np.copy(group_gs)
    # for i in range(len(switched_gs)):
    #     switched_gs[i,5] = group_gs[i,4] 
    #     switched_gs[i,4] = group_gs[i,5]
    #     switched_gs[i,3] = group_gs[i,5]
    
    
    print(group_stiff[18:24,18:24]*0.34)

    # print(marginal_schur_complement(group_stiff[18:24,18:24]*0.34,retained_ids=[0,1,2]))
    # print(marginal_schur_complement(group_stiff[24:30,24:30]*0.34,retained_ids=[0,1,2]))
    # print(marginal_schur_complement(group_stiff[30:36,30:36]*0.34,retained_ids=[0,1,2]))
    # sys.exit()
    
    print('gs')
    print(group_gs)
    print('switched')
    print(switched_gs)
    # sys.exit()
    
    
    # fac = 20
    # for i in range(len(group_stiff)//6):
    # #     group_stiff[i*6+3,:] *= fac
    # #     group_stiff[:,i*6+3] *= fac
    # #     group_stiff[i*6+4,:] *= fac
    # #     group_stiff[:,i*6+4] *= fac
    # #     group_stiff[i*6+5,:] *= fac
    # #     group_stiff[:,i*6+5] *= fac
    #     # group_stiff[i*6+0,i*6+0] *= fac
    #     group_stiff[i*6+3,i*6+3] *= fac
    #     group_stiff[i*6+4,i*6+4] *= fac
        
    Tv = composite_matrix(switched_gs)
    Ti = inv_composite_matrix(switched_gs)
    
    # print(np.sum(np.linalg.inv(Tv)-Ti))
    # sys.exit()
    
    # print(Tv.shape)
    # print(Ti.shape)
    # np.set_printoptions(linewidth=250,precision=2,suppress=True)
    # print(Tv @ Ti)
    # print(np.linalg.det(Tv @ Ti))
    
    # sys.exit()
    
    group_cov = np.linalg.inv(group_stiff)
    group_dx = np.random.multivariate_normal(np.zeros(len(group_cov)), group_cov)
    
    print(group_dx.shape)
    group_dx = statevec2vecs(group_dx,vdim=6)
    print(group_dx.shape)
    
    Ss = euler2rotmat(group_gs)
    Ds = euler2rotmat(group_dx)
    
    Rn = np.eye(4)
    Sn = np.eye(4)
    for i in range(len(Ss)):
        Rn = Rn @ Ss[i] @ Ds[i]
        Sn = Sn @ Ss[i]
    
    Dn = np.linalg.inv(Sn) @ Rn
    sum = so3.se3_rotmat2euler(Dn)
    comp_block = composite_block(group_gs)
    group_vec = group_dx.flatten()
    comp = comp_block @ group_vec
    print('sum')
    print(sum)
    print('comp')
    print(comp)

    
    
    print('relative difference')
    print(np.abs((comp-sum)/group_gs[0]))
    
    Mcomp = Ti.T @ group_stiff @ Ti
    
    Msum = marginal_schur_complement(Mcomp,retained_ids=[i+len(Mcomp)-6 for i in range(6)])
    
    # print('calculated stiffness matrix')
    # print(Msum)
    
    
    # num_confs = 10000
    group_dx = np.random.multivariate_normal(np.zeros(len(group_cov)), group_cov,num_confs)
    group_dx = statevec2vecs(group_dx,vdim=6)
    
    sums = np.zeros((num_confs,6))
    
    for c in range(num_confs):
        Ds = euler2rotmat(group_dx[c])
        Rn = np.eye(4)
        for i in range(len(Ss)):
            Rn = Rn @ Ss[i] @ Ds[i]
        Dn = np.linalg.inv(Sn) @ Rn
        sums[c] = so3.se3_rotmat2euler(Dn)
    
    
    
    
    rise_var = 1./marginal_schur_complement(Msum,retained_ids=[5])[0]
    rise_std = np.sqrt(rise_var)
    maxrise = np.max(np.abs(sums[:,5]))*0.66
    maxrise = rise_std*3
    xvals = np.linspace(-maxrise,maxrise,500)
    norm = sp.stats.norm.pdf(xvals,0,rise_std)    
    
    print('skewness')
    print(sp.stats.skew(sums[:,3]))
    print(sp.stats.skew(sums[:,4]))
    print(sp.stats.skew(sums[:,5]))
    
    print('kurtosis')
    print(sp.stats.kurtosis(sums[:,3]))
    print(sp.stats.kurtosis(sums[:,4]))
    print(sp.stats.kurtosis(sums[:,5]))    
    
    
    
    fig = plt.figure(figsize=(17./2.54,7./2.54))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    axes = [ax1,ax2,ax3,ax4,ax5,ax6]
    labels = ['Tilt','Roll','Twist','Shift','Slide','Rise']
    
    tick_pad            = 2
    axlinewidth         = 0.9
    axtick_major_width  = 0.6
    axtick_major_length = 1.6
    tick_labelsize      = 6
    label_fontsize      = 7
    
    for i,ax in enumerate(axes):
        ax.hist(sums[:,i],bins=100,density=True)
        
        var = 1./marginal_schur_complement(Msum,retained_ids=[i])[0]
        std = np.sqrt(var)
        rnge = std*4
        xvals = np.linspace(-rnge,rnge,500)
        norm = sp.stats.norm.pdf(xvals,0,std) 
        ax.plot(xvals,norm,c='black',lw=1.0)
        
        ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
        ax.set_xlabel(labels[i],size = label_fontsize,labelpad=1)
        ax.set_ylabel('Probability Density',size = label_fontsize,labelpad=1)
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.7)
        
    # ax6.set_xlim([-2.1,2.1])  
    
    plt.subplots_adjust(left=0.06,
                    right=0.98,
                    bottom=0.08,
                    top=0.99,
                    wspace=0.2,
                    hspace=0.22)
    
    plt.savefig(f'Figs/Distributions.png',dpi=300,facecolor='white')
    plt.close()

    
    print(sums.shape)
    
    sums_mean = np.mean(sums,axis=0)
    print(f'sum_mean = {sums_mean}')
    print(f'sum_std  = {np.std(sums,axis=0)}')
    diffs = sums - sums_mean
    
    print('calculate covariance matrix')
    cov = covmat(sums)
    print('done')
    print('calculate diffs covariance matrix')
    diffs_cov = covmat(diffs)
    print('done')
    
    stiff_sampled = np.linalg.inv(cov)
    stiff_sampled_diffs = np.linalg.inv(diffs_cov)
    print('sampled stiffness matrix')
    print(stiff_sampled)
    print('sampled stiffness matrix diffs')
    print(stiff_sampled_diffs)
    print('calculated stiffness matrix')
    print(Msum)
    
    print('##############################')
    print('Marginals')
    print('rotations')
    print(marginal_schur_complement(stiff_sampled,retained_ids=[0,1,2])*nsteps*0.34)
    print(marginal_schur_complement(Msum,retained_ids=[0,1,2])*nsteps*0.34)
    
    print('translations')
    print(marginal_schur_complement(stiff_sampled,retained_ids=[3,4,5])*nsteps)
    print(marginal_schur_complement(Msum,retained_ids=[3,4,5])*nsteps)
    
    print('shift')
    print(marginal_schur_complement(stiff_sampled,retained_ids=[3])*nsteps)
    print(marginal_schur_complement(Msum,retained_ids=[3])*nsteps)
    print('slide')
    print(marginal_schur_complement(stiff_sampled,retained_ids=[4])*nsteps)
    print(marginal_schur_complement(Msum,retained_ids=[4])*nsteps)
    print('rise')
    print(marginal_schur_complement(stiff_sampled,retained_ids=[5])*nsteps)
    print(marginal_schur_complement(Msum,retained_ids=[5])*nsteps)
    
    print('##############################')
    print('Eliminate Rise')
    print(marginal_schur_complement(stiff_sampled,retained_ids=[0,1,2,3,4]))
    print(marginal_schur_complement(Msum,retained_ids=[0,1,2,3,4]))
    print(marginal_schur_complement(marginal_schur_complement(Msum,retained_ids=[0,1,2,3,4]),retained_ids=[0,1,2]))
    
    
    
    
     
    
    
    
    

def test_composite_rot(seq: str = 'ACGATC',num_confs = 10000):
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    print('#########################################')
    print('#########################################')
    print('#########################################')
    
    # generate stiffness
    cayley_gs,cayley_stiff = cgnaplus_bps_params(seq,translations_in_nm=True)
    print('stiff generated')
    
    translation_as_midstep = True
    
    # convert to eulers
    algebra_gs  = cayley2euler(cayley_gs)
    algebra_stiff = cayley2euler_stiffmat(cayley_gs,cayley_stiff,rotation_first=True)
    
    group_gs = np.copy(algebra_gs)
    if translation_as_midstep:
        for i,vec in enumerate(group_gs):
            Phi_0 = vec[:3]
            zeta_0 = vec[3:]
            sqrtS = so3.euler2rotmat(0.5*Phi_0)
            s = sqrtS @ zeta_0
            group_gs[i,3:] = s
    
    group_stiff = algebra2group_stiffmat(algebra_gs,algebra_stiff,rotation_first=True,translation_as_midstep=translation_as_midstep)
    
    idfrom = 10
    idto   = 20
    
    group_gs = group_gs[idfrom:idto]
    group_stiff = group_stiff[idfrom*6:idto*6,idfrom*6:idto*6]
    
    group_stiff = matrix_rotmarginal(group_stiff)
    group_gs = group_gs[:,:3]
    
    Tv = composite_matrix(group_gs)
    Ti = inv_composite_matrix(group_gs)
    
    group_cov = np.linalg.inv(group_stiff)
    group_dx = np.random.multivariate_normal(np.zeros(len(group_cov)), group_cov)
    
    print(group_dx.shape)
    group_dx = statevec2vecs(group_dx,vdim=3)
    # group_dx = group_dx[:,:3]
    print(group_dx.shape)
    
    Ss = euler2rotmat(group_gs)
    Ds = euler2rotmat(group_dx)
    
    Rn = np.eye(3)
    Sn = np.eye(3)
    for i in range(len(Ss)):
        Rn = Rn @ Ss[i] @ Ds[i]
        Sn = Sn @ Ss[i]
    
    Dn = np.linalg.inv(Sn) @ Rn
    sum = so3.rotmat2euler(Dn)
    
    comp_block = composite_block(group_gs)
    
    group_vec = group_dx.flatten()
    
    comp = comp_block @ group_vec
    print('sum')
    print(sum)
    print('comp')
    print(comp)

    
    # print(group_gs*180/np.pi)
    # print(np.sum(group_gs,axis=0))
        
    print('relative difference')
    print(np.abs((comp-sum)/group_gs[0]))
    
    Mcomp = Ti.T @ group_stiff @ Ti
    Msum = marginal_schur_complement(Mcomp,retained_ids=[i+len(Mcomp)-3 for i in range(3)])
    print(Msum)
    
    
    # num_confs = 100000
    group_dx = np.random.multivariate_normal(np.zeros(len(group_cov)), group_cov,num_confs)
    group_dx = statevec2vecs(group_dx,vdim=3)
    
    sums = np.zeros((num_confs,3))
    
    for c in range(num_confs):
        Ds = euler2rotmat(group_dx[c])
        Rn = np.eye(3)
        for i in range(len(Ss)):
            Rn = Rn @ Ss[i] @ Ds[i]
        Dn = np.linalg.inv(Sn) @ Rn
        sums[c] = so3.rotmat2euler(Dn)
    
    print(sums.shape)
    
    print('calculate covariance matrix')
    cov = covmat(sums)
    print('done')
    
    print('sampled stiffness matrix')
    stiff_sampled = np.linalg.inv(cov)
    print(stiff_sampled*10*0.34)
    print('calculated stiffness matrix')
    print(Msum*10*0.34)
    
    
    
    
    
    
    
    
    
    
    
@njit
def covmat(vecs):
    dim = vecs.shape[-1]
    cov = np.zeros((dim,dim))
    for i in range(len(vecs)):
        cov += np.outer(vecs[i],vecs[i])
    cov /= len(vecs)
    return cov
    




if __name__ == "__main__":
    
    seq = 'ACGATCGATCGGAATCCGATCATACTGGC'*5
    num_confs = 5000000
     
    print(f'len = {len(seq)}')
    
    # test_block(seq,num_confs=num_confs)
    
    test_composite(seq,num_confs=num_confs)
    
    # test_composite_rot(seq,num_confs=num_confs)