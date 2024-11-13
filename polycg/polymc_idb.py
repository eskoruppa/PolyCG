import sys, os
import time
import argparse
import numpy as np
import scipy as sp
from typing import Any, Callable, Dict, List, Tuple

from .utils.bmat import BlockOverlapMatrix

from .IOPolyMC.iopolymc import write_idb
from .utils.seq import (
    sequence_file,
    unique_oli_seq,
    unique_olis_in_seq,
    seq2oliseq,
    all_oligomers,
    randseq,
)

from .utils.load_seq import load_sequence
from .partials import partial_stiff
from .cg import coarse_grain
from .transforms.transform_marginals import matrix_rotmarginal,vector_rotmarginal
from .transforms.transform_statevec import statevec2vecs, vecs2statevec

from .cgnaplus import cgnaplus_bps_params
from .models.RBPStiff.read_params import GenStiffness

from .transforms.transform_SE3 import euler2rotmat_se3
from .IOPolyMC.iopolymc import write_xyz, gen_pdb

"""
 TO DO:
    - For now the ground state vectors are always multiplied by 180/pi regardless 
      of whether gs_units was set to 'rad' or 'deg'. This needs to be conditional.
"""


##########################################################################################################
############### Generate IDB and seq files ###############################################################
##########################################################################################################


def stiff2idb(
    basefilename: str,
    gs: np.ndarray,
    stiff: BlockOverlapMatrix,
    couprange: int,
    closed: bool,
    seq: str | None = None,
    disc_len: float = 0.34,
    avg_inconsist: bool = True,
    generate_missing: bool = True,
    unique_sequence: bool = True,
    boundary_char: str = "x",
    exclude_chars: str = "y",
) -> None:
    ndims = 3
    Nbps = len(gs) // ndims
    if closed:
        Nbp = Nbps
    else:
        Nbp = Nbps + 1
    olisize = couprange2olisize(couprange)

    if seq is None or (unique_sequence and not unique_olis_in_seq(seq, olisize)):
        assignseq, chars = unique_oli_seq(
            Nbp, olisize, closed=closed, boundary=boundary_char, exclude=exclude_chars
        )
    else:
        assignseq = str(seq)
        chars = "".join(sorted(set(assignseq)))
        
    # allow cross boundary assignment
    stiff.check_bounds_on_read = False
    params = dict()
    
    for i in range(Nbps):
        oliseq = seq2oliseq(assignseq, i, couprange, closed)
        T0 = gs[i * ndims : (i + 1) * ndims]
        cl = (i - couprange) * ndims
        cu = (i + couprange + 1) * ndims
        
        M = _matassign(stiff ,cl ,cu, closed)
        if cl > 0 and cu < stiff.shape[0]:
            Mc = stiff[cl:cu, cl:cu]
            if (np.sum(Mc-M) > 1e-10):
                raise ValueError('Inconsistent matrix assignment')
                
        coups = _mat2idbcoups(M)
        seqparams = {"seq": oliseq, "vec": T0, "interaction": coups}
        params[oliseq] = seqparams

    if generate_missing:
        params = _add_missing_params(params, couprange, chars, ndims=ndims)

    idbdict = dict()
    idbdict["interaction_range"] = couprange
    idbdict["monomer_types"] = chars
    idbdict["disc_len"] = disc_len
    idbdict["avg_inconsist"] = avg_inconsist
    idbdict["params"] = params

    idbfn = str(basefilename)
    if ".idb" not in idbfn.lower():
        idbfn += ".idb"
    write_idb(idbfn, idbdict, decimals=3)
    sequence_file(basefilename, assignseq, add_extension=True)
    sequence_file(basefilename + ".origseq", seq, add_extension=False)

def _matassign(stiff ,cl: int ,cu: int, closed: bool):
    
    def _select_dense(stiff,cl,cu):
        P = stiff[cl:cu,cl:cu]
        if sp.sparse.issparse(P):
            P = P.toarray()
        return P
    
    size = cu-cl
    if size > stiff.shape[0]:
        raise ValueError(f'selection range larger than matrix')
    al = 0
    au = size
    M = np.zeros((size,size))
    if cl < 0:
        if closed:
            raise ValueError(f'closed not yet implemented: properly implement this with bmat')
        al = -cl 
        cl = 0
    if cu > stiff.shape[0]:
        if closed:
            raise ValueError(f'closed not yet implemented: properly implement this with bmat')
        au = size - (cu - stiff.shape[0])
        cu = stiff.shape[0]
    
    M[al:au,al:au] = _select_dense(stiff,cl,cu)
    return M
    
def _mat2idbcoups(M: np.ndarray):
    ndims = 3
    N = M.shape[0] // ndims
    couprange = (N - 1) // 2
    mats = [
        M[couprange * ndims : (couprange + 1) * ndims, i * ndims : (i + 1) * ndims]
        for i in range(N)
    ]
    for i in range(couprange):
        mats[i] = mats[i].T
    return [_mat2idb_entry(mat) for mat in mats]


def _mat2idb_entry(mat: np.ndarray):
    entry = ["stiffmat"]
    for i in range(3):
        for j in range(3):
            entry.append(mat[i, j])
    return entry


def couprange2olisize(couprange: int) -> int:
    return (1 + couprange) * 2


def olisize2couprange(olisize: int) -> int:
    return olisize // 2 - 1


def _add_missing_params(
    params: dict[str, Any], coup_range: int, chars: str, ndims: int = 3
):
    num_coup = 1 + 2 * coup_range
    oli_size = (coup_range + 1) * 2
    contained_olis = [key for key in params.keys()]
    for oli in all_oligomers(oli_size, chars):
        if oli not in contained_olis:
            seqparams = {
                "seq": oli,
                "vec": np.zeros(3),
                "interaction": [
                    _mat2idb_entry(np.zeros((3, 3))) for i in range(num_coup)
                ],
            }
            params[oli] = seqparams
    return params

def gen_gs_config(gs,disc_len):
    n = len(gs)
    ext_gs = np.zeros((n,6))
    ext_gs[:,:3] = gs
    ext_gs[:,5]  = disc_len
    se3trans = euler2rotmat_se3(ext_gs)
    taus = np.zeros((n+1,4,4))
    taus[0] = np.eye(4)
    for i in range(n):
        taus[i+1] = taus[i] @ se3trans[i]
    return taus


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate PolyMC input files")
    parser.add_argument("-m", "--model", type=str, default = 'cgnaplus',choices=['cgnaplus','lankas','olson'])
    parser.add_argument("-cg", "--composite_size", type=int, default = [1], nargs='*')
    parser.add_argument("-cr", "--coupling_range", type=int, default = 4)
    parser.add_argument('-seqfns', '--sequence_files', type=str, required=True, nargs='*') 
    parser.add_argument('-closed', '--closed', action='store_true') 
    parser.add_argument('-sc', '--scale_factor', type=float, default=1) 
    
    parser.add_argument('-nc', '--no_crop', action='store_true') 
    parser.add_argument('-dl', '--disc_len', type=float, default=0.34) 
    parser.add_argument('-ai', '--avg_inconsistency', action='store_true') 
    parser.add_argument('-gm', '--generate_missing', action='store_true') 
    
    parser.add_argument('-fid', '--first_id', type=int, default=0) 
    parser.add_argument('-gsu', '--gs_units', type=str, default='deg', choices=['rad','deg','Rad','Deg']) 
    parser.add_argument('-rm', '--rotation_map', type=str, default='euler', choices=['euler','cayley']) 
    parser.add_argument('-sf', '--split_fluctuations', type=str, default='matrix', choices=['matrix','vector']) 
    parser.add_argument('-ss', '--simple_stiff', action='store_true')
    parser.add_argument('-C',  '--Cvalue', type=float, default=100) 
    parser.add_argument('-A',  '--Avalue', type=float, default=40) 
    parser.add_argument('-xyz',  '--gen_xyz', action='store_true') 
    parser.add_argument('-gs',  '--dump_gs', action='store_true') 
     
    
    args = parser.parse_args()
    
    model           = args.model
    composite_sizes = args.composite_size
    couprange       = args.coupling_range
    seqfns          = args.sequence_files
    closed          = args.closed
    
    scale_factor    = args.scale_factor
    
    allow_crop          = not args.no_crop
    disc_len            = args.disc_len
    avg_inconsist       = args.avg_inconsistency
    generate_missing    = args.generate_missing
    
    first_id            = args.first_id
    gs_units            = args.gs_units
    rotation_map        = args.rotation_map
    split_fluctuations  = args.split_fluctuations
        
    
    print('############################################')
    print('Settings:')
    print(f'model:                  {model}')
    print(f'sequence files:         {seqfns}')
    print(f'composite size:         {composite_sizes}')
    print(f'coupling ranges:        {couprange}')
    print(f'closed:                 {closed}')
    print(f'scale factor:           {scale_factor}')
    print(f'allow crop:             {allow_crop}')
    print(f'first first_id:         {first_id}')
    print(f'rotation map:           {rotation_map}')
    print(f'split fluctuations:     {split_fluctuations}')
    print(f'groundstate units:      {gs_units}')
    print(f'average inconsistency:  {avg_inconsist}')
    print(f'generate_missing:       {generate_missing}')
    print('')
    
    include_deactivate_static = False
    
    # ps_set = 'cgDNA+ps1'
    ps_set = 'cgDNA+_Curves_BSTJ_10mus_FS'
    # ps_set = 'cgDNA+_MLE_ends12mus_ends_CGF_BSC1'
    # ps_set = 'Di_hmethyl_methylated-hemi_combine'
    ndims               = 3
    
    for seqfn in seqfns:
        # load seq:       
        seq = load_sequence(seqfn)
        if len(seq) == 0:
            raise IOError('Empty sequence found')
        print(f'found sequence of length {len(seq)}')
        seq = seq.lower()
        # seq = seq[:1500]
        unique_sequence = True
        
        ##########################################################
        ##########################################################
        # select models
        
        ################################
        # cgNAplus
        if model.lower() == 'cgnaplus':
            print('generating stiffness with cgNA+')
            
            unique_sequence = True
            method = cgnaplus_bps_params
            stiffgen_args = {
                'translations_in_nm': True, 
                'euler_definition': True, 
                'group_split' : True,
                'parameter_set_name' : 'curves_plus',
                'remove_factor_five' : True,
                'rotations_only': True
                }
            
            block_size = 120
            overlap_size = 20
            tail_size = 20
            nbps = len(seq)-1
            
            if overlap_size > nbps:
                overlap_size = nbps-1
            if block_size > nbps:
                block_size = nbps
            
            print('Generating partial stiffness matrix with')    
            print(f'block_size:   {block_size}')
            print(f'overlap_size: {overlap_size}')
            print(f'tail_size:    {tail_size}')

            gs,stiff = partial_stiff(seq,method,stiffgen_args,block_size=block_size,overlap_size=overlap_size,tail_size=tail_size,closed=closed,ndims=3)
        
        ################################
        # RBPStiff
        
        if model.lower() in ['rbp','rbpstiff','lankas']:
            print('generating stiffness with RBPStiff')
            genstiff = GenStiffness(method='md')
            stiff, gs = genstiff.gen_params(seq,use_group=True,sparse=True)
            gs    = statevec2vecs(vector_rotmarginal(vecs2statevec(gs)),vdim=3)
            stiff = matrix_rotmarginal(stiff)
            
        if model.lower() in ['crystal','olson']:
            print('generating stiffness with RBPStiff')
            genstiff = GenStiffness(method='crystal')
            stiff, gs = genstiff.gen_params(seq,use_group=True,sparse=True)
            gs    = statevec2vecs(vector_rotmarginal(vecs2statevec(gs)),vdim=3)
            stiff = matrix_rotmarginal(stiff)
            
        ##########################################################
        ##########################################################
        # Rescale Stiffness
        if scale_factor != 1:
            stiff *= scale_factor  

        ##########################################################
        ##########################################################
        # Generate Composite Stiffness

        for composite_size in composite_sizes:
            
            outfn = os.path.splitext(seqfn)[0] + f'_{model}_{composite_size}bp' + f'_{couprange}cr'
            if scale_factor != 1:
                outfn += ('_rescaled_%.3f'%scale_factor).replace('.','p')
            
            block_ncomp   = np.max([int(np.ceil(160/composite_size)),2*couprange])
            overlap_ncomp = int(np.max([couprange,2]))
            tail_ncomp    = np.max([int(np.ceil(40/composite_size)),couprange])
          
            ##########################################################
            ##########################################################
            # Coarse grain and write to file
            
            if composite_size == 1:
                
                ##########################################################
                # Marginalize translations
                # print('Calculating rotational marginals')
                # gs_rot = vector_rotmarginal(vecs2statevec(gs))
                # stiff_rot = matrix_rotmarginal(stiff)
                gs_rot = vecs2statevec(gs)
                stiff_rot=stiff.copy()

                ##########################################################
                # Transform groundstate units
                if gs_units.lower() in ['deg','degree','degrees']:
                    gs_rot = np.rad2deg(gs_rot)
                
                ##########################################################
                # Express stiffness in nm
                stiff_nm = stiff_rot * disc_len
                
                if args.simple_stiff:
                    print('Using simple stiffness')
                    A = args.Avalue
                    C = args.Cvalue
                    for i in range(stiff_nm.shape[0]//3):
                        stiff_nm[i*3:(i+1)*3,i*3:(i+1)*3] = [[A,0,0],[0,A,0],[0,0,C]]
                     
                ##########################################################
                # Write to IDB
                print('Create IDB file')
                stiff2idb(
                    outfn,gs_rot,stiff_nm,couprange,closed,seq,disc_len=disc_len,avg_inconsist=avg_inconsist,generate_missing=generate_missing,unique_sequence=True
                )
                if include_deactivate_static:
                    outfn_ns =  outfn + '_no_static'
                    gs_ns = np.copy(gs_rot)
                    gs_ns[0::3] = 0
                    gs_ns[1::3] = 0
                    stiff2idb(
                        outfn_ns,gs_ns,stiff_nm,couprange,closed,seq,disc_len=disc_len,avg_inconsist=avg_inconsist,generate_missing=generate_missing,unique_sequence=True
                    )
                    
                if args.gen_xyz:
                    taus = gen_gs_config(gs,disc_len)
                    
                    taus[:,:3,3] = taus[:,:3,3] - np.mean(taus[:,:3,3],axis=0)
                    
                    pdbfn = outfn + '_gs.pdb' 
                    gen_pdb(pdbfn, taus[:,:3,3], taus[:,:3,:3], sequence=seq, center=False)
                    
                    xyz = {
                        'types': ['C']*(len(taus)),
                        'pos'  : [taus[:,:3,3]]
                        }
                    xyzfn = outfn + '_gs'
                    write_xyz(xyzfn,xyz)
                    
                    taus = taus[::10]
                    xyz = {
                        'types': ['C']*(len(taus)),
                        'pos'  : [taus[:,:3,3]]
                        }
                    xyzfn = outfn + '_gs_10bp'
                    write_xyz(xyzfn,xyz)
                
                if args.dump_gs:
                    fn_gs = outfn + '_gs.npy'
                    np.save(fn_gs,gs)    
                
            else:
                ##########################################################
                # Coarse-Grain 
                print('Coarse-graining stiffness')
                cg_gs,cg_stiff = coarse_grain(gs,stiff,composite_size,start_id=first_id,allow_partial=True)
                
                ##########################################################
                ##########################################################
                # Marginalize translations
                print('Calculating rotational marginals')
                # gs_rot = vector_rotmarginal(vecs2statevec(cg_gs))
                # stiff_rot = matrix_rotmarginal(cg_stiff)
                gs_rot = vecs2statevec(cg_gs)
                stiff_rot=cg_stiff.copy()

                ##########################################################
                ##########################################################
                # Transform groundstate units
                
                if gs_units.lower() in ['deg','degree','degrees']:
                    gs_rot = np.rad2deg(gs_rot)
                
                ##########################################################
                # Express stiffness in nm
                stiff_nm = stiff_rot * disc_len * composite_size
                
                if args.simple_stiff:
                    print('Using simple stiffness')
                    A = args.Avalue
                    C = args.Cvalue
                    for i in range(stiff_nm.shape[0]//3):
                        stiff_nm[i*3:(i+1)*3,i*3:(i+1)*3] = [[A,0,0],[0,A,0],[0,0,C]]
                
                ##########################################################
                # Write to IDB
                cgNbps = len(gs_rot) // 3
                if not closed:
                    cgNbps += 1
                else:
                    raise ValueError(f'Closed CG not properly implemented')
                    
                assignseq, chars = unique_oli_seq(
                    cgNbps, couprange2olisize(couprange), closed=closed, boundary="x", exclude="y"
                )
                
                if cgNbps != len(assignseq):
                    raise ValueError(f'Length of assign sequence does not match required sequence length.')
                
                print('Create IDB file')
                stiff2idb(
                    outfn, gs_rot, stiff_nm, couprange, closed, assignseq, disc_len=disc_len * composite_size, avg_inconsist=avg_inconsist,generate_missing=generate_missing,unique_sequence=True
                )

                if include_deactivate_static:
                    outfn_ns =  outfn + '_no_static'
                    gs_ns = np.copy(gs_rot)
                    gs_ns[0::3] = 0
                    gs_ns[1::3] = 0
                    stiff2idb(
                        outfn_ns, gs_ns, stiff_nm, couprange, closed, assignseq, disc_len=disc_len * composite_size, avg_inconsist=avg_inconsist,generate_missing=generate_missing,unique_sequence=True
                    )
                
                if args.gen_xyz:
                    taus = gen_gs_config(cg_gs,disc_len * composite_size)
                    xyz = {
                        'types': ['C']*(len(taus)),
                        'pos'  : [taus[:,:3,3]]
                        }
                    xyzfn = outfn + '_gs'
                    write_xyz(xyzfn,xyz)
                    
                if args.dump_gs:
                    fn_gs = outfn + '_gs.npy'
                    np.save(fn_gs,cg_gs)    