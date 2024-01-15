"""
cgStiff
=====

Module to coarse-grain sequence-dependent elastic constants and structure parameters

"""

from .SO3 import so3
from .pyConDec import pycondec
from .IOPolyMC import iopolymc

from .bch_coeffs import *
from .bch import BakerCampbellHaussdorff, Lambda_minus, Lambda_plus

from .cgNA_plus.modules.cgDNAUtils import constructSeqParms, seq_edit
from .pyConDec.pycondec import cond_jit
from .IOPolyMC.iopolymc import dna_oligomers, write_idb

from .sequence import randseq
from .marginals import var_assign, gen_select, marginal, rot_marginal, marginal_gs

from .conversions import fifth2rad, fifth2deg, gs2rad, gs2deg, stiff2rad, stiff2deg
from .conversions import statevec2vecs, vecs2statevec
from .conversions import eulers2rotmats, rotmats2eulers
from .conversions import cayleys2rotmats, rotmats2cayleys
from .conversions import vecs2rotmats, rotmats2vecs
from .conversions import rotmats2triads, triads2rotmats
from .conversions import (
    eulers2cayleys,
    cayleys2eulers,
    cayleys2eulers_lintrans,
    eulers2cayleys_lintrans,
)
from .conversions import splittransform_group2algebra, splittransform_algebra2group
from .conversions import rotbps_cayley2euler, rotbps_euler2cayley
from .conversions import rotbps_algebra2group, rotbps_group2algebra


from .bps_params import seq2rotbps, seq2rotbps_partials
from .gen_polymc_idb import polymc_full, polymc_seq
from .gen_polymc_idb import oli_coupling, polymc_couplings, seq_couplings


from .so3_composites import hat_map, vec_map
