"""
PolyCG
=====

Module to coarse-grain sequence-dependent elastic constants and structure parameters

"""

from .SO3 import so3
from .pyConDec.pycondec import cond_jit
# from .Models.cgNA_plus.modules.cgDNAUtils import constructSeqParms, seq_edit

from .IOPolyMC import iopolymc
from .IOPolyMC.iopolymc import dna_oligomers, write_idb

############################
# Generation methods
from .gen_params import gen_params, gen_config
from .cgnaplus import cgnaplus_bps_params

############################
# Coarse-graining methods
from .cg import coarse_grain, cg_groundstate, cg_stiffmat, cg_stiff_partial

############################
# Aux 
from .utils.load_seq import load_sequence
from .utils.bmat import BlockOverlapMatrix
from .utils.seq import randseq, unique_oli_seq, all_oligomers, unique_seq_of_chars, unique_olis_in_seq, sequence_file, seq2oliseq, write_seqfile

############################
# Transforms
from .transforms.transform_cayley2euler import cayley2euler, cayley2euler_lintrans, cayley2euler_stiffmat

############################
# Evals

# IMPORT METHODS! <-----------
