"""
PolyCG
=====

Module to coarse-grain sequence-dependent elastic constants and structure parameters

"""

from .SO3 import so3
from .pyConDec.pycondec import cond_jit
from .Models.cgNA_plus.modules.cgDNAUtils import constructSeqParms, seq_edit

from .IOPolyMC import iopolymc
from .IOPolyMC.iopolymc import dna_oligomers, write_idb

############################
# Aux 
from .Aux.aux import load_sequence
from .Aux.bmat import BlockOverlapMatrix
from .Aux.seq import randseq, unique_oli_seq, all_oligomers, unique_seq_of_chars, unique_olis_in_seq, sequence_file, seq2oliseq

############################
# Transforms

# IMPORT METHODS! <-----------

############################
# Evals

# IMPORT METHODS! <-----------
