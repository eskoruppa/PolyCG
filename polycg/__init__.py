"""
cgStiff
=====

Module to coarse-grain sequence-dependent elastic constants and structure parameters

"""

from .SO3 import so3
from .pyConDec import pycondec
from .IOPolyMC import iopolymc

from .cgNA_plus.modules.cgDNAUtils import constructSeqParms, seq_edit
from .pyConDec.pycondec import cond_jit
from .IOPolyMC.iopolymc import dna_oligomers, write_idb
