import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .bps_params import seq2rotbps, seq2rotbps_partials
from .cg import cg_bpsrot
from .IOPolyMC.iopolymc import dna_oligomers, write_idb
from .sequence import (
    all_oligomers,
    randseq,
    sequence_file,
    unique_oli_seq,
    unique_olis_in_seq,
)

from gen_polymc_idb import GENERATOR_DEFAULT_ARGS


##########################################################################################################
############### Generate IDB and seq files ###############################################################
##########################################################################################################

