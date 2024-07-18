import numpy as np
import sys
import string
from typing import Tuple, Any, List, Dict


def randseq(N: int, nucleotides="atcg") -> str:
    seq = ""
    for i in range(N):
        seq += nucleotides[np.random.randint(4)]
    return seq

def unique_oli_seq(
    N: int, oli_size: int, boundary="x", exclude="y", closed=False
) -> Tuple[List[str], str, str]:
    if oli_size < 2:
        raise ValueError(f"Oligomers size needs to be at least two.")
    if oli_size % 2 != 0:
        raise ValueError(f"Expected even size of oligomer")

    if oli_size == 2 and N > 576:
        raise ValueError(f'The maximum size of a sequence with unique steps is 576.')

    # num_olis = N-oli_size+1
    # N = num_olis + oli_size - 1
    # hl = oli_size//2

    # build chars repertoire
    all_chars = string.ascii_lowercase
    valid_chars = str(all_chars)
    for c in boundary + exclude:
        valid_chars = valid_chars.replace(c, "")
        
    num_chars = int(np.ceil(np.log(N) / np.log(oli_size)))    
    cut_chars = valid_chars[:num_chars]
    seq = unique_seq_of_chars(oli_size, cut_chars, N=N)
    while len(seq) < N:
        num_chars += 1
        cut_chars = valid_chars[:num_chars]
        seq = unique_seq_of_chars(oli_size, cut_chars, N=N)
    
    if not unique_olis_in_seq(seq,oli_size):
        raise ValueError(f'Sequence contains non-unique oligomers')
       
    # match terminal seq to impose closure
    if closed:
        next_char = valid_chars[num_chars]
        seq = seq[:-1] + next_char
        cut_chars += next_char
    return seq, cut_chars


def all_oligomers(oli_size: int, chars: str) -> List[str]:
    def _genloop(oli_size, oli, chars, olis):
        if len(oli) == oli_size:
            olis.append(oli)
            return
        for i in range(len(chars)):
            _genloop(oli_size, oli + chars[i], chars, olis)
    olis = list()
    _genloop(oli_size, "", chars, olis)
    return olis


def unique_seq_of_chars(oli_size: int, chars: str, N=None) -> str:
    seq = ""
    if N is None:
        N = oli_size ** len(chars)

    def fit2seq(seq, oli):
        n = len(oli)
        if oli in seq:
            return seq
        for i in range(1, n):
            if seq[-n + i :] == oli[: n - i]:
                return seq + oli[-i:]
        return seq + oli

    def _genloop(oli_size, seq, chars, oli, curid):
        if len(oli) == oli_size:
            if len(set(oli)) > 1:
                return fit2seq(seq, oli)
            return seq
        if len(oli) == oli_size - 1 and curid == 0:
            curid += 1
        for i in range(curid, len(chars)):
            seq = _genloop(oli_size, seq, chars, oli + chars[i], curid)
            if len(seq) >= N:
                return seq
        return seq

    for i in range(len(chars)):
        seq = fit2seq(seq, chars[i] * oli_size)
        seq = _genloop(oli_size, seq, chars, chars[i], i)
        if len(seq) >= N:
            seq = seq[:N]
            break
    return seq


def unique_olis_in_seq(seq: str, oli_size: int):
    olis = list()
    for i in range(len(seq) - oli_size + 1):
        oli = seq[i : i + oli_size]
        if oli in olis:
            return False
        olis.append(oli)
    return True

def write_seqfile(filename: str, seq, add_extension=True):
    sequence_file(filename,seq,add_extension=add_extension)

def sequence_file(filename: str, seq, add_extension=True):
    if add_extension:
        if ".seq" not in filename.lower():
            filename += ".seq"
    with open(filename, "w") as f:
        f.write(seq.lower())

        
def seq2oliseq(seq: str, id: int, couprange: int, closed: bool, boundary_char: str='x'):
    if not isinstance(seq,str):
        raise TypeError(f'seq needs to be a string. Found type {type(str)}')
    id1 = id-couprange
    id2 = id+couprange+2
    N = len(seq)
    # open
    if not closed:
        m1 = id1
        m2 = id2
        lseq = ''
        mseq = ''
        rseq = ''
        if couprange > 0:
            if id1 < 0:
                lseq = -id1*boundary_char
                m1 = 0
            if id2 > N:
                rseq = (id2-N)*boundary_char
                m2 = N
        mseq = seq[m1:m2]
        return lseq+mseq+rseq
    # closed
    ladd = 0
    if id1 < 0:
        ladd = int(np.ceil(-id1 / N))
    uadd = 0
    if id2 > N:
        uadd = int(np.floor(id2 / N))
    extseq = seq*(1 + ladd + uadd)
    return extseq[ladd * N + id1 : ladd * N + id2]


if __name__ == "__main__":
    
    N = 20
    oli_size = 6
    boundary = "x"
    exclude = "y"
    closed = True

    seq, chars = unique_oli_seq(N, oli_size, boundary, exclude, closed)
    
    print(''.join(sorted(set(seq))))
    print(chars)    
    print(len(seq))
    print(seq)