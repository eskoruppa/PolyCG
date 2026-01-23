from __future__ import annotations

from pathlib import Path
import numpy as np
import sys
import string


def randseq(N: int, nucleotides: str = "atcg") -> str:
    """Generate a random DNA sequence.
    
    Args:
        N: Length of the sequence to generate.
        nucleotides: String of allowed nucleotide characters. Default is 'atcg'.
    
    Returns:
        Random DNA sequence of length N.
    """
    seq = ""
    for i in range(N):
        seq += nucleotides[np.random.randint(4)]
    return seq


def unique_oli_seq(
    N: int, 
    oli_size: int, 
    boundary: str = "x", 
    exclude: str = "y", 
    closed: bool = False
) -> tuple[list[str], str, str]:
    """Generate a sequence with unique oligomers of specified size.
    
    Creates a sequence where all oligomers (subsequences) of length oli_size are unique.
    Uses lowercase letters as sequence characters, excluding specified boundary and 
    exclude characters.
    
    Args:
        N: Target length of the sequence.
        oli_size: Size of oligomers that must be unique. Must be at least 2 and even.
        boundary: Character(s) reserved as boundary markers. Default is 'x'.
        exclude: Character(s) to exclude from the sequence alphabet. Default is 'y'.
        closed: If True, modify sequence to support circular topology.
    
    Returns:
        Tuple containing (sequence, character_set) where sequence is the generated
        string and character_set are the characters used.
    
    Raises:
        ValueError: If oli_size < 2, oli_size is odd, or N > 576 for oli_size=2.
        ValueError: If generated sequence contains non-unique oligomers.
    """
    if oli_size < 2:
        raise ValueError(f"Oligomers size needs to be at least two.")
    if oli_size % 2 != 0:
        raise ValueError(f"Expected even size of oligomer")

    if oli_size == 2 and N > 576:
        raise ValueError(f'The maximum size of a sequence with unique steps is 576.')

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


def all_oligomers(oli_size: int, chars: str) -> list[str]:
    """Generate all possible oligomers of given size from character set.
    
    Creates all possible combinations of characters of length oli_size.
    
    Args:
        oli_size: Length of each oligomer.
        chars: String of characters to use for generating oligomers.
    
    Returns:
        List of all possible oligomer strings of length oli_size.
    """
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
    """Generate a sequence with unique oligomers from given character set.
    
    Creates a sequence where all oligomers of length oli_size are unique, using
    only the provided characters. The algorithm attempts to maximize sequence length
    by overlapping oligomers when possible.
    
    Args:
        oli_size: Length of oligomers that must be unique.
        chars: String of characters to use in the sequence.
        N: Target sequence length. If None, generates maximum possible length.
    
    Returns:
        Sequence string with unique oligomers, truncated to length N if specified.
    """
    seq = ""
    if N is None:
        N = oli_size ** len(chars)

    def fit2seq(seq: str, oli: str) -> str:
        n = len(oli)
        if oli in seq:
            return seq
        for i in range(1, n):
            if seq[-n + i :] == oli[: n - i]:
                return seq + oli[-i:]
        return seq + oli

    def _genloop(oli_size: int, seq: str, chars: str, oli: str, curid: int) -> str:
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


def unique_olis_in_seq(seq: str, oli_size: int) -> bool:
    """Check if all oligomers of given size in sequence are unique.
    
    Args:
        seq: Sequence string to check.
        oli_size: Length of oligomers to verify uniqueness.
    
    Returns:
        True if all oligomers are unique, False if any duplicate is found.
    """
    olis = list()
    for i in range(len(seq) - oli_size + 1):
        oli = seq[i : i + oli_size]
        if oli in olis:
            return False
        olis.append(oli)
    return True


def write_seqfile(filename: Path | str, seq: str, add_extension: bool = True) -> None:
    """Write DNA sequence to file.
    
    Convenience wrapper for sequence_file().
    
    Args:
        filename: Path to output file (string or Path object).
        seq: DNA sequence string to write.
        add_extension: If True, adds .seq extension if not present.
    """
    sequence_file(filename,seq,add_extension=add_extension)

def sequence_file(filename: Path | str, seq: str, add_extension: bool = True) -> None:
    """Write DNA sequence to file in lowercase.
    
    Args:
        filename: Path to output file (string or Path object).
        seq: DNA sequence string to write.
        add_extension: If True, ensures filename has .seq extension.
    """
    filename = Path(filename)
    if add_extension:
        if filename.suffix.lower() != ".seq":
            filename = filename.with_suffix('.seq')
    with open(filename, "w") as f:
        f.write(seq.lower())

        
def seq2oliseq(seq: str, id: int, couprange: int, closed: bool, boundary_char: str = 'x') -> str:
    """Extract oligomer sequence with coupling range around specified position.
    
    Extracts a subsequence centered at position id, extended by couprange on each side.
    For open sequences, boundary characters are added if extraction extends beyond
    sequence bounds. For closed sequences, wraps around periodically.
    
    Args:
        seq: DNA sequence string.
        id: Central position index for extraction.
        couprange: Number of positions to include on each side of id.
        closed: If True, treat sequence as circular for periodic wrapping.
        boundary_char: Character to use for boundary padding in open sequences.
    
    Returns:
        Extracted oligomer sequence with coupling range.
    
    Raises:
        TypeError: If seq is not a string.
    """
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