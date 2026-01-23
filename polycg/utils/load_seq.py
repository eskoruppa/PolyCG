from __future__ import annotations

from pathlib import Path

def load_sequence(fn: Path | str) -> str:
    """Load DNA sequence from a text file.
    
    Reads a sequence file and returns the DNA sequence as a single string,
    removing whitespace and newlines from each line.
    
    Args:
        fn: Path to the sequence file (accepts Path object or string).
    
    Returns:
        DNA sequence as a single string with all whitespace removed.
    """
    with open(fn, 'r') as f:
        return ''.join(line.strip() for line in f)