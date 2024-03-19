def load_sequence(fn: str) -> str:
    seq = ''
    with open(fn,'r') as f:
        seq = ''.join([line.strip() for line in f.readlines()])
    return seq