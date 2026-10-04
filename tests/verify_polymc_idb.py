#!/usr/bin/env python3
"""Verify the closed-chain IDB output of polycg/polymc_idb.py and that its open-chain output is unchanged.

open    Regression test against a baseline revision of polymc_idb.py (default 8b64f69, the last commit
        before closed output was added). Both versions are run on the same inputs and must give identical
        results: exit codes, errors (tracebacks without paths and line numbers), stdout (progress-bar
        timings and the block-wise coarse-graining progress masked), the names of all files written and
        their contents, byte for byte. Covers the command-line tool for the cgNA+, lankas and olson models
        over many sequences, composite sizes, coupling ranges and options, including inputs on which the
        tool fails, and stiff2idb, _matassign and the other module functions called directly with dense,
        sparse and BlockOverlapMatrix input. Only polymc_idb.py is taken from the baseline revision, every
        other module comes from the checkout, so the test isolates the changes to polymc_idb.py.
        Intentional changes: IDB files of composites (cg > 1) differ from the baseline, which split the
        coarse-graining into blocks with the coarse_grain defaults; cases that crashed in the baseline
        (lankas/olson, chains short enough to be coarse-grained in one piece) must fail there and succeed
        in the checkout. Every IDB file the checkout writes is read back and compared with an exact
        reference: the chain coarse-grained as a whole, with the options of the run applied.
closed  Correctness of the closed output. The IDB files written for rings are read back and the ring
        stiffness is rebuilt the way PolyMC assembles it: the oligomer of step i is bp i-R..i+R+1 mod N,
        line j of an entry couples step i to step i+j-R, and the block on that line is K[left, right]
        (PolyMC never transposes). The result is compared with exact references: one cgNA+ calculation
        of the repeated ring sequence with the couplings of the middle copy folded, and RBPStiff for
        lankas/olson, coarse-grained for composite_size > 1.
polymc  Optional, needs a PolyMC executable. Runs PolyMC on IDB files of rings (mode = plasmid) and of
        open chains (mode = open) and compares the elastic energies PolyMC reports with the energies of
        the same configurations computed from the rebuilt stiffness matrix; for rings also without the
        couplings across the seam.

Usage:
    python tests/verify_polymc_idb.py --root ~/Dev/PolyCG open [--baseline-rev 8b64f69] [--quick] [--jobs N]
    python tests/verify_polymc_idb.py --root ~/Dev/PolyCG closed [--quick]
    python tests/verify_polymc_idb.py --root ~/Dev/PolyCG polymc --polymc ~/Dev/PolyMC/bin/Release/PolyMC
    python tests/verify_polymc_idb.py --root ~/Dev/PolyCG all [--polymc ...]

--root is the directory that contains the polycg package. Nothing in the repository is modified, all
files are written to a temporary directory (kept with --keep). Exit code 0 if all checks pass. Run time:
a few minutes with --quick.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.sparse

warnings.filterwarnings('ignore')

BASELINE_REV = '8b64f69'    # last commit before closed IDB output was added to polymc_idb.py
ND = 3
KT_REF = 4.114              # PolyMC: kT in pN nm at its reference temperature of 300 K

SEQ = ''.join(np.random.default_rng(11).choice(list('acgt'), 3000))    # fixed random test sequence

# rings get their own IDB entry per site; the generated assignment sequence is used if a seam window
# of the real sequence repeats an interior one. COLLISION is such a ring: its windows are unique as
# linear 10-mers, but the window of the last step equals an interior window (step 39).
_W = SEQ[2000:2010]
COLLISION = _W[5:] + SEQ[2100:2130] + _W + SEQ[2200:2230] + _W[:5]

# Tolerances
TOL_ABS = 1e-3              # closed IDB vs exact reference, absolute (IDB values have 3 decimals)
TOL_REL_LONG = 3e-4         # rings of 250 bp, relative to max|Kref|: block-assembly truncation of cgNA+
TOL_GS_DEG = 6e-4           # ground state in degrees, absolute
TOL_POLYMC = 1e-4           # PolyMC energies vs rebuilt matrix, relative (PolyMC prints 6 significant digits)
TOL_PARTITION = 1e-6        # block-wise vs whole-chain coarse-graining, relative (IDB values resolve ~4e-6)


#######################################################################################
# helpers


class Report:
    def __init__(self):
        self.failed = 0
        self.passed = 0

    def check(self, name, cond, detail=''):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f'  [{detail}]' if detail else ''), flush=True)
        self.failed += not cond
        self.passed += bool(cond)

    @staticmethod
    def info(name, detail=''):
        print(f'  INFO  {name}' + (f'  [{detail}]' if detail else ''), flush=True)

    @staticmethod
    def section(title):
        print(f'\n{title}', flush=True)


def quiet(fn, *args, **kwargs):
    """Call fn with PolyCG's progress output suppressed (exceptions propagate)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def dense(K) -> np.ndarray:
    if hasattr(K, 'to_sparse'):          # BlockOverlapMatrix
        K = K.to_sparse()
    return K.toarray() if sp.sparse.issparse(K) else np.asarray(K)


def env_for(pkg_root: Path) -> dict:
    env = dict(os.environ)
    env['PYTHONPATH'] = str(pkg_root) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def run_py(pkg_root: Path, cwd: Path, argv: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], cwd=cwd, env=env_for(pkg_root), capture_output=True,
                          text=True, timeout=timeout)


def last_line(text: str) -> str:
    return (text.strip().splitlines() or [''])[-1][:110]


def write_inputs(d: Path, inputs: dict[str, str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, content in inputs.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(content)


def tree(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob('*')) if p.is_file()}


def mask_stdout(text: str) -> str:
    text = re.sub(r' ETA [0-9:]+', ' ETA *', text)
    return re.sub(r' \(elapsed [0-9:]+\)', ' (elapsed *)', text)


def mask_cg_progress(text: str) -> str:
    """The block-wise coarse-graining progress, printed between these two lines, depends on the block sizes."""
    return re.sub(r'(Coarse-graining stiffness\n).*?(?=Calculating rotational marginals)', r'\1<coarse-graining>\n',
                  text, flags=re.S)


def mask_stderr(text: str, roots: list[Path]) -> str:
    """Tracebacks reduced to the functions called and the error: paths, line numbers and the source lines
    shown under each frame are dropped (they change whenever the code is edited)."""
    for r in roots:
        text = text.replace(str(r), '<ROOT>')
    text = re.sub(r'\.py:\d+:', '.py:*:', text)
    out, in_frame = [], False
    for line in text.splitlines():
        frame = re.match(r'\s*File ".*", line \d+, (in .*)$', line)
        if frame:
            out.append(frame.group(1))
        elif not (in_frame and line.startswith('    ')):
            out.append(line)
        in_frame = bool(frame) or (in_frame and line.startswith('    '))
    return '\n'.join(out)


#######################################################################################
# open chains: regression against the baseline revision


def build_baseline(root: Path, rev: str, dest: Path) -> Path:
    """polycg with polymc_idb.py from `rev` and all other entries symlinked to the checkout."""
    code = subprocess.run(['git', '-C', str(root), 'show', f'{rev}:polycg/polymc_idb.py'], capture_output=True,
                          text=True)
    if code.returncode != 0:
        sys.exit(f'cannot read polycg/polymc_idb.py at revision {rev}: {code.stderr.strip()}')
    pkg = dest / 'polycg'
    pkg.mkdir(parents=True)
    for entry in (root / 'polycg').iterdir():
        if entry.name not in ('polymc_idb.py', '__pycache__'):
            (pkg / entry.name).symlink_to(entry)
    (pkg / 'polymc_idb.py').write_text(code.stdout)
    return dest


@dataclass
class Case:
    label: str
    inputs: dict
    args: list
    ok: bool | None = True     # both versions are expected to succeed / to fail with `err` / None: no expectation
    err: str = ''
    quick: bool = True         # included in --quick
    changed: str = ''          # intentional change: the baseline fails with `err`, the checkout succeeds


GENSTIFF_FIX = 'lankas/olson read the dict returned by GenStiffness.gen_params'
SHORT_FIX = 'chains short enough to be coarse-grained in one piece'

# IDB files of composites (cg > 1) intentionally differ from the baseline: the block-wise coarse-graining
# now uses the partition polymc_idb computes instead of the coarse_grain defaults
CG_IDB = re.compile(r'_(\d+)bp_\d+cr.*\.idb$')


def cg_idb(name: str) -> bool:
    m = CG_IDB.search(name)
    return bool(m) and int(m.group(1)) > 1


def parse_cli(argv: list[str]) -> argparse.Namespace:
    """The command line of polymc_idb.py, parsed with the same options (all of them, so that no option is
    taken for an abbreviation of another)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('-m', '--model', default='cgnaplus')
    ap.add_argument('-cg', '--composite_size', type=int, default=[1], nargs='*')
    ap.add_argument('-cr', '--coupling_range', type=int, default=4)
    ap.add_argument('-seqfns', '--sequence_files', nargs='*', default=[])
    ap.add_argument('-closed', '--closed', action='store_true')
    ap.add_argument('-sc', '--scale_factor', type=float, default=1)
    ap.add_argument('-nc', '--no_crop', action='store_true')
    ap.add_argument('-dl', '--disc_len', type=float, default=0.34)
    ap.add_argument('-ai', '--avg_inconsistency', action='store_true')
    ap.add_argument('-gm', '--generate_missing', action='store_true')
    ap.add_argument('-fid', '--first_id', type=int, default=0)
    ap.add_argument('-gsu', '--gs_units', default='deg')
    ap.add_argument('-rm', '--rotation_map', default='euler')
    ap.add_argument('-sf', '--split_fluctuations', default='matrix')
    ap.add_argument('-ss', '--simple_stiff', action='store_true')
    ap.add_argument('-C', '--Cvalue', type=float, default=100)
    ap.add_argument('-A', '--Avalue', type=float, default=40)
    ap.add_argument('-xyz', '--gen_xyz', action='store_true')
    ap.add_argument('-gs', '--dump_gs', action='store_true')
    ap.add_argument('-nostatic', '--no_static', action='store_true')
    return ap.parse_known_args(argv)[0]


def expected_idbs(o: argparse.Namespace) -> list[tuple[str, str, int, bool]]:
    """(IDB file, sequence file, composite size, no_static) written by a successful run."""
    out = []
    for seqfn in o.sequence_files:
        for cg in o.composite_size:
            base = os.path.splitext(seqfn)[0] + f'_{o.model}_{cg}bp_{o.coupling_range}cr'
            if o.scale_factor != 1:
                base += ('_rescaled_%.3f' % o.scale_factor).replace('.', 'p')
            base += '_closed' if o.closed else ''
            out.append((base + '.idb', seqfn, cg, False))
            if o.no_static:
                out.append((base + '_no_static.idb', seqfn, cg, True))
    return out


def open_cli_cases() -> list[Case]:
    s = lambda n, off=0: SEQ[off:off + n]
    seq = lambda n, off=0: {'s.seq': s(n, off)}
    a = lambda *x: ['-seqfns', 's.seq', *x]
    return [
        Case('cgNA+ 300 bp, -cg 1 2 5', seq(300), a('-cg', '1', '2', '5')),
        Case('cgNA+ 21 bp', seq(21, 7), a()),
        Case('cgNA+ 12 bp', seq(12, 30), a()),
        Case('cgNA+ 10 bp, 9 steps = one coupling window', seq(10, 50), a()),
        Case('cgNA+ 8 bp, shorter than a coupling window', seq(8, 60), a(), ok=False, err='selection range larger'),
        Case('cgNA+ 6 bp, -cr 1', seq(6, 70), a('-cr', '1')),
        Case('cgNA+ 1000 bp, -cg 1 3 10 -cr 2', seq(1000, 100), a('-cg', '1', '3', '10', '-cr', '2'), quick=False),
        Case('cgNA+ 2000 bp, -cg 1 4 20', seq(2000, 900), a('-cg', '1', '4', '20'), quick=False),
        Case('cgNA+ 400 bp, -cg 1 2 3 4 5 6 7 8 9 10', seq(400, 50), a('-cg', *map(str, range(1, 11))), quick=False),
        Case('cgNA+ 299 bp, -cg 5 1 3 (cropped composites)', seq(299, 500), a('-cg', '5', '1', '3')),
        Case('cgNA+ poly-A 150 bp, -cg 1 2', {'s.seq': 'a' * 150}, a('-cg', '1', '2')),
        Case('cgNA+ (AG)n 150 bp, -cr 1 -gm', {'s.seq': 'ag' * 75}, a('-cr', '1', '-gm')),
        Case('cgNA+ 120 bp, -cr 2 -gm', seq(120, 800), a('-cr', '2', '-gm')),
        Case('cgNA+ 120 bp, -cr 0', seq(120, 1000), a('-cr', '0')),
        Case('cgNA+ 120 bp, -cr 1 -cg 1 3', seq(120, 1100), a('-cr', '1', '-cg', '1', '3')),
        Case('cgNA+ 150 bp, -cr 6 -cg 1 2', seq(150, 1200), a('-cr', '6', '-cg', '1', '2')),
        Case('cgNA+ 300 bp, -sc 2.5 -dl 0.5 -ai -cg 1 5', seq(300, 1400), a('-sc', '2.5', '-dl', '0.5', '-ai', '-cg', '1', '5')),
        Case('cgNA+ 300 bp, -cg 2 5 -fid 3', seq(300, 1700), a('-cg', '2', '5', '-fid', '3')),
        Case('cgNA+ 301 bp, -cg 4 -nc', seq(301, 2000), a('-cg', '4', '-nc')),
        Case('cgNA+ 200 bp, -gsu rad -rm cayley -sf vector -cg 1 4', seq(200, 2300), a('-gsu', 'rad', '-rm', 'cayley', '-sf', 'vector', '-cg', '1', '4')),
        Case('cgNA+ 200 bp, -ss -C 90 -A 45 -cg 1 5', seq(200, 2500), a('-ss', '-C', '90', '-A', '45', '-cg', '1', '5')),
        Case('cgNA+ 200 bp, -xyz -gs -nostatic -cg 1 5', seq(200, 2700), a('-xyz', '-gs', '-nostatic', '-cg', '1', '5')),
        Case('cgNA+ two sequence files, -cg 1 2', {'a.seq': s(150, 200), 'b.seq': s(180, 400)},
             ['-seqfns', 'a.seq', 'b.seq', '-cg', '1', '2']),
        Case('cgNA+ sequence file in a subdirectory', {'sub/s.seq': s(90, 600)}, ['-seqfns', 'sub/s.seq']),
        Case('cgNA+ mixed case and whitespace in the sequence file', {'s.seq': 'ACGTacgtTTGA\n  ggccaTTAG \nCATG\n' * 4}, a()),
        Case('cgNA+ 100 bp, -cg 1', seq(100, 2900), a('-cg', '1')),
        Case('cgNA+ 300 bp, -cg 5 -fid -2', seq(300, 300), a('-cg', '5', '-fid', '-2'), ok=None),
        Case('cgNA+ 60 bp, -cg 5', seq(60, 2800), a('-cg', '5'), changed=SHORT_FIX, err='to_sparse'),
        Case('cgNA+ 111 bp, -cg 1 5 (110 steps, one-piece threshold of the old partition)', seq(111, 1500),
             a('-cg', '1', '5'), changed=SHORT_FIX, err='to_sparse'),
        Case('cgNA+ 46 bp, -cg 2 -cr 2', seq(46, 1600), a('-cg', '2', '-cr', '2'), changed=SHORT_FIX, err='to_sparse'),
        Case('cgNA+ 1000 bp, -cg 80 -cr 1 (block size above the overlap)', seq(1000, 1900), a('-cg', '80', '-cr', '1'),
             changed=SHORT_FIX, err='to_sparse', quick=False),
        Case('lankas 60 bp', seq(60), a('-m', 'lankas'), changed=GENSTIFF_FIX, err="no attribute 'shape'"),
        Case('olson 60 bp', seq(60), a('-m', 'olson'), changed=GENSTIFF_FIX, err="no attribute 'shape'"),
        Case('lankas 300 bp, -cg 1 5', seq(300, 400), a('-m', 'lankas', '-cg', '1', '5'), changed=GENSTIFF_FIX,
             err="no attribute 'shape'"),
        Case('olson 299 bp, -cg 1 4 -cr 2', seq(299, 700), a('-m', 'olson', '-cg', '1', '4', '-cr', '2'),
             changed=GENSTIFF_FIX, err="no attribute 'shape'"),
        Case('lankas 60 bp, -cg 5', seq(60, 1700), a('-m', 'lankas', '-cg', '5'), changed=GENSTIFF_FIX + '; ' + SHORT_FIX,
             err="no attribute 'shape'"),
        Case('empty sequence file', {'s.seq': ''}, a(), ok=False, err='Empty sequence'),
        Case('missing sequence file', {}, ['-seqfns', 'missing.seq'], ok=False, err='No such file'),
        Case('unknown model', seq(30), a('-m', 'foo'), ok=False, err='invalid choice'),
        Case('no arguments', {}, [], ok=False, err='required'),
        Case('help', {}, ['-h']),
    ]


LIB_DRIVER = r'''
import sys, inspect, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import scipy.sparse as sps
import polycg.polymc_idb as m
from polycg.utils.bmat import BlockOverlapMatrix

out = Path(sys.argv[1])
out.mkdir(parents=True)
print(m.__file__)
counts = {'calls': 0, 'errors': 0}

def record(name, fn):
    counts['calls'] += 1
    try:
        value = fn()
    except Exception as e:
        counts['errors'] += 1
        (out / f'{name}.err').write_text(f'{type(e).__name__}: {e}')
        return
    if isinstance(value, np.ndarray):
        np.save(out / f'{name}.npy', value)
    elif value is not None:
        (out / f'{name}.txt').write_text(repr(value))

def banded(n, band, rng):
    A = rng.normal(size=(3 * n, 3 * n))
    K = A + A.T + 6 * n * np.eye(3 * n)
    i = np.arange(3 * n) // 3
    K[np.abs(i[:, None] - i[None, :]) > band] = 0
    return K

def as_format(K, fmt):
    if fmt == 'dense':
        return K.copy()
    if fmt in ('csc', 'csr', 'coo'):
        return getattr(sps, fmt + '_matrix')(K)
    n = K.shape[0]
    bm = BlockOverlapMatrix(average=True, periodic=False, fixed_size=True, xlo=0, xhi=n, ylo=0, yhi=n)
    for x1 in range(0, n, 24):
        x2 = min(x1 + 36, n)
        bm.add_block(K[x1:x2, x1:x2], x1, x2, y1=x1, y2=x2)
        if x2 == n:
            break
    return bm

rng = np.random.default_rng(3)
letters = np.array(list('acgt'))
options = {
    'default': {},
    'nouniq': dict(unique_sequence=False),
    'opts': dict(disc_len=0.5, avg_inconsist=False, boundary_char='z', exclude_chars='q'),
    'missing': dict(generate_missing=True),
}

# stiff2idb on open chains
for n in (9, 10, 12, 20, 41):
    for cr in (0, 1, 2, 4):
        K = banded(n, cr + 1, rng)
        gs = rng.normal(size=3 * n)
        seqs = {'none': None, 'real': ''.join(rng.choice(letters, n + 1)), 'polya': 'a' * (n + 1)}
        for fmt in ('dense', 'csc', 'csr', 'bmat'):
            for sname, seq in seqs.items():
                for oname, kw in options.items():
                    if oname == 'missing' and cr > 1:
                        continue
                    base = out / f'stiff2idb_n{n}_cr{cr}_{fmt}_{sname}_{oname}'
                    record(base.name, lambda: m.stiff2idb(str(base), gs, as_format(K, fmt), cr, False, seq, **kw))
K = banded(300, 5, rng)
base = out / 'stiff2idb_n300_cr4_csc_real'
record(base.name, lambda: m.stiff2idb(str(base), rng.normal(size=900), sps.csc_matrix(K), 4, False,
                                      ''.join(rng.choice(letters, 301))))

# _matassign on open chains: all windows of every step, and some arbitrary ranges
for n in (9, 12, 30):
    K = banded(n, 4, rng)
    for fmt in ('dense', 'csc', 'csr', 'coo', 'bmat'):
        S = as_format(K, fmt)
        if fmt == 'bmat':
            S.check_bounds_on_read = False
        for cr in (0, 1, 2, 4, 6):
            for i in range(-cr - 2, n + cr + 2):
                record(f'matassign_n{n}_{fmt}_cr{cr}_i{i}', lambda: m._matassign(S, (i - cr) * 3, (i + cr + 1) * 3, False))
    for cl, cu in ((-5, 7), (0, 0), (10, 5), (-40, 2), (3, 400), (1, 4), (-3, 3 * n + 3)):
        record(f'matassign_n{n}_raw_{cl}_{cu}', lambda: m._matassign(sps.csc_matrix(K), cl, cu, False))

# remaining module functions
for nb in (1, 3, 5, 7, 9):
    M = rng.normal(size=(3 * nb, 3 * nb))
    record(f'mat2idbcoups_{nb}', lambda: m._mat2idbcoups(M))
    record(f'mat2idb_entry_{nb}', lambda: m._mat2idb_entry(M[:3, :3]))
record('couprange2olisize', lambda: [m.couprange2olisize(c) for c in range(10)])
record('olisize2couprange', lambda: [m.olisize2couprange(o) for o in range(2, 22)])
for cr, chars in ((0, 'ab'), (1, 'abc'), (1, 'ac')):
    params = {'ab' * (cr + 1): {'seq': 'ab' * (cr + 1), 'vec': np.ones(3), 'interaction': []}}
    record(f'add_missing_{cr}_{chars}', lambda: sorted(m._add_missing_params(params, cr, chars).keys()))
for n in (1, 5, 50):
    g = 0.1 * rng.normal(size=(n, 3))
    record(f'gen_gs_config_{n}', lambda: m.gen_gs_config(g, 0.34))

sigs = sorted(f'{name}{inspect.signature(obj)}' for name, obj in vars(m).items()
              if inspect.isfunction(obj) and obj.__module__ == m.__name__)
(out.parent / (out.name + '.signatures')).write_text('\n'.join(sigs))
print(counts['calls'], counts['errors'])
'''


PARTITION_DRIVER = r'''
import contextlib, io, json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polycg.polymc_idb as m
from polycg.partials import partial_stiff
from polycg.cgnaplus import cgnaplus_bps_params
from polycg.cg import coarse_grain
from polycg.utils.bmat import BlockOverlapMatrix

res = {'file': m.__file__, 'invalid': [], 'accuracy': []}
for cr in range(0, 11):
    for cg in range(1, 201):
        b, o, t = m.cg_partition(cg, cr)
        if not (b > o >= max(cr, 2) and t >= cr and t * cg >= 40 and b * cg >= 160):
            res['invalid'].append((cg, cr, b, o, t))
args = dict(translations_in_nm=True, euler_definition=True, group_split=True, parameter_set_name='curves_plus',
            remove_factor_five=True, rotations_only=True)
with contextlib.redirect_stdout(io.StringIO()):
    gs, K = partial_stiff(sys.argv[1], cgnaplus_bps_params, args, 120, 20, 20, closed=False, ndims=3)
    K = K.to_sparse()
    dense = lambda M: (M.to_sparse() if isinstance(M, BlockOverlapMatrix) else M).toarray()
    for cg in (2, 3, 4, 5, 8, 10, 20):
        Kx = dense(coarse_grain(gs, K, cg, allow_partial=False)[1])
        n = Kx.shape[0] // 3
        d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        for cr in (1, 4, 6):
            b, o, t = m.cg_partition(cg, cr)
            Kp = coarse_grain(gs, K, cg, allow_partial=True, block_ncomp=b, overlap_ncomp=o, tail_ncomp=t)[1]
            band = np.kron(d <= cr, np.ones((3, 3), dtype=bool))
            err = float(np.abs(np.where(band, dense(Kp) - Kx, 0)).max() / np.abs(Kx).max())
            res['accuracy'].append((cg, cr, err, isinstance(Kp, BlockOverlapMatrix)))
print(json.dumps(res))
'''


def phase_open(root: Path, args, rep: Report, tmp: Path, pool: ThreadPoolExecutor) -> None:
    rep.section(f'Open chains: polymc_idb.py of the checkout against revision {args.baseline_rev}')
    base_root = build_baseline(root, args.baseline_rev, tmp / 'baseline')
    other = subprocess.run(['git', '-C', str(root), 'diff', '--name-only', args.baseline_rev, '--', 'polycg',
                            ':!polycg/polymc_idb.py'], capture_output=True, text=True).stdout.split()
    if other:
        rep.info('other files under polycg/ differ from the baseline revision; the checkout version is used for '
                 'both runs', ', '.join(other))
    roots = [root, base_root]
    for label, r in (('checkout', root), ('baseline', base_root)):
        p = run_py(r, tmp, ['-c', 'import polycg.polymc_idb as m; print(m.__file__)'])
        loaded = p.stdout.strip()
        rep.check(f'{label} run imports its own polymc_idb.py', p.returncode == 0 and loaded.startswith(str(r)),
                  loaded or last_line(p.stderr))

    # 1) command-line tool
    rep.section('1) Command-line tool, open chains: checkout vs baseline (exit code, stdout, stderr, all files)')
    cases = [c for c in open_cli_cases() if c.quick or not args.quick]

    def run_case(k: int, case: Case, which: str, pkg: Path):
        d = tmp / 'cli' / which / f'{k:02d}'
        write_inputs(d, case.inputs)
        r = run_py(pkg, d, ['-m', 'polycg.polymc_idb', *case.args])
        return r, tree(d)

    futures = {(k, w): pool.submit(run_case, k, c, w, pkg) for k, c in enumerate(cases)
               for w, pkg in (('new', root), ('base', base_root))}
    n_same = n_cg = 0
    for k, case in enumerate(cases):
        rn, tn = futures[(k, 'new')].result()
        rb, tb = futures[(k, 'base')].result()
        outputs = len(tn) - len(case.inputs)
        if case.changed:
            rep.check(f'{case.label} [intentional change: {case.changed}]',
                      rb.returncode != 0 and case.err in rb.stderr and rn.returncode == 0 and outputs > 0,
                      f'baseline: {last_line(rb.stderr)} | checkout: exit {rn.returncode}, {outputs} output files')
            continue
        diffs = []
        if rn.returncode != rb.returncode:
            diffs.append(f'exit code {rb.returncode} -> {rn.returncode}')
        if mask_cg_progress(mask_stdout(rn.stdout)) != mask_cg_progress(mask_stdout(rb.stdout)):
            diffs.append('stdout differs')
        if mask_stderr(rn.stderr, roots) != mask_stderr(rb.stderr, roots):
            diffs.append('stderr differs')
        if set(tn) != set(tb):
            diffs.append(f'file sets differ: {sorted(set(tn) ^ set(tb))[:3]}')
        same = [f for f in tn if f in tb and not cg_idb(f)]
        differ = [f for f in same if tn[f] != tb[f]]
        if differ:
            diffs.append(f'{len(differ)} files differ, e.g. {differ[0]}')
        n_same += len(same) - len(case.inputs)
        n_cg += sum(cg_idb(f) for f in tn)
        if case.ok is None:
            expected = True
            what = f'exit code {rn.returncode}, {outputs} output files' + (f': {last_line(rn.stderr)}' if rn.returncode else '')
        elif case.ok:
            expected = rn.returncode == 0 and (outputs > 0 or '-h' in case.args)
            what = f'{outputs} output files'
        else:
            expected = rn.returncode != 0 and case.err in rn.stderr
            what = f'fails in both: {last_line(rn.stderr)}'
        rep.check(case.label, not diffs and expected, what + ('' if not diffs else ' | ' + '; '.join(diffs)))
    rep.info(f'{len(cases)} command-line runs per version; {n_same} output files compared byte for byte, '
             f'{n_cg} IDB files of composites checked in 1b')

    # every IDB file written by the checkout is checked against an independent reference
    rep.section('1b) All IDB files written by the checkout, read back as PolyMC does, vs exact references '
                '(coarse-graining of the whole chain at once)')
    P = import_polycg(root)
    cache = {}
    for k, case in enumerate(cases):
        rn, tn = futures[(k, 'new')].result()
        if rn.returncode != 0:
            continue
        o = parse_cli(case.args)
        d = tmp / 'cli' / 'new' / f'{k:02d}'
        errs, worst_base, problems = [], 0.0, []
        for idbname, seqfn, cg, no_static in expected_idbs(o):
            try:
                r = rebuild(d / idbname, P.seq2oliseq, closed=False)
            except Exception as e:
                problems.append(f'{idbname}: {type(e).__name__}: {e}'[:110])
                continue
            seq = ''.join(line.strip() for line in case.inputs[seqfn].splitlines()).lower()
            gsx, Kx = open_reference(P, cache, o.model, seq, cg, o.coupling_range, o.scale_factor, o.disc_len,
                                     o.first_id, o.gs_units, (o.Avalue, o.Cvalue) if o.simple_stiff else None, no_static)
            if r['K'].shape != Kx.shape or r['inexact'] or (r['entries'] != r['N'] and not o.generate_missing):
                problems.append(f"{idbname}: {r['N']} steps (reference {Kx.shape[0] // ND}), {r['entries']} entries, "
                                f"{r['inexact']} inexact pairs")
                continue
            errs.append(max(np.abs(r['K'] - Kx).max() / TOL_ABS, np.abs(r['gs'] - gsx).max() / TOL_GS_DEG))
            base_file = tmp / 'cli' / 'base' / f'{k:02d}' / idbname
            if cg > 1 and base_file.exists():
                rb_ = rebuild(base_file, P.seq2oliseq, closed=False)
                worst_base = max(worst_base, np.abs(rb_['K'] - Kx).max())
        if not errs and not problems:
            continue
        detail = f'{len(errs)} IDB files, worst stiffness/ground-state error {max(errs, default=0):.2f} x tolerance'
        if worst_base:
            detail += f'; baseline composites deviated up to {worst_base:.1e}'
        rep.check(case.label, not problems and max(errs, default=0) <= 1, '; '.join(problems) or detail)

    # the IDB files hold 3 decimals; the partition itself is checked at full precision
    rep.section('1c) Block partition of the coarse-graining (cg_partition): guaranteed sizes for cg 1..200, '
                'cr 0..10, and block-wise vs whole-chain coarse-graining of a 1000 bp chain at full precision')
    driver = tmp / 'partition_driver.py'
    driver.write_text(PARTITION_DRIVER)
    p = run_py(root, tmp, [str(driver), SEQ[:1000]])
    if p.returncode != 0:
        rep.check('partition driver runs', False, last_line(p.stderr))
    else:
        res = json.loads(p.stdout.strip().splitlines()[-1])
        rep.check('block > overlap >= max(cr, 2), tails >= cr and >= 40 bp, blocks >= 160 bp', not res['invalid'],
                  f"violated for (cg, cr, block, overlap, tail) = {res['invalid'][:3]}" if res['invalid'] else
                  '2200 combinations')
        worst = max(res['accuracy'], key=lambda a: a[2])
        blockwise = sum(a[3] for a in res['accuracy'])
        rep.check('block-wise result equals the whole-chain coarse-graining within the coupling range',
                  worst[2] <= TOL_PARTITION and blockwise == len(res['accuracy']),
                  f"{len(res['accuracy'])} cases (cg 2..20, cr 1/4/6), {blockwise} of them block-wise, worst relative "
                  f"error {worst[2]:.1e} (cg={worst[0]}, cr={worst[1]})")

    # 2) module functions
    rep.section('2) Module functions on open chains: checkout vs baseline (stiff2idb, _matassign, ...)')
    driver = tmp / 'lib_driver.py'
    driver.write_text(LIB_DRIVER)
    res = {w: pool.submit(run_py, pkg, tmp, [str(driver), str(tmp / 'lib' / w)])
           for w, pkg in (('new', root), ('base', base_root))}
    rn, rb = res['new'].result(), res['base'].result()
    ok = rn.returncode == 0 and rb.returncode == 0
    rep.check('driver runs with both versions', ok, '' if ok else last_line(rn.stderr + rb.stderr))
    if ok:
        tn, tb = tree(tmp / 'lib' / 'new'), tree(tmp / 'lib' / 'base')
        changed = [f for f in tn if f in tb and tn[f] != tb[f]]
        calls, errors = rn.stdout.split()[-2:]
        rep.check(f'all results identical ({calls} calls, {errors} of them raising, {len(tn)} result files)',
                  set(tn) == set(tb) and not changed and rn.stdout.split()[-2:] == rb.stdout.split()[-2:],
                  f'{len(changed)} differ, e.g. {changed[:2]}' if changed else
                  ('' if set(tn) == set(tb) else f'file sets differ: {sorted(set(tn) ^ set(tb))[:3]}'))
        sn = (tmp / 'lib' / 'new.signatures').read_text().splitlines()
        sb = (tmp / 'lib' / 'base.signatures').read_text().splitlines()
        rep.check('every function of the baseline exists with the same signature', set(sb) <= set(sn),
                  f'new: {sorted(set(sn) - set(sb))}' if set(sb) <= set(sn) else f'missing: {sorted(set(sb) - set(sn))}')


#######################################################################################
# closed chains: correctness


def read_idb(fn: Path):
    lines = fn.read_text().splitlines()
    head = {}
    for line in lines:
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            head[key.strip()] = value.strip()
    cr = int(head['interaction_range'])
    i = max(k for k, line in enumerate(lines) if 'INTERACTIONS' in line) + 2
    params = {}
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        sub = lines[i + 1:i + 3 + 2 * cr]
        if len(sub) != 2 * cr + 2 or sub[-1].split()[0] != 'vec':
            raise ValueError(f'{fn}: malformed entry {lines[i].strip()}')
        coups = [np.array(s.split()[1:], dtype=float).reshape(3, 3) for s in sub[:-1]]
        params[lines[i].strip()] = (coups, np.array(sub[-1].split()[1:], dtype=float))
        i += 2 * cr + 3
    return head, params


def rebuild(idbfn: Path, seq2oliseq, closed: bool = True):
    """Stiffness and ground state as PolyMC assembles them from an IDB file and its sequence file
    (N sites: N monomers for a ring, N+1 for an open chain, whose end steps have fewer neighbours)."""
    head, params = read_idb(idbfn)
    cr, disc = int(head['interaction_range']), float(head['discretization'])
    seq = idbfn.with_suffix('.seq').read_text().strip()
    N = len(seq) if closed else len(seq) - 1
    keys = [seq2oliseq(seq, i, cr, closed) for i in range(N)]
    missing = sorted(set(k for k in keys if k not in params))
    if missing:
        raise KeyError(f'{len(missing)} oligomers missing from {idbfn.name}, e.g. {missing[0]}')
    ent = [params[k] for k in keys]
    K = np.zeros((N * ND, N * ND))
    inexact = 0
    for i in range(N):
        K[i * ND:(i + 1) * ND, i * ND:(i + 1) * ND] += ent[i][0][cr]
        for k in range(1, cr + 1):
            if not closed and i + k >= N:
                break
            j = (i + k) % N
            right, left = ent[i][0][cr + k], ent[j][0][cr - k]     # step i's +k block, step j's -k block
            inexact += not np.array_equal(right, left)
            B = 0.5 * (right + left)                               # avg_inconsist
            K[i * ND:(i + 1) * ND, j * ND:(j + 1) * ND] += B
            K[j * ND:(j + 1) * ND, i * ND:(i + 1) * ND] += B.T
    gs = np.array([e[1] for e in ent])
    return dict(K=K, gs=gs, cr=cr, disc=disc, seq=seq, N=N, entries=len(params), inexact=inexact,
                avg=head['avg_inconsist'])


def ring_mask(N: int, cr: int) -> np.ndarray:
    i = np.arange(N)
    d = np.abs(i[:, None] - i[None, :])
    return np.kron(np.minimum(d, N - d) <= cr, np.ones((ND, ND), dtype=bool))


CGNAP_ARGS = dict(translations_in_nm=True, euler_definition=True, group_split=True, parameter_set_name='curves_plus',
                  remove_factor_five=True, rotations_only=True)


def open_reference(P, cache: dict, model: str, seq: str, cg: int, cr: int, scale: float = 1.0,
                   disc_len: float = 0.34, first_id: int = 0, gs_units: str = 'deg', simple=None,
                   no_static: bool = False):
    """Expected content of an open-chain IDB file: the rotational parameters of the chain (cgNA+ assembled
    as by polymc_idb, or RBPStiff), rescaled, cropped to whole composites from first_id on and coarse-grained
    as one piece (no blocks), in IDB units and with the options of polymc_idb.py applied."""
    if (model, seq) not in cache:
        if model == 'cgnaplus':
            gs, K = quiet(P.partial_stiff, seq, P.cgnaplus_bps_params, CGNAP_ARGS, 120, 20, 20, closed=False, ndims=ND)
        else:
            p = P.GenStiffness(method='md' if model == 'lankas' else 'crystal').gen_params(seq, use_group=True,
                                                                                         sparse=True)
            gs = P.statevec2vecs(P.vector_rotmarginal(P.vecs2statevec(p['groundstate'])), vdim=3)
            K = P.matrix_rotmarginal(p['stiffness'])
        cache[(model, seq)] = (np.asarray(gs), sp.sparse.csc_matrix(dense(K)))
    gs, K = cache[(model, seq)]
    K = K * scale
    if cg > 1:
        start = first_id % len(gs) if first_id < 0 else first_id
        end = start + (len(gs) - start) // cg * cg
        K = dense(quiet(P.cg_stiffmat, gs[start:end], K[start * ND:end * ND, start * ND:end * ND], cg))
        gs = P.cg_groundstate(gs[start:end], cg)
    else:
        K = dense(K)
    K = K * disc_len * cg
    n = K.shape[0] // ND
    if simple is not None:
        for i in range(n):
            K[i * ND:(i + 1) * ND, i * ND:(i + 1) * ND] = np.diag([simple[0], simple[0], simple[1]])
    gs = np.array(gs, dtype=float)
    if gs_units.lower() in ('deg', 'degree', 'degrees'):
        gs = np.rad2deg(gs)
    if no_static:
        gs[:, :2] = 0
    i = np.arange(n)
    band = np.kron(np.abs(i[:, None] - i[None, :]) <= cr, np.ones((ND, ND), dtype=bool))
    return gs, np.where(band, K, 0.0)


class Refs:
    """Exact rotational ring parameters: cgNA+ from one calculation of the repeated ring, folded."""

    def __init__(self, P):
        self.P, self._cache = P, {}

    def ring(self, model: str, seq: str):
        key = (model, seq)
        if key not in self._cache:
            P = self.P
            if model == 'cgnaplus':
                N = len(seq)
                mid = max(1, -(-150 // N))
                reps = 2 * mid + 1
                gs, K = quiet(P.cgnaplus_bps_params, seq * reps, translations_in_nm=True, euler_definition=True,
                              group_split=True, parameter_set_name='curves_plus', remove_factor_five=True,
                              rotations_only=True)
                rows = np.pad(dense(K)[ND * mid * N:ND * (mid + 1) * N], ((0, 0), (0, ND)))
                self._cache[key] = (np.asarray(gs)[mid * N:(mid + 1) * N],
                                    rows.reshape(ND * N, reps, ND * N).sum(axis=1))
            else:
                p = P.GenStiffness(method='md' if model == 'lankas' else 'crystal').gen_params(
                    seq + seq[0], use_group=True, sparse=True)
                gs = P.statevec2vecs(P.vector_rotmarginal(P.vecs2statevec(p['groundstate'])), vdim=3)
                self._cache[key] = (gs, dense(P.matrix_rotmarginal(p['stiffness'])))
        return self._cache[key]

    def idb_units(self, model: str, seq: str, cg: int, cr: int, disc_len: float = 0.34):
        gs, K = self.ring(model, seq)
        if cg > 1:
            K = dense(quiet(self.P.cg_stiffmat, gs, K, cg, use_sparse=False))
            gs = self.P.cg_groundstate(gs, cg)
        n = K.shape[0] // ND
        return np.rad2deg(gs), np.where(ring_mask(n, cr), K, 0.0) * disc_len * cg


def import_polycg(root: Path):
    sys.path.insert(0, str(root))
    import polycg
    if Path(polycg.__file__).resolve().parent != (root / 'polycg').resolve():
        sys.exit(f'imported polycg from {polycg.__file__}, expected {root / "polycg"}')
    from polycg import polymc_idb
    from polycg.partials import partial_stiff
    from polycg.cgnaplus import cgnaplus_bps_params
    from polycg.cg import cg_stiffmat, cg_groundstate
    from polycg.models.RBPStiff.read_params import GenStiffness
    from polycg.transforms.transform_marginals import matrix_rotmarginal, vector_rotmarginal
    from polycg.transforms.transform_statevec import statevec2vecs, vecs2statevec
    from polycg.utils.seq import seq2oliseq, unique_olis_in_seq
    from polycg.utils.bmat import BlockOverlapMatrix
    return argparse.Namespace(**{k: v for k, v in locals().items() if k not in ('root', 'polycg')})


def check_closed_idb(rep: Report, refs: Refs, P, label: str, idbfn: Path, model: str, seq: str, cg: int,
                     tol_rel: float = 0.0, factor: float = 1.0):
    """Rebuilt ring vs reference: |K - Kref| <= TOL_ABS + tol_rel * max|Kref| entrywise, ground state within
    TOL_GS_DEG, one entry per site, both copies of every coupling block identical, no boundary characters."""
    try:
        r = rebuild(idbfn, P.seq2oliseq)
    except Exception as e:
        rep.check(label, False, f'{type(e).__name__}: {e}'[:110])
        return None
    gsx, Kx = refs.idb_units(model, seq, cg, r['cr'])
    Kx = factor * Kx
    diff = np.abs(r['K'] - Kx)
    scale = np.abs(Kx).max()
    dgs = np.abs(r['gs'] - gsx).max()
    structure = r['entries'] == r['N'] and r['inexact'] == 0 and 'x' not in r['seq'] and r['N'] >= 2 * r['cr'] + 2
    rep.check(label, structure and diff.max() <= TOL_ABS + tol_rel * scale and dgs <= TOL_GS_DEG,
              f"{r['N']} sites, {r['entries']} entries, inexact pairs {r['inexact']}, stiffness {diff.max():.1e} "
              f"({diff.max() / scale:.1e} rel.), closing pair {diff[-ND:, :ND].max():.1e}, ground state {dgs:.0e} deg")
    return r


def phase_closed(root: Path, args, rep: Report, tmp: Path, pool: ThreadPoolExecutor) -> None:
    P = import_polycg(root)
    refs = Refs(P)
    rep.section('Closed chains: IDB files read back, ring rebuilt as PolyMC does, compared with exact references')
    print(f'     stiffness error = max|K - Kref| / max|Kref| (couplings up to the interaction range); '
          f'IDB values have 3 decimals', flush=True)

    # 1) command-line tool
    rep.section('1) Command-line tool, closed chains')
    s100, s250 = SEQ[:100], SEQ[300:550]
    runs = {
        'cgnaplus': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '1', '5', '10']),
        'options': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '1', '5', '-sc', '2',
                                            '-nostatic', '-xyz', '-gs']),
        'lankas': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '1', '5', '-m', 'lankas']),
        'olson': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '1', '4', '-m', 'olson']),
        'collision': ({'ring80.seq': COLLISION}, ['-seqfns', 'ring80.seq', '-closed']),
        'minimal': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '20', '-cr', '1']),
    }
    s480 = SEQ[1500:1980]
    if not args.quick:
        runs['long'] = ({'ring250.seq': s250}, ['-seqfns', 'ring250.seq', '-closed', '-cg', '1', '5'])
        runs['coarse'] = ({'ring480.seq': s480}, ['-seqfns', 'ring480.seq', '-closed', '-cg', '80', '-cr', '1'])
    errors = {
        'not a multiple of the composite size (-cg 1 3)': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '1', '3'], 'multiple of the composite size'),
        '5 composites for coupling range 4 (-cg 1 20)': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '1', '20'], 'too short for coupling range'),
        '5 composites for coupling range 2 (-cg 20 -cr 2, minimum 6)': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '20', '-cr', '2'], 'too short for coupling range'),
        '9 bp for coupling range 4 (minimum 10)': ({'ring9.seq': SEQ[:9]}, ['-seqfns', 'ring9.seq', '-closed'], 'too short for coupling range'),
        'first_id != 0 (-cg 5 -fid 2)': ({'ring100.seq': s100}, ['-seqfns', 'ring100.seq', '-closed', '-cg', '5', '-fid', '2'], 'first_id=0'),
    }

    def run(name, inputs, argv):
        d = tmp / 'closed' / name
        write_inputs(d, inputs)
        return d, run_py(root, d, ['-m', 'polycg.polymc_idb', *argv])

    fut = {name: pool.submit(run, name, *spec) for name, spec in runs.items()}
    fut.update({name: pool.submit(run, 'err_' + str(k), *spec[:2]) for k, (name, spec) in enumerate(errors.items())})
    # build the references while the command-line runs proceed
    for model, seq in (('cgnaplus', s100), ('lankas', s100), ('olson', s100), ('cgnaplus', COLLISION)) + \
            ((('cgnaplus', s250), ('cgnaplus', s480)) if not args.quick else ()):
        refs.ring(model, seq)

    d, r = fut['cgnaplus'].result()
    rep.check('cgNA+ 100 bp ring, -cg 1 5 10 runs', r.returncode == 0, last_line(r.stderr) if r.returncode else '')
    for cg in (1, 5, 10):
        check_closed_idb(rep, refs, P, f'cgNA+ 100 bp ring, cg={cg}', d / f'ring100_cgnaplus_{cg}bp_4cr_closed.idb',
                         'cgnaplus', s100, cg)

    d, r = fut['minimal'].result()
    rep.check('cgNA+ 100 bp ring, -cg 20 -cr 1 (5 composites, the minimum for cr=1 is 4) runs', r.returncode == 0,
              last_line(r.stderr) if r.returncode else '')
    check_closed_idb(rep, refs, P, 'cgNA+ 100 bp ring, cg=20, cr=1', d / 'ring100_cgnaplus_20bp_1cr_closed.idb',
                     'cgnaplus', s100, 20)

    d, r = fut['options'].result()
    rep.check('cgNA+ 100 bp ring, -sc 2 -nostatic -xyz -gs runs', r.returncode == 0, last_line(r.stderr) if r.returncode else '')
    for cg in (1, 5):
        base = d / f'ring100_cgnaplus_{cg}bp_4cr_rescaled_2p000_closed'
        rs = check_closed_idb(rep, refs, P, f'cgNA+ 100 bp ring, cg={cg}, -sc 2 gives twice the stiffness',
                              Path(str(base) + '.idb'), 'cgnaplus', s100, cg, factor=2.0)
        try:
            ns = rebuild(Path(str(base) + '_no_static.idb'), P.seq2oliseq)
            ok = rs is not None and np.array_equal(ns['K'], rs['K']) and np.all(ns['gs'][:, :2] == 0) \
                and np.array_equal(ns['gs'][:, 2], rs['gs'][:, 2])
            rep.check(f'cg={cg}, -nostatic: tilt and roll zeroed, twist and stiffness unchanged', ok)
        except Exception as e:
            rep.check(f'cg={cg}, -nostatic', False, f'{type(e).__name__}: {e}'[:110])
        gsfile = Path(str(base) + '_gs.npy')
        written = [p.name for p in d.glob(base.name + '_gs*')]
        expect = 4 if cg == 1 else 2       # cg=1: .npy, .pdb, .xyz, _10bp.xyz; cg>1: .npy, .xyz
        rep.check(f'cg={cg}, -xyz -gs: ground-state files written', len(written) == expect and gsfile.exists()
                  and np.load(gsfile).shape == (100 // cg, 3), ', '.join(sorted(written)))

    for model, cgs in (('lankas', (1, 5)), ('olson', (1, 4))):
        d, r = fut[model].result()
        rep.check(f'{model} 100 bp ring, -cg {" ".join(map(str, cgs))} runs', r.returncode == 0,
                  last_line(r.stderr) if r.returncode else '')
        for cg in cgs:
            check_closed_idb(rep, refs, P, f'{model} 100 bp ring, cg={cg}', d / f'ring100_{model}_{cg}bp_4cr_closed.idb',
                             model, s100, cg)

    olis = [P.seq2oliseq(COLLISION, i, 4, True) for i in range(len(COLLISION))]
    rep.check('test ring: linear 10-mers unique, but a seam window repeats an interior window',
              P.unique_olis_in_seq(COLLISION, 10) and len(set(olis)) < len(olis),
              f'steps sharing an oligomer: {[i for i, o in enumerate(olis) if olis.count(o) > 1]}')
    d, r = fut['collision'].result()
    rep.check('that ring runs', r.returncode == 0, last_line(r.stderr) if r.returncode else '')
    res = check_closed_idb(rep, refs, P, 'that ring gets one entry per site (generated assignment sequence)',
                           d / 'ring80_cgnaplus_1bp_4cr_closed.idb', 'cgnaplus', COLLISION, 1)
    if res is not None:
        rep.check('  ... and the real sequence is kept in .origseq', res['seq'] != COLLISION and
                  (d / 'ring80_cgnaplus_1bp_4cr_closed.origseq').read_text() == COLLISION)

    if not args.quick:
        d, r = fut['long'].result()
        rep.check('cgNA+ 250 bp ring, -cg 1 5 runs', r.returncode == 0, last_line(r.stderr) if r.returncode else '')
        for cg in (1, 5):
            check_closed_idb(rep, refs, P, f'cgNA+ 250 bp ring, cg={cg}', d / f'ring250_cgnaplus_{cg}bp_4cr_closed.idb',
                             'cgnaplus', s250, cg, tol_rel=TOL_REL_LONG)
        d, r = fut['coarse'].result()
        rep.check('cgNA+ 480 bp ring, -cg 80 -cr 1 runs (the blocks used to be no longer than the overlap)',
                  r.returncode == 0, last_line(r.stderr) if r.returncode else '')
        check_closed_idb(rep, refs, P, 'cgNA+ 480 bp ring, cg=80, cr=1', d / 'ring480_cgnaplus_80bp_1cr_closed.idb',
                         'cgnaplus', s480, 80, tol_rel=TOL_REL_LONG)

    rep.section('2) Invalid closed inputs fail before anything is computed or written')
    for name, (inputs, _, msg) in errors.items():
        d, r = fut[name].result()
        written = sorted(p.name for p in d.iterdir() if p.name not in inputs)
        rep.check(name, r.returncode != 0 and msg in r.stderr and not written,
                  last_line(r.stderr) + (f' | written: {written}' if written else ''))

    # 3) stiff2idb called directly
    rep.section('3) stiff2idb(closed=True) called directly with synthetic ring matrices')
    rng = np.random.default_rng(5)
    for N, cr in ((30, 3), (10, 4), (12, 2), (7, 1)):
        A = np.round(rng.normal(size=(N * ND, N * ND)), 3)
        K = np.where(ring_mask(N, cr + 1), A + A.T, 0.0) + np.kron(np.eye(N), 20 * np.eye(ND))
        gs = np.round(rng.normal(size=N * ND), 3)
        for fmt in ('dense', 'csc', 'bmat'):
            M = K.copy() if fmt == 'dense' else sp.sparse.csc_matrix(K)
            if fmt == 'bmat':
                M = P.BlockOverlapMatrix(average=True, periodic=False, fixed_size=True, xlo=0, xhi=N * ND, ylo=0,
                                         yhi=N * ND)
                M.add_block(K, 0, N * ND, y1=0, y2=N * ND)
            base = tmp / 'closed' / 'lib' / f'ring_N{N}_cr{cr}_{fmt}'
            base.parent.mkdir(parents=True, exist_ok=True)
            try:
                P.polymc_idb.stiff2idb(str(base), gs, M, cr, True)
                r = rebuild(Path(str(base) + '.idb'), P.seq2oliseq)
                err = np.abs(r['K'] - np.where(ring_mask(N, cr), K, 0.0)).max()
                dgs = np.abs(r['gs'].ravel() - gs).max()
                rep.check(f'N={N}, cr={cr}, {fmt}: ring matrix reproduced', err < 1e-9 and dgs < 1e-9 and
                          r['entries'] == N and r['inexact'] == 0, f'max error {err:.0e}, {r["entries"]} entries')
            except Exception as e:
                rep.check(f'N={N}, cr={cr}, {fmt}', False, f'{type(e).__name__}: {e}'[:110])
    for label, call, msg in (
            ('ring shorter than 2*cr+2 sites is rejected',
             lambda: P.polymc_idb.stiff2idb(str(tmp / 'x'), np.zeros(27), np.eye(27), 4, True), 'needs at least'),
            ('sequence of the wrong length is rejected',
             lambda: P.polymc_idb.stiff2idb(str(tmp / 'x'), np.zeros(30), np.eye(30), 1, True, 'acgtacgta'),
             'requires a sequence of length')):
        try:
            call()
            rep.check(label, False, 'no error raised')
        except ValueError as e:
            rep.check(label, msg in str(e), str(e)[:110])


#######################################################################################
# PolyMC


POLYMC_INPUT = """mode = {mode}
use_cluster_twist = {cluster}
num_twist = 0
check_link = 0
check_consistency_every = 1000
seed = 1234
IDB = {idb}
sequence = {seq}
subtract_T0 = {subtract}
T = 300
num_bp = {nbp}
sigma = 0
torque = 0
force = 0
EV = 0
steps = 20000
equi = 0
print_every = 1000000
copy_input = 0
dump_dir = dump/run
Thetasn = 2000
En = 2000
"""


def phase_polymc(root: Path, args, rep: Report, tmp: Path, pool: ThreadPoolExecutor) -> None:
    rep.section('PolyMC on IDB files (rings: mode = plasmid, open chains: mode = open): energies reported by PolyMC '
                'vs rebuilt stiffness matrix')
    if not args.polymc:
        rep.info('skipped: pass --polymc <PolyMC executable>')
        return
    exe = Path(args.polymc).expanduser().resolve()
    if not exe.exists():
        rep.check(f'PolyMC executable {exe}', False, 'not found')
        return
    P = import_polycg(root)
    cases = [('cgNA+ 100 bp ring, cg=1', 'cgnaplus', SEQ[:100], 1, True),
             ('cgNA+ 100 bp ring, cg=5', 'cgnaplus', SEQ[:100], 5, True),
             ('cgNA+ 120 bp ring, cg=10', 'cgnaplus', SEQ[1000:1120], 10, True),
             ('lankas 100 bp ring, cg=1', 'lankas', SEQ[:100], 1, True),
             ('cgNA+ 300 bp open chain, cg=1', 'cgnaplus', SEQ[:300], 1, False),
             ('cgNA+ 300 bp open chain, cg=5 (coarse-grained in blocks)', 'cgnaplus', SEQ[:300], 5, False),
             ('cgNA+ 101 bp open chain, cg=5 (coarse-grained in one piece)', 'cgnaplus', SEQ[200:301], 5, False),
             ('olson 200 bp open chain, cg=4', 'olson', SEQ[400:600], 4, False)]

    def run(k, label, model, seq, cg, closed):
        d = tmp / 'polymc' / f'{k}'
        write_inputs(d, {'dna.seq': seq})
        r = run_py(root, d, ['-m', 'polycg.polymc_idb', '-seqfns', 'dna.seq', '-m', model, '-cg', str(cg)]
                   + (['-closed'] if closed else []))
        if r.returncode != 0:
            return d, r, None
        base = f'dna_{model}_{cg}bp_4cr' + ('_closed' if closed else '')
        nbp = len((d / f'{base}.seq').read_text().strip())
        mode = 'plasmid' if closed else 'open'
        (d / 'input').write_text(POLYMC_INPUT.format(mode=mode, cluster=int(closed), subtract=int(not closed),
                                                     idb=f'{base}.idb', seq=f'{base}.seq', nbp=nbp))
        (d / 'dump').mkdir()
        pm = subprocess.run([str(exe), mode, '-in', 'input'], cwd=d, capture_output=True, text=True, timeout=600)
        return d, r, pm

    fut = [pool.submit(run, k, *case) for k, case in enumerate(cases)]
    for (label, model, seq, cg, closed), f in zip(cases, fut):
        d, r, pm = f.result()
        if pm is None:
            rep.check(label, False, 'polymc_idb failed: ' + last_line(r.stderr))
            continue
        out = pm.stdout + pm.stderr
        clean = pm.returncode == 0 and not re.search(r'Conflict|inconsistent|missing entry|Error', out)
        if not clean:
            rep.check(label, False, f'PolyMC exit {pm.returncode}: ' + last_line(out))
            continue
        base = f'dna_{model}_{cg}bp_4cr' + ('_closed' if closed else '')
        mat = rebuild(d / f'{base}.idb', P.seq2oliseq, closed=closed)
        N, cr, a = mat['N'], mat['cr'], mat['disc']
        theta = np.loadtxt(d / 'dump' / 'run.thetas')[:, 1:].reshape(-1, N, ND)
        energy = np.loadtxt(d / 'dump' / 'run.en', ndmin=2)
        dev = (theta - np.deg2rad(mat['gs'])).reshape(len(theta), -1)
        e_full = KT_REF * np.einsum('si,ij,sj->s', dev, mat['K'], dev) / (2 * a)
        e_pmc = energy[:, 1]
        err = np.abs(e_full - e_pmc).max() / np.abs(e_pmc).min()
        ok = len(e_pmc) == len(theta) > 0 and err <= TOL_POLYMC
        detail = f'{len(e_pmc)} configurations of {N} steps, energy error {err:.1e}'
        if closed:
            noseam = mat['K'].copy()
            for i in range(N):
                for k in range(1, cr + 1):
                    if i + k >= N:
                        j = (i + k) % N
                        noseam[i * ND:(i + 1) * ND, j * ND:(j + 1) * ND] = 0
                        noseam[j * ND:(j + 1) * ND, i * ND:(i + 1) * ND] = 0
            e_noseam = KT_REF * np.einsum('si,ij,sj->s', dev, noseam, dev) / (2 * a)
            seam = np.mean(np.abs(e_noseam - e_pmc) / np.abs(e_pmc))
            has_seam = np.abs(mat['K'] - noseam).max() > 0
            ok = ok and (not has_seam or seam > 20 * err)
            detail += '; without the couplings across the seam ' + (f'{seam:.1e}' if has_seam else
                                                                     'identical (no couplings between steps)')
        rep.check(label, ok, detail)


#######################################################################################


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='PolyCG checkout (directory containing the polycg package)')
    ap.add_argument('phase', choices=['open', 'closed', 'polymc', 'all'])
    ap.add_argument('--baseline-rev', default=BASELINE_REV, help='revision of polymc_idb.py for the open regression test')
    ap.add_argument('--polymc', default=None, help='PolyMC executable (phase polymc)')
    ap.add_argument('--quick', action='store_true', help='fewer and shorter cases')
    ap.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1), help='parallel processes')
    ap.add_argument('--keep', action='store_true', help='keep the temporary directory')
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not (root / 'polycg' / '__init__.py').exists():
        sys.exit(f'{root} does not contain the polycg package')

    rep = Report()
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix='verify_polymc_idb_'))
    print(f'testing polycg from {root}; temporary files in {tmp}')
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            if args.phase in ('open', 'all'):
                phase_open(root, args, rep, tmp, pool)
            if args.phase in ('closed', 'all'):
                phase_closed(root, args, rep, tmp, pool)
            if args.phase in ('polymc', 'all'):
                phase_polymc(root, args, rep, tmp, pool)
    finally:
        if not args.keep:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{rep.passed} passed, {rep.failed} failed: '
          f'{"ALL CHECKS PASSED" if rep.failed == 0 else "CHECKS FAILED"}  ({time.time() - t0:.0f} s)')
    return 1 if rep.failed else 0


if __name__ == '__main__':
    sys.exit(main())
