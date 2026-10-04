#!/usr/bin/env python3
"""Verify the closed (periodic) stiffness assembly of PolyCG.

Closed rings are assembled by folding (polycg/partials.py: _partial_stiff_closed, and the closed
composite_size > 1 branch of gen_params). The ring is unrolled into an open chain that extends
R = overlap_size + tail_size steps beyond the ring on either side, this chain is assembled with the
open-chain code, and the rows of the ring are folded back onto the ring by adding up all periodic
images. This script checks the results against exact references:

  * A synthetic banded model (BandedModel). Its couplings depend only on the dinucleotides of the
    two steps and on their distance, so the model has no end effects. Every assembled matrix must
    therefore equal the exact periodic fold to rounding precision, for any ring length and any
    block/overlap/tail setting. No cgNA+ calculation is needed.
  * The exact periodic cgNA+ ring (exact_ring_reference). One full cgNA+ calculation, without block
    assembly, of the ring sequence repeated several times; the couplings of the middle copy to all
    periodic images are summed. Rings that fit into a single cgNA+ call after unrolling
    (N < 2*block - 3*overlap - 2*tail, i.e. N < 140 with the defaults) must be exact. Longer rings
    must be exact for pairs at most overlap_size steps apart, and at the truncation level beyond.

Usage:
    python verify_closed_partial_stiff.py --root ~/Dev/PolyCG check [--quick] [--cli] [--baseline FILE]
    python verify_closed_partial_stiff.py --root <checkout before the fix> baseline --out FILE

--root is the directory that contains the polycg package. The optional baseline, stored with a
checkout from before the fix, lets check 5 report how much the results for rings of 180 bp and more
have changed. Exit code 0 if all checks pass. Run time: a few minutes with --quick, about 10-20
minutes in full (single core).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.sparse

warnings.filterwarnings('ignore')

SEQ = ''.join(np.random.default_rng(5).choice(list('ACGT'), 1200))   # fixed random test sequence
POLY_A = 'A' * 1200
AG = 'AG' * 600
LARGE_RINGS = [180, 200, 250, 300, 400]   # rings compared with the baseline (check 5)

# Tolerances, relative to the largest entry of the reference matrix
TOL_EXACT = 1e-6          # cgNA+: pairs at most overlap_size apart, and whole rings in the single-call regime
TOL_TRUNC = 2e-4          # cgNA+: whole matrix of longer rings (first dropped coupling is <= ~1e-4)
TOL_GS = 1e-8             # cgNA+: ground state (absolute)
TOL_CG = 2e-3             # cgNA+: coarse-grained stiffness of the closed composite path
TOL_CG_SEAM = 5e-3        # cgNA+: closing composite pair, relative to its own largest entry
TOL_SYNTH = 1e-12         # synthetic model: base level, everything is exact up to rounding
TOL_SYNTH_CG = 1e-5       # synthetic model: coarse-grained (end effects of the open-chain coarse-graining)


def load_polycg(root: str):
    root = Path(root).expanduser().resolve()
    if not (root / 'polycg' / '__init__.py').exists():
        sys.exit(f'{root} does not contain the polycg package')
    sys.path.insert(0, str(root))
    import polycg
    if Path(polycg.__file__).resolve().parent != root / 'polycg':
        sys.exit(f'imported polycg from {polycg.__file__}, expected {root / "polycg"}')
    G = importlib.import_module('polycg._gen_params')
    P = importlib.import_module('polycg.partials')
    C = importlib.import_module('polycg.cg')
    return root, G, P, C


class Report:
    def __init__(self):
        self.failed = 0

    def check(self, name, cond, detail=''):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f'  [{detail}]' if detail else ''), flush=True)
        self.failed += not cond

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


def err(e: Exception) -> str:
    return f'{type(e).__name__}: {e}'[:120]


def dense(K) -> np.ndarray:
    if hasattr(K, 'to_sparse'):          # BlockOverlapMatrix
        K = K.to_sparse()
    return K.toarray() if sp.sparse.issparse(K) else np.asarray(K)


def ring_distance(N: int, ndims: int) -> np.ndarray:
    """Distance the short way round the ring between the steps of every matrix entry."""
    i = np.arange(N)
    d = np.abs(i[:, None] - i[None, :])
    return np.kron(np.minimum(d, N - d), np.ones((ndims, ndims), dtype=int))


def compare(K, Kref, N, overlap, ndims=6):
    """Relative errors of K against Kref: whole matrix, and pairs at most `overlap` apart."""
    K, Kref = dense(K), dense(Kref)
    scale = np.abs(Kref).max()
    diff = np.abs(K - Kref)
    short = diff[ring_distance(N, ndims) <= overlap]
    return diff.max() / scale, (short.max() if short.size else 0.0) / scale, np.abs(K - K.T).max() / scale


def gen_defaults(G):
    par = inspect.signature(G.gen_params).parameters
    return par['block_size'].default, par['overlap_size'].default, par['tail_size'].default


#######################################################################################
# exact references


def exact_ring_reference(G, seq: str, margin: int = 150, _cache: dict = {}):
    """Exact cgNA+ ring: fold of one full cgNA+ calculation of the repeated ring sequence.

    The ring (N bp, N steps; step N-1 joins the last and the first bp) is repeated reps times (reps
    odd, the middle copy at least `margin` bp from both chain ends) and computed without block
    assembly. The rows of the middle copy are folded: Kref[i, j] = sum_m K[mid*N+i, mid*N+j+m*N].
    """
    if seq in _cache:
        return _cache[seq]
    N = len(seq)
    mid = max(1, -(-margin // N))
    reps = 2 * mid + 1
    p = quiet(G.gen_params, 'cgnaplus', seq * reps, composite_size=1, allow_partial=False)
    K = p.stiffmat
    K = K.tocsr() if sp.sparse.issparse(K) else sp.sparse.csr_matrix(np.asarray(K))
    rows = K[6 * mid * N:6 * (mid + 1) * N, :].toarray()
    rows = np.pad(rows, ((0, 0), (0, 6)))            # an open chain of reps*N bp has reps*N-1 steps
    Kref = rows.reshape(6 * N, reps, 6 * N).sum(axis=1)
    gsref = np.asarray(p.shape_params)[mid * N:(mid + 1) * N]
    _cache[seq] = (gsref, Kref)
    return gsref, Kref


class BandedModel:
    """Synthetic stand-in for cgnaplus_bps_params with exactly local, sequence-dependent parameters.

    Step k of a sequence is characterised by its dinucleotide. The ground state of a step and the
    stiffness block of a pair of steps (k, l), 0 < l-k <= r, depend only on the dinucleotides of the
    two steps and on l-k; steps further apart are uncoupled. The matrix of any subsequence is
    therefore exactly the restriction of the matrix of a longer chain, and the exact stiffness of a
    ring is the sum over its periodic images (ring()).
    """

    BASES = {b: i for i, b in enumerate('ACGT')} | {b: i for i, b in enumerate('acgt')}

    def __init__(self, r: int, ndims: int = 6):
        self.r, self.ndims = r, ndims
        rng = np.random.default_rng(1)
        a = rng.normal(size=(16, ndims, ndims))
        self.diag = 10.0 * np.eye(ndims) + 0.25 * (a + a.transpose(0, 2, 1))
        self.coup = 0.2 * rng.normal(size=(r + 1, ndims, ndims))
        self.w = rng.uniform(0.5, 1.5, size=16)
        self.gs = 0.1 * rng.normal(size=(16, ndims))

    def _block(self, dk: int, dl: int, d: int) -> np.ndarray:
        return 0.6 ** d * self.w[dk] * self.w[dl] * self.coup[d]

    def _dinucs(self, seq: str) -> np.ndarray:
        return np.array([4 * self.BASES[a] + self.BASES[b] for a, b in zip(seq[:-1], seq[1:])], dtype=int)

    def __call__(self, seq: str, **kwargs):
        dn, nd = self._dinucs(seq), self.ndims
        n = len(dn)
        K = np.zeros((n * nd, n * nd))
        for k in range(n):
            K[k * nd:(k + 1) * nd, k * nd:(k + 1) * nd] = self.diag[dn[k]]
            for d in range(1, min(self.r, n - 1 - k) + 1):
                B = self._block(dn[k], dn[k + d], d)
                K[k * nd:(k + 1) * nd, (k + d) * nd:(k + d + 1) * nd] = B
                K[(k + d) * nd:(k + d + 1) * nd, k * nd:(k + 1) * nd] = B.T
        return self.gs[dn].copy(), K

    def ring(self, seq: str):
        """Exact ground state and stiffness of the ring seq (sum over all periodic images)."""
        N, nd = len(seq), self.ndims
        dn = self._dinucs(seq + seq[0])
        K = np.zeros((N * nd, N * nd))
        for k in range(N):
            K[k * nd:(k + 1) * nd, k * nd:(k + 1) * nd] += self.diag[dn[k]]
            for d in range(1, self.r + 1):
                l = (k + d) % N
                B = self._block(dn[k], dn[l], d)
                K[k * nd:(k + 1) * nd, l * nd:(l + 1) * nd] += B
                K[l * nd:(l + 1) * nd, k * nd:(k + 1) * nd] += B.T
        return self.gs[dn].copy(), K


@contextlib.contextmanager
def stand_in(G, model):
    real = G.cgnaplus_bps_params
    G.cgnaplus_bps_params = model
    try:
        yield
    finally:
        G.cgnaplus_bps_params = real


#######################################################################################
# phases


def save_baseline(G, out):
    data = {}
    for N in LARGE_RINGS:
        p = quiet(G.gen_params, 'cgnaplus', SEQ[:N], composite_size=1, closed=True)
        K = sp.sparse.csr_matrix(p.stiffmat)
        data.update({f'gs_{N}': np.asarray(p.shape_params), f'K_{N}_data': K.data, f'K_{N}_indices': K.indices,
                     f'K_{N}_indptr': K.indptr, f'K_{N}_shape': np.array(K.shape)})
        print(f'  stored closed cgNA+ parameters for N={N}', flush=True)
    np.savez_compressed(out, **data)
    print(f'baseline written to {out}')


def check(root, G, P, C, args):
    rep = Report()
    t0 = time.time()
    block, overlap, tail = gen_defaults(G)
    n_single = 2 * block - 3 * overlap - 2 * tail    # rings shorter than this need a single cgNA+ call
    print(f'gen_params defaults: block_size={block}, overlap_size={overlap}, tail_size={tail}')

    ###################################################################################
    rep.section('1) Inputs that failed before the fix run and return the right shapes (real cgNA+)')
    cases = [
        ("closed N=100 'ACGT'*25 (document section 4)", 'ACGT' * 25, dict(closed=True), (600, None)),
        ("open 21 bp 'ACGT'*5+'A' (document section 4)", 'ACGT' * 5 + 'A', dict(), (120, None)),
        ('closed N=2', SEQ[:2], dict(closed=True), (12, None)),
        ('closed N=12', SEQ[:12], dict(closed=True), (72, None)),
        ('closed N=20', SEQ[:20], dict(closed=True), (120, None)),
        ('closed N=21', SEQ[:21], dict(closed=True), (126, None)),
        ('closed N=30', SEQ[:30], dict(closed=True), (180, None)),
        ('closed N=41', SEQ[:41], dict(closed=True), (246, None)),
        ('closed N=300, block 30 / overlap 20', SEQ[:300], dict(closed=True, block_size=30, overlap_size=20), (1800, None)),
        ('closed N=100, tail_size=0', SEQ[:100], dict(closed=True, tail_size=0), (600, None)),
        ('closed composite_size=5, N=20', SEQ[:20], dict(closed=True, composite_size=5), (120, 24)),
        ('closed composite_size=5, N=5 (one composite)', SEQ[:5], dict(closed=True, composite_size=5), (30, 6)),
        ('closed composite_size=20, N=40', SEQ[:40], dict(closed=True, composite_size=20), (240, 12)),
    ]
    for name, seq, kw, (n_base, n_cg) in cases:
        try:
            p = quiet(G.gen_params, 'cgnaplus', seq, **kw)
            shape_ok = p.stiffmat.shape == (n_base, n_base) and np.asarray(p.shape_params).shape == (n_base // 6, 6)
            if n_cg is not None:
                shape_ok &= p.cg_stiffmat.shape == (n_cg, n_cg)
            rep.check(name, shape_ok, f'stiffmat {p.stiffmat.shape}' + (f', cg_stiffmat {p.cg_stiffmat.shape}' if n_cg else ''))
        except Exception as e:
            rep.check(name, False, err(e))

    ###################################################################################
    rep.section('2) Exactness with a synthetic banded model (no cgNA+; error relative to max|K|)')
    nmax = 120 if args.quick else 300
    settings = [((120, 20, 20), 15), ((60, 10, 10), 8), ((30, 20, 20), 15), ((120, 20, 0), 15), ((5, 3, 1), 3)]
    for (b, o, t), r in settings:
        model = BandedModel(r)
        worst, worst_gs, bad = 0.0, 0.0, []
        Ns = range(1, (nmax if (b, o, t) == (120, 20, 20) else nmax * 2 // 3) + 1)
        with stand_in(G, model):
            for N in Ns:
                seq = SEQ[:N]
                try:
                    p = quiet(G.gen_params, 'cgnaplus', seq, closed=True, block_size=b, overlap_size=o, tail_size=t)
                    gsx, Kx = model.ring(seq)
                    e, _, _ = compare(p.stiffmat, Kx, N, o)
                    worst, worst_gs = max(worst, e), max(worst_gs, np.abs(np.asarray(p.shape_params) - gsx).max())
                except Exception as ex:
                    bad.append((N, err(ex)))
        rep.check(f'closed cg=1, block {b} / overlap {o} / tail {t}, N={Ns.start}..{Ns.stop - 1}',
                  not bad and worst <= TOL_SYNTH and worst_gs <= TOL_SYNTH,
                  f'stiffness {worst:.1e}, ground state {worst_gs:.1e}' + (f', {len(bad)} failures, first {bad[0]}' if bad else ''))

    model = BandedModel(15)
    for cg in (2, 5, 10, 20):
        Ns = [n for n in range(cg, (100 if args.quick else 200) + 1, cg) if cg < 20 or n <= 80]
        worst = worst_cg = worst_gs = 0.0
        bad = []
        with stand_in(G, model):
            for N in Ns:
                seq = SEQ[:N]
                try:
                    p = quiet(G.gen_params, 'cgnaplus', seq, closed=True, composite_size=cg)
                    gsx, Kx = model.ring(seq)
                    Kcx = quiet(C.cg_stiffmat, gsx, Kx, cg, use_sparse=False)
                    worst = max(worst, compare(p.stiffmat, Kx, N, overlap)[0])
                    worst_cg = max(worst_cg, compare(p.cg_stiffmat, Kcx, N // cg, 0)[0])
                    worst_gs = max(worst_gs, np.abs(np.asarray(p.cg_shape_params) - C.cg_groundstate(gsx, cg)).max())
                except Exception as ex:
                    bad.append((N, err(ex)))
        rep.check(f'closed composite_size={cg}, N={Ns[0]}..{Ns[-1]}',
                  not bad and worst <= TOL_SYNTH and worst_cg <= TOL_SYNTH_CG and worst_gs <= 1e-10,
                  f'stiffmat {worst:.1e}, cg_stiffmat {worst_cg:.1e}, cg ground state {worst_gs:.1e}'
                  + (f', {len(bad)} failures, first {bad[0]}' if bad else ''))

    # open chains start at 3 bp: a single-step chain (2 bp) is rejected by DNAParameters (shape_params
    # of shape (6,)), independently of the assembly; cgNA+ itself needs at least 4 bp
    worst, bad = 0.0, []
    with stand_in(G, model):
        for N in range(3, nmax + 1):
            try:
                p = quiet(G.gen_params, 'cgnaplus', SEQ[:N], composite_size=1)
                gsx, Kx = model(SEQ[:N])
                worst = max(worst, np.abs(dense(p.stiffmat) - Kx).max() / np.abs(Kx).max())
            except Exception as ex:
                bad.append((N, err(ex)))
    rep.check(f'open cg=1, N=3..{nmax} bp', not bad and worst <= TOL_SYNTH,
              f'stiffness {worst:.1e}' + (f', {len(bad)} failures, first {bad[0]}' if bad else ''))

    # partial_stiff called directly with polymc_idb.py's settings (ndims=3, block 120 / overlap 20 / tail 20)
    model3 = BandedModel(15, ndims=3)
    for closed in (True, False):
        worst, bad = 0.0, []
        for N in range(2, (nmax if not args.quick else 80) + 1):
            try:
                gs, K = quiet(P.partial_stiff, SEQ[:N], model3, {}, block_size=120, overlap_size=20, tail_size=20,
                              closed=closed, ndims=3)
                Kx = model3.ring(SEQ[:N])[1] if closed else model3(SEQ[:N])[1]
                worst = max(worst, np.abs(dense(K) - Kx).max() / np.abs(Kx).max())
            except Exception as ex:
                bad.append((N, err(ex)))
        rep.check(f'partial_stiff direct, ndims=3, closed={closed}', not bad and worst <= TOL_SYNTH,
                  f'stiffness {worst:.1e}' + (f', {len(bad)} failures, first {bad[0]}' if bad else ''))

    ###################################################################################
    rep.section('3) Closed cgNA+ rings against the exact periodic reference (composite_size=1)')
    print(f'     error = max|K - Kref| / max|Kref|; "short" = pairs at most {overlap} steps apart; '
          f'rings with N < {n_single} need a single cgNA+ call and must be exact', flush=True)
    if args.quick:
        cases = [('random', SEQ, N) for N in (4, 12, 36, 43, 100, 180)] + [('poly-A', POLY_A, 43), ('(AG)n', AG, 43)]
    else:
        cases = [('random', SEQ, N) for N in (2, 4, 8, 12, 20, 25, 34, 36, 40, 41, 43, 50, 62, 63, 100, 139, 140, 179,
                                               180, 250, 400)]
        cases += [(name, s, N) for name, s in (('poly-A', POLY_A), ('(AG)n', AG)) for N in (4, 12, 36, 43, 100, 180)]
    for name, s, N in cases:
        seq = s[:N]
        try:
            p = quiet(G.gen_params, 'cgnaplus', seq, composite_size=1, closed=True)
            gsref, Kref = exact_ring_reference(G, seq)
            tot, short, asym = compare(p.stiffmat, Kref, N, overlap)
            dgs = np.abs(np.asarray(p.shape_params) - gsref).max()
            K = dense(p.stiffmat)
            ev, evref = np.linalg.eigvalsh((K + K.T) / 2)[0], np.linalg.eigvalsh(Kref)[0]
            limit = TOL_EXACT if N < n_single else TOL_TRUNC
            ok = short <= TOL_EXACT and tot <= limit and dgs <= TOL_GS and asym <= 1e-12 and ev > 0 \
                and abs(ev - evref) <= 1e-4 * evref
            rep.check(f'{name} N={N}', ok, f'total {tot:.1e} (limit {limit:.0e}), short {short:.1e}, ground state {dgs:.0e}, '
                                           f'min eigenvalue {ev:.3f} (exact {evref:.3f})')
        except Exception as e:
            rep.check(f'{name} N={N}', False, err(e))

    ###################################################################################
    rep.section('4) Closed composite_size > 1 against the coarse-grained exact ring (real cgNA+)')
    cases = [(2, 12), (5, 20), (5, 100)] if args.quick else \
        [(2, 12), (2, 50), (2, 100), (5, 20), (5, 100), (5, 155), (5, 200), (10, 100), (10, 160)]
    for cg, N in cases:
        seq = SEQ[:N]
        try:
            p = quiet(G.gen_params, 'cgnaplus', seq, composite_size=cg, closed=True)
            gsref, Kref = exact_ring_reference(G, seq)
            Kcref = quiet(C.cg_stiffmat, gsref, Kref, cg, use_sparse=False)
            Kcref = Kcref.toarray() if sp.sparse.issparse(Kcref) else np.asarray(Kcref)
            tot, short, _ = compare(p.stiffmat, Kref, N, overlap)
            Kc = dense(p.cg_stiffmat)
            cg_err = np.abs(Kc - Kcref).max() / np.abs(Kcref).max()
            seam = np.abs(Kc[-6:, :6] - Kcref[-6:, :6]).max() / np.abs(Kcref[-6:, :6]).max()
            dgs = np.abs(np.asarray(p.cg_shape_params) - C.cg_groundstate(gsref, cg)).max()
            ev = np.linalg.eigvalsh((Kc + Kc.T) / 2)[0]
            ok = short <= TOL_EXACT and tot <= TOL_TRUNC and cg_err <= TOL_CG and seam <= TOL_CG_SEAM \
                and dgs <= TOL_GS and ev > 0
            rep.check(f'composite_size={cg}, N={N}', ok,
                      f'stiffmat {tot:.1e} (short {short:.1e}), cg_stiffmat {cg_err:.1e}, closing pair {seam:.1e}, '
                      f'cg ground state {dgs:.0e}, cg min eigenvalue {ev:.2f}')
        except Exception as e:
            rep.check(f'composite_size={cg}, N={N}', False, err(e))

    ###################################################################################
    rep.section('5) Rings >= 180 bp against the baseline from before the fix')
    if not args.baseline:
        rep.info('skipped: store a baseline with a checkout from before the fix and pass --baseline')
    else:
        b = np.load(args.baseline)
        for N in LARGE_RINGS:
            Kb = sp.sparse.csr_matrix((b[f'K_{N}_data'], b[f'K_{N}_indices'], b[f'K_{N}_indptr']),
                                      shape=tuple(b[f'K_{N}_shape'])).toarray()
            p = quiet(G.gen_params, 'cgnaplus', SEQ[:N], composite_size=1, closed=True)
            tot, short, _ = compare(p.stiffmat, Kb, N, overlap)
            dgs = np.abs(np.asarray(p.shape_params) - b[f'gs_{N}']).max()
            rep.check(f'N={N}: changes only beyond {overlap} steps, at the truncation level',
                      short <= TOL_EXACT and tot <= TOL_TRUNC and dgs <= TOL_GS,
                      f'change {tot:.1e} (pairs <= {overlap} apart: {short:.1e}), ground state change {dgs:.0e}')

    ###################################################################################
    if args.cli:
        rep.section('6) Command-line tools (real cgNA+)')
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def run(module, *cli_args):
                return subprocess.run([sys.executable, '-m', module, *cli_args], cwd=tmp, env=env,
                                      capture_output=True, text=True)

            for N in (21, 300):
                (tmp / f's{N}.seq').write_text(SEQ[:N])
            r = run('polycg.polymc_idb', '-seqfns', str(tmp / 's21.seq'))
            rep.check('polymc_idb, open 21 bp', r.returncode == 0 and any(tmp.glob('s21_cgnaplus_1bp*.idb')),
                      '' if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:110])
            r = run('polycg.polymc_idb', '-seqfns', str(tmp / 's300.seq'), '-cg', '1', '2', '5')
            n_idb = len(list(tmp.glob('s300_cgnaplus_*bp*.idb')))
            rep.check('polymc_idb, open 300 bp, -cg 1 2 5', r.returncode == 0 and n_idb == 3,
                      f'{n_idb} IDB files' + ('' if r.returncode == 0 else ', ' + r.stderr.strip().splitlines()[-1][:110]))
            (tmp / 'ring100.seq').write_text(SEQ[:100])
            r = run('polycg.polymc_idb', '-seqfns', str(tmp / 'ring100.seq'), '-closed', '-cg', '1', '5')
            n_idb = len(list(tmp.glob('ring100_cgnaplus_*bp_4cr_closed.idb')))
            rep.check('polymc_idb, closed 100 bp, -cg 1 5 (contents checked by tests/verify_polymc_idb.py)',
                      r.returncode == 0 and n_idb == 2,
                      f'{n_idb} IDB files' + ('' if r.returncode == 0 else ', ' + r.stderr.strip().splitlines()[-1][:110]))
            for closed in (1, 0):
                r = run('polycg.cgnaplus', '-seqfn', str(tmp / 's21.seq'), '-closed', str(closed))
                base = tmp / ('s21.seq_params' + ('_closed' if closed else ''))
                try:
                    gs = np.load(str(base) + '_gs.npy')
                    K = sp.sparse.load_npz(str(base) + '_stiff.npz')
                    if closed:
                        gsref, Kref = exact_ring_reference(G, SEQ[:21])
                        tot = compare(K, Kref, 21, overlap)[0]
                        rep.check('cgnaplus CLI, closed 21 bp gives the ring', gs.shape == (21, 6) and tot <= TOL_EXACT,
                                  f'ground state {gs.shape}, error {tot:.1e}')
                    else:
                        rep.check('cgnaplus CLI, open 21 bp', gs.shape == (20, 6) and K.shape == (120, 120),
                                  f'ground state {gs.shape}')
                except Exception as e:
                    rep.check(f'cgnaplus CLI, {"closed" if closed else "open"} 21 bp', False,
                              err(e) + ' | ' + (r.stderr.strip().splitlines() or [''])[-1][:80])

    print(f'\n{"ALL CHECKS PASSED" if rep.failed == 0 else f"{rep.failed} CHECK(S) FAILED"}  ({time.time() - t0:.0f} s)')
    return rep.failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='PolyCG checkout (directory containing the polycg package)')
    sub = ap.add_subparsers(dest='phase', required=True)
    b = sub.add_parser('baseline', help='store results for rings >= 180 bp (run with a checkout from before the fix)')
    b.add_argument('--out', required=True)
    c = sub.add_parser('check', help='run all checks')
    c.add_argument('--baseline', default=None, help='baseline file from the "baseline" phase (enables check 5)')
    c.add_argument('--quick', action='store_true', help='fewer ring lengths (a few minutes)')
    c.add_argument('--cli', action='store_true', help='also run the polymc_idb and cgnaplus command-line tools')
    args = ap.parse_args()
    root, G, P, C = load_polycg(args.root)
    print(f'testing polycg from {root}')
    if args.phase == 'baseline':
        save_baseline(G, args.out)
        return 0
    return 1 if check(root, G, P, C, args) else 0


if __name__ == '__main__':
    sys.exit(main())
