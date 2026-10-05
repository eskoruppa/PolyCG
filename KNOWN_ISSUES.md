# Known issues

## 1. `gen_params` and `polymc_idb.py` duplicate the parameter pipeline, and `gen_params` coarse-grains less accurately

**Status:** open. Documented and measured on 2026-10-05 at commit 43f1059.

### Summary

PolyCG has two separate pipelines that turn a sequence into (coarse-grained) elastic parameters: `gen_params` and `polymc_idb.py`. They re-implement the same steps, so every fix has to be made twice, and they give different numbers:

- With block-wise processing, `gen_params`' coarse-grained stiffness deviates from the exact result by up to 3e-3 (relative to the largest entry) at composite size 20. `polymc_idb.py` stays within 1.2e-8.
- For several composite sizes, `gen_params` is about 4× slower (80 s vs 18 s for a 5000 bp chain at composite sizes 1, 2, 5 and 10).
- The two treat leftover base pairs at the end of an open chain differently. The last composites then differ by up to a few percent.

Proposed resolution: make `gen_params` the single engine and give it a rotations-only mode that follows `polymc_idb.py`'s order of operations. `polymc_idb.py` then becomes a front-end that writes IDB files (see [Proposed resolution](#proposed-resolution)).

### The two pipelines

- **`gen_params`** (`polycg/_gen_params.py`) is the documented API and returns `DNAParameters`. It uses 6 degrees of freedom per step (rotations and translations) and supports open and closed chains, coarse-graining (`composite_size`), sequence ranges (`start_id`, `end_id`) and per-DOF rescaling.
- **`polymc_idb.py`** writes PolyMC IDB files, which only hold rotations. It integrates the translations out inside every cgNA+ block (`rotations_only=True` in `cgnaplus_bps_params`), then coarse-grains the three rotational DOF.

Both implement each of these steps separately:

| Step | `gen_params` | `polymc_idb.py` |
|---|---|---|
| Model selection: cgNA+ settings; lankas/olson via `GenStiffness` | `_build_cgnaplus_args`, `_generate_local_model_params` | settings inline in `__main__`, `_cgnaplus_chain`, `_local_chain` |
| Block-wise cgNA+ assembly (`partial_stiff`, blocks 120/20/20 bp) | `_gen_params_open`, `_gen_params_closed` | `__main__`, `_cgnaplus_chain` |
| Closed rings: unroll into an open chain, coarse-grain, fold back (`periodic_extension`, `fold_periodic`) | `_gen_params_closed` | `_closed_coarse_grain` |
| Block sizes of the coarse-graining | `_calculate_coarse_grain_params` | `cg_partition` |
| Cropping to whole composites | `coarse_grain(..., allow_crop)` | `coarse_grain(..., start_id=first_id)` |

### Both orders give the same result in exact arithmetic

A composite's rotation depends only on the rotations of its steps. Integrating the translations out before or after coarse-graining therefore gives the same Gaussian marginal.

Measured: one cgNA+ calculation over the whole chain, coarse-grained in one piece, with 3 DOF and with 6 DOF followed by the rotational marginal. The two agree to ≤ 1.4e-12 (601 and 1201 bp, composite sizes 2–20). All differences reported below come from the block-wise approximations.

### Measurements

**Method.** Open chains of 601 and 1201 bp, with a random sequence from `np.random.default_rng(11)`. The 600 and 1200 steps divide by every composite size tested, so no steps are left over.

- **Reference:** one cgNA+ calculation over the whole chain (no blocks), coarse-grained in one piece.
- **Error:** max |K − K_ref| / max |K_ref| over couplings at most 4 composites apart. A default IDB holds exactly those couplings.
- **polymc_idb route:** `partial_stiff` with rotations only, then `coarse_grain` with the block sizes from `cg_partition(cg, 4)`.
- **gen_params route:** `gen_params('cgnaplus', seq, composite_size=cg)`, then `matrix_rotmarginal(p.cg_stiffmat)`.

| Composite size | polymc_idb, 601 bp | gen_params, 601 bp | polymc_idb, 1201 bp | gen_params, 1201 bp |
|---|---|---|---|---|
| 2 | 6.3e-10 | 6.6e-8 | 6.3e-10 | 6.6e-8 |
| 5 | 1.6e-9 | 1.1e-4 | 3.9e-9 | 1.2e-4 |
| 10 | 4.3e-9 | 7.8e-4 | 1.2e-8 | 1.0e-3 |
| 20 | 2.2e-9 | 2.0e-3 | 6.9e-9 | 2.9e-3 |

Rings (`closed=True`) use the same block sizes in `gen_params` and show the same effect. For example, a 300 bp ring at composite size 10 is off by 7.7e-4 (2.3e-6 with polymc_idb's route).

**Runtime** for a 5000 bp open chain at composite sizes 1, 2, 5 and 10 (one machine, 2026-10-05):

| | Base | cg=2 | cg=5 | cg=10 | Total |
|---|---|---|---|---|---|
| polymc_idb route | 11.1 s, computed once | 2.9 s | 2.2 s | 2.2 s | 18 s |
| gen_params route | 6.5 s, recomputed for every size | 41.9 s | 16.6 s | 15.1 s | 80 s |

The gen_params times per composite size include recomputing the base. The 3-DOF base is the slower one, because cgNA+ integrates out the translations in every block. The coarse-graining, however, is 4–12× slower in 6 DOF than in 3 DOF.

### Causes of the gen_params error

**1. Thin coarse-graining blocks.** `_calculate_coarse_grain_params` divides the bp settings by the composite size:
- blocks: ⌈120/cg⌉ composites;
- overlap: ⌈max(20, cg)/cg⌉ composites;
- tails: ⌈20/cg⌉ composites.

At composite size 20 that leaves a single composite of overlap and of tail. Couplings further apart than the overlap that cross a block boundary are never computed. In the 6-DOF matrix they are missing; after the rotational marginal they reappear with wrong values.

Coarse-graining the same 6-DOF base with `cg_partition` block sizes (601 bp) instead:

| Composite size | gen_params blocks (block/overlap/tail) | Error | `cg_partition` blocks | Error |
|---|---|---|---|---|
| 5 | 24/4/4 | 1.1e-4 | 32/4/8 | 6.6e-5 |
| 10 | 12/2/2 | 7.8e-4 | 16/4/4 | 1.8e-5 |
| 20 | 6/1/1 | 2.0e-3 | 8/4/4 | 1.2e-5 |

**2. A 20 bp base-level overlap.** When the cgNA+ blocks are assembled, couplings more than 20 steps apart are cut.
- **Rotations alone are barely affected.** The assembled 6-DOF base has a rotational marginal within 8e-9 of exact, the 3-DOF base within 7e-10.
- **The 6-DOF route is affected.** Translation couplings are long-ranged, and coarse-graining in 6 DOF sums over them.

The same 6-DOF base, coarse-grained in one piece (601 bp):

| Base blocks (block/overlap/tail, bp) | cg=5 | cg=20 |
|---|---|---|
| 120/20/20 (current) | 1.7e-5 | 1.2e-5 |
| 120/40/20 | 5.0e-9 | 2.3e-9 |
| 120/40/40 | 1.2e-11 | 2.3e-9 |
| 160/60/40 | 1.5e-14 | 6.9e-13 |

The 3-DOF route is accurate with 20 bp because it integrates the translations out inside each block, where the cgNA+ model is complete.

### Leftover base pairs at the end of an open chain

When the number of steps is not a multiple of the composite size, both tools drop the leftover steps, but in different ways:
- **polymc_idb.py** removes their rows and columns from the 3-DOF matrix, after the translations have been integrated out.
- **gen_params** removes them from the 6-DOF matrix.

Either way the leftover steps are held fixed rather than integrated out. The last one or two composites then differ between the tools. The difference decays to 1e-11–1e-15 at the first composite; without leftover steps the tools agree to 1e-14.

| Chain | Composite size | Leftover steps | Difference in the last composite |
|---|---|---|---|
| 60 bp | 5 | 4 | 2e-2 |
| 46 bp | 2 | 1 | 7e-2 |
| 200 bp | 10 | 9 | 9e-3 |
| 400 bp | 5 | 4 | 3e-2 |

Neither rule is clearly right. Integrating the leftover steps out would be the principled choice; requiring the length to be a multiple of the composite size, as for rings, is the simple one.

### Related

`coarse_grain`'s default block sizes (16/4/2 composites) have the same problem for callers that pass `allow_partial=True` without block sizes:
- errors up to ~4e-4 at composite size 2;
- couplings more than 4 composites apart are dropped where they cross a block boundary.

No caller in this repository has relied on these defaults since 43f1059.

### Proposed resolution

Make `gen_params` the single engine, add polymc_idb's order as its rotations-only mode, and turn `polymc_idb.py` into a front-end.

1. **Extend `gen_params`.**
   - Add a `rotations_only=False` argument that runs 3 DOF end to end. `partial_stiff`, `coarse_grain`, `periodic_extension` and `fold_periodic` already take `ndims`. What is hard-coded to 6 DOF:
     - `_GEN_PARAMS_NDIMS = 6` and `_GEN_PARAMS_CGNAPLUS_ROT_ONLY`;
     - the shape checks in `DNAParameters`;
     - the local models, which would need the rotational marginal as in `_local_chain`.
   - Add a `coupling_range` argument that sets the minimum overlap, in composites, of the coarse-graining blocks.
   - Replace `_calculate_coarse_grain_params` with `cg_partition`, moved from `polymc_idb.py` to `cg.py`.
   - Use a base-level overlap of at least 40 bp in the 6-DOF route.
2. **Compute the base once for several composite sizes.** Accept a list of sizes, or add a separate coarse-graining step. For rings, unroll the chain once with the largest margin needed.
3. **Choose one rule for leftover base pairs** (see above).
4. **Slim down `polymc_idb.py`.**
   - For each sequence and composite size, call `gen_params(..., rotations_only=True, coupling_range=cr)`.
   - Then apply the IDB units (× discretization length × composite size, degrees), `-sc`, `-ss` and `-nostatic`, and write with `stiff2idb`.
   - `_cgnaplus_chain`, `_local_chain`, `_closed_coarse_grain` and the inline cgNA+ settings go away. The command line and output names stay the same.
5. **Validate.**
   - `tests/verify_polymc_idb.py` compares every IDB file with exact references, so it remains valid across the refactor. Its byte-for-byte baseline (default 8b64f69) would move to 43f1059.
   - `gen_params` needs a matching check against exact references: the snippet below for open chains, and for rings the repeated-sequence reference used in `tests/verify_closed_partial_stiff.py`.

**Expected effect:**
- `polymc_idb.py`: output unchanged to ~1e-9. A few IDB values may flip in their last digit, and the last composites of chains with leftover steps change if the rule for them changes.
- `gen_params`: coarse-grained 6-DOF output moves toward the exact values, by up to ~3e-3 at composite size 20. Base-level values change only at the ~1e-8 level, from the wider overlap.

### Reproducing the gen_params numbers

```python
import numpy as np
from polycg import gen_params, cgnaplus_bps_params, cg_stiffmat
from polycg.transforms.transform_marginals import matrix_rotmarginal

seq = ''.join(np.random.default_rng(11).choice(list('acgt'), 601))    # 600 steps: no leftover steps for cg 2..20
args = dict(translations_in_nm=True, euler_definition=True, group_split=True,
            parameter_set_name='curves_plus', remove_factor_five=True)
gs, K = cgnaplus_bps_params(seq, **args)                       # one cgNA+ calculation, no blocks
for cg in (2, 5, 10, 20):
    exact = matrix_rotmarginal(cg_stiffmat(np.asarray(gs), K, cg)).toarray()
    p = gen_params('cgnaplus', seq, composite_size=cg)
    approx = matrix_rotmarginal(p.cg_stiffmat).toarray()
    n = exact.shape[0] // 3
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    band = np.kron(d <= 4, np.ones((3, 3), dtype=bool))       # couplings up to 4 composites apart
    print(cg, np.abs(np.where(band, approx - exact, 0)).max() / np.abs(exact).max())
```

Output, rounded: `2 6.6e-08`, `5 1.1e-04`, `10 7.8e-04`, `20 2.0e-03`.
