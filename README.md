# PolyCG

Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA.

## Overview

**PolyCG** is a Python package for coarse-graining rigid base pair (RBP) models of DNA to arbitrary resolutions while preserving sequence-dependent structural and elastic properties. This implementation is based on the analytical framework developed in:

> E. Skoruppa and H. Schiessel. *Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA*. [Phys. Rev. Research **7**, 013044 (2025)](https://doi.org/10.1103/PhysRevResearch.7.013044)

Traditional coarse-grained DNA models, such as the wormlike chain, rely on homogeneous, sequence-independent approaches. However, DNA sequence significantly impacts structure and mechanics even at kilobase scales. PolyCG enables efficient sampling of large DNA molecules by systematically reducing the resolution of sequence-dependent base pair step models ([cgNA+](#ref-cgnaplus), [MD](#ref-lankas), [Crystal](#ref-olson)) while maintaining essential structural and dynamic features.

The coarse-graining procedure retains every k-th base pair reference frame while computing effective ground states and stiffness matrices that faithfully reproduce the relative fluctuations of the original system. For rotational and translational degrees of freedom combined, excellent results are achieved up to approximately one helical repeat (~10 bp). When considering only rotational degrees of freedom (appropriate when local stretch modulus is not critical), faithful representation extends to several helical repeats.

## Installation

### Requirements
- Python 3.9 or higher
- NumPy
- SciPy
- Numba (for JIT compilation)

### Download

Clone the repository with recursive submodules:

```bash
git clone --recurse-submodules -j4 git@github.com:eskoruppa/PolyCG.git
```

The recursive clone is necessary to include the cgNA+ parameter library and other dependencies.

## Quick Start

Generate coarse-grained parameters for a random 101 bp DNA sequence with 10 bp composite size:

```python
import numpy as np
import polycg

# Generate random sequence
nbp = 101
seq = ''.join(['ATCG'[np.random.randint(0,4)] for _ in range(nbp)])

# Get base pair step parameters from cgNA+ model
shape, stiff = polycg.cgnaplus_bps_params(
    seq,
    translations_in_nm=True, 
    euler_definition=True, 
    group_split=True,
    parameter_set_name='curves_plus',
    remove_factor_five=True
)

# Coarse-grain to 10 bp resolution
composite_size = 10
cg_shape, cg_stiff = polycg.coarse_grain(shape, stiff, composite_size)

print(f"Original: {len(seq)-1} base pair steps")
print(f"Coarse-grained: {len(cg_shape)} composite steps")
print(f"Ground state shape: {cg_shape.shape}")
print(f"Stiffness matrix: {cg_stiff.shape}")
```

**See also:**
- [cg.ipynb](cg.ipynb) - Complete example demonstrating the coarse-graining workflow
- [transforms.ipynb](transforms.ipynb) - Guide for transforming parameters (shape and stiffness) to the correct format required for coarse-graining

## Usage

### Python API: `coarse_grain()`

The core coarse-graining function transforms base pair step parameters to lower resolution:

```python
import polycg

cg_shape, cg_stiff = polycg.coarse_grain(
    shape,           # Ground state configuration (N×6 array)
    stiff,           # Stiffness matrix (6N×6N array or sparse)
    composite_size,  # Number of base pairs per coarse-grained bead
    start_id=0,      # Starting base pair index (optional)
    end_id=None,     # Ending base pair index (optional)
    allow_partial=True,   # Allow partial block assembly (optional)
    allow_crop=True       # Allow sequence cropping (optional)
)
```

**Parameters:**
- `shape`: Ground state configuration as (N, 6) array where N is the number of base pair steps
- `stiff`: Stiffness matrix as (6N, 6N) array (can be sparse)
- `composite_size`: Coarse-graining factor (k-fold reduction in resolution)
- `start_id` (optional): Starting base pair index for selecting a subsequence. This may be used to shift the position of the first triad. Default: 0
- `end_id` (optional): Ending base pair index for selecting a subsequence. If `None`, uses full sequence. Default: None
- `allow_partial` (optional): If `True`, enables coarse-graining over overlapping blocks to speed up computation. Coarse-graining requires inverting the entire stiffness matrix, which becomes computationally challenging for large sequences. This method divides the matrix into more manageable sub-matrices with overlaps, ensuring results remain very close to the exact full coarse-graining (typically <0.1% difference per entry, often much smaller) as long as couplings in the original system do not span too far (calibrated for ~4th-order couplings). Default: True
- `allow_crop` (optional): If `True`, automatically crops the sequence to fit `composite_size` evenly (removes trailing base pairs). If `False` and length is not divisible, raises an error. Default: True

**Returns:**
- `cg_shape`: Coarse-grained ground state as (N/k, 6) array
- `cg_stiff`: Coarse-grained stiffness matrix as (6N/k, 6N/k) array

### Command-Line Interface: `gen_params`

Generate parameters directly from sequence files or strings:

```bash
# Basic generation (cgNA+ model by default)
python -m polycg.gen_params -seqfn Examples/1kbp

# Coarse-grained parameters (5 bp per bead)
python -m polycg.gen_params -seqfn Examples/200bp -cg 5

# Closed (circular) DNA
python -m polycg.gen_params -seqfn Examples/40bp -cg 10 -closed

# With ChimeraX visualization
python -m polycg.gen_params -seqfn Examples/40bp -cg 5 -pdb -vis -bpst

# Using MD parameters
python -m polycg.gen_params -seqfn Examples/1kbp -m md

# Using Crystal parameters
python -m polycg.gen_params -seqfn Examples/1kbp -m crystall

# Direct sequence input
python -m polycg.gen_params -seq ATCGATCG -cg 1
```

View all options:
```bash
python -m polycg.gen_params --help
```

## Available Models

PolyCG supports three base pair step stiffness libraries:

- **cgNA+** <a name="ref-cgnaplus"></a> (default): Most comprehensive model based on all-atom simulations, including intra-base pair coordinates. Parameters from Sharma et al. (2023). Recommended for most applications.

- **MD** <a name="ref-lankas"></a>: Parameters derived from molecular dynamics simulations by Lankaš et al. (2003). Provides sequence-dependent elastic properties at base pair step resolution.

- **Crystal** <a name="ref-olson"></a>: Parameters based on crystallographic data from Olson et al. (1998). Represents average structural properties from X-ray structures.

## Output Files

The `gen_params` command generates the following files:

- **`*_gs.npy`**: Ground state configuration (NumPy array, shape N×6)
- **`*_stiff.npz`**: Stiffness matrix (SciPy sparse matrix format, shape 6N×6N)
- **`*_cg{k}_gs.npy`**: Coarse-grained ground state (for composite_size k > 1)
- **`*_cg{k}_stiff.npz`**: Coarse-grained stiffness matrix (for composite_size k > 1)
- **`.seq`**: Sequence file (plain text)
- **`.pdb`**: PDB structure file (if `-pdb` flag used)
- **`.cxc`**: ChimeraX visualization script (if `-vis` flag used)
- **`.xyz`**: XYZ coordinate file (if `-xyz` flag used)

## Parameter Format Requirements

PolyCG requires parameters in a specific format for coarse-graining. Input parameters must satisfy two critical conditions:

1. **Rotations as Euler vectors**: Rotational coordinates must be represented as Euler vectors (rotation vectors), not Euler angles, Cayley vectors, quaternions, or rotation matrices.

2. **SE(3) group split**: Ground state and fluctuations must be split at the SE(3) group level through multiplication (g = s·d), not through additive decomposition in coordinate space (X ≠ X₀ + Xₛ). This means the ground state transformation and dynamic fluctuations are composed as rigid body transformations before extracting tangent space coordinates.

For detailed mathematical definitions, see Section II of [Skoruppa & Schiessel (2025)](https://doi.org/10.1103/PhysRevResearch.7.013044).

### Generating Parameters with Correct Format

The built-in parameter generation functions automatically provide parameters in the required format:

**cgNA+ parameters:**
```python
import polycg

seq = "ATCGATCG"
shape, stiff = polycg.cgnaplus_bps_params(
    seq,
    translations_in_nm=True,      # Use nanometers for translations
    euler_definition=True,         # Use Euler vectors (required)
    group_split=True,              # Split at SE(3) level (required)
    parameter_set_name='curves_plus',
    remove_factor_five=True
)
```

**MD parameters:**
```python
import polycg

seq = "ATCGATCG"
genstiff = polycg.GenStiffness(method='md')
shape, stiff = genstiff.gen_params(seq, use_group=True, sparse=True)
```

**Crystal parameters:**
```python
import polycg

seq = "ATCGATCG"
genstiff = polycg.GenStiffness(method='crystall')
shape, stiff = genstiff.gen_params(seq, use_group=True, sparse=True)
```

**Important:** The `use_group=True` flag ensures parameters are generated with the SE(3) group split required for coarse-graining.

### Units
- **Translations**: nanometers (nm) when `translations_in_nm=True`
- **Rotations**: radians
- **Stiffness**: Energies are expressed in units of $k_BT$, thus stiffness matrices have units of $k_BT/\text{nm}^2$ for translations, $k_BT/\text{rad}^2$ for rotations, $k_BT/(\text{rad}\cdot\text{nm})$ for cross-terms

## Limitations

Based on benchmark results from the paper:

- **Full 6D coarse-graining** (rotations + translations): Excellent accuracy up to ~10 bp (one helical repeat)
- **Rotational-only coarse-graining**: Faithful representation up to several helical repeats (30+ bp)
- **Translational fluctuations**: Less accurately reproduced than rotations, particularly for rise (extension) due to emergent asymmetry from bending fluctuations
- **Variance accuracy**: Rotational degrees of freedom show <2% deviation for up to 40-fold coarse-graining

For applications where local stretch modulus is critical, limit coarse-graining to ~10 bp. For phenomena dominated by bending and twisting (e.g., supercoiling, plectonemes), larger composite sizes are appropriate.

## References

1. <a name="ref-skoruppa"></a>**E. Skoruppa and H. Schiessel**, *Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA*, Physical Review Research **7**, 013044 (2025). [DOI: 10.1103/PhysRevResearch.7.013044](https://doi.org/10.1103/PhysRevResearch.7.013044)

2. <a name="ref-cgnaplus"></a>**R. Sharma, J. H. Maddocks, and others**, *cgDNA+: A sequence-dependent coarse-grain model of double-stranded DNA*, (2023).

3. <a name="ref-lankas"></a>**F. Lankaš, J. Šponer, J. Langowski, and T. E. Cheatham III**, *DNA basepair step deformability inferred from molecular dynamics simulations*, Biophysical Journal **85**, 2872 (2003). [DOI: 10.1016/S0006-3495(03)74710-9](https://doi.org/10.1016/S0006-3495(03)74710-9)

4. <a name="ref-olson"></a>**W. K. Olson, A. A. Gorin, X.-J. Lu, L. M. Hock, and V. B. Zhurkin**, *DNA sequence-dependent deformability deduced from protein-DNA crystal complexes*, Proceedings of the National Academy of Sciences **95**, 11163 (1998). [DOI: 10.1073/pnas.95.19.11163](https://doi.org/10.1073/pnas.95.19.11163)

## License

This project is licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use PolyCG in your research, please cite:

```bibtex
@article{skor2025_cg,
  title={Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded {DNA}},
  author={Skoruppa, Enrico and Schiessel, Helmut},
  journal={Phys. Rev. Res.},
  volume = {7},
  issue = {1},
  pages = {013044},
  numpages = {23},
  year = {2025},
  month = {Jan},
  publisher={American Physical Society},
  doi = {10.1103/PhysRevResearch.7.013044},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.7.013044}
}
```

## Contact

- **Author**: [Enrico Skoruppa](https://github.com/eskoruppa)
- **Repository**: [https://github.com/eskoruppa/PolyCG](https://github.com/eskoruppa/PolyCG)