from __future__ import annotations

import numpy as np
import scipy as sp
from typing import List, Tuple
from scipy.sparse import lil_matrix, csc_matrix


# from scipy.sparse import csc_matrix, csr_matrix, spmatrix, coo_matrix, bsr_matrix, lil_matrix

# DOES NOT SUPPORT NEGATIVE INDEXING TO AS COUNTED BACKWARDS FROM THE END
# TODO: For periodic boundary condition implement averaging in all directions.
#       Currently averaging is done only along the diagonal


class BOMat:
    """
    Dense matrix block with global index ranges, used by BlockOverlapMatrix.

    A BOMat represents a dense submatrix defined over global index ranges
    [x1, x2) x [y1, y2). It stores the numerical values for that block and
    provides methods to extract its contribution into a target matrix region,
    optionally accounting for periodic boundary conditions.

    The block itself is stored only once; periodicity is handled by generating
    shifted "images" of the block at extraction time.

    Parameters
    ----------
    mat : np.ndarray
        Dense 2D array of shape (x2 - x1, y2 - y1) containing the block data.
        Sparse matrices are accepted and converted to dense on construction.
    x1, x2 : int
        Global index range in the x-direction (half-open interval [x1, x2)).
    y1, y2 : int or None
        Global index range in the y-direction (half-open interval [y1, y2)).
        If None, y1 defaults to x1 and y2 defaults to x2 (square block).
    copy : bool
        If True, the input matrix is copied on construction. If False, the
        matrix reference is used directly.
    periodic : bool
        If True, the block is treated as periodically repeating with periods
        xrge and yrge. The block is not physically wrapped or split; instead,
        periodic images are generated during extraction.
    xrge, yrge : int
        Period lengths in x and y directions. Must be positive if periodic=True.
        Typically these are the full matrix dimensions.
    weight : int or float
        Weight used when accumulating overlapping blocks. During extraction,
        contributions from this block are multiplied by `weight`, and the
        corresponding counter array is incremented by `weight`. This enables
        weighted averaging of overlapping blocks.

    Periodic behavior
    -----------------
    When periodic=True, the block may extend beyond the base domain defined by
    the period. The block is stored with its original (x1, x2, y1, y2) indices,
    which may lie partially outside the fundamental domain.

    During extraction, the block contributes not only from its base placement
    but also from periodic images shifted by integer multiples of the period
    (±n*xrge, ±m*yrge). All images that overlap the requested extraction window
    are accumulated.

    Important assumptions
    ---------------------
    - The block extent (x2-x1, y2-y1) should not exceed the period in periodic
      mode. Larger blocks can lead to multiple overlapping images and double
      counting.
    - Index ranges follow Python slicing conventions: half-open intervals with
      x1 < x2 and y1 < y2.
    - This class does not perform bounds checking against a global matrix; it
      relies on BlockOverlapMatrix to define the global domain and extraction
      windows.

    Notes
    -----
    BOMat is a low-level helper class. It does not implement matrix algebra on
    its own; its role is to contribute local dense blocks into a larger matrix
    assembly via extraction and accumulation.
    """

    def __init__(
        self,
        mat: np.ndarray,
        x1: int,
        x2: int,
        y1: int | None = None,
        y2: int | None = None,
        copy=True,
        periodic: bool = False,
        xrge: int = 0,
        yrge: int = 0,
        weight: int = 1,
    ):
        if sp.sparse.issparse(mat):
            mat = mat.toarray()
        
        if copy:
            self.mat = np.copy(mat)
        else:
            self.mat = mat
        self.x1 = x1
        self.x2 = x2
        if y1 is None:
            self.y1 = x1
        else:
            self.y1 = y1
        if y2 is None:
            self.y2 = x2
        else:
            self.y2 = y2
            
        arr = np.asarray(self.mat)
        if arr.ndim != 2:
            raise ValueError(f"mat must be a 2D array, got ndim={arr.ndim}.")

        expected_shape = (self.x2 - self.x1, self.y2 - self.y1)
        if arr.shape != expected_shape:
            raise IndexError(
                f"mat shape {arr.shape} inconsistent with specified range "
                f"({expected_shape[0]} x {expected_shape[1]}) from "
                f"x:[{self.x1},{self.x2}) y:[{self.y1},{self.y2})."
            )
        self.mat = arr
                    
        self.overlap_mat = np.ones(self.mat.shape)

        self.periodic = periodic
        self.xrge = xrge
        self.yrge = yrge
        self.weight = weight

        if self.periodic:
            if self.xrge <= 0 or self.yrge <= 0:
                raise ValueError("Periodic BOMat requires positive xrge and yrge.")
            
            dx = self.x2 - self.x1
            dy = self.y2 - self.y1
            if dx > self.xrge or dy > self.yrge:
                raise ValueError(
                    "Periodic BOMat requires block extent <= period "
                    f"(got dx={dx}, dy={dy}, xrge={self.xrge}, yrge={self.yrge})."
                )

    def extract(
        self,
        extr_mat: np.ndarray | sp.spmatrix,
        cnt: np.ndarray | sp.spmatrix,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        use_weight: bool = True,
    ) -> tuple[np.ndarray | sp.spmatrix, np.ndarray | sp.spmatrix]:
        """
        Accumulate this block's contribution into (extr_mat, cnt) over
        the query window [x1, x2) × [y1, y2). Supports dense or sparse targets.
        """

        # Non-periodic: simple overlap test
        if not self.periodic:
            if self.x2 <= x1 or self.x1 >= x2 or self.y2 <= y1 or self.y1 >= y2:
                return extr_mat, cnt
            return self._extract(extr_mat, cnt, x1, x2, y1, y2, use_weight=use_weight)

        # Periodic: generate wrapped images
        xshifts = self._periodic_shifts(self.x1, self.x2, x1, x2, self.xrge)
        yshifts = self._periodic_shifts(self.y1, self.y2, y1, y2, self.yrge)

        for xshift in xshifts:
            for yshift in yshifts:
                extr_mat, cnt = self._extract(
                    extr_mat,
                    cnt,
                    x1 - xshift,
                    x2 - xshift,
                    y1 - yshift,
                    y2 - yshift,
                    use_weight=use_weight,
                )

        return extr_mat, cnt


    def _extract(
        self,
        extr_mat: np.ndarray | sp.spmatrix,
        cnt: np.ndarray | sp.spmatrix,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        use_weight: bool = True,
    ) -> tuple[np.ndarray | sp.spmatrix, np.ndarray | sp.spmatrix]:
        # assert (x1 <= self.x1 <= x2) or (x1 <= self.x2 <= x2), "x out of range"
        # assert (y1 <= self.y1 <= y2) or (y1 <= self.y2 <= y2), "y out of range"

        # Determine overlap region
        
        xlo = max(self.x1, x1)
        xhi = min(self.x2, x2)

        ylo = max(self.y1, y1)
        yhi = min(self.y2, y2)
        
        if xlo >= xhi or ylo >= yhi:
            return extr_mat, cnt
        
        weight = self.weight if use_weight else 1
        
        # Target slices (in extraction window coordinates)
        xs = slice(xlo - x1, xhi - x1)
        ys = slice(ylo - y1, yhi - y1)

        # Source slices (in block-local coordinates)
        bxs = slice(xlo - self.x1, xhi - self.x1)
        bys = slice(ylo - self.y1, yhi - self.y1)
        
        block_vals = self.mat[bxs, bys]

        if sp.sparse.issparse(extr_mat):
            # Sparse-safe accumulation
            extr_mat[xs, ys] = extr_mat[xs, ys] + block_vals * weight
            cnt[xs, ys] = cnt[xs, ys] + np.ones(block_vals.shape) * weight
        else:
            # Dense accumulation
            extr_mat[xs, ys] += block_vals * weight
            cnt[xs, ys] += weight

        return extr_mat, cnt
            

    def _periodic_shifts(self, x1: int, x2, b1: int, b2: int, rge: int) -> List[int]:
        sx2 = (x2 - b1) % rge + b1
        baseshift = sx2 - x2
        sx1 = x1 + baseshift
        num = (b2 - 1 - sx1) // rge
        shifts = [baseshift + (i + 1) * rge for i in range(num)]
        if sx1 < b2:
            shifts += [baseshift]
        return shifts
    

#######################################################################################
#######################################################################################
#######################################################################################



class BlockOverlapMatrix:
    """
    A sparse-like matrix representation assembled from overlapping dense blocks.

    This class stores a list of dense submatrices ("blocks") with associated
    index ranges. Querying a region returns the weighted average of all blocks
    that overlap that region. This is useful for progressively constructing
    very large matrices (e.g., stiffness matrices) from local contributions
    while blending boundary effects via overlap averaging.

    Coordinates and indexing
    ------------------------
    All indices (x1, x2, y1, y2) follow Python slice conventions:
        - ranges are half-open: [x1, x2) and [y1, y2)
        - x1 < x2 and y1 < y2 are required
        - negative indexing relative to the upper bound is not supported

    Blocks are added in global coordinates. The matrix "domain" is defined by
    (xlo, xhi) and (ylo, yhi). The matrix shape is:
        shape = (xhi - xlo, yhi - ylo)

    Overlap handling
    ----------------
    When multiple blocks contribute to the same entry (i, j), the returned value
    is the weighted average of those contributions:
        A(i,j) = sum_k w_k * block_k(i,j) / sum_k w_k
    where the sum runs over all blocks overlapping (i,j). By default each block
    has uniform weight 1 (weights can be extended).

    Periodic behavior
    -----------------
    If periodic=True, the matrix is interpreted as defined on a torus with
    periods:
        xrge = xhi - xlo
        yrge = yhi - ylo

    Important: blocks are NOT physically split or relocated when they cross a
    boundary. Instead, periodicity is implemented at access time:

      - When adding blocks, the lower bounds (x1, y1) are wrapped into the base
        domain [xlo, xhi) x [ylo, yhi) by shifting the entire block by an
        integer multiple of the period. The upper bounds (x2, y2) are shifted
        by the same amount, so the block extent is preserved. As a result, a
        stored block may still have x2 > xhi and/or y2 > yhi.

      - When extracting (reading) a matrix region, each stored block contributes
        not only from its base placement but also from periodic "images" shifted
        by ±n*xrge and ±m*yrge (integers n, m) whenever those images overlap the
        requested region. This is what connects the "front" and "back" of the
        matrix for ring-closure / periodic boundary conditions.

    Invariants and assumptions
    --------------------------
    - For periodic matrices, xhi > xlo and yhi > ylo must hold (positive period).
    - Intended use is local blocks (extent not exceeding the period in periodic
      mode), consistent with banded / local stiffness matrices.
    - fixed_size=True enforces that all reads/writes remain within the bounds.
      If fixed_size=False and periodic=False, bounds grow as blocks are added.

    Notes
    -----
    This object is designed for incremental assembly and averaging. It does not
    implement true matrix multiplication semantics; scalar scaling is supported.
    """

    def __init__(
        self,
        ndims: int,
        average: bool = True,
        xlo: int = None,
        xhi: int = None,
        ylo: int = None,
        yhi: int = None,
        periodic: bool = False,
        fixed_size: bool = False,
        check_bounds: bool = True,
        check_bounds_on_read: bool = True
    ):
        self.ndims = ndims
        self.average = average
        self.matblocks = list()

        if None in [xlo, xhi, ylo, yhi]:
            if fixed_size:
                raise ValueError(
                    "For fixed size matrix all bounds need to be specified!"
                )
            if periodic:
                raise ValueError("For periodic matrix all bounds need to be specified!")

        self.fixed_size = fixed_size
        self.periodic = periodic
        if periodic:
            self.fixed_size = False
        self.check_bounds = check_bounds
        self.check_bounds_on_read = check_bounds_on_read

        def set_val(x):
            if x is None:
                return 0
            else:
                return x

        self.xlo = set_val(xlo)
        self.xhi = set_val(xhi)
        self.ylo = set_val(ylo)
        self.yhi = set_val(yhi)
        
        if self.periodic or self.fixed_size:
            if self.xhi <= self.xlo:
                raise ValueError(f"xhi ({self.xhi}) must be > xlo ({self.xlo})")
            if self.yhi <= self.ylo:
                raise ValueError(f"yhi ({self.yhi}) must be > ylo ({self.ylo})")

        # self.xrge = self.xhi - self.xlo
        # self.yrge = self.yhi - self.ylo
        # self.shape = (self.xrge, self.yrge)

    ###################################################################################

    def _convert_bounds(
        self, x1: int, x2: int, y1: int | None, y2: int | None
    ) -> Tuple[int, int, int, int]:
        if y1 is None:
            y1 = x1
        if y2 is None:
            y2 = x2
        if self.periodic:
            shift_x1 = (x1 - self.xlo) % self.xrge + self.xlo
            dx1 = shift_x1 - x1
            x1 = shift_x1
            x2 = x2 + dx1

            shift_y1 = (y1 - self.ylo) % self.yrge + self.ylo
            dy1 = shift_y1 - y1
            y1 = shift_y1
            y2 = y2 + dy1
        return x1, x2, y1, y2

    ###################################################################################

    def _valid_arg_order(self, x1: int, x2: int, y1: int, y2: int) -> None:
        if x1 >= x2:
            raise ValueError(
                f"lower bound x1 ({x1}) needs to be strictly smaller than upper bound x2 ({x2}). Negative indexing relative to \
                             upper array bound is not supported!"
            )
        if y1 >= y2:
            raise ValueError(
                f"lower bound y1 ({y1}) needs to be strictly smaller than upper bound y2 ({y2}). Negative indexing relative to \
                             upper array bound is not supported!"
            )

    def _check_bounds(self, x1: int, x2: int, y1: int, y2: int) -> None:
        if self.fixed_size and self.check_bounds:
            if x1 < self.xlo:
                raise ValueError(f"x1 ({x1}) is out of bounds with xlo={self.xlo}.")
            if x2 > self.xhi:
                raise ValueError(f"x2 ({x2}) is out of bounds with xhi={self.xhi}.")
            if y1 < self.ylo:
                raise ValueError(f"y1 ({y1}) is out of bounds with ylo={self.ylo}.")
            if y2 > self.yhi:
                raise ValueError(f"y2 ({y2}) is out of bounds with yhi={self.yhi}.")

    ###################################################################################

    def _update_bounds(self, x1: int, x2: int, y1: int, y2: int) -> None:
        if self.periodic or self.fixed_size:
            return
        # update bounds
        if x1 < self.xlo:
            self.xlo = x1
        if x2 > self.xhi:
            self.xhi = x2
        if y1 < self.ylo:
            self.ylo = y1
        if y2 > self.yhi:
            self.yhi = y2
        # self.xrge = self.xhi - self.xlo
        # self.yrge = self.yhi - self.ylo
        # self.shape = (self.xrge, self.yrge)

    ###################################################################################

    def _slice2ids(
        self, ids: Tuple[slice]
    ) -> Tuple[int, int, int, int]:
        x1 = ids[0].start
        x2 = ids[0].stop
        y1 = ids[1].start
        y2 = ids[1].stop

        if x1 == None:
            x1 = self.xlo
        if x2 == None:
            x2 = self.xhi
        if y1 == None:
            y1 = self.ylo
        if y2 == None:
            y2 = self.yhi

        x1, x2, y1, y2 = self._convert_bounds(x1, x2, y1, y2)
        self._valid_arg_order(x1,x2,y1,y2)
        # self._check_bounds(x1, x2, y1, y2)
        return x1, x2, y1, y2

    ###################################################################################

    def __len__(self) -> int:
        """
        Return the size of the matrix along the x-dimension.

        This corresponds to the extent `xhi - xlo` of the underlying domain.
        """
        return self.xhi - self.xlo

    def __contains__(self, elem: BOMat) -> bool:
        return elem in self.matblocks
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix shape as (nx, ny)."""
        return (self.xhi - self.xlo, self.yhi - self.ylo)
    
    @property
    def xrge(self) -> int:
        """Return the x-extent of the matrix domain (xhi - xlo)."""
        return self.xhi - self.xlo

    @property
    def yrge(self) -> int:
        """Return the y-extent of the matrix domain (yhi - ylo)."""
        return self.yhi - self.ylo

    ###################################################################################

    def _new_block(
        self,
        mat: np.ndarray,
        x1: int,
        x2: int,
        y1: int = None,
        y2: int = None,
    ) -> BOMat:

        new_block = BOMat(
            mat,
            x1,
            x2,
            y1,
            y2,
            periodic=self.periodic,
            xrge=self.xrge,
            yrge=self.yrge,
        )
        self.matblocks.append(new_block)
        return new_block

    ###################################################################################

    def __setitem__(self, ids: Tuple[slice, slice], mat: np.ndarray | float | int) -> None:
        """
        Assign a dense block (or scalar-filled block) into the matrix over a slice window.

        This creates a new contributing block over the region specified by `ids`
        (two slice objects). The new block is appended to `self.matblocks`, and
        then any *existing* blocks are updated so that, in the regions covered by
        the new block, their stored values are overwritten by the new block's values.

        This means `__setitem__` behaves more like "write/overwrite" semantics
        (on already-stored blocks), unlike `add_block` which only adds another
        contributor and relies on averaging at read-time.

        Parameters
        ----------
        ids : tuple of slice
            Tuple `(x_slice, y_slice)` defining the half-open assignment window
            `[x1:x2, y1:y2)`. Only slice-based indexing is supported.
        mat : numpy.ndarray or float or int
            If an array, it must have shape `(x2-x1, y2-y1)` and provides the values
            for the new block. If a scalar, a constant block of that value is created
            with the appropriate shape.

        Raises
        ------
        ValueError
            If `ids` is not a tuple of two slices, or if `mat` is neither a NumPy array
            nor convertible to a float, or if the slice bounds are invalid.
        ValueError
            If bounds checking is enabled and the assignment region is out of bounds.
        IndexError
            If `mat` is an array whose shape does not match the specified slice region
            (raised indirectly by BOMat construction).
        """
        
        if not isinstance(ids, tuple):
            raise ValueError(
                f"Expected tuple of two slices, but received argument of type {type(ids)}."
            )
        for sl in ids:
            if not isinstance(sl, slice):
                raise ValueError(f"Expected slice but encountered {type(sl)}.")

        x1, x2, y1, y2 = self._slice2ids(ids)
        self._check_bounds(x1, x2, y1, y2)

        # if mat is scalar generate unform matrix of that scalar
        if not isinstance(mat, np.ndarray):
            try:
                val = float(mat)
            except:
                raise ValueError("mat should be a scalar or numpy ndarray")
            mat = np.ones((x2 - x1, y2 - y1)) * val

        new_block = self._new_block(mat, x1, x2, y1=y1, y2=y2)
        self._update_bounds(x1, x2, y1, y2)

        # set values in existing blocks
        for block in self.matblocks:
            if block == new_block:
                continue

            # skip blocks that cannot overlap new_block at all
            if not self.periodic:
                if block.x2 <= x1 or block.x1 >= x2 or block.y2 <= y1 or block.y1 >= y2:
                    continue

            extr_mat = np.zeros((block.x2 - block.x1, block.y2 - block.y1))
            extr_cnt = np.zeros(extr_mat.shape)
            extr_mat, extr_cnt = new_block.extract(
                extr_mat,
                extr_cnt,
                block.x1,
                block.x2,
                block.y1,
                block.y2,
                use_weight=False,
            )
            extr_cnt = np.clip(extr_cnt, 0, 1)
            block.mat = block.mat * (1 - extr_cnt) + extr_mat

    ###################################################################################

    def add_block(
        self, mat: np.ndarray, x1: int, x2: int, y1: int = None, y2: int = None
    ) -> None:
        """
        Add a matrix block contributing to the specified index range.

        The block is appended as an additional contributor. In regions where blocks
        overlap, values are combined via averaging during extraction. For periodic
        matrices, block indices are wrapped to the base domain.

        Parameters
        ----------
        mat : numpy.ndarray
            2D array containing the block values.
        x1, x2 : int
            Half-open index range `[x1:x2)` along the x-dimension.
        y1, y2 : int, optional
            Half-open index range `[y1:y2)` along the y-dimension. If omitted, the
            y-range defaults to the x-range.
        """
        x1, x2, y1, y2 = self._convert_bounds(x1, x2, y1, y2)
        self._valid_arg_order(x1,x2,y1,y2)
        self._check_bounds(x1, x2, y1, y2)
        self._new_block(mat, x1, x2, y1=y1, y2=y2)
        self._update_bounds(x1, x2, y1, y2)

    ###################################################################################

    def __getitem__(self, ids: Tuple[slice, slice]) -> np.ndarray:
        """
        Extract a dense submatrix defined by two slice objects.

        The result is assembled from all contributing blocks. In regions where
        multiple blocks overlap, values are averaged. For periodic matrices,
        contributions are wrapped across the domain boundaries.

        Only slice-based indexing is supported; negative indexing is not allowed.

        Parameters
        ----------
        ids : tuple of slice
            Tuple `(x_slice, y_slice)` defining the half-open ranges
            `[x1:x2, y1:y2)`.

        Returns
        -------
        numpy.ndarray
            Dense array containing the assembled submatrix.
        """
        x1, x2, y1, y2 = self._slice2ids(ids)
        if self.check_bounds_on_read:
            self._check_bounds(x1, x2, y1, y2)
        
        mat = np.zeros((x2 - x1, y2 - y1))
        cnt = np.zeros(mat.shape)
        for block in self.matblocks:
            mat, cnt = block.extract(mat, cnt, x1, x2, y1, y2)
        cnt[cnt == 0] = 1
        return mat / cnt

    def to_array(self):
        """
        Return the full matrix as a dense NumPy array.

        This is equivalent to extracting the full domain via
        `self[self.xlo:self.xhi, self.ylo:self.yhi]`, including averaging in overlap
        regions and periodic wrapping if enabled.
        """
        return self[self.xlo : self.xhi, self.ylo : self.yhi]
    
    def __mul__(self, B):
        if not np.isscalar(B):
            raise ValueError(f'Matrix multiplication is currently not supported for instances of BlockOverlapMatrix.')
        for block in self.matblocks:
            block.mat *= B
        return self
    
    def __rmul__(self, B):
        if not np.isscalar(B):
            raise ValueError(f'Matrix multiplication is currently not supported for instances of BlockOverlapMatrix.')
        for block in self.matblocks:
            block.mat *= B
        return self
    
    
    def to_sparse(self):
        """
        Assemble the full matrix into a sparse representation.

        This method has the same semantics as extracting the full matrix via
        `self[self.xlo:self.xhi, self.ylo:self.yhi]`, but stores the result as a
        sparse matrix. Block contributions are accumulated and averaged in overlap
        regions.

        If the matrix is periodic, blocks that extend beyond the base domain are
        wrapped via periodic images using each block’s extraction logic.
        """
        nx, ny = self.shape
        # Accumulate numerator and counts
        S = lil_matrix((nx, ny))
        C = lil_matrix((nx, ny))
        for block in self.matblocks:
            S, C = block.extract(
                S, C,
                self.xlo, self.xhi,
                self.ylo, self.yhi,
                use_weight=True
            )
        # Convert to CSC for efficient arithmetic
        S = S.tocsc()
        C = C.tocsc()

        # Elementwise division: only where C > 0
        if C.nnz == 0:
            return S 
        C.data = 1.0 / C.data
        S = S.multiply(C)
        return S
    
    
    def to_periodic(
        self,
        xlo: int | None = None,
        xhi: int | None = None,
        ylo: int | None = None,
        yhi: int | None = None,
    ) -> "BlockOverlapMatrix":
        """
        Return a new BlockOverlapMatrix with periodic=True, containing the same blocks.

        This is the safe way to switch from open (non-periodic) to closed (periodic)
        behavior: create a fresh periodic matrix with well-defined bounds and re-add
        all BOMat blocks so periodic invariants (xrge/yrge, wrapping, etc.) are
        consistent.

        Parameters
        ----------
        xlo, xhi, ylo, yhi:
            Bounds of the periodic fundamental domain. If not provided, the current
            object's bounds are used.

        Notes
        -----
        - Blocks are re-added through `add_block`, so periodic wrapping/conversion is
        handled by the new object.
        - This preserves block contents and ranges. If your BOMat uses per-block
        weights, ensure BlockOverlapMatrix.add_block/_new_block passes `weight`
        through to BOMat (currently it likely does not).
        """
        nxlo = self.xlo if xlo is None else xlo
        nxhi = self.xhi if xhi is None else xhi
        nylo = self.ylo if ylo is None else ylo
        nyhi = self.yhi if yhi is None else yhi

        if nxhi <= nxlo or nyhi <= nylo:
            raise ValueError(
                "Periodic bounds must satisfy xhi > xlo and yhi > ylo. "
                f"Got x:[{nxlo},{nxhi}) y:[{nylo},{nyhi})."
            )

        new = BlockOverlapMatrix(
            self.ndims,
            average=self.average,
            xlo=nxlo, xhi=nxhi,
            ylo=nylo, yhi=nyhi,
            periodic=True,
            fixed_size=False,
            check_bounds=self.check_bounds,
            check_bounds_on_read=self.check_bounds_on_read,
        )

        for block in self.matblocks:
            new.add_block(
                block.mat,
                block.x1, block.x2,
                y1=block.y1, y2=block.y2,
            )

        return new


    
    ###################################################################################
    
    # def invert(self) -> BlockOverlapMatrix:
    #     invbmat = BlockOverlapMatrix(
    #         self.ndims,
    #         average=self.average,
    #         xlo=self.xlo,xhi=self.xhi,
    #         ylo=self.ylo,yhi=self.yhi,
    #         periodic=self.periodic,
    #         fixed_size=self.fixed_size,
    #         check_bounds=self.check_bounds,
    #         check_bounds_on_read=self.check_bounds_on_read
    #     )
    #     for i,block in enumerate(self.matblocks):
    #         print(f'Inverting block {i+1}/{len(self.matblocks)}')
    #         invblock = np.linalg.inv(block.mat)
    #         invbmat.add_block(invblock,block.x1,block.x2,block.y1,block.y2)
    #     return invbmat
    