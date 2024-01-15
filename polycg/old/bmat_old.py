from __future__ import annotations
import numpy as np
import scipy as sp
import sys
from typing import List, Tuple, Callable, Any, Dict

# DOES NOT SUPPORT NEGATIVE INDEXING TO AS COUNTED BACKWARDS FROM THE END
# TODO: For non-periodic matrices, the behavior should be equivalent to standard lib indexing

class BlockOverlapMatrix:
    ndims: int
    average: bool = True
    matblocks: List[BOMat]
    ranges: List[List[int]]

    xlo: int
    xhi: int
    ylo: int
    yhi: int
    
    periodic: bool=False
    fixed_size: bool=False

    xrge: int
    yrge: int
    shape: tuple[int, int]

    def __init__(
        self,
        ndims: int,
        average: bool = True,
        xlo: int = None,
        xhi: int = None,
        ylo: int = None,
        yhi: int = None,
                
        periodic=False,
        fixed_size=False
    ):
        
        self.ndims = ndims
        self.average = average
        self.matblocks = list()

        if fixed_size:
            if xlo is None or xhi is None or ylo is None or yhi is None:
                raise ValueError("For fixed size matrix all bounds need to be specified!")
        if periodic:
            fixed_size=True
            if xlo is None or xhi is None or ylo is None or yhi is None:
                raise ValueError("For periodic matrix all bounds need to be specified!")
        self.fixed_size=fixed_size
        self.periodic=periodic

        def set_val(x):
            if x is None:
                return 0
            else:
                return x

        self.xlo = set_val(xlo)
        self.xhi = set_val(xhi)
        self.ylo = set_val(ylo)
        self.yhi = set_val(yhi)
        
        self.xrge = self.xhi-self.xlo
        self.yrge = self.yhi-self.ylo
        self.shape = (self.xrge, self.yrge)


    def __setitem__(
        self, ids: Tuple[slice] | int, mat: np.ndarray | float | int
    ) -> None:
        if not isinstance(ids, tuple):
            raise ValueError(
                f"Expected tuple of two slices, but received argument of type {type(ids)}."
            )
        for sl in ids:
            if not isinstance(sl, slice):
                raise ValueError(f"Expected slice but encountered {type(sl)}.")

        x1,x2,y1,y2 = self._slice2ids(ids)
        x1,x2,y1,y2 = self._process_bounds(x1,x2,y1,y2)
        
        if not isinstance(mat, np.ndarray):
            try:
                val = float(mat)
            except:
                raise ValueError('mat should be a scalar or numpy ndarray')
            mat = np.ones((x2 - x1, y2 - y1)) * val
            
        if not self.periodic:   
            self._setblock(mat,x1,x2,y1=y1,y2=y2,image=False)
            self._update_bounds(x1,x2,y1,y2)
        
        if x1 >= self.xhi:
            raise ValueError(f'x1 ({x1}) larger or equal to upper matrix lim ({self.xhi}).')
        if x2 <= self.xlo:
            raise ValueError(f'x1 ({x2}) smaller or equal to lower matrix lim ({self.xlo}).')
        if y1 >= self.yhi:
            raise ValueError(f'y1 ({y1}) larger or equal to upper matrix lim ({self.yhi}).')
        if y2 <= self.ylo:
            raise ValueError(f'y1 ({y2}) smaller or equal to lower matrix lim ({self.ylo}).')

        self._setblock(mat,x1,x2,y1=y1,y2=y2,image=False)
        if x2 > self.xhi or y2 > self.yhi:
            self._setblock(mat,x1-self.xrge,x2-self.xrge,y1-self.yrge,y2-self.yrge,image=True)
        if x1 < 0 or y1 < 0:
            self._setblock(mat,x1+self.xrge,x2+self.xrge,y1+self.yrge,y2+self.yrge,image=True)
            

    def _setblock(
        self, mat: np.ndarray, x1: int, x2: int, y1: int = None, y2: int = None, image=False
    ) -> bool:        
        new_block = BOMat(mat, x1, x2, y1, y2, image=image)
        self._update_bounds(new_block.x1, new_block.x2, new_block.y1, new_block.y2)

        add = True
        remove = []
        for i, block in enumerate(self.matblocks):
            new_block.set_overlap(block)
            if block.inclusion(new_block):
                if add:
                    self.matblocks[i] = new_block
                    add = False
                else:
                    remove.append(i)
            elif new_block.inclusion(block):
                add = False
        if add:
            self.matblocks.append(new_block)
        for rem in np.sort(remove)[::-1]:
            del self.matblocks[rem]
    
    
    def add_block(
        self, mat: np.ndarray, x1: int, x2: int, y1: int = None, y2: int = None
    ) -> bool:
        
        x1,x2,y1,y2 = self._process_bounds(x1,x2,y1,y2)
                 
        if not self.periodic:   
            self._add_block(mat,x1,x2,y1=y1,y2=y2,image=False)
            self._update_bounds(x1,x2,y1,y2)

        if x1 >= self.xhi:
            raise ValueError(f'x1 ({x1}) larger or equal to upper matrix lim ({self.xhi}).')
        if x2 <= self.xlo:
            raise ValueError(f'x1 ({x2}) smaller or equal to lower matrix lim ({self.xlo}).')
        if y1 >= self.yhi:
            raise ValueError(f'y1 ({y1}) larger or equal to upper matrix lim ({self.yhi}).')
        if y2 <= self.ylo:
            raise ValueError(f'y1 ({y2}) smaller or equal to lower matrix lim ({self.ylo}).')

        self._add_block(mat,x1,x2,y1=y1,y2=y2,image=False)
        if x2 > self.xhi or y2 > self.yhi:
            self._add_block(mat,x1-self.xrge,x2-self.xrge,y1-self.yrge,y2-self.yrge,image=True)
        if x1 < 0 or y1 < 0:
            self._add_block(mat,x1+self.xrge,x2+self.xrge,y1+self.yrge,y2+self.yrge,image=True)
                    
        
    def _add_block(
        self, mat: np.ndarray, x1: int, x2: int, y1: int = None, y2: int = None, image=False
    ) -> bool:
        new_block = BOMat(mat, x1, x2, y1, y2, image=image)
        add = True
        remove = []
        for i, block in enumerate(self.matblocks):
            if self.average:
                new_block.avg_overlap(block, avg_other=True)
            else:
                new_block.set_overlap(block)
            if block.inclusion(new_block):
                if add:
                    self.matblocks[i] = new_block
                    add = False
                else:
                    remove.append(i)
            elif new_block.inclusion(block):
                add = False
        if add:
            self.matblocks.append(new_block)
        for rem in np.sort(remove)[::-1]:
            del self.matblocks[rem]
        return add
    
    
    def _process_bounds(self, x1: int, x2: int, y1: int, y2: int):
        if y1 is None:
            y1 = x1
        else:
            y1 = y1
        if y2 is None:
            y2 = x2
        else:
            y2 = y2
            
        if x2 <= x1:
            raise ValueError(f'For now upper bound needs to be strictly larger than lower bound. Encountered x2<=x1')
        if y2 <= y1:
            raise ValueError(f'For now upper bound needs to be strictly larger than lower bound. Encountered y2<=y1')
            
        return x1,x2,y1,y2

    def _update_bounds(self, x1: int, x2: int, y1: int, y2: int) -> None:
        if self.periodic:
            return
        if x1 < self.xlo:
            self.xlo = x1
        if x2 > self.xhi:
            self.xhi = x2
        if y1 < self.ylo:
            self.ylo = y1
        if y2 > self.yhi:
            self.yhi = y2
            
        self.xrge = self.xhi-self.xlo
        self.yrge = self.yhi-self.ylo
        self.shape = (self.xrge, self.yrge)

    def __getitem__(self, ids: Tuple[slice] | int) -> float | np.ndarray:
        x1, x2, y1, y2 = self._slice2ids(ids)

        if x1 < self.xlo:
            raise ValueError(f"Lower x ({x1}) smaller than lower x bound ({self.xlo}).")
        if x2 > self.xhi:
            raise ValueError(f"Upper x ({x2}) larger than upper x bound ({self.xhi}).")

        if y1 < self.ylo:
            raise ValueError(f"Lower y ({y2}) smaller than lower y bound ({self.ylo}).")
        if y2 > self.yhi:
            raise ValueError(f"Upper y ({y2}) larger than upper y bound ({self.yhi}).")

        mat = np.zeros((x2 - x1, y2 - y1))
        for block in self.matblocks:
            pmat, xl, xh, yl, yh = block.extract(x1, x2, y1, y2)
            if pmat is None:
                continue
            mat[xl - x1 : xh - x1, yl - y1 : yh - y1] = pmat
        return mat

    def _slice2ids(self, ids: Tuple[slice],check_bounds: bool=True) -> Tuple[int, int, int, int]:
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
        
        # print(x1,x2,y1,y2)
        # TODO: CHECK IF INDICES ARE WITHIN BOUNDS      
        return x1, x2, y1, y2

    def __len__(self) -> Tuple[int, int]:
        return self.xhi - self.xlo

    def __contains__(self, elem: BOMat) -> bool:
        return elem in self.matblocks

    def to_array(self):
        return self[self.xlo : self.xhi, self.ylo : self.yhi]


class BOMat:
    mat: np.ndarray
    x1: int
    x2: int
    y1: int
    y2: int
    overlap_mat: np.ndarray
    image: bool = False

    def __init__(
        self,
        mat: np.ndarray,
        x1: int,
        x2: int,
        y1: int | None = None,
        y2: int | None = None,
        copy=True,
        image=False,
    ):
        self.image = image
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
        if len(mat) != self.x2 - self.x1:
            raise ValueError("Size of x-dimension inconsistent with specified x range")
        if len(mat[0]) != self.y2 - self.y1:
            raise ValueError("Size of y-dimension inconsistent with specified y range")
        self.overlap_mat = np.ones(self.mat.shape)

    def set_overlap(self, otherblock: BOMat):
        xrgs = self._overlap_coords(self.x1, self.x2, otherblock.x1, otherblock.x2)
        if xrgs is None:
            return
        yrgs = self._overlap_coords(self.y1, self.y2, otherblock.y1, otherblock.y2)
        if yrgs is None:
            return
        otherblock.mat[xrgs[2] : xrgs[3], yrgs[2] : yrgs[3]] = self.mat[
            xrgs[0] : xrgs[1], yrgs[0] : yrgs[1]
        ]

    def avg_overlap(self, otherblock: BOMat, avg_other: bool = True):
        xrgs = self._overlap_coords(self.x1, self.x2, otherblock.x1, otherblock.x2)
        if xrgs is None:
            return
        yrgs = self._overlap_coords(self.y1, self.y2, otherblock.y1, otherblock.y2)
        if yrgs is None:
            return
        mat = self.mat * self.overlap_mat
        self.overlap_mat[xrgs[0] : xrgs[1], yrgs[0] : yrgs[1]] += 1
        mat[xrgs[0] : xrgs[1], yrgs[0] : yrgs[1]] += otherblock.mat[
            xrgs[2] : xrgs[3], yrgs[2] : yrgs[3]
        ]
        self.mat = mat / self.overlap_mat
        if avg_other:
            otherblock.mat[xrgs[2] : xrgs[3], yrgs[2] : yrgs[3]] = self.mat[
                xrgs[0] : xrgs[1], yrgs[0] : yrgs[1]
            ]

    def _overlap_coords(self, a1: int, a2: int, b1: int, b2: int) -> Tuple[int] | None:
        if a1 > b1:
            x1 = a1
        else:
            x1 = b1
        if a2 < b2:
            x2 = a2
        else:
            x2 = b2
        if x2 <= x1:
            return None
        return x1 - a1, x2 - a1, x1 - b1, x2 - b1

    def inclusion(self, otherblock) -> bool:
        if self.x1 < otherblock.x1:
            return False
        if self.x2 > otherblock.x2:
            return False
        if self.y1 < otherblock.y1:
            return False
        if self.y2 > otherblock.y2:
            return False
        return True

    def extract(
        self, x1: int, x2: int, y1: int, y2: int
    ) -> Tuple[np.ndarray, int, int, int, int]:
        if x1 < self.x1:
            xlo = self.x1
        else:
            xlo = x1
        if x2 > self.x2:
            xhi = self.x2
        else:
            xhi = x2
        xrge = xhi - xlo
        if xrge <= 0:
            return None, 0, 0, 0, 0

        if y1 < self.y1:
            ylo = self.y1
        else:
            ylo = y1
        if y2 > self.y2:
            yhi = self.y2
        else:
            yhi = y2
        yrge = yhi - ylo
        if yrge <= 0:
            return None, 0, 0, 0, 0

        mat = self.mat[xlo - self.x1 : xhi - self.x1, ylo - self.y1 : yhi - self.y1]
        return mat, xlo, xhi, ylo, yhi


if __name__ == "__main__":
    m1 = np.ones((4, 4))
    m2 = np.ones((4, 4)) * 0
    m3 = np.ones((4, 4)) * -1

    block1 = BOMat(m1, 0, 4)
    block2 = BOMat(m2, 2, 6)

    bom = BlockOverlapMatrix(3, average=True)

    bom.add_block(m1, 0, 4)
    bom.add_block(m2, 2, 6)
    bom.add_block(m3, 4, 8)

    bom[6:10, 6:10] = np.ones((4,) * 2) * 2

    # bom[2:8,2:8] = 5
    # bom[0:10,0:10] = -5

    for bmat in bom.matblocks:
        print(bmat.mat)

    print(bom[0:10, 0:12])

    print(bom.to_array())
