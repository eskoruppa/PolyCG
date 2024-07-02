# PolyCG
Module to coarse-grain elastic parameters of DNA models 

----
----
# Download <a name=download></a>

------
Requires recursive cloning of submodules
```console
git clone --recurse-submodules -j4 git@github.com:eskoruppa/PolyCG.git
```


### <span style="color:red">Documentation coming soon</span>

----
----
# Basic Functionality <a name=functionality></a>

---- 
### Coarse-Grain 


```python
import polycg


cg_gs, cg_stiff = polycg.coarse_grain(
    gs,
    stiff,
    composite_size,
    start_id=start_id,
    end_id=end_id,
    allow_partial=allow_partial,
    block_ncomp=block_ncomp,
    overlap_ncomp=overlap_ncomp,
    tail_ncomp=tail_ncomp
)
```