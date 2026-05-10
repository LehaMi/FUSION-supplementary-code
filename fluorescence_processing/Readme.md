# CaImAn Fixed-A Temporal Re-optimization (`fit_v2`)

This repository documents a CaImAn modification that adds a `fit_v2` workflow for re-optimizing temporal components `C` while keeping existing spatial components `A` fixed. The modification is intended for cases where reliable ROIs/spatial components are already available and the user wants to recompute temporal traces on the same imaging data or on another recording with the same field of view.

## 1. System Requirements

### Operating system

The modification was developed and tested on:

- Windows 10, 64-bit

The code is based on CaImAn and should also be portable to other operating systems supported by CaImAn, but only the Windows 10 environment above has been explicitly tested for this modification.

### Software dependencies

The modification is based on the CaImAn Python package and its CNMF modules. The following software environment is recommended:

- Python 3.10
- CaImAn 13.0
- NumPy
- SciPy
- scikit-image
- scikit-learn
- OpenCV Python (`opencv-python`)
- h5py
- tifffile
- matplotlib
- pandas
- psutil
- tqdm
- Jupyter Notebook or JupyterLab, optional but recommended for interactive testing

Use the dependency versions required by CaImAn 13.0 unless your project has a locked environment file. If exact package versions are required for reproduction, export the working environment after installation.

### Tested hardware

The modification has been tested on a desktop/workstation with:

- Operating system: Windows 10, 64-bit
- System memory: 256 GB RAM

### Required non-standard hardware

No special hardware is required to run the code itself. However, calcium imaging datasets can be large, and the practical memory requirement depends on the movie size, number of frames, number of components, and whether memory mapping is used.

A normal desktop computer can run a small demo dataset, but large one-photon calcium imaging movies may require substantially more RAM and disk space.

## 2. Installation Guide

### Installation instructions

1. Install Anaconda or Miniconda.

2. Create and activate a clean Python environment:

```bash
conda create -n caiman_fixedA python=3.10 -y
conda activate caiman_fixedA
```

3. Install CaImAn 13.0 following the official CaImAn installation procedure for your platform. 

4. Apply the modified files described in this README to the corresponding CaImAn source files:

- `cnmf.py`
- `initialization.py`
- `spatial.py`
- `temporal.py`

5. Verify that the environment can import CaImAn:

```bash
python -c "import caiman as cm; print(cm.__version__)"
```

6. Start Python, Jupyter Notebook, or JupyterLab and run your CNMF workflow.

### Typical installation time

On a normal desktop computer with a stable internet connection, creating the environment and installing CaImAn and its Python dependencies typically takes about 15-45 minutes. Installation may take longer if packages need to be built from source or if the network connection is slow.

## 3. Demo

### Demo purpose

The demo verifies that `fit_v2` can reuse a pre-existing spatial component matrix `A` and recompute temporal traces `C` without changing the number or structure of spatial components.

### Demo data

Use a small calcium imaging movie that can be loaded into memory, for example a short `.tif`, `.hdf5`, or CaImAn-compatible demo movie. The movie should be suitable for one-photon CNMF processing with `method_init='corr_pnr'`.

### Instructions to run the demo

```python
import numpy as np
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf.params import CNMFParams

# Replace this with your own movie-loading step.
# images should have shape: (T, height, width)
# Example:
# images = cm.load('demo_movie.tif')

params = CNMFParams(params_dict={
    'init': {
        'method_init': 'corr_pnr',
        'ring_size_factor': 1.5,
    },
    'data': {
        # Add dataset-specific options here, such as dims, fr, decay_time, etc.
    }
})

# Step 1: Run the standard CNMF workflow once to obtain A.
cnm = cnmf.CNMF(n_processes=1, params=params)
cnm.fit(images)

A_fixed = cnm.estimates.A.copy()
C_original = cnm.estimates.C.copy()

# Step 2: Run fit_v2 with A fixed.
cnm_fixed = cnmf.CNMF(n_processes=1, params=params)
cnm_fixed.estimates.A = A_fixed
cnm_fixed.fit_v2(images)

C_new = cnm_fixed.estimates.C

# Step 3: Basic checks.
print('Original A shape:', A_fixed.shape)
print('New A shape:', cnm_fixed.estimates.A.shape)
print('Original C shape:', C_original.shape)
print('New C shape:', C_new.shape)
print('Mean absolute C difference:', np.mean(np.abs(C_original - C_new)))
```

### Expected output

The exact numerical values depend on the input movie and CNMF parameters. A successful run should show:

- `fit_v2` completes without raising an error.
- The shape of `A` before and after `fit_v2` remains the same.
- The shape of `C_new` is compatible with the fixed number of spatial components.
- The temporal traces `C_new` are recomputed under the fixed-`A` constraint.

Example output format:

```text
Original A shape: (pixels, components)
New A shape: (pixels, components)
Original C shape: (components, frames)
New C shape: (components, frames)
Mean absolute C difference: <dataset-dependent value>
```

### Expected demo run time

For a small demo movie on a normal desktop computer, the demo is expected to take several minutes to tens of minutes. The runtime depends strongly on image dimensions, number of frames, number of components, preprocessing settings, and whether memory mapping or parallel processing is enabled.

On the tested Windows 10 workstation with 256 GB RAM, small demo data should run comfortably in memory. Large calcium imaging movies may take substantially longer.

## 4. Instructions for Use

### When to use `fit_v2`

Use `fit_v2` when:

- You already have reliable spatial components `A`, such as manually curated ROIs or components obtained from a previous CNMF run.
- You want to recompute temporal activity traces `C` while preserving the same ROI/spatial component structure.
- You want to compare temporal extraction results under a fixed spatial footprint.
- You are working primarily with one-photon data using `method_init='corr_pnr'`.

### How to run the software on your own data

1. Load your calcium imaging movie into a CaImAn-compatible array or memory-mapped object. The expected movie format is typically `(time, height, width)`.

2. Configure CNMF parameters. The fixed-`A` workflow is recommended with:

```python
params = CNMFParams(params_dict={
    'init': {
        'method_init': 'corr_pnr',
        'ring_size_factor': 1.5,
    }
})
```

3. Obtain or load the spatial component matrix `A`.

You can obtain `A` by first running the standard CNMF workflow:

```python
cnm = cnmf.CNMF(n_processes=1, params=params)
cnm.fit(images)
A_fixed = cnm.estimates.A.copy()
```

Alternatively, assign an externally generated sparse matrix `A` if it follows the same dimensions and format expected by CaImAn.

4. Run the fixed-`A` workflow:

```python
cnm_fixed = cnmf.CNMF(n_processes=1, params=params)
cnm_fixed.estimates.A = A_fixed
cnm_fixed.fit_v2(images)

C_new = cnm_fixed.estimates.C
```

5. Save or analyze the output temporal components:

```python
np.save('C_fixedA.npy', C_new)
```

### Important notes

- `cnm_fixed.estimates.A` must not be `None` before calling `fit_v2`.
- The spatial dimensions of `A` must match the spatial dimensions of the input movie.
- This modification is based on CaImAn 13.0.
- The current implementation is mainly intended for one-photon calcium imaging workflows with `method_init='corr_pnr'`.
- For large datasets, use CaImAn memory mapping and adjust parallel processing settings according to available RAM and CPU resources.

---

## Detailed Modification Notes

## Overview

This modification introduces the `fit_v2` method, which re-optimizes temporal components C while **keeping spatial components A fixed**. This is particularly useful for scenarios where high-quality spatial components are already available (e.g., manually annotated ROIs).

## Modified Files List

### 1. **cnmf.py** - Core Interface File

#### Modification 1.1:  `fit_v2` Method

**Functionality**:

- Similar workflow to standard `fit` method, but uses fixed-A versions
- Calls `initialize_with_A_fixed` instead of `initialize` during initialization
- Uses specialized fixed-A functions for spatial/temporal updates

**Key Code Segment**:

```python
def fit_v2(self, images, indices=(slice(None), slice(None))) -> None:
    # ... data preparation code same as fit ...
    
    if self.estimates.A is None:
        logger.info('initializing ...')
        self.initialize(Y)
    elif (not (self.estimates.A is None)):
        logger.info('initializing: A fixed mode...')
        self.initialize_with_A_fixed(Y)  # <-- Key difference
    
    # ... subsequent update workflow ...
```

**Main Differences from `fit`**:

1. Check if A exists, if so call `initialize_with_A_fixed`
2. Rest of the workflow is similar but calls fixed-A versions of update functions

#### Modification 1.2:  `initialize_with_A_fixed` Method

**Functionality**:

- Initialize using existing A
- Calls `initialize_components_fixed_A` function

**Key Code**:

```python
def initialize_with_A_fixed(self, Y, **kwargs) -> None:
    """Component initialization with fixed A"""
    self.params.set('init', kwargs)
    estim = self.estimates
    
    if (self.params.get('init', 'method_init') == 'corr_pnr' and
            self.params.get('init', 'ring_size_factor') is not None):
        A_temp = self.estimates.A.copy()  # Copy existing A
        estim.A, estim.C, estim.b, estim.f, estim.center, \
            extra_1p = initialize_components_fixed_A(
            Y, A=A_temp, sn=estim.sn, 
            options_total=self.params.to_dict(),
            **self.params.get_group('init'))
        # ... handle extra outputs ...
```

### 2. **initialization.py** - Initialization Module

#### Modification 2.1:  `initialize_components_fixed_A` 

**Functionality**:

- Initialize other components (C, b, f, etc.) with A fixed
- Primarily for 1-photon data (corr_pnr method)

**Signature**:

```python
def initialize_components_fixed_A(Y, A, K=30, gSig=[5, 5], 
                                  # ... other parameters ...
                                  ):
```

**Key Parameters**:

- `Y`: Raw data
- `A`: Fixed spatial components (required)
- Other parameters same as standard `initialize_components`

**Workflow**:

```python
if method == 'corr_pnr':
    Ain, Cin, _, b_in, f_in, extra_1p = greedyROI_corr_fixed_A(
        Y, Y_ds, A_fixed=A,  # <-- Pass fixed A
        max_number=K, gSiz=gSiz[0], gSig=gSig[0],
        # ... other parameters ...
    )
```

#### Modification 2.2:  `greedyROI_corr_fixed_A` Function

**Functionality**:

- Initialize using corr_pnr method with fixed A
- Calls `init_neurons_corr_pnr_fixed_A` for initialization
- Uses fixed-A versions of spatial and temporal update functions

**Key Call Chain**:

```python
def greedyROI_corr_fixed_A(Y, Y_ds, A_fixed, ...):
    # 1. Initialize neurons with fixed A
    A, C, _, _, center = init_neurons_corr_pnr_fixed_A(
        Y_ds, A_fixed=A_fixed, gSiz=gSiz, gSig=gSig,
        # ...
    )
    
    # 2. Update spatial components (fixed-A version)
    A, _, C, _ = caiman.source_extraction.cnmf.spatial.update_spatial_components_fixed_A_ljr(
        B, C=C, f=..., A_in=A,
        # ...
    )
    
    # 3. Update temporal components (fixed-A version)
    C, A = caiman.source_extraction.cnmf.temporal.update_temporal_components_fixed_A(
        B, spr.csc_matrix(A, dtype=np.float32),
        # ...
    )
```

**Multiple Iterations**:

- First iteration: On downsampled data (Y_ds)
- Second iteration: On full resolution data (Y)
- Each iteration updates spatial then temporal components

### 3. **spatial.py** - Spatial Component Update Module

#### Modification:  `update_spatial_components_fixed_A` Function

**Functionality**:

- Update spatial components with A fixed as constraint

**Signature**:

```python
def update_spatial_components_fixed_A(Y, C=None, f=None, A_in=None, 
                                         sn=None, dims=None, ...):
```

**Key Features**:

- Accepts `A_in` as the fixed spatial component input
- Does not change A's shape or sparse structure
- Mainly used for iterative optimization during initialization

### 4. **temporal.py** - Temporal Component Update Module

#### Modification:  `update_temporal_components_fixed_A` Function (Lines 384-464)

**Functionality**:

- **Core Modification**: Fix A and b, only update temporal components C and f
- Removed "delete empty components" code block to ensure A shape remains unchanged

**Workflow**:

```python
def update_temporal_components_fixed_A(Y, A, b, Cin, fin, ...):
    # 1. Data preparation
    A_tot = scipy.sparse.hstack((A, b)).tocsc()
    
    # 2. Calculate projections
    YA = A_tot.T.dot(Y).T * diags(1. / nA)
    AA = (A_tot.T.dot(A_tot)) * diags(1. / nA)
    YrA = YA - AA.T.dot(Cin_tot).T
    
    # 3. Iterative optimization (calls update_iteration)
    C_tot, S, bl, YrA, c1, sn, g, lam = update_iteration(
        parrllcomp, len_parrllcomp, nb, C_tot, S, bl, nr,
        ITER, YrA, c1, sn, g, Cin_tot, T, nA, dview, debug, AA, kwargs
    )
    
    # 4. [KEY] Don't delete empty components, directly split results
    C_out = C_tot[:nr, :]
    # ... return results ...
```

## Core Design Philosophy

### 1. **Fixed-A Constraint Propagation**

```
fit_v2()
  └─> initialize_with_A_fixed()
       └─> initialize_components_fixed_A()
            └─> greedyROI_corr_fixed_A()
                 ├─> init_neurons_corr_pnr_fixed_A()
                 ├─> update_spatial_components_fixed_A()
                 └─> update_temporal_components_fixed_A()
```

Throughout the entire call chain, A is always passed as a fixed parameter and never modified.

### 2. **Key Constraint Points**

1. **Initialization Phase**: Use existing A, don't regenerate
2. **Spatial Update**: Use special functions to keep A unchanged
3. **Temporal Update**: Remove delete-empty-components logic to ensure A's count and shape unchanged
4. **Deconvolution**: Performed under fixed-A constraint

### 3. **Applicable Scenarios**

- Already have high-quality spatial components A (e.g., manually annotated ROIs)
- Need to re-analyze using same ROIs across different time periods
- Want to test the impact of fixed spatial components on results
- 1-photon calcium imaging data (corr_pnr method)



## Usage Notes

### 1. **Prerequisites**

- Must first run standard `fit()` once, or obtain A through other means
- `estimates.A` cannot be `None`
- Recommended to use `method_init='corr_pnr'`
- These modifications are based on CaImAn version 13.0.
- `fit_v2`is primarily an optimization for 1p data (corresponding to the case where nb=0 in `fit`).

### 2. **Parameter Settings**

```python
params = CNMFParams(params_dict={
    'init': {
        'method_init': 'corr_pnr',  # Required
        'ring_size_factor': 1.5,     # Must not be None
        # ... other parameters ...
    }
})
```

### 3. **Workflow Example**

```python
# Step 1: Standard fit
cnm = CNMF(n_processes=1, params=params)
cnm.fit(images)
A_original = cnm.estimates.A.copy()
C_original = cnm.estimates.C.copy()

# Step 2: fit_v2 (using existing A)
cnm2 = CNMF(n_processes=1, params=params)
cnm2.estimates.A = A_original  # Set fixed A
cnm2.fit_v2(images)
C_new = cnm2.estimates.C  # New temporal traces

# Step 3: Compare
difference = np.abs(C_original - C_new)
```