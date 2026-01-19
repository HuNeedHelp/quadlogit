# Copilot Instructions for Quadlogit

## Project Overview

**Quadlogit** is a Python package for **quadruple logit regression** in **network formation models**. It estimates parameters of directed networks using a specialized quasi-differencing approach that constructs informative quadruples from network data.

The project also includes scaffolding for **twowaypanel** (nonlinear panel data models with fixed effects), though this is in development with incomplete examples.

## Architecture & Key Components

### Core Data Flow

1. **Input**: Adjacency matrix `G` (N×N binary network) + optional covariates `X` (N×N×K)
2. **Quadruple Construction** (`quadlogit.api.quadlogit.py`): Generates all possible 4-tuples of agents, filtering for informative contrasts
3. **Standard Error Computation** (`quadlogit.utils.helpers.py`): Uses pre-loaded permutation indices for fast tensor operations
4. **Estimation**: Logit regression on quadruple-level data via `quadlogit.lib.ipt.logit`

### Module Structure

- **`quadlogit/api/`**: Public interface—main `fit()` class orchestrates entire workflow
- **`quadlogit/lib/ipt/`**: Graham's **netrics** econometric methods (logit, OLS, IV, etc.)
- **`quadlogit/utils/`**: Helpers for quadruple construction, permutation indices, standard errors
- **`quadlogit/demo/`**: Data generating processes for Monte Carlo studies (`NetworkGenDataMC`, `GenData`)

### Critical Design Patterns

1. **Pre-computed Indices**: Network analysis requires many 4-tuples; rearrangement/permutation indices are **pre-loaded from `.mat` files** (`N100_*.mat`) for N ≤ 100 to avoid expensive computation
   - See: `quadlogit.utils.helpers.loadindex()`
   - Falls back to on-the-fly computation for N > 100

2. **Tensor Operations**: Heavy use of NumPy advanced indexing for speed (e.g., `G[rearragement_index[:,0],rearragement_index[:,1]]`)
   - This allows vectorized quadruple construction; don't refactor to loops

3. **Silent Mode**: Output controlled via `silent=False` flag—respects verbose/quiet user preferences

## Developer Workflows

### Installation

```bash
# Development install
pip install -e .

# For Jupyter use (ensures same environment as kernel):
%cd /path/to/quadlogit
%pip install -e .
```

### Running Examples

- [examples/examples.ipynb](examples/examples.ipynb): Demonstrates basic workflow (incomplete, uses twowaypanel API)
- [ShiranNetwork - Final.ipynb](ShiranNetwork%20-%20Final.ipynb): Full network formation example

### Key Dependencies

- **numpy, scipy, pandas**: Core numerical computing
- **torch** (in setup.cfg): Optional, for future panel model extensions
- **netrics** (vendored in `lib/ipt/`): Graham's econometric methods

## Important Conventions & Gotchas

1. **Data Shape Requirements**:
   - Adjacency matrix: `G` must be (N, N) binary, float64
   - Covariates: `X` as (N, N, K) or auto-converted from (N, N) → (N, N, 1)
   - Only the **first covariate** (`X[:,:,0]`) is currently used; multiple covariates require refactoring

2. **Estimation Limitations**:
   - Current implementation: **directed networks only** with **single covariate**
   - Undirected/mutual extension exists (referenced in `_readme.txt`) but not in main API
   - `silent=True` suppresses output tables; `silent=False` prints detailed results

3. **Error Handling**:
   - Missing indices trigger auto-load from `.mat` files
   - Estimation failures silently set `self.success = 0` and exit
   - No graceful recovery—improve error messages if extending

4. **Documentation State**:
   - Main docstrings are placeholders (`---to be add a helpfile---`)
   - Code comments in English; docstrings in progress
   - README mentions nonexistent features (paper link, full API ref)

## Code Quality & Testing

- No unit tests found—validation currently manual via notebooks
- Version: 0.2.1 (beta, under active development)
- Authors: Zizhong Yan (primary), Shiran Hu (network extensions)

## When Adding Features

1. **Maintain tensor operations**: Don't refactor vectorized indexing to loops
2. **Preserve silent mode**: Add flag to new methods if user-facing
3. **Test with N ≤ 100 first**: Use pre-loaded indices for quick iteration; N > 100 is slower
4. **Update examples**: Keep [examples/examples.ipynb](examples/examples.ipynb) and docstrings synchronized
5. **Handle multi-covariate case carefully**: Current code comments indicate refactoring needed

## Useful Entry Points

- **Main workflow**: [quadlogit/api/quadlogit.py#L27](quadlogit/api/quadlogit.py#L27) (`fit.__init__`)
- **Quadruple logic**: [quadlogit/utils/helpers.py#L14](quadlogit/utils/helpers.py#L14) (`rearragement_fast`)
- **Standard errors**: [quadlogit/utils/helpers.py#L22](quadlogit/utils/helpers.py#L22) (`standarderror_fast`)
- **Data generation**: [quadlogit/demo/networkDGP.py#L14](quadlogit/demo/networkDGP.py#L14) (`NetworkGenDataMC`)
