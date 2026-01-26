# Quadlogit: Quadruple Logit Regression for Network Formation Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**Quadlogit** is a Python package for **quadruple logit regression** in **network formation models**. It estimates parameters of directed networks using a specialized quasi-differencing approach that constructs informative quadruples from network data.

The key innovation is the **quadruple construction** method: instead of analyzing individual links, the method forms 4-tuples of agents and uses differential comparisons of link patterns to eliminate agent-specific fixed effects. This approach provides consistent parameter estimates in networks with agent heterogeneity.

### Key Features

- ✅ **Directed network support** with optional covariates
- ✅ **Fast tensor-based computation** using broadcast masking for efficient quadruple construction
- ✅ **Automatic standard error calculation** via fully vectorized tensor ops
- ✅ **Data generation utilities** for Monte Carlo studies
- ✅ **Flexible API** with silent/verbose modes

## Installation

### From PyPI (Coming Soon)

```bash
pip install quadlogit
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/zizhongyan/quadlogit.git
cd quadlogit

# Install in development mode
pip install -e .
```

### Jupyter/Notebook Users

To ensure the package is installed in the same environment as your Jupyter kernel:

```python
%cd /path/to/quadlogit
%pip install -e .
```

## Quick Start

### 1. Basic Network Estimation

```python
import numpy as np
import quadlogit

# Generate synthetic data
G, X, _, _, _, _, _, _ = quadlogit.demo.GenData(N=50, directed=True, seed=42)

# Estimate network formation model
result = quadlogit.fit(G, X)

# Access results
print(f"Coefficient: {result.paras}")
print(f"Standard Error: {result.se}")
print(f"Estimation Success: {result.success}")
```

### 2. Silent Mode (Suppress Output)

```python
# Default: prints detailed output table
result = quadlogit.fit(G, X, silent=False)

# Silent mode: suppresses output
result = quadlogit.fit(G, X, silent=True)
```

## API Reference

### Core Function: `quadlogit.fit()`

```python
quadlogit.fit(G, X=None, X_names=None, silent=False, indices=None)
```

#### Parameters

| Parameter   | Type                              | Default            | Description                                                                                                                                                   |
| ----------- | --------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `G`       | numpy.ndarray (N×N)              | **Required** | Binary adjacency matrix of the network. Must be float64. Shape (N, N) where N is the number of agents. Values must be 0 or 1.                                 |
| `X`       | numpy.ndarray (N×N×K) or (N×N) | None               | Dyadic covariates. Can be 2D (N×N) which is auto-expanded to (N×N×1), or 3D (N×N×K).**Note:** Only the first covariate X[:,:,0] is currently used. |
| `X_names` | list of str                       | None               | Names of covariates for output display. If None, auto-generates names as "X1", "X2", etc.                                                                     |
| `silent`  | bool                              | False              | If True, suppresses the output table. If False, prints estimation results including coefficients, standard errors, and p-values.                              |

#### Returns

Returns a `fit` object with the following attributes:

| Attribute    | Type          | Description                                                                                             |
| ------------ | ------------- | ------------------------------------------------------------------------------------------------------- |
| `paras`    | numpy.ndarray | Estimated coefficients (1D array). First element is covariate coefficient, remaining are fixed effects. |
| `se`       | numpy.ndarray | Standard errors of the coefficients.                                                                    |
| `success`  | int           | 1 if estimation succeeded, 0 if failed.                                                                 |
| `N`        | int           | Number of agents in the network.                                                                        |
| `n_valid_quads` | int           | Number of informative quadruples used in estimation.                                                    |

#### Example

```python
import quadlogit
import numpy as np

# Generate network data
G, X, _, _, _, _, _, _ = quadlogit.demo.GenData(N=50, directed=True, seed=42)

# Fit model
result = quadlogit.fit(G, X, X_names=["Covariate1"], silent=False)

# Extract results
coef = result.paras[0]          # Coefficient on covariate
se = result.se[0]              # Standard error
t_stat = coef / se             # t-statistic
```

---

### Data Generation: `quadlogit.demo.GenData()`

```python
quadlogit.demo.GenData(N, directed=True, mutual=False, specification="A1", seed=111)
```

Generates synthetic network data from a network formation DGP based on Graham (2017) and extended to directed networks.

#### Parameters

| Parameter         | Type | Default            | Description                                                                                               |
| ----------------- | ---- | ------------------ | --------------------------------------------------------------------------------------------------------- |
| `N`             | int  | **Required** | Number of agents.                                                                                         |
| `directed`      | bool | True               | If True, generates directed networks. If False, generates undirected networks.                            |
| `mutual`        | bool | False              | If True, allows mutual link formation. Only used when `directed=True`.                                  |
| `specification` | str  | "A1"               | DGP specification: "A1", "A2", "A3", "B1", "B2", "B3". Affects fixed effects distribution and parameters. |
| `seed`          | int  | 111                | Random seed for reproducibility.                                                                          |

#### Returns

Tuple containing:

```python
(G, X, Z, density, degree, transitivity, separation, true_parameters)
```

| Return              | Type                    | Description                                     |
| ------------------- | ----------------------- | ----------------------------------------------- |
| `G`               | numpy.ndarray (N×N)    | Binary directed adjacency matrix.               |
| `X`               | numpy.ndarray (N×N×1) | Dyadic covariate (directed utility component).  |
| `Z`               | numpy.ndarray (N×N×1) | Dyadic covariate (mutual utility component).    |
| `density`         | float                   | Network density (proportion of possible links). |
| `degree`          | float                   | Average out-degree.                             |
| `transitivity`    | float                   | Network transitivity (clustering coefficient).  |
| `separation`      | int                     | 1 if network has isolation issues, 0 otherwise. |
| `true_parameters` | numpy.ndarray           | True parameters used in DGP (for evaluation).   |

#### Example

```python
import quadlogit

# Generate directed network with 45 agents
G, X, Z, density, degree, transitivity, sep, true_params = quadlogit.demo.GenData(
    N=45, 
    directed=True, 
    mutual=False,
    specification="A1",
    seed=42
)

print(f"Network density: {density:.3f}")
print(f"Average degree: {degree:.2f}")
print(f"True parameter: {true_params[0]}")
```

---

### Monte Carlo Data Generation: `quadlogit.demo.NetworkGenDataMC()`

```python
quadlogit.demo.NetworkGenDataMC(N, directed=True, mutual=False, specification="A1", mc_sweep=1000)
```

Generates multiple independent samples from the network formation DGP for Monte Carlo studies.

#### Parameters

| Parameter         | Type | Default            | Description                           |
| ----------------- | ---- | ------------------ | ------------------------------------- |
| `N`             | int  | **Required** | Number of agents.                     |
| `directed`      | bool | True               | If True, generates directed networks. |
| `mutual`        | bool | False              | If True, allows mutual links.         |
| `specification` | str  | "A1"               | DGP specification (see GenData).      |
| `mc_sweep`      | int  | 1000               | Number of Monte Carlo replications.   |

#### Returns

Tuple containing:

```python
(G_bank, X_bank, Z_bank, density_mc, degree_mc, transitivity_mc, separation_mc, 
 thetaEst_mc, thetaSE_mc, CP95_mc, CP90_mc, networkFE_succ_mc, clogit_succ_mc,
 apethetaEst_mc, apethetaSE_mc, apeCP95_mc, apeCP90_mc)
```

| Return              | Type          | Shape            | Description                           |
| ------------------- | ------------- | ---------------- | ------------------------------------- |
| `G_bank`          | numpy.ndarray | (N, N, mc_sweep) | Network samples.                      |
| `X_bank`          | numpy.ndarray | (N, N, mc_sweep) | Covariate samples.                    |
| `Z_bank`          | numpy.ndarray | (N, N, mc_sweep) | Mutual covariate samples.             |
| `density_mc`      | numpy.ndarray | (mc_sweep,)      | Density for each sample.              |
| `degree_mc`       | numpy.ndarray | (mc_sweep,)      | Average degree for each sample.       |
| `transitivity_mc` | numpy.ndarray | (mc_sweep,)      | Transitivity for each sample.         |
| `separation_mc`   | numpy.ndarray | (mc_sweep,)      | Separation indicator for each sample. |

#### Example

```python
import quadlogit
import numpy as np

# Generate 500 Monte Carlo samples
G_bank, X_bank, Z_bank, densities, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
    quadlogit.demo.NetworkGenDataMC(N=50, directed=True, mc_sweep=500, seed=42)

# Estimate model on each sample
coefs = []
for mc in range(500):
    result = quadlogit.fit(G_bank[:,:,mc], X_bank[:,:,mc], silent=True)
    if result.success == 1:
        coefs.append(result.paras[0])

print(f"Mean estimate: {np.mean(coefs):.4f}")
print(f"Std deviation: {np.std(coefs):.4f}")
```

---

## Detailed Usage Examples

### Example 1: Basic Network Formation

```python
import quadlogit
import numpy as np

# Generate synthetic network
G, X, _, _, _, _, _, true_params = quadlogit.demo.GenData(
    N=50, 
    directed=True, 
    specification="A1",
    seed=123
)

# Estimate model with verbose output
result = quadlogit.fit(
    G, 
    X, 
    X_names=["Link Preference"],
    silent=False
)

# Print results
print(f"\nTrue parameter: {true_params[0]:.4f}")
print(f"Estimated coefficient: {result.paras[0]:.4f}")
print(f"Standard error: {result.se[0]:.4f}")
print(f"95% CI: [{result.paras[0] - 1.96*result.se[0]:.4f}, {result.paras[0] + 1.96*result.se[0]:.4f}]")
```

**Output:**

```
--------------------------------------------------------------------------------
---------------------------- ESTIMATION RESULTS --------------------------------
        DIRECTED NETWORK FORMATION MODEL -- QUADRUPLE LOGIT REGRESSION
--------------------------------------------------------------------------------
Number of agents: 50                        Number of obs: [M]
                                            Time spent (seconds):  X.XXX
--------------------------------------------------------------------------------
Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
--------------------------------------------------------------------------------
Link Preference             X.XXXX        X.XXXX      0.000    [X.XXXX, X.XXXX]
```

### Example 2: Multiple Networks (Monte Carlo)

```python
import quadlogit
import numpy as np
import matplotlib.pyplot as plt

# Generate 200 Monte Carlo samples
G_bank, X_bank, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
    quadlogit.demo.NetworkGenDataMC(N=40, mc_sweep=200, seed=42)

# Estimate on each sample
estimates = []
ses = []
successes = []

for mc in range(200):
    result = quadlogit.fit(
        G_bank[:,:,mc], 
        X_bank[:,:,mc],
        silent=True
    )
    if result.success == 1:
        estimates.append(result.paras[0])
        ses.append(result.se[0])
        successes.append(1)
    else:
        successes.append(0)

estimates = np.array(estimates)
ses = np.array(ses)

# Summary statistics
print(f"Estimation success rate: {np.mean(successes):.1%}")
print(f"Mean estimate: {np.mean(estimates):.4f}")
print(f"Mean SE: {np.mean(ses):.4f}")
print(f"Empirical std dev: {np.std(estimates):.4f}")

# Plot
plt.hist(estimates, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(np.mean(estimates), color='red', linestyle='--', label='Mean')
plt.xlabel('Coefficient Estimate')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

### Example 3: Batch Estimation (Broadcasted Quadruples)

```python
import quadlogit
import numpy as np
import time

N = 80

# Fit multiple models without any pre-computed indices
results = []
fit_times = []
for seed in range(5):
    G, X, _, _, _, _, _, _ = quadlogit.demo.GenData(N=N, seed=seed)
    start = time.time()
    result = quadlogit.fit(G, X, silent=True)
    fit_times.append(time.time() - start)
    results.append(result)
    print(f"Model {seed+1} fit in {fit_times[-1]:.3f}s (success={result.success})")

print(f"Average runtime: {np.mean(fit_times):.3f}s")
```

### Example 4: Custom Covariates

```python
import quadlogit
import numpy as np

# Create custom network and covariates
N = 60
np.random.seed(42)

# Binary adjacency matrix
G = np.random.binomial(1, 0.15, (N, N)).astype('float64')
np.fill_diagonal(G, 0)  # No self-loops

# Multiple covariates (only first is used currently)
X1 = np.random.normal(0, 1, (N, N)).astype('float64')  # Continuous covariate
X2 = np.random.binomial(1, 0.5, (N, N)).astype('float64')  # Binary covariate

X = np.stack([X1, X2], axis=2)

# Estimate (only X1 is used)
result = quadlogit.fit(
    G, 
    X,
    X_names=["Continuous Feature", "Binary Feature (unused)"],
    silent=False
)

print(f"\nNote: Only 'Continuous Feature' is used in current version.")
print(f"      'Binary Feature' is included for future multi-covariate support.")
```

---

## Important Notes & Limitations

### Data Requirements

1. **Adjacency Matrix `G`**:

   - Must be (N, N) square matrix
   - Values must be binary (0 or 1)
   - Data type must be float64 (auto-converted if needed)
   - No self-loops (diagonal should be 0)
2. **Covariates `X`**:

   - Can be 2D (N, N) or 3D (N, N, K)
   - Data type must be float64 (auto-converted if needed)
   - **Currently only first covariate `X[:,:,0]` is used**

### Current Limitations

1. **Single Covariate**: The implementation currently uses only the first covariate. Multi-covariate support requires refactoring of the quadruple construction logic.
2. **Directed Networks Only**: Current API supports directed networks. Undirected/mutual extensions exist (see `_readme.txt`) but are not exposed in the main API.
3. **Network Size**: Tensor construction scales as O(N^4); runtimes and memory grow quickly with N. N ≤ 150 is practical; larger N may be costly.
4. **Fixed Effects**: The method automatically handles agent-level fixed effects through quadruple differencing. Dyadic-level fixed effects are not supported.

### Error Handling

- Missing index files (for N ≤ 100) trigger automatic loading from `.mat` files
- Estimation failures silently set `result.success = 0`
- Check `result.success == 1` to verify successful estimation

### Silent Mode

```python
# Default (silent=False): prints detailed output
result = quadlogit.fit(G, X, silent=False)

# Silent mode (silent=True): no printed output
result = quadlogit.fit(G, X, silent=True)
```

---

## Architecture Overview

### Core Computation Pipeline

```
Input: G (N×N), X (N×N×K)
    ↓
[1] Broadcasted Quadruple Construction
    - Tensor reshaping + masks enforce i1 < i2 and j1 ≠ j2
    - Compute differences zz (links) and rr (covariates)
    - Keep informative contrasts (ss==1)
    ↓
[2] Logit Regression
    - LHS: Transformed network differences (zz)
    - RHS: Covariate differences (rr)
    - Method: Maximum likelihood via Graham's netrics package
    ↓
[3] Standard Error Computation
    - Fully vectorized tensor operations (no permutation index files)
    - Accounts for network structure via masks
    ↓
Output: result.paras, result.se, result.success
```

### Key Functions (Internal)

- `rearrangement_fast()`: Broadcast + masking quadruple construction (numpy)
- `standarderror_fast()`: Tensor-based SE calculation (no external indices)

---

## Performance Tips

1. **Use Broadcasted Pipeline**: No need to manage index files—just call `fit` directly.
2. **Silent Mode**: Use `silent=True` for batch processing to suppress output.
3. **Network Size & Memory**: Full tensor construction scales as O(N^4); moderate N (≤150) remains practical, larger N may need substantial RAM/time.
4. **Seed Management**: Use consistent seeds for reproducibility in Monte Carlo studies.

---

## References

### Methodological Foundation

- Graham, B. S. (2017). "An econometric model of link formation with degree heterogeneity." *Econometrica*, 85(4), 1033-1063.
- Jochmans, K. (2019). "Semiparametric estimation of network formation games." *Journal of Econometrics*, 211(2), 472-487.

### Related Packages

- **netrics** (Graham, 2016): https://github.com/bryangraham/netrics
  - Provides the `logit` and other econometric methods used internally

---

## Development & Citation

### Code Contributors

- Prof. **Zizhong Yan** (Advisor)
- **Shiran Hu**
- **Muyang Guo**
- **Xinran Cheng**
- **Xuan Zhou**

### Version

- Current Version: 0.2.1 (Beta)
- Status: Active development
- License: MIT

### Citing Quadlogit

If you use Quadlogit in your research, please cite:

```bibtex
@software{yan2025quadlogit,
  title={Quadlogit: Quadruple Logit Regression for Network Formation Models},
  author={Hu, Shiran and Guo, Muyang and Cheng, Xinran and Zhou, Xuan},
  year={2025},
  url={https://github.com/HuNeedHelp/quadlogit}
}
```

---

## Troubleshooting

### Common Issues

**Q: "Error: adjacency matrix G is not binary"**

- Ensure G contains only 0 and 1 values
- Check for floating-point precision issues
- Example: `G = (G > 0.5).astype(float)`

**Q: "Estimation failed" (result.success == 0)**

- Check for perfect separation in quadruple data
- Try different network sizes or specifications
- Verify that X is properly formatted

**Q: Memory error with large N**

- Standard error computation creates (N, N, N, N) tensors
- For N > 100, consider computational limitations
- Use `silent=True` to reduce overhead

---

## FAQ

**Q: Can I use multiple covariates?**

- Currently, only the first covariate is used. Multi-covariate support is planned for future versions.

**Q: Does this work with undirected networks?**

- The main API is for directed networks. Undirected extensions exist (see `_readme.txt`) but are not in the standard API.

**Q: How do I interpret the coefficients?**

- The first coefficient is the effect of the dyadic covariate on link formation probability.
- Remaining coefficients are fixed effects for each agent.

**Q: What if my network has isolated nodes?**

- The method is designed to handle this, but the `separation` flag will be set to 1. Check your network structure.

---

## Support & Contributions

- **Issues**: https://github.com/zizhongyan/quadlogit/issues
- **Documentation**: https://quadlogit.readthedocs.io/
- **Discussions**: GitHub Discussions (to be enabled)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

Graham, Bryan S. (2017). "ipt: a Python 3.7 package for causal inference by inverse 
 probability tilting," (Version 0.2.2) [Computer program]. Available at 
 https://github.com/bryangraham/ipt (Accessed 04 Oct 2018) 

