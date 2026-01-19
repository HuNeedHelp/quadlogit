# twowaypanel

**twowaypanel** is a Python package for **bias-corrected estimation and inference** in **nonlinear panel data models with additive individual and time fixed effects**.  
It accompanies the paper:

> Yan, Zizhong., Zhang, Zhengyu., Chen, Mingli., and Li, Jingrong. (2026), *Robust Priors in Nonlinear Panel Models with Individual and Time Effects*. *arXiv e-prints*, `<arXiv id to be added>`.

**Quick links**
- **Paper**: `<arXiv link to be added>`
- **Documentation (Read the Docs)**: [twowaypanel.readthedocs.io](https://twowaypanel.readthedocs.io)
- **Replication materials**: see **Replication materials for Yan et al. (2026)** below (notebook + PDF)

## ▶ Overview

Nonlinear panel models with additive individual and time effects are a workhorse in empirical economics. However, fixed-effects maximum likelihood estimation is subject to the **incidental parameter problem**, which can induce substantial **finite-sample bias** in both parameter estimates and economically relevant functionals (e.g., average partial effects).

`twowaypanel` provides **bias-corrected** estimation and inference for:
- model parameters, and  
- **average partial effects (APEs)**, for both **continuous regressors** and **discrete regressors** (finite changes).

The current release supports four model classes—**binary logit**, **binary probit**, **multinomial logit**, and **ordered logit**—covering **static** and **dynamic** specifications. Here, “static” refers to specifications with exogenous regressors, while “dynamic” allows lagged dependent variables among regressors.

## ▶ Replication materials for Yan et al. (2026)

This repository includes replication materials for the **Monte Carlo experiments** and the **empirical illustration** in Yan et al. (2026).

- **Replication notebook**: [Replication Notebook.ipynb](https://github.com/zizhongyan/twowaypanel/blob/main/replications/Replication%20Notebook.ipynb)
- **Compiled PDF (generated from the notebook)**: [Replication Notebook.pdf](https://github.com/zizhongyan/twowaypanel/blob/main/replications/Replication%20Notebook.pdf)
- **All replication files**: [replications.zip](https://github.com/zizhongyan/twowaypanel/raw/main/replications.zip)


**Suggested workflow**
1. Install the package (see below).
2. Open the replication notebook and run cells in order to reproduce tables/figures.

> For readers who prefer a guided introduction before running replication, please consult the Read the Docs tutorial and examples gallery first.

## ▶  Documentation

User-facing documentation—tutorials, examples (by model class), and API reference—is maintained on Read the Docs:

- **Documentation homepage**: [twowaypanel.readthedocs.io](https://twowaypanel.readthedocs.io/)
- **Tutorials**: [Quick Start and tutorial pages](https://twowaypanel.readthedocs.io/en/latest/tutorial/tutorial.html)
- **Examples gallery**: [Worked examples by model class](https://twowaypanel.readthedocs.io/en/latest/examples/examples.html)

The documentation explains expected data formats, model options, and how to interpret the printed output tables (including APEs and, when relevant, MCMC diagnostics).


## ▶  Supported models and bias-correction methods

`twowaypanel` implements the likelihood-based bias-correction methods proposed in Yan et al. (2026), including:
- **Integrated-likelihood-based correction** (via *priors*; “prior correction”), and  
- **Joint-likelihood-based correction** (via *penalties*; “penalty correction”).
For logit and probit panels, the package also provides the **analytical bias correction** for fixed-effects MLE developed by Fernández-Val and Weidner (2016).

| Model class       | Specification  | Generic prior / penalty |   Model-specific prior / penalty    | Analytical correction (FW16) |
| ----------------- | :------------: | :---------------------: | :---------------------------------: | :--------------------------: |
| Binary logit      |     Static     |            ✓            |            ✓ (Prior SE)             |              ✓               |
| Binary logit      |    Dynamic     |            ✓            | ✓ (Prior DE; Prior SE + analytical) |              ✓               |
| Probit            | Static/Dynamic |            ✓            |                  –                  |              ✓               |
| Multinomial logit |     Static     |            ✓            |            ✓ (Prior SML)            |              –               |
| Multinomial logit |    Dynamic     |            ✓            |            ✓ (Prior DML)            |              –               |
| Ordered logit     | Static/Dynamic |            ✓            |                  –                  |              –               |

**Notes.**
- *Model-specific* priors/penalties are defined in Yan et al. (2026) and are designed to deliver improved finite-sample performance within the corresponding model class.
- The *generic* prior/penalty is intended to be **robust across a broad class of nonlinear panel specifications**; see Yan et al. (2026) for details and motivation.



## ▶  Installation 

1. **Installation from PyPI** (recommended)
```bash
pip install twowaypanel
````

2. **Install the development version** from GitHub
```bash
git clone https://github.com/zizhongyan/twowaypanel.git
cd twowaypanel
pip install -e .
```

3. **Install from a local copy**
   
If you downloaded the source code (e.g., as a ZIP file) and unzipped it locally, install from the repository root:
```bash
cd /path/to/twowaypanel
pip install -e .
```

4. **Jupyter / notebook users**
   
To ensure the package is installed into the **same environment as the current Jupyter kernel**, run the following from the repository root directory:
```python
%cd /path/to/twowaypanel
%pip install -e .
```



## ▶ Citing this work

Please cite both the underlying paper and the software package when using this code in your research.

1. **Paper**
   - Yan, Zizhong., Zhang, Zhengyu., Chen, Mingli., and Li, Jingrong (2026), “Robust priors in nonlinear panel models with individual and time effects.”       *arXiv e-prints*, `<arXiv id to be added>`.

2. **Software**
   - Yan, Zizhong., Zhang, Zhengyu., Chen, Mingli., and Li, Jingrong (2026),  “twowaypanel: A Python package for econometric analysis of nonlinear panel models with two-way fixed effects.“  (Version 0.9.4) [Computer software].  Source code: `https://github.com/zizhongyan/twowaypanel`.


## References

- Fernández-Val, Iván and Martin Weidner (2016),    “Individual and time effects in nonlinear panel models with large N, T.”    *Journal of Econometrics*, 192(1), 291–312.

---

**Code maintainer:** Zizhong Yan, Institute for Economic and Social Research (IESR), Jinan University, Guangzhou, China.  
Email: `helloyzz@gmail.com`



