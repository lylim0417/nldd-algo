# Normalized Least Dependent Difference (NLDD)

## Overview

This repository provides an implementation of the **Normalized Least Dependent Difference (NLDD)**, a novel statistical framework designed to **jointly assess linearity and homoskedasticity** within a unified formulation.

Conventional approaches typically evaluate these properties separately—using correlation coefficients (e.g., Pearson, Spearman, Kendall) for linearity, and statistical tests (e.g., Breusch–Pagan, White, Goldfeld–Quandt) for heteroskedasticity. In contrast, NLDD introduces a **normalized variance-based dependency measure** that captures both structural conformity and residual dispersion simultaneously.

---

## Key Contributions

- **Unified Diagnostic Metric**  
  A single formulation that integrates linearity and heteroskedasticity assessment.

- **Normalized Dependency Structure**  
  Reduces sensitivity to scale and improves interpretability across datasets.

- **Outlier-Aware Iterative Refinement**  
  Incorporates an iterative mechanism to mitigate the influence of extreme observations.

- **Comparative Robustness**  
  Demonstrates competitive or superior performance against classical correlation measures and heteroskedasticity tests.

---

## Methodological Insight

NLDD is based on the principle of **minimizing dependence between transformed residual structures**, normalized to ensure comparability across varying data distributions.

Unlike correlation-based methods that focus solely on monotonic relationships, NLDD explicitly accounts for:

- Residual variance behavior  
- Structural deviations from linearity  
- Distributional inconsistencies  

This allows NLDD to function as a **comprehensive diagnostic tool** rather than a single-purpose statistic.

---

## Repository Structure

```text
nldd-algo/
├── nldd.py              # Core NLDD implementation
├── dataset/             # Synthetic and benchmark datasets
├── result_graph/        # Plotting, results, and visualization
│   ├── linearity/       # PCC, SRCC, KTCC, WLS, NLDD comparisons
│   ├── hetero/          # WT, BPT, GQT, NLDD comparisons
│   └── outlier/         # IQR, LOF, NLDD vs ground truth (GT)
├── README.md            # Project documentation
├── LICENSE              # Non-commercial license
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/lylim0417/nldd-algo.git
cd nldd-algo
pip install -r requirements.txt
```

---

## Usage

```bash
from nldd import run_nldd_alone

run_nldd_alone()
```

---

## Experimental Evaluation

The repository includes comparative experiments against:

**Linearity Measures:**
- Pearson Correlation Coefficient (PCC)
- Spearman's Rank Correlation Coefficient (SRCC)
- Kendall’s Tau Correlation Coefficient (KTCC)
- Weighted Least Squares (WLS)

**Heteroskedasticity Tests:**
- Breusch–Pagan Test (BPT)
- White’s Test (WT)
- Goldfeld–Quandt Test (GQT)

**Outlier Tests:**
- Interquartile Range (IQR)
- Local Outlier Factor (LOF)

Results demonstrate that NLDD provides **consistent and interpretable diagnostics** across varying data conditions, including:

- Linear vs non-linear relationships  
- Homoskedastic vs heteroskedastic noise structures  
- Presence of outliers  

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nldd2026,
  title        = {Normalized Least Dependent Difference (NLDD)},
  author       = {Herrick Han Lin Yeap and Le Ying Lim and Brandon Chen Hong Chow and Kok Seng Eu and Zhengying Ho and Kian Meng Yap},
  year         = {2026},
  howpublished = {\url{https://github.com/lylim0417/nldd-algo}},
  note         = {Accessed: YYYY-MM-DD}
}
```

---

## License

This project is licensed under the **NLDD Non-Commercial Research License (NCRL v1.0)**.

- ✅ Free for academic, research, and educational use  
- ❌ Commercial use requires explicit permission  

For commercial licensing inquiries, please contact: **kmyap@sunway.edu.my**

---

## Acknowledgements

This work has been developed as part of ongoing research in statistical dependency modeling and diagnostic frameworks.

---

## Disclaimer

This software is provided for research purposes only. The authors make no guarantees regarding correctness, completeness, or suitability for any specific application.

---
