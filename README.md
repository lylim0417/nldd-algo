# Normalized Least Dependent Difference (NLDD)

## Overview

This repository provides an implementation of the **Normalized Least Dependent Difference (NLDD)**, a novel statistical framework designed to **jointly assess linearity, heteroskedasticity, and outlier behavior** within a unified formulation.

Traditional approaches typically evaluate these properties independently, requiring multiple statistical tools to diagnose structural relationships, variance patterns, and anomalous observations. In contrast, NLDD introduces a **normalized variance-based dependency measure** that captures structural conformity, residual dispersion, and deviation irregularities simultaneously.

By integrating these aspects into a single metric, NLDD enables a more **holistic and interpretable assessment of data behavior**, reducing methodological fragmentation and improving diagnostic consistency across diverse data conditions. The framework further incorporates an **iterative refinement mechanism** to enhance robustness against outliers and distributional inconsistencies.

This unified perspective positions NLDD as a **comprehensive diagnostic tool** for analyzing complex data relationships beyond conventional single-purpose methods.

---

## Key Contributions

- **Unified Diagnostic Framework**  
  NLDD provides a single, coherent formulation for jointly assessing linearity, heteroskedasticity, and outlier behavior, eliminating the need for multiple independent statistical tests.

- **Normalized Dependency Measure**  
  Introduces a variance-based normalization strategy that ensures scale invariance and enables consistent interpretation across heterogeneous datasets.

- **Integrated Outlier Sensitivity Mechanism**  
  Embeds an iterative refinement process that systematically reduces the influence of anomalous observations while preserving underlying structural patterns.

- **Enhanced Diagnostic Interpretability**  
  Produces a consolidated metric that reflects both structural conformity and residual irregularities, facilitating more intuitive and reliable data analysis.

- **Robustness Across Data Conditions**  
  Demonstrates stable performance under varying relationship structures, noise distributions, and data irregularities.
  
---

## Methodological Insight

NLDD is grounded in the principle of **minimizing dependence within normalized residual structures**, where both structural alignment and variance behavior are jointly evaluated.

The framework operates by transforming observed relationships into a normalized space in which:

- Structural deviations from linearity are explicitly quantified  
- Variance heterogeneity is incorporated into the dependency formulation  
- Irregular observations are iteratively down-weighted through refinement  

This formulation departs from conventional correlation-based approaches by extending beyond monotonic association and explicitly integrating variance-driven characteristics of the data.

As a result, NLDD captures a broader spectrum of data behavior, enabling it to function as a **comprehensive diagnostic measure** rather than a single-purpose statistic.

---

## Mathematical Formulation

Let the fitted linear response be defined as

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i,
$$

with residuals

$$
e_i = y_i - \hat{y}_i, \qquad i = 1,2,\dots,n.
$$

The Normalized Least Dependent Difference (NLDD) is defined as

$$
\mathrm{NLDD}(X,Y)
=
1
-
\frac{
\sum_{i=1}^{n} w_i \left(e_i - \bar{e}_w\right)^2
}{
\sum_{i=1}^{n} \left(y_i - \bar{y}\right)^2
},
$$

where

$$
\bar{e}_w = \frac{\sum_{i=1}^{n} w_i e_i}{\sum_{i=1}^{n} w_i},
\qquad
\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i,
$$

and \(w_i\) denotes an adaptive weight used to reduce the influence of anomalous observations during iterative refinement.

This formulation evaluates the proportion of normalized residual variability remaining after fitting the structural relationship between \(X\) and \(Y\). Larger NLDD values indicate stronger structural conformity with more stable variance behavior, whereas smaller values reflect greater deviation from linearity, heteroskedasticity, or outlier contamination.

---

## Diagnostic Scope

NLDD is designed to provide a unified assessment across three key aspects of data behavior:

- **Linearity** – Evaluates the degree of structural conformity between variables  
- **Heteroskedasticity** – Captures variability patterns in residual dispersion  
- **Outliers** – Identifies and mitigates the influence of anomalous observations  

By integrating these components, NLDD offers a consolidated perspective that reduces fragmentation in traditional diagnostic workflows.

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
├── LICENSE.md           # Non-commercial license
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
