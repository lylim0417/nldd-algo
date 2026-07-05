# Normalized Least Dependent Difference (NLDD)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21203715.svg)](https://doi.org/10.5281/zenodo.21203715)

## Overview

This repository provides an implementation of the **Normalized Least Dependent Difference (NLDD)**, a statistical framework designed to **jointly assess linearity, heteroskedasticity, and outlier behavior** within a unified formulation.

NLDD operates by quantifying the **dependent difference** between observed and predicted values derived from a fitted linear regression model. By analyzing the dispersion of these differences, the method captures both structural conformity and variance stability of the data.

Unlike conventional approaches that separately evaluate structural relationships and variance assumptions, NLDD integrates these aspects through a **range-normalized dispersion measure of dependent differences**, enabling a holistic and interpretable assessment of data behavior. The framework further incorporates an **iterative refinement mechanism** to improve robustness against anomalous observations and progressively stabilize the regression model.

This unified perspective positions NLDD as a **comprehensive diagnostic tool** for evaluating regression validity under complex data conditions.

---

## Key Contributions

- **Unified Diagnostic Framework**  
  NLDD provides a single formulation for jointly evaluating linearity, heteroskedasticity, and outlier behavior based on residual-derived dependent differences.

- **Dependent Difference Formulation**  
  Introduces a novel metric based on the absolute deviation between observed and predicted values, integrating concepts from Mean Absolute Error (MAE) and Mean Absolute Deviation (MAD).

- **Range-Normalized Dispersion Measure**  
  Utilizes the standard deviation of dependent differences and normalizes it using the response range $$\((\max(y) - \min(y))\)$$, enabling consistent interpretation across datasets.

- **Iterative Outlier Refinement Mechanism**  
  Incorporates a dynamic, threshold-based process to detect and remove anomalous observations, improving regression reliability.

- **Robust Performance Across Data Conditions**  
  Demonstrates effectiveness in identifying linearity, variance stability, and outliers under noisy, non-linear, and heteroskedastic scenarios.
  
---

## Mathematical Formulation

Given a dataset $$\( M = \{(x_i, y_i)\}_{i=1}^{n} \)$$, a linear regression model is first fitted:

$$
\hat{y}_i = \beta x_i + \hat{\epsilon}
$$

The dependent difference is defined as:

$$
\Delta Y_i = \left| \hat{y}_i - y_i \right|
$$

Let the collection of dependent differences be:

$$
C(\Delta Y) = \{\Delta Y_1, \Delta Y_2, \dots, \Delta Y_n\}
$$

The mean dependent difference is:

$$
\overline{\Delta Y} = \frac{1}{n} \sum_{i=1}^{n} \Delta Y_i
$$

The dispersion of dependent differences is measured using the standard deviation:

$$
s = \sqrt{
\frac{\sum_{i=1}^{n} \left( \Delta Y_i - \overline{\Delta Y} \right)^2}{n - 1}
}
$$

The Normalized Least Dependent Difference (NLDD) is defined as:

$$
s_{\mathrm{NLDD}} = \frac{s}{\max(y) - \min(y)}
$$

A value of $$\( s_{\mathrm{NLDD}} \)$$ close to 0 indicates strong linearity and homoskedasticity, while larger values reflect increasing structural deviation, variance instability, or the presence of outliers.

---

## Methodological Insight

NLDD is grounded in the analysis of **dependent differences**, defined as the absolute deviation between observed values and their corresponding predictions from a fitted linear regression model.

Rather than directly measuring correlation or variance independently, NLDD evaluates the **dispersion of these dependent differences**, capturing:

- Structural deviation from linearity  
- Variability in residual distribution  
- Irregular deviations induced by outliers  

The dispersion is quantified using the standard deviation of dependent differences and subsequently normalized using the response range $$\((\max(y) - \min(y))\)$$ to ensure comparability across datasets.

This formulation unifies error magnitude (MAE), dispersion (MAD), and variance behavior into a single diagnostic measure, enabling NLDD to function as a **comprehensive indicator of regression validity** rather than a single-purpose statistic.

---

## Diagnostic Scope

NLDD is designed to provide a unified assessment across three key aspects of data behavior:

- **Linearity** – Evaluates the degree of structural conformity between variables  
- **Heteroskedasticity** – Captures variability patterns in residual dispersion  
- **Outliers** – Identifies and mitigates the influence of anomalous observations  

By integrating these components, NLDD offers a consolidated perspective that reduces fragmentation in traditional diagnostic workflows.

---

## Diagnostic Interpretation

NLDD provides a unified interpretation of regression validity:

- **NLDD ≈ 0** → Strong linearity and homoskedasticity  
- **Moderate NLDD** → Mild non-linearity or variance instability  
- **High NLDD** → Significant deviation, heteroskedasticity, or outliers  

This interpretation enables direct and intuitive assessment without requiring multiple statistical tests.

---

## Experimental Evaluation

The repository includes both **synthetic datasets** and an **applied real-world case study**. The synthetic datasets are used for controlled evaluation of linearity, heteroskedasticity, and outlier behavior, with each synthetic dataset containing \(n=120\) observations. The real-world case study uses the Breast Cancer Wisconsin Diagnostic dataset for applied regression-structure assessment.

The repository includes comparative experiments against the following methods:

**Linearity Measures:**
- Pearson Correlation Coefficient (PCC)
- Spearman's Rank Correlation Coefficient (SRCC)
- Kendall’s Tau Correlation Coefficient (KTCC)
- Weighted Pearson Correlation Coefficient (WPCC)
- Blest’s Rank Correlation Coefficient (BRC)

**Heteroskedasticity Tests:**
- Breusch–Pagan Test (BPT)
- White’s Test (WT)
- Goldfeld–Quandt Test (GQT)

**Outlier Tests:**
- Z-Score
- Interquartile Range (IQR)
- Local Outlier Factor (LOF)
- Random Sample Consensus Regression (RANSAC)
- Theil–Sen Regression with residual-based detection

The experiments include varying data conditions, including:

- Linear vs non-linear relationships  
- Homoskedastic vs heteroskedastic noise structures  
- Presence of outliers
- A real-world biomedical case study using the Breast Cancer Wisconsin Diagnostic dataset

---

## Repository Structure

```text
nldd-algo/
├── nldd.py              # Core NLDD implementation
├── dataset/             # Synthetic and benchmark datasets
├── result_graph/        # Plotting, results, and visualization
│   ├── linearity/       # PCC, SRCC, KTCC, WPCC, BPC, NLDD comparisons
│   ├── hetero/          # WT, BPT, GQT, NLDD comparisons
│   └── outlier/         # IQR, LOF, RANSAC, Theil-Sen, NLDD vs ground truth (GT)
├── README.md            # Project documentation
├── CITATION.cff         # Citation metadata for GitHub and Zenodo
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

## Citation

If you use the Normalized Least Dependent Difference (NLDD) algorithm or this repository in your research, please cite the archived Zenodo release:

```bibtex
@software{nldd2026,
  title     = {{Normalized Least Dependent Difference (NLDD)}},
  author    = {Herrick Han Lin Yeap and Le Ying Lim and Brandon Chen Hong Chow and Kok Seng Eu and Zhengying Ho and Kian Meng Yap},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21203715},
  url       = {https://doi.org/10.5281/zenodo.21203715}
}
```

The source code is available at:

[https://github.com/lylim0417/nldd-algo](https://github.com/lylim0417/nldd-algo)

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
