# TMU Research Assistant 
# Prioritizing Road Safety Interventions: A Robust Multi-Model Ranking of Fatal Collision Risk Factors Using Canadian National Data
## Overview

This repository contains the implementation and analysis from my **Research Assistant** for the **Ted Rogers School of Management** at Toronto Metropolitan University. The project focuses on identifying and prioritizing **fatal road collision risk factors** using a robust, multi-model, and multi-year machine learning framework.

A key challenge in road safety analytics is that different machine learning models often produce **inconsistent feature-importance rankings,** even when trained on the same data. This project addresses that challenge by developing and evaluating **consensus-based ranking methods** that integrate information across multiple models and reporting years. The goal is to provide **stable, interpretable, and policy-relevant insights** to support evidence-based road safety interventions.

## Objectives

- Identify collision-related risk factors that are **consistently associated with fatal outcomes** across models and time.
- Reduce reliance on single-model feature-importance interpretations.
- Develop a **consensus ranking framework** that accounts for model-specific bias and temporal heterogeneity.
- Compare multiple consensus-ranking approaches and evaluate their agreement and stability.
- Provide interpretable results that can support **road safety policy and intervention prioritization.**

## Methodology
- **Data Source:** Analyzed eight years (2014–2021) of person-level data from the **Canadian National Collision Database (NCDB).**
- **Model Selection:** Trained multiple supervised models including **Random Forest, XGBoost, and Neural Networks** to predict fatal injury outcomes.
- **Feature Importance Extraction:** Gini importance for Random Forest, Gain-based importance for XGBoost, and SHAP values for Neural Networks
- **Consensus Ranking Frameworks:** **Eigenvectors-Weighted Consensus Ranking (EWCR)** based on PCA, Non-negative PCA (NPCA) extension for interpretability, **TOPSIS-based** multi-criteria ranking, and **NMF-based** latent structure consensus ranking
- **Robustness Analysis:** Evaluated rank concordance, rank-shift distributions, sparsity-controlled selection, and cross-method stability.
- **Interpretability Focus:** Distinguished **core risk factors** from secondary or unstable predictors based on persistence across methods and years.

## Key Results
- Identified a **stable core set of fatal collision risk factors** that remain highly ranked across models and reporting years.
- Demonstrated that consensus-based approaches provide **more reliable and defensible prioritization** than single-model feature importance.
- PCA-based EWCR preserved global ranking structure while down-weighting noisy or model-specific effects.
- TOPSIS and NMF revealed complementary insights into mid-ranked and latent factor structures.
- Sparsity-controlled analysis successfully separated **major risk factors** from secondary influences.
- The proposed framework improves transparency and confidence in **ML-driven road safety decision support.**

## Dataset

**Source:** Canadian National Collision Database (NCDB)
**Coverage:** 2014–2021

The NCDB contains police-reported collision data from all Canadian provinces and territories, including detailed information on crash circumstances, vehicles, roadway conditions, and injury severity. The dataset presents real-world challenges such as high dimensionality, temporal heterogeneity, and class imbalance, making it well-suited for robust machine learning analysis.

## Contact

**Author:** Nguyen Duy Anh Luong  
**Supervisor:** Dr. Shengkun Xie  
**Email:** [nguyenduyanh.luong@torontomu.ca]
