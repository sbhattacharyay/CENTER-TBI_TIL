# Clinimetric assessment of the Therapy Intensity Level (TIL) scale for traumatic brain injury
The Therapy Intensity Level scale for traumatic brain injury: clinimetric assessment on neuro-monitored patients across 52 European intensive care units

## Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Code](#code)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

This repository contains the code underlying the article entitled **The Therapy Intensity Level scale for traumatic brain injury: clinimetric assessment on neuro-monitored patients across 52 European intensive care units** from the Collaborative European NeuroTrauma Effectiveness Research in TBI ([CENTER-TBI](https://www.center-tbi.eu/)) consortium. In this file, we present the abstract, to outline the motivation for the work and the findings, and then a brief description of the code with which we generate these finding and achieve this objective.\
\
The code on this repository is commented throughout to provide a description of each step alongside the code which achieves it.

## Abstract
The Therapy Intensity Level (TIL) scale and its abridged version (TIL<sup>(Basic)</sup>) are used to record the intensity of daily management for raised intracranial pressure (ICP) after traumatic brain injury (TBI). However, it is uncertain: (1) whether TIL is valid across the wide variation in modern ICP treatment strategies, (2) if TIL performs better than its predecessors, (3) how TIL's component therapies contribute to the overall score, and (4) whether TIL<sup>(Basic)</sup> may capture sufficient information. We aimed to answer these questions by assessing TIL on a contemporary population of ICP-monitored TBI patients (*n*=873) in 52 intensive care units (ICUs) across 18 European countries and Israel. From the observational, prospective Collaborative European NeuroTrauma Effectiveness Research in TBI (CENTER-TBI) study, we extracted first-week daily TIL scores (TIL<sub>24</sub>), ICP values, physician-based impressions of aberrant ICP, clinical markers of injury severity, and six-month functional outcome scores. We evaluated the construct and criterion validity of TIL against that of its predecessors, an unweighted version of TIL, and TIL<sup>(Basic)</sup>. We calculated the median score of each TIL component therapy for each total score as well as associations between each component score and markers of injury severity. Moreover, we calculated the information coverage of TIL by TIL<sup>(Basic)</sup>,defined by the mutual information of TIL and TIL<sup>(Basic)</sup> divided by the entropy of TIL. The statistical validity measures of TIL were significantly greater or similar to those of alternative scales, and TIL integrated the widest range of modern ICP treatments. First-week median TIL<sub>24</sub> (TIL<sub>median</sub>) outperformed first-week maximum TIL<sub>24</sub> (TIL<sub>max</sub>) in discriminating refractory intracranial hypertension (RIC) during ICU stay, and the thresholds which maximised the sum of sensitivity and specificity for RIC detection were TIL<sub>median</sub>≥7.5 (sensitivity: 81% [95% CI: 77-87%], specificity: 72% [95% CI: 70-75%]) and TIL<sub>max</sub>≥14 (sensitivity: 68% [95% CI: 62-74%], specificity: 79% [95% CI: 77-81%]). The sensitivity-specificity-optimising TIL<sub>24</sub> threshold for detecting surgical ICP control was TIL<sub>24</sub>≥9 (sensitivity: 87% [95% CI: 83-91%], specificity: 74% [95% CI: 72-76%]). The median component scores for each TIL<sub>24</sub> reflected a credible staircase approach to treatment intensity escalation, from head positioning to surgical ICP control, as well as considerable variability in the use of cerebrospinal fluid drainage and decompressive craniectomy. First-week maximum TIL<sup>(Basic)</sup> (TIL<sup>(Basic)</sup><sub>max</sub>) suffered from a strong ceiling effect and could not replace TIL<sub>max</sub>. TIL<sup>(Basic)</sup><sub>24</sub> and first-week median TIL<sup>(Basic)</sup> (TIL<sup>(Basic)</sup><sub>median</sub>) could be a suitable replacement for TIL<sub>24</sub> and TIL<sub>median</sub>, respectively (up to 33% [95% CI: 31-35%] information coverage). Numerical ranges were derived for categorising TIL<sub>24</sub> scores into TIL<sup>(Basic)</sup><sub>24</sub> scores. Our results validate the TIL scale across a spectrum of ICP management and monitoring approaches and support its use as a surrogate outcome after TBI.

## Code 
All of the code used in this work can be found in the `./scripts` directory as Python (`.py`), R (`.R`), or bash (`.sh`) scripts. Moreover, custom functions have been saved in the `./scripts/functions` sub-directory.

### 1. Extract, prepare, and characterise CENTER-TBI variables used for study

<ol type="a">
  <li><h4><a href="scripts/01a_prepare_study_sample.py">Extract and prepare study sample covariates from CENTER-TBI dataset</a></h4> In this <code>.py</code> file, we extract all appropriate covariates from the CENTER-TBI dataset and calculate summarised metrics.</li>
  <li><h4><a href="scripts/01b_calculate_summary_stats.py">Calculate summary statistics and missingness of different study sub-samples</a></h4> In this <code>.py</code> file, we characterise the study dataset by calculating summary statistics and assessing variable missingness. </li>
  <li><h4><a href="scripts/01c_missing_value_imputation.R">Perform multiple imputation of missing study values before analysis</a></h4> In this <code>.R</code> file, we create 100 stochastically imputed sets of our study variables to account for the uncertainty due to variable missingness. Static variables were imputed with the <code>mice</code> package while longitudinal variables were imputed with the <code>Amelia II</code> package </li>
</ol>

### 2. Calculate study statistics and confidence intervals with bootstrapping

<ol type="a">
  <li><h4><a href="scripts/02a_prepare_for_TIL_stats_bootstrapping.py">Prepare study resamples for bootstrapping TIL-based statistics</a></h4> In this <code>.py</code> file, we draw resamples for statistical bootstrapping per validation population.</li>
  <li><h4><a href="scripts/02b_calculate_correlations_and_stats.py">Calculate TIL correlations and statistics of different study sub-samples</a></h4> In this <code>.py</code> file, we calculate study statistics in each bootstrapping resample. This is run, with multi-array indexing, on the HPC using a <a href="scripts/02b_calculate_correlations_and_stats.sh">bash script</a>.</li>
  <li><h4><a href="scripts/02c_compile_correlations_and_stats.py">Compile TIL correlations and statistics of different study sub-samples</a></h4> In this <code>.py</code> file, we compile the study statistics calculated across bootstrapping resamples and calculate 95% confidence intervals for statistical inference. </li>
</ol>

### 3. [Visualise study results for manuscript](scripts/03_manuscript_visualisations.R)
In this `.R` file, we produce the figures for the manuscript and the supplementary figures. The quantitative figures in the manuscript are produced using the `ggplot` package.

## Citation
```
```
