# CENTER-TBI_TIL
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

### 1. Tokenise all CENTER-TBI variables and place into discretised ICU stay time windows

<ol type="a">
  <li><h4><a href="scripts/01a_prepare_study_sample.py">Format CENTER-TBI variables for tokenisation</a></h4> In this <code>.py</code> file, we extract all heterogeneous types of variables from CENTER-TBI and fix erroneous timestamps and formats.</li>
  <li><h4><a href="scripts/01b_calculate_summary_stats.py">Convert full patient records over ICU stays into tokenised time windows</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/01c_missing_value_imputation.R">Convert full patient records over ICU stays into tokenised time windows</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
</ol>

In this `.py` file, we extract the study sample from the CENTER-TBI dataset, filter patients by our study criteria, and determine ICU admission and discharge times for time window discretisation. We also perform proportional odds logistic regression analysis to determine significant effects among summary characteristics.

### 2. Partition CENTER-TBI for stratified, repeated k-fold cross-validation

<ol type="a">
  <li><h4><a href="scripts/02a_prepare_for_TIL_stats_bootstrapping.py">Format CENTER-TBI variables for tokenisation</a></h4> In this <code>.py</code> file, we extract all heterogeneous types of variables from CENTER-TBI and fix erroneous timestamps and formats.</li>
  <li><h4><a href="scripts/02b_calculate_correlations_and_stats.py">Convert full patient records over ICU stays into tokenised time windows</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/02c_compile_correlations_and_stats.py">Convert full patient records over ICU stays into tokenised time windows</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
</ol>

In this `.py` file, we create 100 partitions, stratified by 6-month GOSE, for repeated k-fold cross-validation, and save the splits into a dataframe for subsequent scripts.

### 4. Train and evaluate full-context ordinal-trajectory-generating models

<ol type="a">
  <li><h4><a href="scripts/04a_train_full_set_models.py">Train full-context trajectory-generating models</a></h4> In this <code>.py</code> file, we train the trajectory-generating models across the repeated cross-validation splits and the hyperparameter configurations. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04a_train_full_set_models.sh">bash script</a>.</li>
  <li><h4><a href="scripts/04b_compile_full_set_model_predictions.py">Compile generated trajectories across repeated cross-validation and different hyperparameter configurations</a></h4> In this <code>.py</code> file, we compile the training, validation, and testing set trajectories generated by the models and creates bootstrapping resamples for validation set dropout.</li>
  <li><h4><a href="scripts/04c_validation_set_bootstrapping_for_dropout.py">Calculate validation set calibration and discrimination of generated trajectories for hyperparameter configuration dropout</a></h4> In this <code>.py</code> file, we calculate validation set trajectory calibration and discrimination based on provided bootstrapping resample row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04c_validation_set_bootstrapping_for_dropout.sh">bash script</a>.</li>
  <li><h4><a href="scripts/04d_dropout_configurations.py">Compile validation set performance metrics and dropout under-performing hyperparameter configurations</a></h4> In this <code>.py</code> file, we compiled the validation set performance metrics and perform bias-corrected bootstrapping dropout for cross-validation (BBCD-CV) to reduce the number of hyperparameter configurations. We also create testing set resamples for final performance calculation bootstrapping. </li>
  <li><h4><a href="scripts/04e_test_set_performance.py">Calculate calibration and discrimination performance metrics of generated trajectories of the testing set with bootstrapping</a></h4> In this <code>.py</code> file, calculate the model calibration and explanation metrics to assess model reliability and information, respectively. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04e_test_set_performance.sh">bash script</a>.</li>
  <li><h4><a href="scripts/04f_test_set_confidence_intervals.py">Compile testing set trajectory performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile the performance metrics and summarise them across bootstrapping resamples to define the 95% confidence intervals for statistical inference. </li>
</ol>

### 3. [Visualise study results for manuscript](scripts/03_manuscript_visualisations.R)
In this `.R` file, we produce the figures for the manuscript and the supplementary figures. The quantitative figures in the manuscript are produced using the `ggplot` package.

## Citation
```
```
