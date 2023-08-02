## Scripts
All of the code used in this work can be found in this directory as Python (`.py`), R (`.R`), or bash (`.sh`) scripts. Moreover, custom functions have been saved in the `./functions` sub-directory.

### 1. Extract, prepare, and characterise CENTER-TBI variables used for study

<ol type="a">
  <li><h4><a href="01a_prepare_study_sample.py">Extract and prepare study sample covariates from CENTER-TBI dataset</a></h4> In this <code>.py</code> file, we extract all appropriate covariates from the CENTER-TBI dataset and calculate summarised metrics.</li>
  <li><h4><a href="01b_calculate_summary_stats.py">Calculate summary statistics and missingness of different study sub-samples</a></h4> In this <code>.py</code> file, we characterise the study dataset by calculating summary statistics and assessing variable missingness. </li>
  <li><h4><a href="01c_missing_value_imputation.R">Perform multiple imputation of missing study values before analysis</a></h4> In this <code>.R</code> file, we create 100 stochastically imputed sets of our study variables to account for the uncertainty due to variable missingness. Static variables were imputed with the <code>mice</code> package while longitudinal variables were imputed with the <code>Amelia II</code> package </li>
</ol>

### 2. Calculate study statistics and confidence intervals with bootstrapping

<ol type="a">
  <li><h4><a href="02a_prepare_for_TIL_stats_bootstrapping.py">Prepare study resamples for bootstrapping TIL-based statistics</a></h4> In this <code>.py</code> file, we draw resamples for statistical bootstrapping per validation population.</li>
  <li><h4><a href="02b_calculate_correlations_and_stats.py">Calculate TIL correlations and statistics of different study sub-samples</a></h4> In this <code>.py</code> file, we calculate study statistics in each bootstrapping resample. This is run, with multi-array indexing, on the HPC using a <a href="02b_calculate_correlations_and_stats.sh">bash script</a>.</li>
  <li><h4><a href="02c_compile_correlations_and_stats.py">Compile TIL correlations and statistics of different study sub-samples</a></h4> In this <code>.py</code> file, we compile the study statistics calculated across bootstrapping resamples and calculate 95% confidence intervals for statistical inference. </li>
</ol>

### 3. [Visualise study results for manuscript](03_manuscript_visualisations.R)
In this `.R` file, we produce the figures for the manuscript and the supplementary figures. The quantitative figures in the manuscript are produced using the `ggplot` package.
