#### Master Script 2d: Perform analyses to quantify information content stored across TIL scales ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate Shannon's Entropy and mutual information between TILBasic and other TIL scores
# III. Calculate cutoffs for each TIL scale to designate refractory IC hypertension status
# IV. Calculate cutoffs for each TIL scale to map onto TILBasic
# V. Calculate detection capabilities of daily TIL for 4-hourly TIL

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import random
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as cp
import pingouin as pg
from tqdm import tqdm
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.utils import resample
from sklearn.metrics import mutual_info_score, roc_curve, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.feature_selection._mutual_info import mutual_info_regression, _estimate_mi

# Custom methods
from functions.analysis import calc_MI_entropy, calc_AUC, calc_MCC_accuracy

# Set number of resamples for bootstrapping-based inference
NUM_RESAMP = 1000

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Calculate Shannon's Entropy and mutual information between TILBasic and other TIL scores
## Load different TIL scale dataframes
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')
formatted_TIL_max = pd.read_csv('../formatted_data/formatted_TIL_max.csv')
formatted_TIL_mean = pd.read_csv('../formatted_data/formatted_TIL_mean.csv')

# Load formatted TILBasic scores
formatted_TIL_Basic_scores = pd.read_csv('../formatted_data/formatted_TIL_Basic_scores.csv')
formatted_TIL_Basic_max = pd.read_csv('../formatted_data/formatted_TIL_Basic_max.csv')
formatted_TIL_Basic_mean = pd.read_csv('../formatted_data/formatted_TIL_Basic_mean.csv')

# Load unweighted TIL scores
formatted_unweighted_TIL_scores = pd.read_csv('../formatted_data/formatted_unweighted_TIL_scores.csv')
formatted_unweighted_TIL_scores['uwTILSum'] = formatted_unweighted_TIL_scores.CSFDrainage + formatted_unweighted_TIL_scores.DecomCraniectomy + formatted_unweighted_TIL_scores.FluidLoading + formatted_unweighted_TIL_scores.Hypertonic + formatted_unweighted_TIL_scores.ICPSurgery + formatted_unweighted_TIL_scores.Mannitol + formatted_unweighted_TIL_scores.Neuromuscular + formatted_unweighted_TIL_scores.Positioning + formatted_unweighted_TIL_scores.Sedation + formatted_unweighted_TIL_scores.Temperature + formatted_unweighted_TIL_scores.Vasopressor + formatted_unweighted_TIL_scores.Ventilation
formatted_unweighted_TIL_scores = formatted_unweighted_TIL_scores[formatted_unweighted_TIL_scores.TILTimepoint<=7].reset_index(drop=True)
formatted_unweighted_TIL_max = formatted_unweighted_TIL_scores.loc[formatted_unweighted_TIL_scores.groupby(['GUPI'])['uwTILSum'].idxmax()].reset_index(drop=True).rename(columns={'uwTILSum':'uwTILmax'})
formatted_unweighted_TIL_mean = pd.pivot_table(formatted_unweighted_TIL_scores.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'uwTILSum':'uwTILmean'})

# Load PILOT scores
formatted_PILOT_scores = pd.read_csv('../formatted_data/formatted_PILOT_scores.csv')
formatted_PILOT_max = pd.read_csv('../formatted_data/formatted_PILOT_max.csv')
formatted_PILOT_mean = pd.read_csv('../formatted_data/formatted_PILOT_mean.csv')

# Load TIL_1987 scores
formatted_TIL_1987_scores = pd.read_csv('../formatted_data/formatted_TIL_1987_scores.csv')
formatted_TIL_1987_max = pd.read_csv('../formatted_data/formatted_TIL_1987_max.csv')
formatted_TIL_1987_mean = pd.read_csv('../formatted_data/formatted_TIL_1987_mean.csv')

# Merge relevent columns of daily TIL scores onto single dataframe
merged_scores_df = formatted_TIL_Basic_scores.merge(formatted_unweighted_TIL_scores[['GUPI','TILTimepoint','TILDate','uwTILSum']],how='left').merge(formatted_PILOT_scores[['GUPI','TILTimepoint','TILDate','PILOTSum']],how='left').merge(formatted_TIL_1987_scores[['GUPI','TILTimepoint','TILDate','TIL_1987Sum']],how='left')
merged_scores_df = merged_scores_df[merged_scores_df.TILTimepoint<=7].reset_index(drop=True)

# Merge relevent columns of max TIL scores onto single dataframe
merged_max_df = formatted_TIL_max[['GUPI','TILmax']].merge(formatted_unweighted_TIL_max[['GUPI','uwTILmax']],how='left').merge(formatted_TIL_Basic_max[['GUPI','TIL_Basicmax']],how='left').merge(formatted_PILOT_max[['GUPI','PILOTmax']],how='left').merge(formatted_TIL_1987_max[['GUPI','TIL_1987max']],how='left')

# Merge relevent columns of mean TIL scores onto single dataframe
merged_mean_df = formatted_TIL_mean[['GUPI','TILmean']].merge(formatted_unweighted_TIL_mean[['GUPI','uwTILmean']],how='left').merge(formatted_TIL_Basic_mean[['GUPI','TIL_Basicmean']],how='left').merge(formatted_PILOT_mean[['GUPI','PILOTmean']],how='left').merge(formatted_TIL_1987_mean[['GUPI','TIL_1987mean']],how='left')

## Calculate mutual information and entropy in parallel
# Create bootstrapping resamples
bs_resamples = [np.unique(resample(merged_scores_df.GUPI.unique(),replace=True,n_samples=len(merged_scores_df.GUPI.unique()))) for _ in range(NUM_RESAMP)]

# Partition dataframes into different cores
s = [NUM_RESAMP // NUM_CORES for _ in range(NUM_CORES)]
s[:(NUM_RESAMP - sum(s))] = [over+1 for over in s[:(NUM_RESAMP - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
resamples_per_core = [(bs_resamples[start_idx[idx]:end_idx[idx]],merged_scores_df,merged_max_df,True,'Calculating mutual information in parallel') for idx in range(len(start_idx))]

# Calculate boostrapped mutual information and entropy in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_MI_entropy = pd.concat(pool.starmap(calc_MI_entropy, resamples_per_core),ignore_index=True)

# Save compiled mutual information and entropy values
compiled_MI_entropy.to_csv('../bootstrapping_results/compiled_MI_entropy_results.csv',index=False)

# Calculate and format 95% confidence intervals
CI_MI_entropy = compiled_MI_entropy.melt(id_vars=['TILTimepoint','METRIC'],var_name='VARIABLE').groupby(['TILTimepoint','METRIC','VARIABLE'],as_index=False)['value'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)

# Add formatted confidence interval
CI_MI_entropy['FormattedCI'] = CI_MI_entropy['median'].round(2).astype(str)+' ('+CI_MI_entropy.lo.round(2).astype(str)+'–'+CI_MI_entropy.hi.round(2).astype(str)+')'

# Save formatted confidence intervals
CI_MI_entropy.to_csv('../bootstrapping_results/CI_MI_entropy_results.csv',index=False)

### III. Calculate cutoffs for each TIL scale to designate refractory IC hypertension status
## Load different max TIL scale dataframes
# Load TIL_max
formatted_TIL_max = pd.read_csv('../formatted_data/formatted_TIL_max.csv')

# Load TIL_Basic_max
formatted_TIL_Basic_max = pd.read_csv('../formatted_data/formatted_TIL_Basic_max.csv')

# Load PILOT_max
formatted_PILOT_max = pd.read_csv('../formatted_data/formatted_PILOT_max.csv')

# Load TIL_1987_max
formatted_TIL_1987_max = pd.read_csv('../formatted_data/formatted_TIL_1987_max.csv')

# Load ICP_max
formatted_ICP_EH_max = pd.read_csv('../formatted_data/formatted_low_resolution_maxes_means.csv')[['GUPI','ICPmax']].rename(columns={'ICPmax':'ICPmax_EH'})
formatted_ICP_HR_max = pd.read_csv('../formatted_data/formatted_high_resolution_maxes_means.csv')[['GUPI','ICPmax']].rename(columns={'ICPmax':'ICPmax_HR'})

# Load and calculate unweighted TIL max
formatted_unweighted_TIL_scores = pd.read_csv('../formatted_data/formatted_unweighted_TIL_scores.csv')
formatted_unweighted_TIL_scores['uwTILSum'] = formatted_unweighted_TIL_scores.CSFDrainage + formatted_unweighted_TIL_scores.DecomCraniectomy + formatted_unweighted_TIL_scores.FluidLoading + formatted_unweighted_TIL_scores.Hypertonic + formatted_unweighted_TIL_scores.ICPSurgery + formatted_unweighted_TIL_scores.Mannitol + formatted_unweighted_TIL_scores.Neuromuscular + formatted_unweighted_TIL_scores.Positioning + formatted_unweighted_TIL_scores.Sedation + formatted_unweighted_TIL_scores.Temperature + formatted_unweighted_TIL_scores.Vasopressor + formatted_unweighted_TIL_scores.Ventilation
formatted_unweighted_TIL_scores = formatted_unweighted_TIL_scores[formatted_unweighted_TIL_scores.TILTimepoint<=7].reset_index(drop=True)
formatted_unweighted_TIL_max = formatted_unweighted_TIL_scores.loc[formatted_unweighted_TIL_scores.groupby(['GUPI'])['uwTILSum'].idxmax()].reset_index(drop=True).rename(columns={'uwTILSum':'uwTILmax'})

# Compile max TIL scores
merged_max_df = formatted_TIL_max[['GUPI','TILmax']].merge(formatted_unweighted_TIL_max[['GUPI','uwTILmax']],how='left').merge(formatted_TIL_Basic_max[['GUPI','TIL_Basicmax']],how='left').merge(formatted_PILOT_max[['GUPI','PILOTmax']],how='left').merge(formatted_TIL_1987_max[['GUPI','TIL_1987max']],how='left').merge(formatted_ICP_EH_max[['GUPI','ICPmax_EH']],how='left').merge(formatted_ICP_HR_max[['GUPI','ICPmax_HR']],how='left')

## Load and merge refractory ICP status
# Load demographic information dataframe with refractory ICP markers
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Merge refractory intracranial hypertension status to max TIL dataframe
merged_max_df = merged_max_df.merge(CENTER_TBI_demo_outcome[['GUPI','RefractoryICP']],how='left').dropna(subset='RefractoryICP')

## Calculate ROC curves at different thresholds
# Calculate ROC curves with thresholds for each TILmax rating
TILmax_fpr, TILmax_tpr, TILmax_thresholds = roc_curve(merged_max_df.RefractoryICP,merged_max_df.TILmax)
TIL_Basicmax_fpr, TIL_Basicmax_tpr, TIL_Basicmax_thresholds = roc_curve(merged_max_df.RefractoryICP,merged_max_df.TIL_Basicmax)
uwTILmax_fpr, uwTILmax_tpr, uwTILmax_thresholds = roc_curve(merged_max_df.RefractoryICP,merged_max_df.uwTILmax)
PILOTmax_fpr, PILOTmax_tpr, PILOTmax_thresholds = roc_curve(merged_max_df.RefractoryICP,merged_max_df.PILOTmax)
TIL_1987max_fpr, TIL_1987max_tpr, TIL_1987max_thresholds = roc_curve(merged_max_df.RefractoryICP,merged_max_df.TIL_1987max)
ICPmax_EH_fpr, ICPmax_EH_tpr, ICPmax_EH_thresholds = roc_curve(merged_max_df[~merged_max_df.ICPmax_EH.isna()].RefractoryICP,merged_max_df[~merged_max_df.ICPmax_EH.isna()].ICPmax_EH)
ICPmax_HR_fpr, ICPmax_HR_tpr, ICPmax_HR_thresholds = roc_curve(merged_max_df[~merged_max_df.ICPmax_HR.isna()].RefractoryICP,merged_max_df[~merged_max_df.ICPmax_HR.isna()].ICPmax_HR)

# Compile curve information into single dataframe
TILmax_curve_df = pd.DataFrame({'Scale':'TILmax','FPR':TILmax_fpr,'TPR':TILmax_tpr,'Threshold':TILmax_thresholds})
TIL_Basicmax_curve_df = pd.DataFrame({'Scale':'TIL_Basicmax','FPR':TIL_Basicmax_fpr,'TPR':TIL_Basicmax_tpr,'Threshold':TIL_Basicmax_thresholds})
uwTILmax_curve_df = pd.DataFrame({'Scale':'uwTILmax','FPR':uwTILmax_fpr,'TPR':uwTILmax_tpr,'Threshold':uwTILmax_thresholds})
PILOTmax_curve_df = pd.DataFrame({'Scale':'PILOTmax','FPR':PILOTmax_fpr,'TPR':PILOTmax_tpr,'Threshold':PILOTmax_thresholds})
TIL_1987max_curve_df = pd.DataFrame({'Scale':'TIL_1987max','FPR':TIL_1987max_fpr,'TPR':TIL_1987max_tpr,'Threshold':TIL_1987max_thresholds})
ICPmax_EH_curve_df = pd.DataFrame({'Scale':'ICPmax_EH','FPR':ICPmax_EH_fpr,'TPR':ICPmax_EH_tpr,'Threshold':ICPmax_EH_thresholds})
ICPmax_HR_curve_df = pd.DataFrame({'Scale':'ICPmax_HR','FPR':ICPmax_HR_fpr,'TPR':ICPmax_HR_tpr,'Threshold':ICPmax_HR_thresholds})
compiled_curve_df = pd.concat([TILmax_curve_df,TIL_Basicmax_curve_df,uwTILmax_curve_df,PILOTmax_curve_df,TIL_1987max_curve_df,ICPmax_EH_curve_df,ICPmax_HR_curve_df],ignore_index=True)

# Save compiled ROC dataframe
compiled_curve_df.to_csv('../bootstrapping_results/compiled_ROC_refractory_results.csv',index=False)

## Calculate AUC values in parallel to form confidence intervals
# Create bootstrapping resamples
bs_resamples = [np.unique(resample(merged_max_df.GUPI.unique(),replace=True,n_samples=len(merged_max_df.GUPI.unique()))) for _ in range(NUM_RESAMP)]

# Partition dataframes into different cores
s = [NUM_RESAMP // NUM_CORES for _ in range(NUM_CORES)]
s[:(NUM_RESAMP - sum(s))] = [over+1 for over in s[:(NUM_RESAMP - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
resamples_per_core = [(bs_resamples[start_idx[idx]:end_idx[idx]],merged_max_df,['TILmax','uwTILmax','TIL_Basicmax','PILOTmax','TIL_1987max'],['RefractoryICP'],True,'Calculating AUC in parallel') for idx in range(len(start_idx))]

# Calculate boostrapped AUC in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_AUC = pd.concat(pool.starmap(calc_AUC, resamples_per_core),ignore_index=True)

# Create bootstrapping resamples for ICP_EH and ICP_HR
EH_bs_resamples = [np.unique(resample(merged_max_df[~merged_max_df.ICPmax_EH.isna()].GUPI.unique(),replace=True,n_samples=len(merged_max_df[~merged_max_df.ICPmax_EH.isna()].GUPI.unique()))) for _ in range(NUM_RESAMP)]
HR_bs_resamples = [np.unique(resample(merged_max_df[~merged_max_df.ICPmax_HR.isna()].GUPI.unique(),replace=True,n_samples=len(merged_max_df[~merged_max_df.ICPmax_HR.isna()].GUPI.unique()))) for _ in range(NUM_RESAMP)]

# Partition dataframes into different cores
EH_resamples_per_core = [(EH_bs_resamples[start_idx[idx]:end_idx[idx]],merged_max_df,['ICPmax_EH'],['RefractoryICP'],True,'Calculating ICP_EH AUC in parallel') for idx in range(len(start_idx))]
HR_resamples_per_core = [(HR_bs_resamples[start_idx[idx]:end_idx[idx]],merged_max_df,['ICPmax_HR'],['RefractoryICP'],True,'Calculating ICP_HR AUC in parallel') for idx in range(len(start_idx))]

# Calculate boostrapped AUC for ICP_EH and ICP_HR in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_EH_AUC = pd.concat(pool.starmap(calc_AUC, EH_resamples_per_core),ignore_index=True)

with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_HR_AUC = pd.concat(pool.starmap(calc_AUC, HR_resamples_per_core),ignore_index=True)

# Concatenate all AUC values
compiled_AUC = pd.concat([compiled_AUC,compiled_EH_AUC,compiled_HR_AUC],ignore_index=True)

# Save compiled AUC values
compiled_AUC.to_csv('../bootstrapping_results/compiled_AUC_refractory_results.csv',index=False)

# Calculate and format 95% confidence intervals
CI_AUC = compiled_AUC.groupby(['Scale','Label'],as_index=False)['AUC'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)

# Add formatted confidence interval
CI_AUC['FormattedCI'] = CI_AUC['median'].round(2).astype(str)+' ('+CI_AUC.lo.round(2).astype(str)+'–'+CI_AUC.hi.round(2).astype(str)+')'

# Save formatted confidence intervals
CI_AUC.to_csv('../bootstrapping_results/CI_AUC_refractory_results.csv',index=False)

### IV. Calculate cutoffs for each TIL scale to map onto TILBasic
## Load different TIL scale dataframes
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Load formatted TILBasic scores
formatted_TIL_Basic_scores = pd.read_csv('../formatted_data/formatted_TIL_Basic_scores.csv')

# Load unweighted TIL scores
formatted_unweighted_TIL_scores = pd.read_csv('../formatted_data/formatted_unweighted_TIL_scores.csv')
formatted_unweighted_TIL_scores['uwTILSum'] = formatted_unweighted_TIL_scores.CSFDrainage + formatted_unweighted_TIL_scores.DecomCraniectomy + formatted_unweighted_TIL_scores.FluidLoading + formatted_unweighted_TIL_scores.Hypertonic + formatted_unweighted_TIL_scores.ICPSurgery + formatted_unweighted_TIL_scores.Mannitol + formatted_unweighted_TIL_scores.Neuromuscular + formatted_unweighted_TIL_scores.Positioning + formatted_unweighted_TIL_scores.Sedation + formatted_unweighted_TIL_scores.Temperature + formatted_unweighted_TIL_scores.Vasopressor + formatted_unweighted_TIL_scores.Ventilation

# Load PILOT scores
formatted_PILOT_scores = pd.read_csv('../formatted_data/formatted_PILOT_scores.csv')

# Load TIL_1987 scores
formatted_TIL_1987_scores = pd.read_csv('../formatted_data/formatted_TIL_1987_scores.csv')

# Merge relevent columns of daily TIL scores onto single dataframe
merged_scores_df = formatted_TIL_Basic_scores.merge(formatted_unweighted_TIL_scores[['GUPI','TILTimepoint','TILDate','uwTILSum']],how='left').merge(formatted_PILOT_scores[['GUPI','TILTimepoint','TILDate','PILOTSum']],how='left').merge(formatted_TIL_1987_scores[['GUPI','TILTimepoint','TILDate','TIL_1987Sum']],how='left')
merged_scores_df = merged_scores_df[merged_scores_df.TILTimepoint<=7].reset_index(drop=True)

## Create labels at each threshold of TIL_Basic
# Identify unique levels of TIL_Basic
uniq_TIL_Basic = np.sort(merged_scores_df.TIL_Basic.unique())

# Iterate through positive TIL_Basic values to add threshold columns
for curr_thresh in uniq_TIL_Basic[1:]:

    # Create new column to store TIL_Basic threshold labels
    merged_scores_df['TIL>='+str(int(curr_thresh))] = (merged_scores_df.TIL_Basic>=curr_thresh).astype(int)

## Calculate ROC curves at different thresholds of TIL_Basic
# Extract new TIL_Basic threshold columns
TIL_Basic_cols = [col for col in merged_scores_df if col.startswith('TIL>=')]

# Initialize empty list to store compiled curve dataframes
compiled_curve_df = []

# Iterate through TIL_Basic thresholds and calculate ROC curves
for curr_thresh_col in tqdm(TIL_Basic_cols,'Calculating ROC curves at each TIL_Basic threshold'):

    # Calculate ROC curves with thresholds for each TIL scale
    TIL_fpr, TIL_tpr, TIL_thresholds = roc_curve(merged_scores_df[curr_thresh_col],merged_scores_df.TotalSum)
    uwTIL_fpr, uwTIL_tpr, uwTIL_thresholds = roc_curve(merged_scores_df[curr_thresh_col],merged_scores_df.uwTILSum)
    PILOT_fpr, PILOT_tpr, PILOT_thresholds = roc_curve(merged_scores_df[curr_thresh_col],merged_scores_df.PILOTSum)
    TIL_1987_fpr, TIL_1987_tpr, TIL_1987_thresholds = roc_curve(merged_scores_df[curr_thresh_col],merged_scores_df.TIL_1987Sum)

    # Compile curve information into single dataframe
    TIL_curve_df = pd.DataFrame({'Scale':'TIL','FPR':TIL_fpr,'TPR':TIL_tpr,'Threshold':TIL_thresholds})
    uwTIL_curve_df = pd.DataFrame({'Scale':'uwTIL','FPR':uwTIL_fpr,'TPR':uwTIL_tpr,'Threshold':uwTIL_thresholds})
    PILOT_curve_df = pd.DataFrame({'Scale':'PILOT','FPR':PILOT_fpr,'TPR':PILOT_tpr,'Threshold':PILOT_thresholds})
    TIL_1987_curve_df = pd.DataFrame({'Scale':'TIL_1987','FPR':TIL_1987_fpr,'TPR':TIL_1987_tpr,'Threshold':TIL_1987_thresholds})
    curr_curve_df = pd.concat([TIL_curve_df,uwTIL_curve_df,PILOT_curve_df,TIL_1987_curve_df],ignore_index=True)
    curr_curve_df.insert(1,'Label',curr_thresh_col)
    compiled_curve_df.append(curr_curve_df)

# Concatenate running list of curves
compiled_curve_df = pd.concat(compiled_curve_df,ignore_index=True)

# Save compiled ROC dataframe
compiled_curve_df.to_csv('../bootstrapping_results/compiled_ROC_TILBasic_results.csv',index=False)

## Calculate AUC values in parallel to form confidence intervals
# Create bootstrapping resamples
bs_resamples = [np.unique(resample(merged_scores_df.GUPI.unique(),replace=True,n_samples=len(merged_scores_df.GUPI.unique()))) for _ in range(NUM_RESAMP)]

# Partition dataframes into different cores
s = [NUM_RESAMP // NUM_CORES for _ in range(NUM_CORES)]
s[:(NUM_RESAMP - sum(s))] = [over+1 for over in s[:(NUM_RESAMP - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
resamples_per_core = [(bs_resamples[start_idx[idx]:end_idx[idx]],merged_scores_df,['TotalSum','uwTILSum','PILOTSum','TIL_1987Sum'],TIL_Basic_cols,True,'Calculating AUC in parallel') for idx in range(len(start_idx))]

# Calculate boostrapped AUC in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_AUC = pd.concat(pool.starmap(calc_AUC, resamples_per_core),ignore_index=True)

# Save compiled AUC values
compiled_AUC.to_csv('../bootstrapping_results/compiled_AUC_TILBasic_results.csv',index=False)

# Calculate and format 95% confidence intervals
CI_AUC = compiled_AUC.groupby(['Scale','Label'],as_index=False)['AUC'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)

# Add formatted confidence interval
CI_AUC['FormattedCI'] = CI_AUC['median'].round(2).astype(str)+' ('+CI_AUC.lo.round(2).astype(str)+'–'+CI_AUC.hi.round(2).astype(str)+')'

# Save formatted confidence intervals
CI_AUC.to_csv('../bootstrapping_results/CI_AUC_TILBasic_results.csv',index=False)

### V. Calculate detection capabilities of daily TIL for 4-hourly TIL
## Load and format 4-hourly delta TIL scores
# Load 4-hourly delta TIL scores
formatted_delta_TIL_scores = pd.read_csv('../formatted_data/formatted_delta_TIL_scores.csv')

# Filter to focus on first week of ICU stay
formatted_delta_TIL_scores = formatted_delta_TIL_scores[formatted_delta_TIL_scores.TILTimepoint<=7].reset_index(drop=True)

# Convert HVTIL column to integer type
formatted_delta_TIL_scores.HVTIL = formatted_delta_TIL_scores.HVTIL.astype(int)

# Count the number of types of changes per patient's day in ICU
count_delta_TIL_scores = pd.pivot_table(formatted_delta_TIL_scores.groupby(['GUPI','TILTimepoint','TILDate','ChangeInTIL','HVTIL'],as_index=False).TotalSum.size(), values = 'size', index=['GUPI','TILTimepoint','TILDate','ChangeInTIL'], columns = 'HVTIL').reset_index().fillna(0)

# Change the column names appropriately
count_delta_TIL_scores = count_delta_TIL_scores.rename(columns={-1:'Decrease',0:'NoChange',1:'Increase'})

# Count number of assessments per patient's ICU day
count_delta_TIL_scores['TotalCount'] = count_delta_TIL_scores.Decrease.astype(int) + count_delta_TIL_scores.Increase.astype(int) + count_delta_TIL_scores.NoChange.astype(int)

# Count number of zero types per patient's ICU day
count_delta_TIL_scores['ZeroTypesCount'] = count_delta_TIL_scores.Decrease.eq(0).astype(int) + count_delta_TIL_scores.Increase.eq(0).astype(int)

# Add another column designating first change in TIL of the day
count_delta_TIL_scores = count_delta_TIL_scores.merge(formatted_delta_TIL_scores.groupby(['GUPI','TILTimepoint','TILDate','ChangeInTIL'],as_index=False).HVTIL.aggregate(lambda s: s[s.ne(0).idxmax()]).rename(columns={'HVTIL':'FirstChange'}),how='left')

# Add a column designating ambiguous rows
count_delta_TIL_scores['Ambiguous'] = ((count_delta_TIL_scores.ZeroTypesCount==0)&(count_delta_TIL_scores.FirstChange!=1)).astype(int)

# Add a column designating daily change in TIL
count_delta_TIL_scores['DailyIncreaseTIL'] = (count_delta_TIL_scores.ChangeInTIL>0).astype(int)

# Add a column designating 4-hourly change in TIL
count_delta_TIL_scores['4HourlyIncreaseTIL'] = ((count_delta_TIL_scores.Increase>0)&(count_delta_TIL_scores.Ambiguous == 0)).astype(int)

# Add a column designating daily change in TIL
count_delta_TIL_scores['DailyDecreaseTIL'] = (count_delta_TIL_scores.ChangeInTIL<0).astype(int)

# Add a column designating 4-hourly change in TIL
count_delta_TIL_scores['4HourlyDecreaseTIL'] = ((count_delta_TIL_scores.Decrease>0)&(count_delta_TIL_scores.Increase == 0)).astype(int)

# Save formatted dataframe comparing daily change in TIL with hourly change
count_delta_TIL_scores.to_csv('../bootstrapping_results/delta_TIL_results.csv',index=False)

## Calculate accuracy and Matthew's Correlation Coefficient values in parallel to form confidence intervals
# Create bootstrapping resamples
bs_resamples = [np.unique(resample(count_delta_TIL_scores[count_delta_TIL_scores.Ambiguous!=1].GUPI.unique(),replace=True,n_samples=len(count_delta_TIL_scores[count_delta_TIL_scores.Ambiguous!=1].GUPI.unique()))) for _ in range(NUM_RESAMP)]

# Partition dataframes into different cores
s = [NUM_RESAMP // NUM_CORES for _ in range(NUM_CORES)]
s[:(NUM_RESAMP - sum(s))] = [over+1 for over in s[:(NUM_RESAMP - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
resamples_per_core = [(bs_resamples[start_idx[idx]:end_idx[idx]],count_delta_TIL_scores,True,'Calculating accuracy and MCC in parallel') for idx in range(len(start_idx))]

# Calculate boostrapped MCC and accuracy in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_MCC_accuracy = pd.concat(pool.starmap(calc_MCC_accuracy, resamples_per_core),ignore_index=True)

# Save compiled MCC and accuracy values
compiled_MCC_accuracy.to_csv('../bootstrapping_results/compiled_MCC_accuracy_results.csv',index=False)

# Calculate and format 95% confidence intervals
CI_MCC_accuracy = compiled_MCC_accuracy.groupby(['TILChange','Metric'],as_index=False)['Value'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)

# Add formatted confidence interval
CI_MCC_accuracy['FormattedCI'] = CI_MCC_accuracy['median'].round(2).astype(str)+' ('+CI_MCC_accuracy.lo.round(2).astype(str)+'–'+CI_MCC_accuracy.hi.round(2).astype(str)+')'

# Save formatted confidence intervals
CI_MCC_accuracy.to_csv('../bootstrapping_results/CI_MCC_accuracy_results.csv',index=False)