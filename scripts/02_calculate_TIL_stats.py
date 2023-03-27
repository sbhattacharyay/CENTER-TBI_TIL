#### Master Script 2: Calculate TIL statistics of different study sub-samples ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and prepare TIL scores of full dataset and designate sub-study assignment
# III. Calculate population-level summary characteristics
# IV. Calculate population-level TIL correlations with overall characteristics
# V. Calculate TIL correlations with ICP measures

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
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# Custom methods
from functions.analysis import spearman_rho

### III. Calculate population-level summary characteristics
# Calculate group-level summary TIL values and round to one decimal point
summarised_TIL_population = summarised_TIL_per_patient.groupby(['Group','TILmetric'],as_index=False)['value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':lambda x: np.median(x),'q3':lambda x: np.quantile(x,.75),'count':lambda x: len(x)}).reset_index(drop=True)
summarised_TIL_population[['q1','median','q3']] = summarised_TIL_population[['q1','median','q3']].round(1)

# Create formatted text IQR
summarised_TIL_population['FormattedIQR'] = summarised_TIL_population['median'].astype(str)+' ('+summarised_TIL_population.q1.astype(str)+'–'+summarised_TIL_population.q3.astype(str)+')'

## Daily TIL scores
# Melt into long form and filter in-group patients
long_formatted_TIL_scores = formatted_TIL_scores[['GUPI','TILTimepoint','TotalTIL','Total','ICP_lo_res','ICP_hi_res']].melt(id_vars=['GUPI','TILTimepoint','TotalTIL'],var_name='Group')
long_formatted_TIL_scores = long_formatted_TIL_scores[long_formatted_TIL_scores['value']==True].drop(columns='value').reset_index(drop=True)

# Save daily TIL score dataframe
long_formatted_TIL_scores.to_csv('../formatted_data/formatted_TIL24_scores.csv',index=False)

# Calculate TIL scores for each ICU stay day
TIL_per_day_population = long_formatted_TIL_scores.groupby(['Group','TILTimepoint'],as_index=False).TotalTIL.aggregate({'q1':lambda x: np.quantile(x,.25),'median':lambda x: np.median(x),'q3':lambda x: np.quantile(x,.75),'count':lambda x: len(x)}).reset_index(drop=True)
TIL_per_day_population[['q1','median','q3']] = TIL_per_day_population[['q1','median','q3']].round(1)

# Create formatted text IQR
TIL_per_day_population['FormattedIQR'] = TIL_per_day_population['median'].astype(str)+' ('+TIL_per_day_population.q1.astype(str)+'–'+TIL_per_day_population.q3.astype(str)+')'

## Number of TIL assessments per patient
# Calculate median number of TIL assessments per patient per group
count_assessments_per_patient = long_formatted_TIL_scores.groupby(['Group','GUPI'],as_index=False).TILTimepoint.aggregate({'count':'count'})

# Calculate group-level summary TIL values and round to one decimal point
TIL_per_patient = count_assessments_per_patient.groupby(['Group'],as_index=False)['count'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':lambda x: np.median(x),'q3':lambda x: np.quantile(x,.75),'count':lambda x: len(x)}).reset_index(drop=True)
TIL_per_patient[['q1','median','q3']] = TIL_per_patient[['q1','median','q3']].astype(int)

# Create formatted text IQR
TIL_per_patient['FormattedIQR'] = TIL_per_patient['median'].astype(str)+' ('+TIL_per_patient.q1.astype(str)+'–'+TIL_per_patient.q3.astype(str)+')'

### IV. Calculate population-level TIL correlations with overall characteristics
## Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

## Correlation with baseline GCS
# Merge GCS information with TILmax and TILmean scores
GCS_scores_with_TIL = summarised_TIL_per_patient.merge(CENTER_TBI_demo_outcome[['GUPI','GCSScoreBaselineDerived']],how='left')

# Remove rows with missing GCS values
GCS_scores_with_TIL = GCS_scores_with_TIL[~GCS_scores_with_TIL.GCSScoreBaselineDerived.isna()].reset_index(drop=True)

# Calculate correlation between TILmetrics and GCS
GCS_corr_with_TIL = GCS_scores_with_TIL.groupby(['Group','TILmetric'],as_index=False).apply(spearman_rho,'GCSScoreBaselineDerived')

# Round correlation and p-values
GCS_corr_with_TIL[['rho','p_val']] = GCS_corr_with_TIL[['rho','p_val']].round(3)

## Correlation with 6-month GOSE
# Merge GOSE information with TILmax and TILmean scores
GOSE_scores_with_TIL = summarised_TIL_per_patient.merge(CENTER_TBI_demo_outcome[['GUPI','GOSE6monthEndpointDerived']],how='left')

# Remove rows with missing GOSE values
GOSE_scores_with_TIL = GOSE_scores_with_TIL[~GOSE_scores_with_TIL.GOSE6monthEndpointDerived.isna()].reset_index(drop=True)

# Calculate correlation between TILmetrics and GOSE
GOSE_corr_with_TIL = GOSE_scores_with_TIL.groupby(['Group','TILmetric'],as_index=False).apply(spearman_rho,'GOSE6monthEndpointDerived')

# Round correlation and p-values
GOSE_corr_with_TIL[['rho','p_val']] = GOSE_corr_with_TIL[['rho','p_val']].round(3)

## Correlation with ordinal prognoses
# Extract names of ordinal prognosis columns
prog_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('Pr(GOSE>')]

# Merge ordinal prognosis information with TILmax and TILmean scores
prog_scores_with_TIL = summarised_TIL_per_patient.merge(CENTER_TBI_demo_outcome[['GUPI']+prog_cols],how='left')

# Melt prognostic scores to long form
prog_scores_with_TIL = prog_scores_with_TIL.melt(id_vars=['GUPI','Group','TILmetric','value'],var_name='Threshold',value_name='Probability')

# Remove rows with missing ordinal prognosis values
prog_scores_with_TIL = prog_scores_with_TIL[~prog_scores_with_TIL.Probability.isna()].reset_index(drop=True)

# Calculate correlation between TILmetrics and ordinal prognosis scores
prog_corr_with_TIL = prog_scores_with_TIL.groupby(['Group','TILmetric','Threshold'],as_index=False).apply(spearman_rho,'Probability')

# Round correlation and p-values
prog_corr_with_TIL[['rho','p_val']] = prog_corr_with_TIL[['rho','p_val']].round(3)

### V. Calculate TIL correlations with ICP and CPP measures
## Prepare low-resolution ICP and CPP information
# Load low-resolution ICP and CPP information
mod_daily_hourly_info = pd.read_csv('../formatted_data/formatted_daily_hourly_values.csv')

# Convert timestamps from string to date format in both low-resolution dataframe and formatted TIL score dataframe
mod_daily_hourly_info.TimeStamp = pd.to_datetime(mod_daily_hourly_info.TimeStamp,format = '%Y-%m-%d %H:%M:%S')
formatted_TIL_scores.TimeStamp = pd.to_datetime(formatted_TIL_scores.TimeStamp,format = '%Y-%m-%d %H:%M:%S')

# Add a column to designate day component in both low-resolution dataframe and formatted TIL score dataframe
mod_daily_hourly_info['DateComponent'] = mod_daily_hourly_info.TimeStamp.dt.date
formatted_TIL_scores['DateComponent'] = formatted_TIL_scores.TimeStamp.dt.date

# Merge TIL scores onto corresponding low-resolution ICP/CPP scores based on 'DateComponent'
formatted_lo_res_values = mod_daily_hourly_info.merge(formatted_TIL_scores[['GUPI','DateComponent','TILTimepoint','TotalTIL']],how='left',on=['GUPI','DateComponent'])

# Filter out rows without TIL scores on the day and select relevant columns
formatted_lo_res_values = formatted_lo_res_values[~formatted_lo_res_values.TotalTIL.isna()][['GUPI','TimeStamp','DateComponent','HVICP','HVCPP','TILTimepoint','TotalTIL']]

# Melt out ICP and CPP values to long-form
formatted_lo_res_values = formatted_lo_res_values.melt(id_vars=['GUPI','TimeStamp','DateComponent','TILTimepoint','TotalTIL'])

# Remove missing CPP values
formatted_lo_res_values = formatted_lo_res_values[~formatted_lo_res_values.value.isna()].reset_index(drop=True)

# Calculate daily ICP and CPP means
formatted_lo_res_values = formatted_lo_res_values.groupby(['GUPI','DateComponent','TILTimepoint','TotalTIL','variable'],as_index=False)['value'].aggregate({'value':'mean','count':'count'})

# Rename variable names to ICP24 and CPP24
formatted_lo_res_values['variable'] = formatted_lo_res_values.variable.str.replace('HV','') + '24'

# Save formatted low-resolution ICP and CPP information
formatted_lo_res_values.to_csv('../formatted_data/formatted_low_resolution_neuromonitoring.csv',index=False)

## Calculate ICP/CPPmax and ICP/CPPmean
# Load low-resolution ICP and CPP information
mod_daily_hourly_info = pd.read_csv('../formatted_data/formatted_daily_hourly_values.csv')

# Convert timestamps from string to date format in both low-resolution dataframe
mod_daily_hourly_info.TimeStamp = pd.to_datetime(mod_daily_hourly_info.TimeStamp,format = '%Y-%m-%d %H:%M:%S')

# Add a column to designate day component in both low-resolution dataframe
mod_daily_hourly_info['DateComponent'] = mod_daily_hourly_info.TimeStamp.dt.date

# Melt out ICP and CPP values to long-form
all_daily_lo_res_values = mod_daily_hourly_info[['GUPI','TimeStamp','DateComponent','HVICP','HVCPP']].melt(id_vars=['GUPI','TimeStamp','DateComponent'])

# Remove missing CPP values
all_daily_lo_res_values = all_daily_lo_res_values[~all_daily_lo_res_values.value.isna()].reset_index(drop=True)

# Calculate daily ICP and CPP means
all_daily_lo_res_values = all_daily_lo_res_values.groupby(['GUPI','DateComponent','variable'],as_index=False)['value'].aggregate({'value':'mean','count':'count'})

# Rename variable names to ICP24 and CPP24
all_daily_lo_res_values['variable'] = all_daily_lo_res_values.variable.str.replace('HV','') + '24'

# Calculate ICP/CPPmean
patient_lo_res_neuro_means = all_daily_lo_res_values.groupby(['GUPI','variable'],as_index=False)['value'].mean()
patient_lo_res_neuro_means['variable'] = patient_lo_res_neuro_means.variable.str.replace('24','mean')

# Calculate ICP/CPPmax
patient_lo_res_neuro_maxes = all_daily_lo_res_values.groupby(['GUPI','variable'],as_index=False)['value'].max()
patient_lo_res_neuro_maxes['variable'] = patient_lo_res_neuro_maxes.variable.str.replace('24','max')

# Isolate TILmean and TILmax
isolated_TIL_mean_max = summarised_TIL_per_patient[['GUPI','TILmetric','value']].drop_duplicates(ignore_index=True)
isolated_TIL_mean = isolated_TIL_mean_max[isolated_TIL_mean_max.TILmetric=='TILmean'].reset_index(drop=True).drop(columns='TILmetric').rename(columns={'value':'TILmean'})
isolated_TIL_max = isolated_TIL_mean_max[isolated_TIL_mean_max.TILmetric=='TILmax'].reset_index(drop=True).drop(columns='TILmetric').rename(columns={'value':'TILmax'})

# Merge TILmean to ICP/CPPmean and merge TILmax to ICP/CPPmax
patient_lo_res_neuro_means = patient_lo_res_neuro_means.merge(isolated_TIL_mean,how='left')
patient_lo_res_neuro_maxes = patient_lo_res_neuro_maxes.merge(isolated_TIL_max,how='left')

# Calculate correlation between TILmetrics and corresponding ICP/CPP summary
lo_res_neuro_corr_with_TIL = pd.concat([patient_lo_res_neuro_means.groupby(['variable'],as_index=False).apply(spearman_rho,'TILmean'),patient_lo_res_neuro_maxes.groupby(['variable'],as_index=False).apply(spearman_rho,'TILmax')],ignore_index=True)

# Round correlation and p-values
lo_res_neuro_corr_with_TIL[['rho','p_val']] = lo_res_neuro_corr_with_TIL[['rho','p_val']].round(3)