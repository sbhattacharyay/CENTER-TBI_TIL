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

### II. Load and prepare TIL scores of full dataset and designate sub-study assignment
## Load formatted TIL score dataframe
# Read CSV
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Convert dates from string to date format
formatted_TIL_scores.TILDate = pd.to_datetime(formatted_TIL_scores.TILDate,format = '%Y-%m-%d')

## Add columns designating sub-study assignment
# Label each row as part of Total analysis group
formatted_TIL_scores['Total'] = True

# Label rows of low-resolution ICP subgroup
mod_daily_hourly_info = pd.read_csv('../formatted_data/formatted_daily_hourly_values.csv')
formatted_TIL_scores['ICP_lo_res'] = False
formatted_TIL_scores.ICP_lo_res[formatted_TIL_scores.GUPI.isin(mod_daily_hourly_info.GUPI)] = True

# Label rows of low-resolution ICP subgroup
hi_res_daily_TIL_info = pd.read_csv('../CENTER-TBI/HighResolution/high_res_TIL_timestamps.csv')
formatted_TIL_scores['ICP_hi_res'] = False
formatted_TIL_scores.ICP_hi_res[formatted_TIL_scores.GUPI.isin(hi_res_daily_TIL_info.GUPI)] = True

## Convert each 'None' timepoint to a number
# Replace 'None' timepoints with NaN
formatted_TIL_scores.TILTimepoint[formatted_TIL_scores.TILTimepoint=='None'] = np.nan

# Determine GUPIs with 'None' timepoints
none_GUPIs = formatted_TIL_scores[formatted_TIL_scores.TILTimepoint.isna()].GUPI.unique()

# Iterate through 'None' GUPIs and impute missing timepoint values
for curr_GUPI in none_GUPIs:
    curr_GUPI_TIL_scores = formatted_TIL_scores[formatted_TIL_scores.GUPI==curr_GUPI].reset_index(drop=True)
    curr_date_diff = int((curr_GUPI_TIL_scores.TILDate.dt.day - curr_GUPI_TIL_scores.TILTimepoint.astype(float)).mode()[0])
    fixed_timepoints_vector = (curr_GUPI_TIL_scores.TILDate.dt.day - curr_date_diff).astype(str)
    fixed_timepoints_vector.index=formatted_TIL_scores[formatted_TIL_scores.GUPI==curr_GUPI].index
    formatted_TIL_scores.TILTimepoint[formatted_TIL_scores.GUPI==curr_GUPI] = fixed_timepoints_vector    

# Convert TILTimepoint variable from string to integer
formatted_TIL_scores.TILTimepoint = formatted_TIL_scores.TILTimepoint.astype(int)

## Fix instances with more than 1 daily TIL score per patient's day in ICU
# Count number of TIL scores available per patient-Timepoint combination
patient_TIL_counts = formatted_TIL_scores.groupby(['GUPI','TILTimepoint'],as_index=False).TotalTIL.count()

# Isolate patients with instances of more than 1 daily TIL per day
more_than_one_GUPIs = patient_TIL_counts[patient_TIL_counts.TotalTIL>1].GUPI.unique()

# Filter dataframe of more-than-one-instance patients to visually examine
more_than_one_TIL_scores = formatted_TIL_scores[formatted_TIL_scores.GUPI.isin(more_than_one_GUPIs)].reset_index(drop=True)

# Select the rows which correspond to the greatest TIL score per ICU stay day per patient
keep_idx = formatted_TIL_scores.groupby(['GUPI','TILTimepoint'])['TotalTIL'].transform(max) == formatted_TIL_scores['TotalTIL']

# Filter to keep selected rows only
formatted_TIL_scores = formatted_TIL_scores[keep_idx].reset_index(drop=True)

### III. Calculate population-level summary characteristics
## TILmax and TILmean
# Calculate TILmax and TILmean per patient
summarised_TIL_per_patient = formatted_TIL_scores.groupby(['GUPI','Total','ICP_lo_res','ICP_hi_res'],as_index=False).TotalTIL.aggregate({'TILmax':'max','TILmean':'mean'}).reset_index(drop=True)

# Melt into long form and filter in-group patients
summarised_TIL_per_patient = summarised_TIL_per_patient.melt(id_vars=['GUPI','TILmax','TILmean'],var_name='Group')
summarised_TIL_per_patient = summarised_TIL_per_patient[summarised_TIL_per_patient['value']==True].drop(columns='value').reset_index(drop=True)

# Melt metrics into long form as well
summarised_TIL_per_patient = summarised_TIL_per_patient.melt(id_vars=['GUPI','Group'],var_name='TILmetric')

# Calculate group-level summary TIL values and round to one decimal point
summarised_TIL_population = summarised_TIL_per_patient.groupby(['Group','TILmetric'],as_index=False)['value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':lambda x: np.median(x),'q3':lambda x: np.quantile(x,.75),'count':lambda x: len(x)}).reset_index(drop=True)
summarised_TIL_population[['q1','median','q3']] = summarised_TIL_population[['q1','median','q3']].round(1)

# Create formatted text IQR
summarised_TIL_population['FormattedIQR'] = summarised_TIL_population['median'].astype(str)+' ('+summarised_TIL_population.q1.astype(str)+'–'+summarised_TIL_population.q3.astype(str)+')'

## Daily TIL scores
# Melt into long form and filter in-group patients
long_formatted_TIL_scores = formatted_TIL_scores[['GUPI','TILTimepoint','TotalTIL','Total','ICP_lo_res','ICP_hi_res']].melt(id_vars=['GUPI','TILTimepoint','TotalTIL'],var_name='Group')
long_formatted_TIL_scores = long_formatted_TIL_scores[long_formatted_TIL_scores['value']==True].drop(columns='value').reset_index(drop=True)

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
mod_daily_hourly_info['DayComponent'] = mod_daily_hourly_info.TimeStamp.dt.day
formatted_TIL_scores['DayComponent'] = formatted_TIL_scores.TimeStamp.dt.day

# Merge TIL scores onto corresponding low-resolution ICP/CPP scores based on 'DayComponent'
formatted_lo_res_values = mod_daily_hourly_info.merge(formatted_TIL_scores[['GUPI','DayComponent','TILTimepoint','TotalTIL']],how='left',on=['GUPI','DayComponent'])

# Filter out rows without TIL scores on the day and select relevant columns
formatted_lo_res_values = formatted_lo_res_values[~formatted_lo_res_values.TotalTIL.isna()][['GUPI','','']]