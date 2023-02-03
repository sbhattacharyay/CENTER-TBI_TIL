#### Master Script 1b: Calculate summary statistics of different study sub-samples ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and prepare summary characteristics of full TIL dataset
# III. Calculate summary statistics and statistical tests of numerical variables
# IV. Calculate summary statistics and statistical tests of categorical variables

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

# StatsModel methods
from statsmodels.stats.anova import AnovaRM

### II. Load and prepare summary characteristics of full TIL dataset
## Prepare summary statistics of full TIL dataset
# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Categorise GCS into severity
CENTER_TBI_demo_outcome['GCSSeverity'] = np.nan
CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.GCSScoreBaselineDerived<=8] = 'Severe'
CENTER_TBI_demo_outcome.GCSSeverity[(CENTER_TBI_demo_outcome.GCSScoreBaselineDerived>=9)&(CENTER_TBI_demo_outcome.GCSScoreBaselineDerived<=12)] = 'Moderate'
CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.GCSScoreBaselineDerived>=13] = 'Mild'

# Merge Marshall CT V and VI into one category
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==1] = '1'
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==2] = '2'
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==3] = '3'
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==4] = '4'
CENTER_TBI_demo_outcome.MarshallCT[(CENTER_TBI_demo_outcome.MarshallCT==5)|(CENTER_TBI_demo_outcome.MarshallCT==6)] = '5_or_6'

# Merge unknown race categories
CENTER_TBI_demo_outcome.Race[(CENTER_TBI_demo_outcome.Race.isna())|(CENTER_TBI_demo_outcome.Race=='Unknown')|(CENTER_TBI_demo_outcome.Race=='NotAllowed')] = 'Unknown'

## Tag patients which belong to different substudy groups
# Load patients with low-resolution ICP
mod_daily_hourly_info = pd.read_csv('../formatted_data/formatted_daily_hourly_values.csv')

# Label rows of low-resolution ICP subgroup
CENTER_TBI_demo_outcome['ICP_lo_res'] = 0
CENTER_TBI_demo_outcome.ICP_lo_res[CENTER_TBI_demo_outcome.GUPI.isin(mod_daily_hourly_info.GUPI)] = 1

# Load patients with high-resolution ICP
hi_res_daily_TIL_info = pd.read_csv('../CENTER-TBI/HighResolution/high_res_TIL_timestamps.csv')

# Label rows of low-resolution ICP subgroup
CENTER_TBI_demo_outcome['ICP_hi_res'] = 0
CENTER_TBI_demo_outcome.ICP_hi_res[CENTER_TBI_demo_outcome.GUPI.isin(hi_res_daily_TIL_info.GUPI)] = 1

### III. Calculate summary statistics and statistical tests of numerical variables
## Count statistics
# Total TIL population
n_total = CENTER_TBI_demo_outcome.shape[0]

# TIL-ICP_lo_res population
n_TIL_ICP_lo_res = CENTER_TBI_demo_outcome.ICP_lo_res.sum()

# TIL-ICP_hi_res population
n_TIL_ICP_hi_res = CENTER_TBI_demo_outcome.ICP_hi_res.sum()

## Number of centres
# Total TIL population
centres_total = CENTER_TBI_demo_outcome.SiteCode.nunique()

# TIL-ICP_lo_res population
centres_TIL_ICP_lo_res = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_lo_res==1].SiteCode.nunique()

# TIL-ICP_hi_res population
centres_TIL_ICP_hi_res = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_hi_res==1].SiteCode.nunique()

## Age
# Total TIL population
age_total = CENTER_TBI_demo_outcome.Age.median().astype(int).astype(str)+' ('+CENTER_TBI_demo_outcome.Age.quantile(.25,interpolation='nearest').astype(int).astype(str)+'–'+CENTER_TBI_demo_outcome.Age.quantile(.75,interpolation='nearest').astype(int).astype(str)+')' 

# TIL-ICP_lo_res population
age_TIL_ICP_lo_res = CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.ICP_lo_res==1].median().astype(int).astype(str)+' ('+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.ICP_lo_res==1].quantile(.25,interpolation='nearest').astype(int).astype(str)+'–'+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.ICP_lo_res==1].quantile(.75,interpolation='nearest').astype(int).astype(str)+')' 

# TIL-ICP_hi_res population
age_TIL_ICP_hi_res = CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.ICP_hi_res==1].median().astype(int).astype(str)+' ('+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.ICP_hi_res==1].quantile(.25,interpolation='nearest').astype(int).astype(str)+'–'+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.ICP_hi_res==1].quantile(.75,interpolation='nearest').astype(int).astype(str)+')' 

# Prepare a compiled dataframe
age_total_group = CENTER_TBI_demo_outcome[['GUPI','Age']].reset_index(drop=True)
age_total_group['Group'] = 'Total'
age_ICP_lo_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_lo_res==1][['GUPI','Age']].reset_index(drop=True)
age_ICP_lo_res_group['Group'] = 'ICP_lo_res'
age_ICP_hi_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_hi_res==1][['GUPI','Age']].reset_index(drop=True)
age_ICP_hi_res_group['Group'] = 'ICP_hi_res'
compiled_age_dataframe = pd.concat([age_total_group,age_ICP_lo_res_group,age_ICP_hi_res_group],ignore_index=True)

## Baseline ordinal prognosis
# Extract names of ordinal prognosis columns
prog_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('Pr(GOSE>')]

# Melt baseline prognostic scores into long dataframe
prog_scores_df = CENTER_TBI_demo_outcome[['GUPI']+prog_cols].melt(id_vars='GUPI',var_name='Threshold',value_name='Probability').dropna().reset_index(drop=True)

# Extract substudy assignment information
substudy_assignment_df = CENTER_TBI_demo_outcome[['GUPI','ICP_lo_res','ICP_hi_res']]
substudy_assignment_df['total'] = 1

# Merge substudy assignment information to prognostic scores and melt to longer form
prog_scores_df = prog_scores_df.merge(substudy_assignment_df).melt(id_vars=['GUPI','Threshold','Probability'],var_name='Group',value_name='Marker')

# Filter patients who meet substudy assignment
prog_scores_df = prog_scores_df[prog_scores_df.Marker==1].drop(columns='Marker').reset_index(drop=True)

# Calculate median and interquartile range for each grouping factor combination
prog_scores_df = prog_scores_df.groupby(['Group','Threshold'],as_index=False).Probability.aggregate({'q1':lambda x: 100*np.quantile(x,.25),'median':lambda x: 100*np.median(x),'q3':lambda x: 100*np.quantile(x,.75),'count':'count'}).reset_index(drop=True)

# Create formatted text IQR
prog_scores_df['FormattedIQR'] = prog_scores_df['median'].round(1).astype(str)+' ('+prog_scores_df.q1.round(1).astype(str)+'–'+prog_scores_df.q3.round(1).astype(str)+')'

### IV. Calculate summary statistics and statistical tests of categorical variables
## Sex
# Prepare a compiled dataframe
sex_total_group = CENTER_TBI_demo_outcome[['GUPI','Sex']].reset_index(drop=True)
sex_total_group['Group'] = 'Total'
sex_ICP_lo_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_lo_res==1][['GUPI','Sex']].reset_index(drop=True)
sex_ICP_lo_res_group['Group'] = 'ICP_lo_res'
sex_ICP_hi_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_hi_res==1][['GUPI','Sex']].reset_index(drop=True)
sex_ICP_hi_res_group['Group'] = 'ICP_hi_res'
compiled_sex_dataframe = pd.concat([sex_total_group,sex_ICP_lo_res_group,sex_ICP_hi_res_group],ignore_index=True)

# Count number of females in each group
group_female_counts = compiled_sex_dataframe.groupby(['Group'],as_index=False).Sex.aggregate({'count':'count','F_count':lambda x: (x=='F').sum()}).reset_index(drop=True)

# Calculate percentage of patients in each group that are female
group_female_counts['F_prop'] = (100*group_female_counts.F_count/group_female_counts['count']).round().astype(int).astype(str)

# Create formatted text proportion
group_female_counts['FormattedProportion'] = group_female_counts['F_count'].astype(str)+' ('+group_female_counts.F_prop+'%)'

## Race
# Prepare a compiled dataframe
race_total_group = CENTER_TBI_demo_outcome[['GUPI','Race']].reset_index(drop=True)
race_total_group['Group'] = 'Total'
race_ICP_lo_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_lo_res==1][['GUPI','Race']].reset_index(drop=True)
race_ICP_lo_res_group['Group'] = 'ICP_lo_res'
race_ICP_hi_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_hi_res==1][['GUPI','Race']].reset_index(drop=True)
race_ICP_hi_res_group['Group'] = 'ICP_hi_res'
compiled_race_dataframe = pd.concat([race_total_group,race_ICP_lo_res_group,race_ICP_hi_res_group],ignore_index=True)

# Count number of patients of each race in each group
group_race_counts = compiled_race_dataframe.groupby(['Group','Race'],as_index=False).Race.aggregate({'count':'count'}).reset_index(drop=True)

# Merge group totals
group_race_counts = group_race_counts.merge(group_race_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# Calculate percentage of patients of each race in each group
group_race_counts['Race_prop'] = (100*group_race_counts['count']/group_race_counts['GroupCount']).round().astype(int).astype(str)

# Create formatted text proportion
group_race_counts['FormattedProportion'] = group_race_counts['count'].astype(str)+' ('+group_race_counts.Race_prop+'%)'

## GCSSeverity
# Prepare a compiled dataframe
severity_total_group = CENTER_TBI_demo_outcome[['GUPI','GCSSeverity']].reset_index(drop=True)
severity_total_group['Group'] = 'Total'
severity_ICP_lo_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_lo_res==1][['GUPI','GCSSeverity']].reset_index(drop=True)
severity_ICP_lo_res_group['Group'] = 'ICP_lo_res'
severity_ICP_hi_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_hi_res==1][['GUPI','GCSSeverity']].reset_index(drop=True)
severity_ICP_hi_res_group['Group'] = 'ICP_hi_res'
compiled_severity_dataframe = pd.concat([severity_total_group,severity_ICP_lo_res_group,severity_ICP_hi_res_group],ignore_index=True).dropna()

# Count number of patients of each severity in each group
group_severity_counts = compiled_severity_dataframe.groupby(['Group','GCSSeverity'],as_index=False).GCSSeverity.aggregate({'count':'count'}).reset_index(drop=True)

# Merge group totals
group_severity_counts = group_severity_counts.merge(group_severity_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# Calculate percentage of patients of each severity in each group
group_severity_counts['GCSSeverity_prop'] = (100*group_severity_counts['count']/group_severity_counts['GroupCount']).round().astype(int).astype(str)

# Create formatted text proportion
group_severity_counts['FormattedProportion'] = group_severity_counts['count'].astype(str)+' ('+group_severity_counts.GCSSeverity_prop+'%)'

## MarshallCT
# Prepare a compiled dataframe
marshall_total_group = CENTER_TBI_demo_outcome[['GUPI','MarshallCT']].reset_index(drop=True)
marshall_total_group['Group'] = 'Total'
marshall_ICP_lo_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_lo_res==1][['GUPI','MarshallCT']].reset_index(drop=True)
marshall_ICP_lo_res_group['Group'] = 'ICP_lo_res'
marshall_ICP_hi_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_hi_res==1][['GUPI','MarshallCT']].reset_index(drop=True)
marshall_ICP_hi_res_group['Group'] = 'ICP_hi_res'
compiled_marshall_dataframe = pd.concat([marshall_total_group,marshall_ICP_lo_res_group,marshall_ICP_hi_res_group],ignore_index=True).dropna()

# Count number of patients of each marshall in each group
group_marshall_counts = compiled_marshall_dataframe.groupby(['Group','MarshallCT'],as_index=False).MarshallCT.aggregate({'count':'count'}).reset_index(drop=True)

# Merge group totals
group_marshall_counts = group_marshall_counts.merge(group_marshall_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# Calculate percentage of patients of each marshall in each group
group_marshall_counts['MarshallCT_prop'] = (100*group_marshall_counts['count']/group_marshall_counts['GroupCount']).round().astype(int).astype(str)

# Create formatted text proportion
group_marshall_counts['FormattedProportion'] = group_marshall_counts['count'].astype(str)+' ('+group_marshall_counts.MarshallCT_prop+'%)'

## GOSE6monthEndpointDerived
# Prepare a compiled dataframe
gose_total_group = CENTER_TBI_demo_outcome[['GUPI','GOSE6monthEndpointDerived']].reset_index(drop=True)
gose_total_group['Group'] = 'Total'
gose_ICP_lo_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_lo_res==1][['GUPI','GOSE6monthEndpointDerived']].reset_index(drop=True)
gose_ICP_lo_res_group['Group'] = 'ICP_lo_res'
gose_ICP_hi_res_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.ICP_hi_res==1][['GUPI','GOSE6monthEndpointDerived']].reset_index(drop=True)
gose_ICP_hi_res_group['Group'] = 'ICP_hi_res'
compiled_gose_dataframe = pd.concat([gose_total_group,gose_ICP_lo_res_group,gose_ICP_hi_res_group],ignore_index=True).dropna()

# Count number of patients of each gose in each group
group_gose_counts = compiled_gose_dataframe.groupby(['Group','GOSE6monthEndpointDerived'],as_index=False).GOSE6monthEndpointDerived.aggregate({'count':'count'}).reset_index(drop=True)

# Merge group totals
group_gose_counts = group_gose_counts.merge(group_gose_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# Calculate percentage of patients of each gose in each group
group_gose_counts['GOSE6monthEndpointDerived_prop'] = (100*group_gose_counts['count']/group_gose_counts['GroupCount']).round().astype(int).astype(str)

# Create formatted text proportion
group_gose_counts['FormattedProportion'] = group_gose_counts['count'].astype(str)+' ('+group_gose_counts.GOSE6monthEndpointDerived_prop+'%)'