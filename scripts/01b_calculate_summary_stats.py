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
# V. Calculate summary statistics of summarised TIL metrics

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

# Load prior study demographic and functional outcome score dataframe
prior_study_demo_outcome = pd.read_csv('../formatted_data/prior_study_formatted_outcome_and_demographics.csv')

# Categorise GCS into severity
CENTER_TBI_demo_outcome['GCSSeverity'] = np.nan
CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.GCSScoreBaselineDerived<=8] = 'Severe'
CENTER_TBI_demo_outcome.GCSSeverity[(CENTER_TBI_demo_outcome.GCSScoreBaselineDerived>=9)&(CENTER_TBI_demo_outcome.GCSScoreBaselineDerived<=12)] = 'Moderate'
CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.GCSScoreBaselineDerived>=13] = 'Mild'

# Categorise GCS into severity for prior study dataset
prior_study_demo_outcome['GCSSeverity'] = np.nan
prior_study_demo_outcome.GCSSeverity[prior_study_demo_outcome.GCSScoreBaselineDerived<=8] = 'Severe'
prior_study_demo_outcome.GCSSeverity[(prior_study_demo_outcome.GCSScoreBaselineDerived>=9)&(prior_study_demo_outcome.GCSScoreBaselineDerived<=12)] = 'Moderate'
prior_study_demo_outcome.GCSSeverity[prior_study_demo_outcome.GCSScoreBaselineDerived>=13] = 'Mild'

# Merge Marshall CT V and VI into one category
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==1] = '1'
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==2] = '2'
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==3] = '3'
CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.MarshallCT==4] = '4'
CENTER_TBI_demo_outcome.MarshallCT[(CENTER_TBI_demo_outcome.MarshallCT==5)|(CENTER_TBI_demo_outcome.MarshallCT==6)] = '5_or_6'

# Merge Marshall CT V and VI into one category
prior_study_demo_outcome.MarshallCT[prior_study_demo_outcome.MarshallCT==1] = '1'
prior_study_demo_outcome.MarshallCT[prior_study_demo_outcome.MarshallCT==2] = '2'
prior_study_demo_outcome.MarshallCT[prior_study_demo_outcome.MarshallCT==3] = '3'
prior_study_demo_outcome.MarshallCT[prior_study_demo_outcome.MarshallCT==4] = '4'
prior_study_demo_outcome.MarshallCT[(prior_study_demo_outcome.MarshallCT==5)|(prior_study_demo_outcome.MarshallCT==6)] = '5_or_6'

# Merge unknown race categories
CENTER_TBI_demo_outcome.Race[(CENTER_TBI_demo_outcome.Race.isna())|(CENTER_TBI_demo_outcome.Race=='Unknown')|(CENTER_TBI_demo_outcome.Race=='NotAllowed')] = 'Unknown'

### III. Calculate summary statistics and statistical tests of numerical variables
## Count statistics
# Total TIL population
n_total = CENTER_TBI_demo_outcome.shape[0]

# TIL-LowResolutionSet population
n_TIL_LowResolutionSet = CENTER_TBI_demo_outcome.LowResolutionSet.sum()

# TIL-HighResolutionSet population
n_TIL_HighResolutionSet = CENTER_TBI_demo_outcome.HighResolutionSet.sum()

## Number of centres
# Total TIL population
centres_total = CENTER_TBI_demo_outcome.SiteCode.nunique()

# TIL-LowResolutionSet population
centres_TIL_LowResolutionSet = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1].SiteCode.nunique()

# TIL-HighResolutionSet population
centres_TIL_HighResolutionSet = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1].SiteCode.nunique()

## Age
# Total TIL population
age_total = CENTER_TBI_demo_outcome.Age.median().astype(int).astype(str)+' ('+CENTER_TBI_demo_outcome.Age.quantile(.25,interpolation='nearest').astype(int).astype(str)+'–'+CENTER_TBI_demo_outcome.Age.quantile(.75,interpolation='nearest').astype(int).astype(str)+')' 

# TIL-LowResolutionSet population
age_TIL_LowResolutionSet = CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.LowResolutionSet==1].median().astype(int).astype(str)+' ('+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.LowResolutionSet==1].quantile(.25,interpolation='nearest').astype(int).astype(str)+'–'+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.LowResolutionSet==1].quantile(.75,interpolation='nearest').astype(int).astype(str)+')' 

# TIL-HighResolutionSet population
age_TIL_HighResolutionSet = CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.HighResolutionSet==1].median().astype(int).astype(str)+' ('+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.HighResolutionSet==1].quantile(.25,interpolation='nearest').astype(int).astype(str)+'–'+CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.HighResolutionSet==1].quantile(.75,interpolation='nearest').astype(int).astype(str)+')' 

# Prepare a compiled dataframe
age_total_group = CENTER_TBI_demo_outcome[['GUPI','Age']].reset_index(drop=True)
age_total_group['Group'] = 'Total'
age_LowResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1][['GUPI','Age']].reset_index(drop=True)
age_LowResolutionSet_group['Group'] = 'LowResolutionSet'
age_HighResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1][['GUPI','Age']].reset_index(drop=True)
age_HighResolutionSet_group['Group'] = 'HighResolutionSet'
compiled_age_dataframe = pd.concat([age_total_group,age_LowResolutionSet_group,age_HighResolutionSet_group],ignore_index=True)

# Calculate p-values for age
age_LowResolution_pval = stats.ttest_ind(CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.LowResolutionSet==1].values,prior_study_demo_outcome.Age.values,equal_var=False)
age_HighResolution_pval = stats.ttest_ind(CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.HighResolutionSet==1].values,prior_study_demo_outcome.Age.values,equal_var=False)

## Baseline ordinal prognosis
# Extract names of ordinal prognosis columns
prog_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('Pr(GOSE>')]

# Melt baseline prognostic scores into long dataframe
prog_scores_df = CENTER_TBI_demo_outcome[['GUPI']+prog_cols].melt(id_vars='GUPI',var_name='Threshold',value_name='Probability').dropna().reset_index(drop=True)

# Extract substudy assignment information
substudy_assignment_df = CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']]
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
sex_LowResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1][['GUPI','Sex']].reset_index(drop=True)
sex_LowResolutionSet_group['Group'] = 'LowResolutionSet'
sex_HighResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1][['GUPI','Sex']].reset_index(drop=True)
sex_HighResolutionSet_group['Group'] = 'HighResolutionSet'
compiled_sex_dataframe = pd.concat([sex_total_group,sex_LowResolutionSet_group,sex_HighResolutionSet_group],ignore_index=True)

# Count number of females in each group
group_female_counts = compiled_sex_dataframe.groupby(['Group'],as_index=False).Sex.aggregate({'count':'count','F_count':lambda x: (x=='F').sum()}).reset_index(drop=True)

# Calculate percentage of patients in each group that are female
group_female_counts['F_prop'] = (100*group_female_counts.F_count/group_female_counts['count']).round().astype(int).astype(str)

# Create formatted text proportion
group_female_counts['FormattedProportion'] = group_female_counts['F_count'].astype(str)+' ('+group_female_counts.F_prop+'%)'

# Calculate p-values for Sex
Sex_LowResolution_pval = stats.chi2_contingency(np.stack((prior_study_demo_outcome.Sex.value_counts().values,CENTER_TBI_demo_outcome.Sex[CENTER_TBI_demo_outcome.LowResolutionSet==1].value_counts().values)))
Sex_HighResolution_pval = stats.chi2_contingency(np.stack((prior_study_demo_outcome.Sex.value_counts().values,CENTER_TBI_demo_outcome.Sex[CENTER_TBI_demo_outcome.HighResolutionSet==1].value_counts().values)))

# ## Race
# # Prepare a compiled dataframe
# race_total_group = CENTER_TBI_demo_outcome[['GUPI','Race']].reset_index(drop=True)
# race_total_group['Group'] = 'Total'
# race_LowResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1][['GUPI','Race']].reset_index(drop=True)
# race_LowResolutionSet_group['Group'] = 'LowResolutionSet'
# race_HighResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1][['GUPI','Race']].reset_index(drop=True)
# race_HighResolutionSet_group['Group'] = 'HighResolutionSet'
# compiled_race_dataframe = pd.concat([race_total_group,race_LowResolutionSet_group,race_HighResolutionSet_group],ignore_index=True)

# # Count number of patients of each race in each group
# group_race_counts = compiled_race_dataframe.groupby(['Group','Race'],as_index=False).Race.aggregate({'count':'count'}).reset_index(drop=True)

# # Merge group totals
# group_race_counts = group_race_counts.merge(group_race_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# # Calculate percentage of patients of each race in each group
# group_race_counts['Race_prop'] = (100*group_race_counts['count']/group_race_counts['GroupCount']).round().astype(int).astype(str)

# # Create formatted text proportion
# group_race_counts['FormattedProportion'] = group_race_counts['count'].astype(str)+' ('+group_race_counts.Race_prop+'%)'

## GCSSeverity
# Prepare a compiled dataframe
severity_total_group = CENTER_TBI_demo_outcome[['GUPI','GCSSeverity']].reset_index(drop=True)
severity_total_group['Group'] = 'Total'
severity_LowResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1][['GUPI','GCSSeverity']].reset_index(drop=True)
severity_LowResolutionSet_group['Group'] = 'LowResolutionSet'
severity_HighResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1][['GUPI','GCSSeverity']].reset_index(drop=True)
severity_HighResolutionSet_group['Group'] = 'HighResolutionSet'
compiled_severity_dataframe = pd.concat([severity_total_group,severity_LowResolutionSet_group,severity_HighResolutionSet_group],ignore_index=True).dropna()

# Count number of patients of each severity in each group
group_severity_counts = compiled_severity_dataframe.groupby(['Group','GCSSeverity'],as_index=False).GCSSeverity.aggregate({'count':'count'}).reset_index(drop=True)

# Merge group totals
group_severity_counts = group_severity_counts.merge(group_severity_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# Calculate percentage of patients of each severity in each group
group_severity_counts['GCSSeverity_prop'] = (100*group_severity_counts['count']/group_severity_counts['GroupCount']).round().astype(int).astype(str)

# Create formatted text proportion
group_severity_counts['FormattedProportion'] = group_severity_counts['count'].astype(str)+' ('+group_severity_counts.GCSSeverity_prop+'%)'

# Calculate p-values for GCS severity
lo_res_severity_counts = CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.LowResolutionSet==1].value_counts().sort_index().values
hi_res_severity_counts = CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.HighResolutionSet==1].value_counts().sort_index().values
prior_study_severity_counts = prior_study_demo_outcome.GCSSeverity.value_counts().sort_index().values
severity_LowResolution_pval = stats.chi2_contingency(np.stack((prior_study_severity_counts,lo_res_severity_counts)))
severity_HighResolution_pval = stats.chi2_contingency(np.stack((prior_study_severity_counts,hi_res_severity_counts)))

## MarshallCT
# Prepare a compiled dataframe
marshall_total_group = CENTER_TBI_demo_outcome[['GUPI','MarshallCT']].reset_index(drop=True)
marshall_total_group['Group'] = 'Total'
marshall_LowResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1][['GUPI','MarshallCT']].reset_index(drop=True)
marshall_LowResolutionSet_group['Group'] = 'LowResolutionSet'
marshall_HighResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1][['GUPI','MarshallCT']].reset_index(drop=True)
marshall_HighResolutionSet_group['Group'] = 'HighResolutionSet'
compiled_marshall_dataframe = pd.concat([marshall_total_group,marshall_LowResolutionSet_group,marshall_HighResolutionSet_group],ignore_index=True).dropna()

# Count number of patients of each marshall in each group
group_marshall_counts = compiled_marshall_dataframe.groupby(['Group','MarshallCT'],as_index=False).MarshallCT.aggregate({'count':'count'}).reset_index(drop=True)

# Merge group totals
group_marshall_counts = group_marshall_counts.merge(group_marshall_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# Calculate percentage of patients of each marshall in each group
group_marshall_counts['MarshallCT_prop'] = (100*group_marshall_counts['count']/group_marshall_counts['GroupCount']).round().astype(int).astype(str)

# Create formatted text proportion
group_marshall_counts['FormattedProportion'] = group_marshall_counts['count'].astype(str)+' ('+group_marshall_counts.MarshallCT_prop+'%)'

# Calculate p-values for Marshall CT
lo_res_marshall_counts = CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.LowResolutionSet==1].value_counts().sort_index().values
hi_res_marshall_counts = CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.HighResolutionSet==1].value_counts().sort_index().values
prior_study_marshall_counts = prior_study_demo_outcome.MarshallCT.value_counts().sort_index().values
marshall_LowResolution_pval = stats.chi2_contingency(np.stack((prior_study_marshall_counts,lo_res_marshall_counts)))
marshall_HighResolution_pval = stats.chi2_contingency(np.stack((prior_study_marshall_counts,hi_res_marshall_counts)))

## GOSE6monthEndpointDerived
# Prepare a compiled dataframe
gose_total_group = CENTER_TBI_demo_outcome[['GUPI','GOSE6monthEndpointDerived']].reset_index(drop=True)
gose_total_group['Group'] = 'Total'
gose_LowResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1][['GUPI','GOSE6monthEndpointDerived']].reset_index(drop=True)
gose_LowResolutionSet_group['Group'] = 'LowResolutionSet'
gose_HighResolutionSet_group = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1][['GUPI','GOSE6monthEndpointDerived']].reset_index(drop=True)
gose_HighResolutionSet_group['Group'] = 'HighResolutionSet'
compiled_gose_dataframe = pd.concat([gose_total_group,gose_LowResolutionSet_group,gose_HighResolutionSet_group],ignore_index=True).dropna()

# Count number of patients of each gose in each group
group_gose_counts = compiled_gose_dataframe.groupby(['Group','GOSE6monthEndpointDerived'],as_index=False).GOSE6monthEndpointDerived.aggregate({'count':'count'}).reset_index(drop=True)

# Merge group totals
group_gose_counts = group_gose_counts.merge(group_gose_counts.groupby('Group',as_index=False)['count'].sum().rename(columns={'count':'GroupCount'}),how='left')

# Calculate percentage of patients of each gose in each group
group_gose_counts['GOSE6monthEndpointDerived_prop'] = (100*group_gose_counts['count']/group_gose_counts['GroupCount']).round().astype(int).astype(str)

# Create formatted text proportion
group_gose_counts['FormattedProportion'] = group_gose_counts['count'].astype(str)+' ('+group_gose_counts.GOSE6monthEndpointDerived_prop+'%)'

# Convert GOSE to GOS for comparison with prior study
CENTER_TBI_demo_outcome['GOS6monthEndpointDerived'] = np.nan
CENTER_TBI_demo_outcome.GOS6monthEndpointDerived[CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived=='1'] = '1'
CENTER_TBI_demo_outcome.GOS6monthEndpointDerived[CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived.isin(['2_or_3','4'])] = '2_or_3'
CENTER_TBI_demo_outcome.GOS6monthEndpointDerived[CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived.isin(['5','6'])] = '4'
CENTER_TBI_demo_outcome.GOS6monthEndpointDerived[CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived.isin(['7','8'])] = '5'
prior_study_demo_outcome.GOS6monthEndpointDerived = prior_study_demo_outcome.GOS6monthEndpointDerived.astype(int).astype(str)
prior_study_demo_outcome.GOS6monthEndpointDerived[prior_study_demo_outcome.GOS6monthEndpointDerived.isin(['2','3'])] = '2_or_3'

# Calculate p-values for GOS
lo_res_GOS_counts = CENTER_TBI_demo_outcome.GOS6monthEndpointDerived[CENTER_TBI_demo_outcome.LowResolutionSet==1].value_counts().sort_index().values
hi_res_GOS_counts = CENTER_TBI_demo_outcome.GOS6monthEndpointDerived[CENTER_TBI_demo_outcome.HighResolutionSet==1].value_counts().sort_index().values
prior_study_GOS_counts = prior_study_demo_outcome.GOS6monthEndpointDerived.value_counts().sort_index().values
GOS_LowResolution_pval = stats.chi2_contingency(np.stack((prior_study_GOS_counts,lo_res_GOS_counts)))
GOS_HighResolution_pval = stats.chi2_contingency(np.stack((prior_study_GOS_counts,hi_res_GOS_counts)))

### V. Calculate summary statistics of summarised TIL metrics
## Load and prepare formatted TIL values of different populations
# Load formatted TIL values and add row index
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Load formatted low-resolution values
formatted_lo_res_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Load formatted high-resolution values
formatted_hi_res_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')

# Merge set assignment to formatted TIL dataframe
formatted_TIL_scores = formatted_TIL_scores.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']],how='left')

# Keep only GUPIs either in low- or high-resolution set
formatted_TIL_scores = formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)|(formatted_TIL_scores.HighResolutionSet==1)].reset_index(drop=True)

# Merge WLST information onto formatted TIL dataframe
formatted_TIL_scores = formatted_TIL_scores.merge(pd.concat([formatted_lo_res_values[['GUPI','WLSTDateComponent','WLST']],formatted_hi_res_values[['GUPI','WLSTDateComponent','WLST']]],ignore_index=True).drop_duplicates(ignore_index=True),how='left')

# Convert dated timestamps to proper data format
formatted_TIL_scores.WLSTDateComponent = pd.to_datetime(formatted_TIL_scores.WLSTDateComponent,format = '%Y-%m-%d')
formatted_TIL_scores.DateComponent = pd.to_datetime(formatted_TIL_scores.DateComponent,format = '%Y-%m-%d')

# Keep only rows before decision to WLST or no WLST at all
formatted_TIL_scores = formatted_TIL_scores[(formatted_TIL_scores.DateComponent<formatted_TIL_scores.WLSTDateComponent)|(formatted_TIL_scores.WLST==0)].reset_index(drop=True)

# Load prior study TIL values
prior_study_database_extraction = pd.read_excel('../prior_study_data/SPSS_Main_Database.xlsx',na_values = ["NA","NaN"," ", "","#NULL!"]).dropna(axis=1,how='all').dropna(subset=['Pt'],how='all').rename(columns={'Pt':'GUPI'}).reset_index(drop=True)
prior_study_database_extraction['RowIdx'] = prior_study_database_extraction.groupby('GUPI').cumcount()

# Load prior study demographic and functional outcome score dataframe
prior_study_demo_outcome = pd.read_csv('../formatted_data/prior_study_formatted_outcome_and_demographics.csv')

# Load prior study ICU-TIL data from separate dataframe and add RowIdx
prior_study_ICP_TIL = pd.read_excel('../prior_study_data/ICP_vs_TIL_in_TBI_ICU.xlsx',na_values = ["NA","NaN"," ", "","#NULL!"]).rename(columns={'CamGro_ID':'GUPI','TIL_4h':'TIL_corrob','meanICP':'MeanICP'})
prior_study_ICP_TIL['RowIdx'] = prior_study_ICP_TIL.groupby('GUPI').cumcount()

# Focus on in-study population
prior_study_database_extraction = prior_study_database_extraction[prior_study_database_extraction.GUPI.isin(prior_study_demo_outcome.GUPI)].reset_index(drop=True)

# Select relevant columns
prior_study_database_extraction = prior_study_database_extraction[['GUPI','Period','RowIdx','Start','End','TIL_sum','TIL_4h_PAT','TIL_4h_WorstPerDay','TIL_24h_PAT','MeanICP','MeanCPP']]

# Determine rows with missing TIL scores
missing_TIL_rows = prior_study_database_extraction[prior_study_database_extraction.TIL_sum.isna()].reset_index(drop=True)

# Keep only ICU-TIL data that matches with missing rows
prior_study_ICP_TIL = prior_study_ICP_TIL.merge(missing_TIL_rows,how='inner').reset_index(drop=True)

# Merge corroborated TIL scores onto inital dataframe
prior_study_database_extraction = prior_study_database_extraction.merge(prior_study_ICP_TIL,how='left')

# In cases for which TIL_sum is missing, replace with `TIL_corrob`
prior_study_database_extraction.TIL_sum[prior_study_database_extraction.TIL_sum.isna()] = prior_study_database_extraction.TIL_corrob[prior_study_database_extraction.TIL_sum.isna()]

# If TIL_sum is still missing, replace with `TIL_4h_PAT`
prior_study_database_extraction.TIL_sum[prior_study_database_extraction.TIL_sum.isna()] = prior_study_database_extraction.TIL_4h_PAT[prior_study_database_extraction.TIL_sum.isna()]

# Remove unneccessary columns
prior_study_database_extraction = prior_study_database_extraction[['GUPI','Period','Start', 'End', 'TIL_sum','MeanICP','MeanCPP']]

# Sort dataframe
prior_study_database_extraction = prior_study_database_extraction.sort_values(by=['GUPI','Period']).reset_index(drop=True)

# Save formatted prior study dataframe
prior_study_database_extraction.to_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv',index=False)

## Calculate daily TIL scores and TILmax and TILmean
## CENTER-TBI TILmax and TILmean
# Calculate TILmax and TILmean per patient
summarised_TIL_per_patient = formatted_TIL_scores.groupby(['GUPI','LowResolutionSet','HighResolutionSet'],as_index=False).TotalTIL.aggregate({'TILmax':'max','TILmean':'mean'}).reset_index(drop=True)

# Melt into long form and filter in-group patients
summarised_TIL_per_patient = summarised_TIL_per_patient.melt(id_vars=['GUPI','TILmax','TILmean'],var_name='Group')
summarised_TIL_per_patient = summarised_TIL_per_patient[summarised_TIL_per_patient['value']==1].drop(columns='value').reset_index(drop=True)

# Melt metrics into long form as well
summarised_TIL_per_patient = summarised_TIL_per_patient.melt(id_vars=['GUPI','Group'],var_name='TILmetric')

# Save dataframe of TILmean/TILmax per patient
summarised_TIL_per_patient.to_csv('../formatted_data/formatted_TIL_means_maxes.csv',index=False)

## Prior study - convert TIL4 to TIL24
# Load formatted prior study TIL scores
prior_study_formatted_TIL_scores = pd.read_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv')

# Convert timestamps to datetime format
prior_study_formatted_TIL_scores['Start'] = pd.to_datetime(prior_study_formatted_TIL_scores['Start'].str[:19],format = '%Y-%m-%d %H:%M:%S')
prior_study_formatted_TIL_scores['End'] = pd.to_datetime(prior_study_formatted_TIL_scores['End'].str[:19],format = '%Y-%m-%d %H:%M:%S')

# Add column representing date component of end timestamp
prior_study_formatted_TIL_scores['DateComponent'] = prior_study_formatted_TIL_scores['End'].dt.date

# Calculate TIL24
prior_study_dailyTILs = prior_study_formatted_TIL_scores.groupby(['GUPI','DateComponent'],as_index=False).TIL_sum.max().rename(columns={'TIL_sum':'TotalTIL'})

# Calculate daily ICP and CPP estimates
mean_count_ICP_CPP = prior_study_formatted_TIL_scores[['GUPI','DateComponent','MeanICP','MeanCPP']].melt(id_vars=['GUPI','DateComponent']).groupby(['GUPI','DateComponent','variable'],as_index=False)['value'].aggregate({'mean':'mean','count':'count'})
mean_count_ICP_CPP = pd.pivot_table(mean_count_ICP_CPP, values = 'mean', index=['GUPI','DateComponent','count'], columns = 'variable').reset_index().rename(columns={'count':'nICP'})

# Merge daily ICP and CPP estimates to TIL24 dataframe
prior_study_dailyTILs = prior_study_dailyTILs.merge(mean_count_ICP_CPP,how='left').reset_index(drop=True)

# Sort dataframe by GUPI and DateComponent
prior_study_dailyTILs = prior_study_dailyTILs.sort_values(by=['GUPI','DateComponent']).reset_index(drop=True)

# Add column marking TILTimepoint
prior_study_dailyTILs['TILTimepoint'] = prior_study_dailyTILs.groupby('GUPI').cumcount()+1

# Rearrange columns
prior_study_dailyTILs = prior_study_dailyTILs[['GUPI','DateComponent','TILTimepoint','TotalTIL','MeanICP','MeanCPP','nICP']]

# Save formatted daily TILs for prior study
prior_study_dailyTILs.to_csv('../formatted_data/prior_study_formatted_TIL24_scores.csv',index=False)

## Prior study - calculate TILmax and TILmean
# Calculate TILmax and TILmean per patient
prior_study_summarised_TIL_per_patient = prior_study_dailyTILs.groupby(['GUPI'],as_index=False).TotalTIL.aggregate({'TILmax':'max','TILmean':'mean'}).reset_index(drop=True)

# Melt metrics into long form as well
prior_study_summarised_TIL_per_patient = prior_study_summarised_TIL_per_patient.melt(id_vars='GUPI',var_name='TILmetric')

# Save dataframe of TILmean/TILmax per patient
prior_study_summarised_TIL_per_patient.to_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv',index=False)

## Format TILmean and TILmax for characteristics table
# Load formatted TILmean and TILmax values from CENTER-TBI
summarised_TIL_per_patient = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')

# Calculate group-level summary TIL values and round to one decimal point
summarised_TIL_population = summarised_TIL_per_patient.groupby(['Group','TILmetric'],as_index=False)['value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':lambda x: np.median(x),'q3':lambda x: np.quantile(x,.75),'count':lambda x: len(x)}).reset_index(drop=True)
summarised_TIL_population[['q1','median','q3']] = summarised_TIL_population[['q1','median','q3']].round(1)

# Create formatted text IQR
summarised_TIL_population['FormattedIQR'] = summarised_TIL_population['median'].astype(str)+' ('+summarised_TIL_population.q1.astype(str)+'–'+summarised_TIL_population.q3.astype(str)+')'

# Load prior study dataframe of TILmean/TILmax per patient
prior_study_summarised_TIL_per_patient = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')

# TILmax t-tests
TILmax_LowResolution_pval = stats.ttest_ind(summarised_TIL_per_patient[(summarised_TIL_per_patient.Group=='LowResolutionSet')&(summarised_TIL_per_patient.TILmetric=='TILmax')]['value'].values,prior_study_summarised_TIL_per_patient[prior_study_summarised_TIL_per_patient.TILmetric=='TILmax']['value'].values,equal_var=False)
TILmax_HighResolution_pval = stats.ttest_ind(summarised_TIL_per_patient[(summarised_TIL_per_patient.Group=='HighResolutionSet')&(summarised_TIL_per_patient.TILmetric=='TILmax')]['value'].values,prior_study_summarised_TIL_per_patient[prior_study_summarised_TIL_per_patient.TILmetric=='TILmax']['value'].values,equal_var=False)

# TILmean t-tests
TILmean_LowResolution_pval = stats.ttest_ind(summarised_TIL_per_patient[(summarised_TIL_per_patient.Group=='LowResolutionSet')&(summarised_TIL_per_patient.TILmetric=='TILmean')]['value'].values,prior_study_summarised_TIL_per_patient[prior_study_summarised_TIL_per_patient.TILmetric=='TILmean']['value'].values,equal_var=False)
TILmean_HighResolution_pval = stats.ttest_ind(summarised_TIL_per_patient[(summarised_TIL_per_patient.Group=='HighResolutionSet')&(summarised_TIL_per_patient.TILmetric=='TILmean')]['value'].values,prior_study_summarised_TIL_per_patient[prior_study_summarised_TIL_per_patient.TILmetric=='TILmean']['value'].values,equal_var=False)

## Format TIL24s for characteristics table
# Melt CENTER-TBI TIL scores by population and filter
melted_TIL_scores = formatted_TIL_scores[['GUPI','TILTimepoint','LowResolutionSet','HighResolutionSet','TotalTIL']].melt(id_vars=['GUPI','TILTimepoint','TotalTIL'],var_name='Group')
melted_TIL_scores = melted_TIL_scores[melted_TIL_scores['value']==1].drop(columns='value').reset_index(drop=True)

# Calculate summarised TIL scores by day and sub-study
melted_TIL_scores = melted_TIL_scores.groupby(['TILTimepoint','Group'],as_index=False).TotalTIL.aggregate({'q1':lambda x: int(np.round(np.quantile(x,.25))),'median':lambda x: int(np.round(np.median(x))),'q3':lambda x: int(np.round(np.quantile(x,.75))),'count':'count'}).reset_index(drop=True)

# Filter to focus on first week
melted_TIL_scores = melted_TIL_scores[melted_TIL_scores.TILTimepoint<=7].reset_index(drop=True)

# Format for table
melted_TIL_scores['FormattedIQR'] = melted_TIL_scores['median'].astype(str)+' ('+melted_TIL_scores.q1.astype(str)+'–'+melted_TIL_scores.q3.astype(str)+')'

# Load prior study formatted TIL24s
prior_study_dailyTILs = pd.read_csv('../formatted_data/prior_study_formatted_TIL24_scores.csv')

# TIL24 low-resolution t-tests
TIL1_LowResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==1)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==1]['TotalTIL'].values,equal_var=False)
TIL2_LowResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==2)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==2]['TotalTIL'].values,equal_var=False)
TIL3_LowResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==3)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==3]['TotalTIL'].values,equal_var=False)
TIL4_LowResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==4)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==4]['TotalTIL'].values,equal_var=False)
TIL5_LowResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==5)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==5]['TotalTIL'].values,equal_var=False)
TIL6_LowResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==6)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==6]['TotalTIL'].values,equal_var=False)
TIL7_LowResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.LowResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==7)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==7]['TotalTIL'].values,equal_var=False)

# TIL24 high-resolution t-tests
TIL1_HighResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.HighResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==1)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==1]['TotalTIL'].values,equal_var=False)
TIL2_HighResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.HighResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==2)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==2]['TotalTIL'].values,equal_var=False)
TIL3_HighResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.HighResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==3)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==3]['TotalTIL'].values,equal_var=False)
TIL4_HighResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.HighResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==4)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==4]['TotalTIL'].values,equal_var=False)
TIL5_HighResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.HighResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==5)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==5]['TotalTIL'].values,equal_var=False)
TIL6_HighResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.HighResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==6)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==6]['TotalTIL'].values,equal_var=False)
TIL7_HighResolution_pval = stats.ttest_ind(formatted_TIL_scores[(formatted_TIL_scores.HighResolutionSet==1)&(formatted_TIL_scores.TILTimepoint==7)]['TotalTIL'].values,prior_study_dailyTILs[prior_study_dailyTILs.TILTimepoint==7]['TotalTIL'].values,equal_var=False)