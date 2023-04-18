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

# Calculate p-values for age
age_LowResolution_pval = stats.ttest_ind(CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.LowResolutionSet==1].values,CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.LowResolutionSet!=1].values,equal_var=False)
age_HighResolution_pval = stats.ttest_ind(CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.HighResolutionSet==1].values,CENTER_TBI_demo_outcome.Age[CENTER_TBI_demo_outcome.HighResolutionSet!=1].values,equal_var=False)

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

# # Filter patients who meet substudy assignment
# prog_scores_df = prog_scores_df[prog_scores_df.Marker==1].drop(columns='Marker').reset_index(drop=True)

# Calculate p-values
prog_scores_pvals = prog_scores_df.groupby(['Group','Threshold','Marker'])['Probability'].agg(['size','mean']).unstack('Marker')
prog_scores_pvals.columns = [f'group{0}_size',f'group{1}_size',f'group{0}_mean',f'group{1}_mean']
prog_scores_pvals['pvalue'] = prog_scores_df.groupby(['Group','Threshold']).apply(lambda dfx: stats.ttest_ind(dfx.loc[dfx['Marker'] == 0, 'Probability'],dfx.loc[dfx['Marker'] == 1, 'Probability'],equal_var=False).pvalue)
prog_scores_pvals = prog_scores_pvals.reset_index()

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
Sex_LowResolution_pval = stats.chi2_contingency(np.stack((CENTER_TBI_demo_outcome.Sex[CENTER_TBI_demo_outcome.LowResolutionSet!=1].value_counts().values,CENTER_TBI_demo_outcome.Sex[CENTER_TBI_demo_outcome.LowResolutionSet==1].value_counts().values)))
Sex_HighResolution_pval = stats.chi2_contingency(np.stack((CENTER_TBI_demo_outcome.Sex[CENTER_TBI_demo_outcome.HighResolutionSet!=1].value_counts().values,CENTER_TBI_demo_outcome.Sex[CENTER_TBI_demo_outcome.HighResolutionSet==1].value_counts().values)))

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
non_lo_res_severity_counts = CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.LowResolutionSet!=1].value_counts().sort_index().values
non_hi_res_severity_counts = CENTER_TBI_demo_outcome.GCSSeverity[CENTER_TBI_demo_outcome.HighResolutionSet!=1].value_counts().sort_index().values
severity_LowResolution_pval = stats.chi2_contingency(np.stack((non_lo_res_severity_counts,lo_res_severity_counts)))
severity_HighResolution_pval = stats.chi2_contingency(np.stack((non_hi_res_severity_counts,hi_res_severity_counts)))

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
non_lo_res_marshall_counts = np.insert(CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.LowResolutionSet!=1].value_counts().sort_index().values,3,0)
non_hi_res_marshall_counts = CENTER_TBI_demo_outcome.MarshallCT[CENTER_TBI_demo_outcome.HighResolutionSet!=1].value_counts().sort_index().values
marshall_LowResolution_pval = stats.chi2_contingency(np.stack((non_lo_res_marshall_counts,lo_res_marshall_counts)))
marshall_HighResolution_pval = stats.chi2_contingency(np.stack((non_hi_res_marshall_counts,hi_res_marshall_counts)))

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

# Calculate p-values for GOSE
lo_res_GOSE_counts = CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived[CENTER_TBI_demo_outcome.LowResolutionSet==1].value_counts().sort_index().values
hi_res_GOSE_counts = CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived[CENTER_TBI_demo_outcome.HighResolutionSet==1].value_counts().sort_index().values
non_lo_res_GOSE_counts = CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived[CENTER_TBI_demo_outcome.LowResolutionSet!=1].value_counts().sort_index().values
non_hi_res_GOSE_counts = CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived[CENTER_TBI_demo_outcome.HighResolutionSet!=1].value_counts().sort_index().values
GOSE_LowResolution_pval = stats.chi2_contingency(np.stack((non_lo_res_GOSE_counts,lo_res_GOSE_counts)))
GOSE_HighResolution_pval = stats.chi2_contingency(np.stack((non_hi_res_GOSE_counts,hi_res_GOSE_counts)))

# Caulcate p-values for TIL maxes and means
combined_TIL_means_maxes = pd.read_csv('../formatted_data/formatted_TIL_max.csv')[['GUPI','TILmax']].merge(pd.read_csv('../formatted_data/formatted_TIL_mean.csv')[['GUPI','TILmean']])
combined_TIL_means_maxes = combined_TIL_means_maxes.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']]).melt(id_vars=['GUPI','LowResolutionSet','HighResolutionSet'])
combined_TIL_means_maxes['Total'] = 1
combined_TIL_means_maxes = combined_TIL_means_maxes.melt(id_vars=['GUPI','variable','value'],var_name='Group',value_name='Marker')

TIL_means_maxes_pvals = combined_TIL_means_maxes.groupby(['Group','variable','Marker'])['value'].agg(['size','mean']).unstack('Marker')
TIL_means_maxes_pvals.columns = [f'group{0}_size',f'group{1}_size',f'group{0}_mean',f'group{1}_mean']
TIL_means_maxes_pvals['pvalue'] = combined_TIL_means_maxes.groupby(['Group','variable']).apply(lambda dfx: stats.ttest_ind(dfx.loc[dfx['Marker'] == 0, 'value'],dfx.loc[dfx['Marker'] == 1, 'value'],equal_var=False).pvalue)
TIL_means_maxes_pvals = TIL_means_maxes_pvals.reset_index()

combined_TIL_means_maxes = combined_TIL_means_maxes.groupby(['Group','Marker','variable'],as_index=False)['value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':lambda x: np.median(x),'q3':lambda x: np.quantile(x,.75),'count':'count'}).reset_index(drop=True)
combined_TIL_means_maxes['FormattedIQR'] = combined_TIL_means_maxes['median'].round(1).astype(str)+' ('+combined_TIL_means_maxes.q1.round(1).astype(str)+'–'+combined_TIL_means_maxes.q3.round(1).astype(str)+')'

# Caulcate p-values for TIL24 scores
combined_TIL_24s = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')[['GUPI','TILTimepoint','TotalSum']]
combined_TIL_24s = combined_TIL_24s.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']]).melt(id_vars=['GUPI','TILTimepoint','LowResolutionSet','HighResolutionSet'])
combined_TIL_24s['Total'] = 1
combined_TIL_24s = combined_TIL_24s.melt(id_vars=['GUPI','TILTimepoint','variable','value'],var_name='Group',value_name='Marker')

TIL_24s_pvals = combined_TIL_24s.groupby(['Group','TILTimepoint','variable','Marker'])['value'].agg(['size','mean']).unstack('Marker')
TIL_24s_pvals.columns = [f'group{0}_size',f'group{1}_size',f'group{0}_mean',f'group{1}_mean']
TIL_24s_pvals['pvalue'] = combined_TIL_24s.groupby(['Group','TILTimepoint','variable']).apply(lambda dfx: stats.ttest_ind(dfx.loc[dfx['Marker'] == 0, 'value'],dfx.loc[dfx['Marker'] == 1, 'value'],equal_var=False).pvalue)
TIL_24s_pvals = TIL_24s_pvals.reset_index()

combined_TIL_24s = combined_TIL_24s.groupby(['Group','TILTimepoint','Marker','variable'],as_index=False)['value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':lambda x: np.median(x),'q3':lambda x: np.quantile(x,.75),'count':'count'}).reset_index(drop=True)
combined_TIL_24s['FormattedIQR'] = combined_TIL_24s['median'].round(1).astype(str)+' ('+combined_TIL_24s.q1.round(1).astype(str)+'–'+combined_TIL_24s.q3.round(1).astype(str)+')'