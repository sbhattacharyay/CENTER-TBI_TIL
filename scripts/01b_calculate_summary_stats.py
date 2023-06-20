#### Master Script 1b: Calculate summary statistics and missingness of different study sub-samples ####
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
# V. Assess missingness of TIL scores

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
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv',na_values = ["NA","NaN","NaT"," ", ""])

# Add column designating "overall set" inclusion
CENTER_TBI_demo_outcome['OverallSet'] = 1

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

# Convert prognostic probabilities to percentages
prog_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('Pr(GOSE>')]
CENTER_TBI_demo_outcome[prog_cols] = CENTER_TBI_demo_outcome[prog_cols]*100

# Load TILmax, TILmedian, TILmean, and TIL24 values
formatted_TIL_max = pd.read_csv('../formatted_data/formatted_TIL_max.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_TIL_mean = pd.read_csv('../formatted_data/formatted_TIL_mean.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_TIL_median = pd.read_csv('../formatted_data/formatted_TIL_median.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv',na_values = ["NA","NaN","NaT"," ", ""])

# Convert first week TIL24 scores to wide form
formatted_TIL_scores = formatted_TIL_scores[(formatted_TIL_scores.TILTimepoint<=7)&(formatted_TIL_scores.TILTimepoint>=1)].reset_index(drop=True)
formatted_TIL_scores['TILTimepoint'] = 'TIL24_Day' + formatted_TIL_scores['TILTimepoint'].astype(str)
formatted_TIL_24 = pd.pivot_table(formatted_TIL_scores[['GUPI','TILTimepoint','TotalSum']], values = 'TotalSum', index=['GUPI'], columns = 'TILTimepoint').reset_index().rename(columns={'TotalSum':'TIL24'})

# Merge summary TIL scores to demographics/outcome dataframe
CENTER_TBI_demo_outcome = CENTER_TBI_demo_outcome.merge(formatted_TIL_max[['GUPI','TILmax']],how='left').merge(formatted_TIL_mean[['GUPI','TILmean']],how='left').merge(formatted_TIL_median[['GUPI','TILmedian']],how='left').merge(formatted_TIL_24,how='left')

## Define summary characteristic columns for analysis
# Extract names of columns which begin with 'TIL'
TIL_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('TIL')]

# Define numeric summary statistics
num_cols = ['Age'] + prog_cols + TIL_cols

# Define categorical summary statistics
cat_cols = ['Sex','Race','GOSE6monthEndpointDerived','GCSSeverity','RefractoryICP','MarshallCT']

## Calculate basic count stats
# Total TIL population
n_total = CENTER_TBI_demo_outcome.shape[0]

# TIL-LowResolutionSet population
n_TIL_LowResolutionSet = CENTER_TBI_demo_outcome.LowResolutionSet.sum()

# TIL-HighResolutionSet population
n_TIL_HighResolutionSet = CENTER_TBI_demo_outcome.HighResolutionSet.sum()

## Calculate number of centres per sub-study
# Total TIL population
centres_total = CENTER_TBI_demo_outcome.SiteCode.nunique()

# TIL-LowResolutionSet population
centres_TIL_LowResolutionSet = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1].SiteCode.nunique()

# TIL-HighResolutionSet population
centres_TIL_HighResolutionSet = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1].SiteCode.nunique()

### III. Calculate summary statistics and statistical tests of numerical variables
## Filter and prepare numeric characteristics
# Convert set assignment to long-form
num_charset = CENTER_TBI_demo_outcome.melt(id_vars=CENTER_TBI_demo_outcome.columns.difference(['HighResolutionSet','LowResolutionSet','OverallSet']),value_vars=['HighResolutionSet','LowResolutionSet','OverallSet'],var_name='Set',value_name='SetIndicator')

# Select numeric characteristics and convert to long-form
num_charset = num_charset[['GUPI','Set','SetIndicator']+num_cols].melt(id_vars=['GUPI','Set','SetIndicator'],value_vars=num_cols).dropna().reset_index(drop=True)

## Calculate summary statistics and p-values for numeric characteristics
# First, calculate summary statistics for each numeric variable
num_summary_stats = num_charset.groupby(['variable','Set','SetIndicator'],as_index=False)['value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':np.median,'q3':lambda x: np.quantile(x,.75),'n':'count'}).reset_index(drop=True)

# Add a formatted confidence interval
num_summary_stats['FormattedCI'] = num_summary_stats['median'].round(1).astype(str)+' ('+num_summary_stats['q1'].round(1).astype(str)+'–'+num_summary_stats['q3'].round(1).astype(str)+')'

# Second, calculate p-value for each numeric variable comparison and add to dataframe
num_summary_stats = num_summary_stats.merge(num_charset.groupby(['variable','Set'],as_index=False).apply(lambda x: stats.ttest_ind(x['value'][x.SetIndicator==1].values,x['value'][x.SetIndicator==0].values,equal_var=False).pvalue).rename(columns={None:'p_val'}),how='left')

# # Filter rows to only include in-set results
# num_summary_stats = num_summary_stats[num_summary_stats.SetIndicator==1].reset_index(drop=True)

## Save results
num_summary_stats.to_excel('../formatted_data/numerical_summary_statistics.xlsx')

### IV. Calculate summary statistics and statistical tests of categorical variables
## Filter and prepare categorical characteristics
# Convert set assignment to long-form
cat_charset = CENTER_TBI_demo_outcome.melt(id_vars=CENTER_TBI_demo_outcome.columns.difference(['HighResolutionSet','LowResolutionSet','OverallSet']),value_vars=['HighResolutionSet','LowResolutionSet','OverallSet'],var_name='Set',value_name='SetIndicator')

# Select categorical characteristics and convert to long-form
cat_charset = cat_charset[['GUPI','Set','SetIndicator']+cat_cols].melt(id_vars=['GUPI','Set','SetIndicator'],value_vars=cat_cols).dropna().reset_index(drop=True)
cat_charset['value'] = cat_charset['value'].astype(str)

# First, calculate summary characteristics for each categorical variable
cat_summary_stats = cat_charset.groupby(['variable','Set','SetIndicator','value'],as_index=False).GUPI.count().rename(columns={'GUPI':'n'}).merge(cat_charset.groupby(['variable','Set','SetIndicator'],as_index=False).GUPI.count().rename(columns={'GUPI':'n_total'}),how='left')
cat_summary_stats['proportion'] = 100*(cat_summary_stats['n']/cat_summary_stats['n_total'])

# Add a formatted proportion entry
cat_summary_stats['FormattedProp'] = cat_summary_stats['n'].astype(str)+' ('+cat_summary_stats['proportion'].round().astype(int).astype(str)+'%)'

# Then, calculate p-value for each categorical variable comparison and add to dataframe
cat_summary_stats = cat_summary_stats.merge(cat_charset.groupby(['variable','Set'],as_index=False).apply(lambda x: stats.chi2_contingency(pd.crosstab(x["value"],x["SetIndicator"])).pvalue).rename(columns={None:'p_val'}),how='left')

## Save results
cat_summary_stats.to_excel('../formatted_data/categorical_summary_statistics.xlsx')

### V. Assess missingness of TIL scores
## Load and prepare pertinent dataframes
# Load formatted TIL score dataframe
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv',na_values = ["NA","NaN","NaT"," ", ""])

# Load ICU timestamp dataframe and filter to study population
CENTER_TBI_datetime = pd.read_csv('../timestamps/adm_disch_timestamps.csv')
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime.GUPI.isin(formatted_TIL_scores.GUPI)].reset_index(drop=True)

