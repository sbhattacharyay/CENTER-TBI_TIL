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
# V. Prepare missingness report in longitudinal measures

### I. Initialisation
## Import libraries and prepare environment
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

# Custom methods
from functions.analysis import long_missingness_analysis

## Initialise directory placeholders
# Initialise directory to store formatted data
formatted_data_dir = '../formatted_data'

# Initialise results directory
results_dir = '../results'

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

## Save results to results dataframe
num_summary_stats.to_excel(os.path.join(results_dir,'numerical_summary_statistics.xlsx'))

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
cat_summary_stats.to_excel(os.path.join(results_dir,'categorical_summary_statistics.xlsx'))

### V. Prepare missingness report in longitudinal measures
## Load and prepare pertinent dataframes
# Load formatted demographics and outcome dataframe to extract set assignment
CENTER_TBI_demo_info = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv',na_values = ["NA","NaN","NaT"," ", ""])
CENTER_TBI_demo_info['OverallSet'] = 1

# Load formatted TIL score dataframe
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv',na_values = ["NA","NaN","NaT"," ", ""])

# Load and prepare formatted low-resolution neuromonitoring values over time
formatted_low_resolution_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])

# Load and prepare formatted high-resolution neuromonitoring values over time
formatted_high_resolution_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])

## Identify expected cases that are missing
# Filter to study window
expected_combinations = formatted_TIL_scores[(formatted_TIL_scores.TILTimepoint>=1)&(formatted_TIL_scores.TILTimepoint<=7)].reset_index(drop=True)[['GUPI','TILTimepoint','TotalSum','TILPhysicianConcernsICP']]

# Add information of missingess of low-resolution ICP
expected_combinations = expected_combinations.merge(formatted_low_resolution_values[['GUPI','TILTimepoint','ICPmean']],how='left').rename(columns={'ICPmean':'ICP_EH'})

# Add information of missingess of high-resolution ICP
expected_combinations = expected_combinations.merge(formatted_high_resolution_values[['GUPI','TILTimepoint','ICPmean']],how='left').rename(columns={'ICPmean':'ICP_HR'})

# Add set assignment information to merged expected combinations
expected_combinations = expected_combinations.merge(CENTER_TBI_demo_info[['GUPI','OverallSet','LowResolutionSet','HighResolutionSet']],how='left')

## Create summary statistics of longitudinal missingess
# Pivot set assignment to longer form
expected_combinations = expected_combinations.melt(id_vars=['GUPI','TILTimepoint','TotalSum','TILPhysicianConcernsICP','ICP_EH','ICP_HR'],value_vars=['OverallSet','LowResolutionSet','HighResolutionSet'],value_name='SetIndicator',var_name='Set')

# Melt variable values to longer form
expected_combinations = expected_combinations.melt(id_vars=['Set','SetIndicator','GUPI','TILTimepoint'],value_vars=['TotalSum','TILPhysicianConcernsICP','ICP_EH','ICP_HR'],value_name='Value',var_name='Variable')

# Add missingness indicator
expected_combinations['MissingValues'] = expected_combinations['Value'].isna().map({True:'Miss',False:'NonMiss'})
expected_combinations['MissingValues'] = expected_combinations['Variable']+expected_combinations['MissingValues']

# Remove implausible combinations
expected_combinations = expected_combinations[(~expected_combinations.Variable.str.startswith('ICP'))|((expected_combinations.Variable.str.endswith('EH'))&(expected_combinations.Set=='LowResolutionSet'))|((expected_combinations.Variable.str.endswith('HR'))&(expected_combinations.Set=='HighResolutionSet'))]
expected_combinations = expected_combinations[expected_combinations.SetIndicator==1].reset_index(drop=True)

# Remove '_HR' and '_EH' suffixes
expected_combinations['MissingValues'] = expected_combinations['MissingValues'].str.replace('_EH','')
expected_combinations['MissingValues'] = expected_combinations['MissingValues'].str.replace('_HR','')

# Create missingness groups per patient
patient_missing_combos = expected_combinations.groupby(['Set','SetIndicator','TILTimepoint','GUPI'],as_index=False)['MissingValues'].apply(';'.join).reset_index()
patient_missing_combos['MissingValues'][patient_missing_combos['MissingValues']=='TotalSumNonMiss;TILPhysicianConcernsICPNonMiss'] = 'TotalSumNonMiss;TILPhysicianConcernsICPNonMiss;ICPNonMiss'
patient_missing_combos['MissingValues'][patient_missing_combos['MissingValues']=='TotalSumNonMiss;TILPhysicianConcernsICPMiss'] = 'TotalSumNonMiss;TILPhysicianConcernsICPMiss;ICPNonMiss'
patient_missing_combos['MissingValues'][patient_missing_combos['MissingValues']=='TotalSumMiss;TILPhysicianConcernsICPMiss'] = 'TotalSumMiss;TILPhysicianConcernsICPMiss;ICPNonMiss'

# Calculate number missing out of total available
missing_combination_counts = patient_missing_combos.groupby(['Set','TILTimepoint','MissingValues'],as_index=False)['GUPI'].count().rename(columns={'GUPI':'Count','MissingValues':'Combination'})
missing_combination_counts['TotalCount'] = missing_combination_counts.groupby(['Set','TILTimepoint'])['Count'].transform('sum')

# Add a formatted column designating percent per combination
missing_combination_counts['Proportion'] = (missing_combination_counts.Count/missing_combination_counts.TotalCount)

# Reorder combined count dataframe
missing_combination_counts = missing_combination_counts.sort_values(['Set','Combination','TILTimepoint'],ignore_index=True)

# Save combined count dataframe
missing_combination_counts.to_csv(os.path.join(results_dir,'longitudinal_data_availability.csv'),index=False)

## Calculate number of patients remaining at points between admission and discharge
# Load timestamp dataframe
CENTER_TBI_datetime = pd.read_csv('../timestamps/adm_disch_timestamps.csv')

# Convert ICU admission/discharge timestamps to datetime variables
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationDays'] = CENTER_TBI_datetime['ICUDurationHours']/24
CENTER_TBI_datetime = CENTER_TBI_datetime.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']],how='left')
CENTER_TBI_datetime['OverallSet'] = 1

# Filter timestamp dataframe to patients in study set
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime.GUPI.isin(formatted_TIL_scores.GUPI)].reset_index(drop=True)
CENTER_TBI_datetime = CENTER_TBI_datetime.melt(id_vars=['GUPI','ICUDurationDays'], value_vars=['LowResolutionSet','HighResolutionSet','OverallSet'],var_name='Set')
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime['value']==1].drop(columns='value').reset_index(drop=True)

# Create a dummy vector for points between 0 and 7 days post-admission
days_vector = np.linspace(0,7,num=1000)

# Create empty running lists to store values
remaining_df = []

# Iterate through time vector
for curr_day in tqdm(days_vector,'Calculating n remaining over time'):
    # Count number of patients remaining at current timepoint
    curr_remaining_count = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUDurationDays>=curr_day].groupby('Set',as_index=False).ICUDurationDays.count().rename(columns={'ICUDurationDays':'NonMissingCount'})
    
    # Format current count dataframe
    curr_remaining_count['DaysSinceICUAdmission'] = curr_day
    curr_remaining_count['Type'] = 'RemainingInICU'

    # Add dataframe to running list
    remaining_df.append(curr_remaining_count)

# Organise lists into dataframe
counts_over_time = pd.concat(remaining_df,ignore_index=True)

# Reorder columns of dataframe and sort
counts_over_time = counts_over_time[['DaysSinceICUAdmission','NonMissingCount','Type','Set']]

# Save remaining in ICU count dataframe
counts_over_time.to_csv(os.path.join(results_dir,'remaining_in_icu_curve.csv'),index=False)

## Calculate significance of baseline characteristic differences associated with longitudinal missingess
# Load and prepare formatted TIL scores over time
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_TIL_scores = formatted_TIL_scores[(formatted_TIL_scores.TILTimepoint<=7)&(formatted_TIL_scores.TILTimepoint>=1)].reset_index(drop=True)
formatted_TIL_scores = formatted_TIL_scores.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']],how='left')
formatted_TIL_scores['OverallSet'] = 1

# Format substudy assignment for TIL dataframe
formatted_TIL_scores = formatted_TIL_scores.melt(id_vars=['GUPI','TILTimepoint','TotalSum','TILPhysicianConcernsCPP','TILPhysicianConcernsICP'], value_vars=['LowResolutionSet','HighResolutionSet','OverallSet'],var_name='Set')
formatted_TIL_scores = formatted_TIL_scores[formatted_TIL_scores['value']==1].drop(columns='value').reset_index(drop=True)

# Load and prepare formatted low-resolution neuromonitoring values over time
formatted_low_resolution_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_low_resolution_values = formatted_low_resolution_values[formatted_low_resolution_values.TILTimepoint<=7].reset_index(drop=True)

# Load and prepare formatted high-resolution neuromonitoring values over time
formatted_high_resolution_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_high_resolution_values = formatted_high_resolution_values[formatted_high_resolution_values.TILTimepoint<=7].reset_index(drop=True)

# Select characteristics for calculation
char_set = CENTER_TBI_demo_outcome[['GUPI', 'SiteCode', 'Age', 'Sex','GCSSeverity', 'GOSE6monthEndpointDerived','RefractoryICP','MarshallCT','Pr(GOSE>1)', 'Pr(GOSE>3)','Pr(GOSE>4)', 'Pr(GOSE>5)', 'Pr(GOSE>6)', 'Pr(GOSE>7)','TILmax','TILmedian']]

# Filter datasets by substudy
overall_dataset = formatted_TIL_scores[formatted_TIL_scores.Set=='OverallSet'].reset_index(drop=True)
lores_dataset = formatted_TIL_scores[formatted_TIL_scores.Set=='LowResolutionSet'].reset_index(drop=True).merge(formatted_low_resolution_values[['GUPI','TILTimepoint','TotalSum','ICPmean']],how='left')
hires_dataset = formatted_TIL_scores[formatted_TIL_scores.Set=='HighResolutionSet'].reset_index(drop=True).merge(formatted_high_resolution_values[['GUPI','TILTimepoint','TotalSum','ICPmean']],how='left')

# Calculate characteristic differences
overall_num_char_diffs, overall_cat_char_diffs = long_missingness_analysis(char_set,overall_dataset,[1,2,3,4,5,6,7],['TotalSum','TILPhysicianConcernsICP'])
lores_num_char_diffs, lores_cat_char_diffs = long_missingness_analysis(char_set,lores_dataset,[1,2,3,4,5,6,7],['TotalSum','TILPhysicianConcernsICP','ICPmean'])
hires_num_char_diffs, hires_cat_char_diffs = long_missingness_analysis(char_set,hires_dataset,[1,2,3,4,5,6,7],['TotalSum','TILPhysicianConcernsICP','ICPmean'])

# Add set code to all result dataframes
overall_num_char_diffs['Substudy'] = 'OverallSet'
overall_cat_char_diffs['Substudy'] = 'OverallSet'
lores_num_char_diffs['Substudy'] = 'LowResolutionSet'
lores_cat_char_diffs['Substudy'] = 'LowResolutionSet'
hires_num_char_diffs['Substudy'] = 'HighResolutionSet'
hires_cat_char_diffs['Substudy'] = 'HighResolutionSet'

# Append character and categorical dataframes
num_char_diffs = pd.concat([overall_num_char_diffs,lores_num_char_diffs,hires_num_char_diffs],ignore_index=False)
cat_char_diffs = pd.concat([overall_cat_char_diffs,lores_cat_char_diffs,hires_cat_char_diffs],ignore_index=False)

# Format salient details
num_char_diffs['FormattedLabel'] = num_char_diffs['median'].round(1).astype(str)+' ('+num_char_diffs['q1'].round(1).astype(str)+'–'+num_char_diffs['q3'].round(1).astype(str)+')'
num_char_diffs['value'] = ''
cat_char_diffs['FormattedLabel'] = cat_char_diffs['n'].astype(str)+' ('+cat_char_diffs['proportion'].round().astype(int).astype(str)+'%)'
cat_char_diffs = cat_char_diffs.drop(columns=['n']).rename(columns={'n_total':'n'})

# Concatenate formatted results dataframe
char_diffs = pd.concat([num_char_diffs[['Substudy','DaysSinceICUAdmission','MissingVariable','Set','variable','value','n','FormattedLabel','p_val']],
                        cat_char_diffs[['Substudy','DaysSinceICUAdmission','MissingVariable','Set','variable','value','n','FormattedLabel','p_val']]],ignore_index=False)

# Calculate number of unique centres per combination
site_code_replacements = char_diffs[char_diffs['variable']=='SiteCode'].groupby(['Substudy','DaysSinceICUAdmission','MissingVariable','Set','variable','n','p_val'],as_index=False)['value'].count().rename(columns={'value':'FormattedLabel'})
site_code_replacements['value'] = ''

# Replace side code information
char_diffs = pd.concat([char_diffs[char_diffs['variable']!='SiteCode'].reset_index(drop=True),site_code_replacements],ignore_index=True)

# Reorder characteristic difference dataframe
char_diffs = char_diffs.sort_values(by=['Substudy','DaysSinceICUAdmission','MissingVariable','variable','value','Set'],ignore_index=True)

# Remove redundant rows
char_diffs = char_diffs[(~((char_diffs['variable']=='Sex')&(char_diffs['value']=='M')))&(~((char_diffs['variable']=='RefractoryICP')&(char_diffs['value']=='0.0')))].reset_index(drop=True)

# Save characteristic differences dataframe
char_diffs.to_csv(os.path.join(results_dir,'longitudinal_missingness_analysis.csv'),index=False)