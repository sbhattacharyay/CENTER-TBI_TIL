#### Master Script 5: Calculate metrics of missingness for analysis of study population ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Prepare missingness reports in static measures
# III. Prepare missingness report in longitudinal measures

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

### II. Prepare missingness reports in static measures
## Load and prepare static measures used in analysis
# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv',na_values = ["NA","NaN","NaT"," ", ""])

### III. Prepare missingness report in longitudinal measures
## Load and prepare longitudinal measures used in analysis
# Load and prepare study admission/discharge timestamps
CENTER_TBI_datetime = pd.read_csv('../timestamps/adm_disch_timestamps.csv')
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationDays'] = CENTER_TBI_datetime['ICUDurationHours']/24
CENTER_TBI_datetime = CENTER_TBI_datetime.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']],how='left')
CENTER_TBI_datetime['FullSet'] = 1

# Load and prepare formatted TIL scores over time
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_TIL_scores = formatted_TIL_scores[(formatted_TIL_scores.TILTimepoint<=7)&(formatted_TIL_scores.TILTimepoint>=1)].reset_index(drop=True)
formatted_TIL_scores = formatted_TIL_scores.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']],how='left')
formatted_TIL_scores['FullSet'] = 1

# Format substudy assignment for TIL dataframe
formatted_TIL_scores = formatted_TIL_scores.melt(id_vars=['GUPI','TILTimepoint','TotalSum','TILPhysicianConcernsCPP','TILPhysicianConcernsICP'], value_vars=['LowResolutionSet','HighResolutionSet','FullSet'],var_name='Substudy')
formatted_TIL_scores = formatted_TIL_scores[formatted_TIL_scores['value']==1].drop(columns='value').reset_index(drop=True)

# Filter timestamp dataframe to patients in study set
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime.GUPI.isin(formatted_TIL_scores.GUPI)].reset_index(drop=True)
CENTER_TBI_datetime = CENTER_TBI_datetime.melt(id_vars=['GUPI','ICUDurationDays'], value_vars=['LowResolutionSet','HighResolutionSet','FullSet'],var_name='Substudy')
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime['value']==1].drop(columns='value').reset_index(drop=True)

# Load and prepare formatted low-resolution neuromonitoring values over time
formatted_low_resolution_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_low_resolution_values = formatted_low_resolution_values[formatted_low_resolution_values.TILTimepoint<=7].reset_index(drop=True)

# Load and prepare formatted high-resolution neuromonitoring values over time
formatted_high_resolution_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_high_resolution_values = formatted_high_resolution_values[formatted_high_resolution_values.TILTimepoint<=7].reset_index(drop=True)

## Calculate number of patients remaining at points between admission and discharge
# Create a dummy vector for points between 0 and 7 days post-admission
days_vector = np.linspace(0,7,num=1000)

# Create empty running lists to store values
remaining_df = []

# Iterate through time vector
for curr_day in tqdm(days_vector,'Calculating n remaining over time'):
    # Count number of patients remaining at current timepoint
    curr_remaining_count = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUDurationDays>=curr_day].groupby('Substudy',as_index=False).ICUDurationDays.count().rename(columns={'ICUDurationDays':'n_value'})
    
    # Format current count dataframe
    curr_remaining_count['DaysSinceICUAdmission'] = curr_day
    curr_remaining_count['Type'] = 'RemainingInICU'

    # Add dataframe to running list
    remaining_df.append(curr_remaining_count)

# Organise lists into dataframe
counts_over_time = pd.concat(remaining_df,ignore_index=True)

# Reorder columns of dataframe and sort
counts_over_time =counts_over_time[['DaysSinceICUAdmission','n_value','Type','Substudy']]

## Create longitudinal TIL missingness profiles for plotting
# Calculate count of TIL scores over time
TIL_counts_over_time = formatted_TIL_scores.groupby(['TILTimepoint','Substudy'],as_index=False).TotalSum.count().rename(columns={'TILTimepoint':'DaysSinceICUAdmission','TotalSum':'n_value'})

# Create copy with shifted timepoint for stepwise plotting
shifted_TIL_counts_over_time = TIL_counts_over_time.copy()
shifted_TIL_counts_over_time['DaysSinceICUAdmission'] = shifted_TIL_counts_over_time['DaysSinceICUAdmission']-1

# Concatenate two dataframes and add column designating TIL type
TIL_counts_over_time = pd.concat([TIL_counts_over_time,shifted_TIL_counts_over_time],ignore_index=True)
TIL_counts_over_time['Type'] = 'TILAvailable'

# Concatenate with patients remaining in ICU over time
counts_over_time = pd.concat([counts_over_time,TIL_counts_over_time],ignore_index=False)

## Create longitudinal ICP missingness profiles for plotting
# Calculate count of low-resolution ICP/CPP scores over time
ICP_EH_counts_over_time = formatted_low_resolution_values.groupby('TILTimepoint',as_index=False).ICPmean.count().rename(columns={'TILTimepoint':'DaysSinceICUAdmission','ICPmean':'n_value'})
ICP_EH_counts_over_time['Type'] = 'ICPAvailable'
ICP_EH_counts_over_time['Substudy'] = 'LowResolutionSet'
CPP_EH_counts_over_time = formatted_low_resolution_values.groupby('TILTimepoint',as_index=False).CPPmean.count().rename(columns={'TILTimepoint':'DaysSinceICUAdmission','CPPmean':'n_value'})
CPP_EH_counts_over_time['Type'] = 'CPPAvailable'
CPP_EH_counts_over_time['Substudy'] = 'LowResolutionSet'

# Calculate count of high-resolution ICP/CPP scores over time
ICP_HR_counts_over_time = formatted_high_resolution_values.groupby('TILTimepoint',as_index=False).ICPmean.count().rename(columns={'TILTimepoint':'DaysSinceICUAdmission','ICPmean':'n_value'})
ICP_HR_counts_over_time['Type'] = 'ICPAvailable'
ICP_HR_counts_over_time['Substudy'] = 'HighResolutionSet'
CPP_HR_counts_over_time = formatted_high_resolution_values.groupby('TILTimepoint',as_index=False).CPPmean.count().rename(columns={'TILTimepoint':'DaysSinceICUAdmission','CPPmean':'n_value'})
CPP_HR_counts_over_time['Type'] = 'CPPAvailable'
CPP_HR_counts_over_time['Substudy'] = 'HighResolutionSet'

# Create shifted dataframe for step-wise count of daily measures
ICP_CPP_counts_over_time = pd.concat([ICP_EH_counts_over_time,CPP_EH_counts_over_time,ICP_HR_counts_over_time,CPP_HR_counts_over_time],ignore_index=False)
shifted_ICP_CPP_counts_over_time = ICP_CPP_counts_over_time.copy()
shifted_ICP_CPP_counts_over_time['DaysSinceICUAdmission'] = shifted_ICP_CPP_counts_over_time['DaysSinceICUAdmission']-1
ICP_CPP_counts_over_time = pd.concat([ICP_CPP_counts_over_time,shifted_ICP_CPP_counts_over_time],ignore_index=True)

# Concatenate with combined count dataframes
counts_over_time = pd.concat([counts_over_time,ICP_CPP_counts_over_time],ignore_index=False)

## Create longitudinal physician concern missingness profiles for plotting
# Calculate count of ICP/CPP concerns over time
ICP_concerns_counts_over_time = formatted_TIL_scores.groupby(['TILTimepoint','Substudy'],as_index=False).TILPhysicianConcernsICP.count().rename(columns={'TILTimepoint':'DaysSinceICUAdmission','TILPhysicianConcernsICP':'n_value'})
ICP_concerns_counts_over_time['Type'] = 'ConcernICPAvailable'
CPP_concerns_counts_over_time = formatted_TIL_scores.groupby(['TILTimepoint','Substudy'],as_index=False).TILPhysicianConcernsCPP.count().rename(columns={'TILTimepoint':'DaysSinceICUAdmission','TILPhysicianConcernsCPP':'n_value'})
CPP_concerns_counts_over_time['Type'] = 'ConcernCPPAvailable'

# Create shifted dataframe for step-wise count of daily measures
ICP_CPP_concern_counts_over_time = pd.concat([ICP_concerns_counts_over_time,CPP_concerns_counts_over_time],ignore_index=False)
shifted_ICP_CPP_concern_counts_over_time = ICP_CPP_concern_counts_over_time.copy()
shifted_ICP_CPP_concern_counts_over_time['DaysSinceICUAdmission'] = shifted_ICP_CPP_concern_counts_over_time['DaysSinceICUAdmission']-1
ICP_CPP_concern_counts_over_time = pd.concat([ICP_CPP_concern_counts_over_time,shifted_ICP_CPP_concern_counts_over_time],ignore_index=True)

# Concatenate with combined count dataframes
counts_over_time = pd.concat([counts_over_time,ICP_CPP_concern_counts_over_time],ignore_index=False)

## Reorder and save combined longitudinal count dataframe for plotting
# Reorder combined count dataframe
counts_over_time = counts_over_time.sort_values(['Substudy','Type','DaysSinceICUAdmission'],ignore_index=True)

# Save combined count dataframe
counts_over_time.to_csv('../formatted_data/longitudinal_data_availability.csv',index=False)

## Calculate differences in TIL and baseline characteristics in each longitudinal population gap
# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv',na_values = ["NA","NaN","NaT"," ", ""])

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

# Convert prognostic probabilities to percentages
prog_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('Pr(GOSE>')]
CENTER_TBI_demo_outcome[prog_cols] = CENTER_TBI_demo_outcome[prog_cols]*100

# Load TILmax and TILmean values
formatted_TIL_max = pd.read_csv('../formatted_data/formatted_TIL_max.csv')
formatted_TIL_mean = pd.read_csv('../formatted_data/formatted_TIL_mean.csv')

char_set = CENTER_TBI_demo_outcome.merge(formatted_TIL_max[['GUPI','TILmax']],how='left').merge(formatted_TIL_mean[['GUPI','TILmean']],how='left')
char_set = char_set[['GUPI', 'SiteCode', 'Age', 'Sex','GCSSeverity', 'GOSE6monthEndpointDerived','RefractoryICP','MarshallCT','Pr(GOSE>1)', 'Pr(GOSE>3)','Pr(GOSE>4)', 'Pr(GOSE>5)', 'Pr(GOSE>6)', 'Pr(GOSE>7)','TILmax','TILmean']]
data_set = formatted_TIL_scores[formatted_TIL_scores.Substudy=='LowResolutionSet'].reset_index(drop=True).merge(formatted_low_resolution_values[['GUPI','TILTimepoint','TotalSum','ICPmean']],how='left')
timepoints = [1,2,3,4,5,6,7]
chosen_cols = ['ICPmean','TILPhysicianConcernsICP']

a_1, a_2 = long_missingness_analysis(char_set,data_set,timepoints,chosen_cols)

char_set = CENTER_TBI_demo_outcome.merge(formatted_TIL_max[['GUPI','TILmax']],how='left').merge(formatted_TIL_mean[['GUPI','TILmean']],how='left')
char_set = char_set[['GUPI', 'SiteCode', 'Age', 'Sex','GCSSeverity', 'GOSE6monthEndpointDerived','RefractoryICP','MarshallCT','Pr(GOSE>1)', 'Pr(GOSE>3)','Pr(GOSE>4)', 'Pr(GOSE>5)', 'Pr(GOSE>6)', 'Pr(GOSE>7)','TILmax','TILmean']]
data_set = formatted_TIL_scores[formatted_TIL_scores.Substudy=='HighResolutionSet'].reset_index(drop=True).merge(formatted_high_resolution_values[['GUPI','TILTimepoint','TotalSum','ICPmean']],how='left')
timepoints = [1,2,3,4,5,6,7]
chosen_cols = ['ICPmean','TILPhysicianConcernsICP']

b_1, b_2 = long_missingness_analysis(char_set,data_set,timepoints,chosen_cols)

# Function to calculate summary statistics stratified by missingness of longitudinal measures
def long_missingness_analysis(char_set,data_set,timepoints,chosen_cols):
    # Create empty running lists to store numeric and categorical variable summary statistics
    num_summary_stats = []
    cat_summary_stats = []
    
    # Iterate through columns of interest
    for curr_col in chosen_cols:

        # Iterate through timepoints of interest
        for curr_tp in timepoints:
            
            # Filter datasets to current timepoint
            curr_filt_dataset = data_set[(data_set.TILTimepoint==curr_tp)].reset_index(drop=True)
            curr_filt_charset = char_set[char_set.GUPI.isin(curr_filt_dataset.GUPI)].reset_index(drop=True)

            # Add daily TIL scores to current, filtered characteristics set
            curr_filt_charset = curr_filt_charset.merge(curr_filt_dataset[['GUPI','TotalSum']],how='left').rename(columns={'TotalSum':'TIL24'})

            # Extract patient GUPIs with non-missing value for variable of interest
            curr_in_set = pd.DataFrame({'Set':'In','GUPI':curr_filt_dataset[~curr_filt_dataset[curr_col].isna()].GUPI.unique()})
            curr_out_set = pd.DataFrame({'Set':'Out','GUPI':curr_filt_dataset[(curr_filt_dataset[curr_col].isna())&(~curr_filt_dataset['TotalSum'].isna())].GUPI.unique()})
            curr_set_key = pd.concat([curr_in_set,curr_out_set],ignore_index=True)

            # Add set placement to characteristic dataframe
            curr_filt_charset = curr_filt_charset.merge(curr_set_key,how='left')

            # Divide characteristics into numeric and categorical and melt into long form
            num_filt_charset = curr_filt_charset[['GUPI','Set','Age','Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)','TILmax','TILmean','TIL24']].melt(id_vars=['GUPI','Set'],value_vars=['Age','Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)','TILmax','TILmean','TIL24']).dropna().reset_index(drop=True)
            cat_filt_charset = curr_filt_charset[['GUPI','Set','SiteCode','Sex','GCSSeverity','GOSE6monthEndpointDerived','RefractoryICP','MarshallCT']].melt(id_vars=['GUPI','Set'],value_vars=['SiteCode','Sex','GCSSeverity','GOSE6monthEndpointDerived','RefractoryICP','MarshallCT']).dropna().reset_index(drop=True)
            cat_filt_charset['value'] = cat_filt_charset['value'].astype(str)

            # First, calculate summary statistics for each numeric variable
            curr_num_summary_stats = num_filt_charset.groupby(['variable','Set'],as_index=False)['value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':np.median,'q3':lambda x: np.quantile(x,.75),'n':'count'}).reset_index(drop=True)
            curr_num_summary_stats.insert(1,'DaysSinceICUAdmission',curr_tp)
            curr_num_summary_stats.insert(2,'MissingVariable',curr_col)

            # Second, calculate p-value for each numeric variable comparison and add to dataframe
            curr_num_summary_stats = curr_num_summary_stats.merge(num_filt_charset.groupby('variable',as_index=False).apply(lambda x: stats.ttest_ind(x['value'][x.Set=='In'].values,x['value'][x.Set=='Out'].values,equal_var=False).pvalue).rename(columns={None:'p_val'}),how='left')

            # Third, calculate summary characteristics for each categorical variable
            curr_cat_summary_stats = cat_filt_charset.groupby(['variable','Set','value'],as_index=False).GUPI.count().rename(columns={'GUPI':'n'}).merge(cat_filt_charset.groupby(['variable','Set'],as_index=False).GUPI.count().rename(columns={'GUPI':'n_total'}),how='left')
            curr_cat_summary_stats['proportion'] = 100*(curr_cat_summary_stats['n']/curr_cat_summary_stats['n_total'])
            curr_cat_summary_stats.insert(1,'DaysSinceICUAdmission',curr_tp)
            curr_cat_summary_stats.insert(2,'MissingVariable',curr_col)
            
            # Fourth, calculate p-value for each categorical variable comparison and add to dataframe
            curr_cat_summary_stats = curr_cat_summary_stats.merge(cat_filt_charset.groupby('variable',as_index=False).apply(lambda x: stats.chi2_contingency(pd.crosstab(x["value"],x["Set"])).pvalue).rename(columns={None:'p_val'}),how='left')

            # Append current summary statistics to running lists
            num_summary_stats.append(curr_num_summary_stats)
            cat_summary_stats.append(curr_cat_summary_stats)

    # Concatenate running lists
    num_summary_stats = pd.concat(num_summary_stats,ignore_index=True)
    cat_summary_stats = pd.concat(cat_summary_stats,ignore_index=True)

    # Return results
    return(num_summary_stats,cat_summary_stats)