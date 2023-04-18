#### Master Script 1a: Extract and prepare study sample covariates from CENTER-TBI dataset ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and prepare initial study set
# III. Load and prepare hourly changes in TIL
# IV. Load and prepare low-resolution ICP and CPP information
# V. Load and prepare high-resolution ICP and CPP information
# VI. Load and prepare demographic information and baseline characteristics
# VII. Calculate TIL_1987, PILOT, and TIL_Basic
# VIII. Calculate summarised TIL metrics
# IX. Load and prepare serum sodium values from CENTER-TBI

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
from functions.analysis import calculate_TILsum, calculate_TIL_1987, calculate_PILOT, calculate_TIL_Basic

### II. Load and prepare initial study set
## Load and prepare timestamp dataframe
# Load timestamp dataframe
CENTER_TBI_datetime = pd.read_csv('../timestamps/adm_disch_timestamps.csv')

# Convert ICU admission/discharge timestamps to datetime variables
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Load demographic information dataframe
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN","NaT"," ", ""])

# Select desired basic demographic information
CENTER_TBI_demo_info = CENTER_TBI_demo_info[['GUPI','SiteCode','Age','Sex']]

# Merge basic demographic information with ICU timestamps
CENTER_TBI_datetime = CENTER_TBI_demo_info.merge(CENTER_TBI_datetime,how='right')

# Apply inclusion criteria no. 1: age >= 16
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime.Age >= 16].reset_index(drop=True)

## Load and prepare TIL information
# Load DailyTIL dataframe
daily_TIL_info = pd.read_csv('../CENTER-TBI/DailyTIL/data.csv',na_values = ["NA","NaN"," ", ""])

# Remove all entries without date or `TILTimepoint`
daily_TIL_info = daily_TIL_info[(daily_TIL_info.TILTimepoint!='None')|(~daily_TIL_info.TILDate.isna())].reset_index(drop=True)

# Remove all TIL entries with NA for all data variable columns
true_var_columns = daily_TIL_info.columns.difference(['GUPI', 'TILTimepoint', 'TILDate', 'TILTime','DailyTILCompleteStatus','TotalTIL','TILFluidCalcStartDate','TILFluidCalcStartTime','TILFluidCalcStopDate','TILFluidCalcStopTime'])
daily_TIL_info = daily_TIL_info.dropna(axis=1,how='all').dropna(subset=true_var_columns,how='all').reset_index(drop=True)

# Remove all TIL entries marked as "Not Performed"
daily_TIL_info = daily_TIL_info[daily_TIL_info.DailyTILCompleteStatus!='INCPT']

# Convert dates from string to date format
daily_TIL_info.TILDate = pd.to_datetime(daily_TIL_info.TILDate,format = '%Y-%m-%d')

# For each patient, and for the overall set, calculate median TIL evaluation time
median_TILTime = daily_TIL_info.copy().dropna(subset=['GUPI','TILTime'])
median_TILTime['TILTime'] = pd.to_datetime(median_TILTime.TILTime,format = '%H:%M:%S')
median_TILTime = median_TILTime.groupby(['GUPI'],as_index=False).TILTime.aggregate('median')
overall_median_TILTime = median_TILTime.TILTime.median().strftime('%H:%M:%S')
median_TILTime['TILTime'] = median_TILTime['TILTime'].dt.strftime('%H:%M:%S')
median_TILTime = median_TILTime.rename(columns={'TILTime':'medianTILTime'})

# Iterate through GUPIs and fix `TILDate` based on `TILTimepoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(daily_TIL_info.GUPI.unique(),'Fixing daily TIL dates if possible'):
    curr_GUPI_daily_TIL = daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_TIL.TILDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_TIL.TILDate.dt.day - curr_GUPI_daily_TIL.TILTimepoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_TIL.TILTimepoint.astype(float)-1)],index=daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].index)
    daily_TIL_info.TILDate[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')] = fixed_date_vector    
    
# Replace 'None' timepoints with NaN
daily_TIL_info.TILTimepoint[daily_TIL_info.TILTimepoint=='None'] = np.nan

# Determine GUPIs with 'None' timepoints
none_GUPIs = daily_TIL_info[daily_TIL_info.TILTimepoint.isna()].GUPI.unique()

# Iterate through 'None' GUPIs and impute missing timepoint values
for curr_GUPI in none_GUPIs:
    curr_GUPI_TIL_scores = daily_TIL_info[daily_TIL_info.GUPI==curr_GUPI].reset_index(drop=True)
    non_missing_timepoint_mask = ~curr_GUPI_TIL_scores.TILTimepoint.isna()
    if non_missing_timepoint_mask.sum() != 1:
        curr_default_date = (curr_GUPI_TIL_scores.TILDate[non_missing_timepoint_mask] - pd.to_timedelta(curr_GUPI_TIL_scores.TILTimepoint.astype(float)[non_missing_timepoint_mask],unit='d')).mode()[0]
    else:
        curr_default_date = (curr_GUPI_TIL_scores.TILDate[non_missing_timepoint_mask] - timedelta(days=curr_GUPI_TIL_scores.TILTimepoint.astype(float)[non_missing_timepoint_mask].values[0])).mode()[0]
    fixed_timepoints_vector = ((curr_GUPI_TIL_scores.TILDate - curr_default_date)/np.timedelta64(1,'D')).astype(int).astype(str)
    fixed_timepoints_vector.index=daily_TIL_info[daily_TIL_info.GUPI==curr_GUPI].index
    daily_TIL_info.TILTimepoint[daily_TIL_info.GUPI==curr_GUPI] = fixed_timepoints_vector

# Convert TILTimepoint variable from string to integer
daily_TIL_info.TILTimepoint = daily_TIL_info.TILTimepoint.astype(int)

# Fix volume and dose variables if incorrectly casted as character types
fix_TIL_columns = [col for col, dt in daily_TIL_info.dtypes.items() if (col.endswith('Dose')|('Volume' in col))&(dt == object)]
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].replace(to_replace='^\D*$', value=np.nan, regex=True)
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace(',','.',regex=False))
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('[^0-9\\.]','',regex=True))
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('\\.\\.','.',regex=True))
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(pd.to_numeric)

# Merge daily TIL dataframe onto CENTER-TBI timestamp dataframe
mod_daily_TIL_info = CENTER_TBI_datetime[['GUPI','PatientType','ICUAdmTimeStamp','ICUDischTimeStamp']].merge(daily_TIL_info,how='inner')

# Rearrange columns
first_cols = ['GUPI','TILTimepoint','TILDate','TotalTIL','ICUAdmTimeStamp','ICUDischTimeStamp']
other_cols = mod_daily_TIL_info.columns.difference(first_cols).to_list()
mod_daily_TIL_info = mod_daily_TIL_info[first_cols+other_cols]

# Find CENTER-TBI patients who experienced WLST
CENTER_TBI_WLST_patients = pd.read_csv('../CENTER-TBI/WLST_patients.csv',na_values = ["NA","NaN"," ", ""])

# Filter WLST patients in current set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients[CENTER_TBI_WLST_patients.GUPI.isin(mod_daily_TIL_info.GUPI)].reset_index(drop=True)

# Find CENTER-TBI patients who died in ICU
CENTER_TBI_death_patients = pd.read_csv('../CENTER-TBI/death_patients.csv',na_values = ["NA","NaN"," ", ""])

# Add ICU death information to WLST set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.merge(CENTER_TBI_death_patients[['GUPI','ICUDischargeStatus']],how='left')

# Add ICU discharge information to WLST set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.merge(mod_daily_TIL_info[['GUPI','ICUDischTimeStamp']].drop_duplicates(ignore_index=True),how='left')

# Convert to long form
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.melt(id_vars=['GUPI','PatientType','DeathERWithdrawalLifeSuppForSeverityOfTBI','WithdrawalTreatmentDecision','DeadSeverityofTBI','DeadAge','DeadCoMorbidities','DeadRequestRelatives','DeadDeterminationOfBrainDeath','ICUDisWithdrawlTreatmentDecision','ICUDischargeStatus','ICUDischTimeStamp'])

# Sort dataframe by date(time) value per patient
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.sort_values(['GUPI','value'],ignore_index=True)

# Find patients who have no non-missing timestamps
no_non_missing_timestamp_patients = CENTER_TBI_WLST_patients.groupby('GUPI',as_index=False)['value'].agg('count')
no_non_missing_timestamp_patients = no_non_missing_timestamp_patients[no_non_missing_timestamp_patients['value']==0].reset_index(drop=True)

# Drop missing values entries given that the paient has at least one non-missing vlaue
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients[(~CENTER_TBI_WLST_patients['value'].isna())|(CENTER_TBI_WLST_patients.GUPI.isin(no_non_missing_timestamp_patients.GUPI))].reset_index(drop=True)

# Extract first timestamp (chronologically) from each patient
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.groupby('GUPI',as_index=False).first()

# If patient has all missing timestamps, insert '1970-01-01'
CENTER_TBI_WLST_patients['value'] = CENTER_TBI_WLST_patients['value'].fillna('1970-01-01')

# Extract 'DateComponent' from timestamp values
CENTER_TBI_WLST_patients['value'] = pd.to_datetime(CENTER_TBI_WLST_patients['value'].str[:10],format = '%Y-%m-%d').dt.date

# Add dummy column marking WLST
CENTER_TBI_WLST_patients['WLST'] = 1

# Append WLST date to formatted low-resolution dataframe
mod_daily_TIL_info = mod_daily_TIL_info.merge(CENTER_TBI_WLST_patients.rename(columns={'value':'WLSTDateComponent'})[['GUPI','WLSTDateComponent','WLST']],how='left')

# Fill in missing dummy-WLST markers
mod_daily_TIL_info.WLST = mod_daily_TIL_info.WLST.fillna(0)

# Filter out columns in which TILDate occurs during or after WLST decision
mod_daily_TIL_info = mod_daily_TIL_info[(mod_daily_TIL_info.TILDate<mod_daily_TIL_info.WLSTDateComponent)|(mod_daily_TIL_info.WLST==0)].reset_index(drop=True)

## Ensure refractory decompressive craniectomy surgery markers carry over onto subsequent TIL assessments for each patient
# First, sort modified daily TIL dataframe
mod_daily_TIL_info = mod_daily_TIL_info.sort_values(by=['GUPI','TILDate'],ignore_index=True)

# Identify GUPIs which contain markers for intracranial operation and decompressive craniectomy
gupis_with_initial_DecomCranectomy = mod_daily_TIL_info[(mod_daily_TIL_info.TILICPSurgeryDecomCranectomy==1)&(mod_daily_TIL_info.TILTimepoint==1)].GUPI.unique()
gupis_with_DecomCranectomy = mod_daily_TIL_info[(mod_daily_TIL_info.TILICPSurgeryDecomCranectomy==1)].GUPI.unique()
gupis_with_refract_DecomCranectomy = np.setdiff1d(gupis_with_DecomCranectomy, gupis_with_initial_DecomCranectomy)

# Iterate through GUPIs with initial decompressive craniectomies and correct surgery indicators and the total TIL score
for curr_GUPI in tqdm(gupis_with_initial_DecomCranectomy, 'Fixing TotalTIL and initial decompressive craniectomy indicators'):

    # Extract TIL assessments of current patient
    curr_GUPI_daily_TIL = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].reset_index(drop=True)

    # Extract total TIL scores of current patient
    curr_TotalTIL = curr_GUPI_daily_TIL.TotalTIL

    # Extract decompressive craniectomy indicators of current patient
    curr_TILICPSurgeryDecomCranectomy = curr_GUPI_daily_TIL.TILICPSurgeryDecomCranectomy

    # Identify first TIL instance of surgery
    firstSurgInstance = curr_TILICPSurgeryDecomCranectomy.index[curr_TILICPSurgeryDecomCranectomy==1].tolist()[0]

    # Fix the decompressive craniectomy indicators of current patient
    fix_TILICPSurgeryDecomCranectomy = curr_TILICPSurgeryDecomCranectomy.copy()
    if firstSurgInstance != (len(fix_TILICPSurgeryDecomCranectomy)-1):
        fix_TILICPSurgeryDecomCranectomy[range(firstSurgInstance+1,len(fix_TILICPSurgeryDecomCranectomy))] = 0
    fix_TILICPSurgeryDecomCranectomy.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Fix the total TIL score of current patient
    fix_TotalTIL = curr_TotalTIL.copy()
    if firstSurgInstance != (len(fix_TotalTIL)-1):
        fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')] = fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')]-5
    fix_TotalTIL.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Place fixed vectors into modified daily TIL dataframe
    mod_daily_TIL_info.TotalTIL[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TotalTIL    
    mod_daily_TIL_info.TILICPSurgeryDecomCranectomy[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TILICPSurgeryDecomCranectomy

# Iterate through GUPIs with refractory decompressive craniectomies and correct surgery indicators and the total TIL score
for curr_GUPI in tqdm(gupis_with_refract_DecomCranectomy, 'Fixing TotalTIL and refractory decompressive craniectomy indicators'):

    # Extract TIL assessments of current patient
    curr_GUPI_daily_TIL = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].reset_index(drop=True)

    # Extract total TIL scores of current patient
    curr_TotalTIL = curr_GUPI_daily_TIL.TotalTIL

    # Extract decompressive craniectomy indicators of current patient
    curr_TILICPSurgeryDecomCranectomy = curr_GUPI_daily_TIL.TILICPSurgeryDecomCranectomy

    # Identify first TIL instance of surgery
    firstSurgInstance = curr_TILICPSurgeryDecomCranectomy.index[curr_TILICPSurgeryDecomCranectomy==1].tolist()[0]

    # Fix the decompressive craniectomy indicators of current patient
    fix_TILICPSurgeryDecomCranectomy = curr_TILICPSurgeryDecomCranectomy.copy()
    if firstSurgInstance != (len(fix_TILICPSurgeryDecomCranectomy)-1):
        fix_TILICPSurgeryDecomCranectomy[range(firstSurgInstance+1,len(fix_TILICPSurgeryDecomCranectomy))] = 1
    fix_TILICPSurgeryDecomCranectomy.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Fix the total TIL score of current patient
    fix_TotalTIL = curr_TotalTIL.copy()
    if firstSurgInstance != (len(fix_TotalTIL)-1):
        fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')] = fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')]+5
    fix_TotalTIL.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Place fixed vectors into modified daily TIL dataframe
    mod_daily_TIL_info.TotalTIL[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TotalTIL    
    mod_daily_TIL_info.TILICPSurgeryDecomCranectomy[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TILICPSurgeryDecomCranectomy

## Carefully recalculate TILsum and get each true item of TIL
# Apply custom function
recalc_daily_TIL_info = calculate_TILsum(mod_daily_TIL_info)

# Load ICP-monitored patients
icp_monitored_patients = pd.read_csv('../CENTER-TBI/ICP_monitored_patients.csv')

# Filter TIL scores to only ICP-monitored patients
recalc_daily_TIL_info = recalc_daily_TIL_info[recalc_daily_TIL_info.GUPI.isin(icp_monitored_patients.GUPI)].sort_values(by=['GUPI','TILTimepoint','TILDate'],ignore_index=True)

# Save modified Daily TIL dataframes in new directory
os.makedirs('../formatted_data/',exist_ok=True)
recalc_daily_TIL_info.to_csv('../formatted_data/formatted_TIL_scores.csv',index=False)

# Remove weighting from dataframe scores and re-save
unweighted_daily_TIL_info = recalc_daily_TIL_info.copy()
unweighted_daily_TIL_info['CSFDrainage'] = unweighted_daily_TIL_info['CSFDrainage'].rank(method='dense') - 1
unweighted_daily_TIL_info['DecomCraniectomy'] = unweighted_daily_TIL_info['DecomCraniectomy'].rank(method='dense') - 1
unweighted_daily_TIL_info['Hypertonic'] = unweighted_daily_TIL_info['Hypertonic'].rank(method='dense') - 1
unweighted_daily_TIL_info['ICPSurgery'] = unweighted_daily_TIL_info['ICPSurgery'].rank(method='dense') - 1
unweighted_daily_TIL_info['Mannitol'] = unweighted_daily_TIL_info['Mannitol'].rank(method='dense') - 1
unweighted_daily_TIL_info['Neuromuscular'] = unweighted_daily_TIL_info['Neuromuscular'].rank(method='dense') - 1
unweighted_daily_TIL_info['Sedation'] = unweighted_daily_TIL_info['Sedation'].rank(method='dense') - 1
unweighted_daily_TIL_info['Temperature'] = unweighted_daily_TIL_info['Temperature'].rank(method='dense') - 1
unweighted_daily_TIL_info['Ventilation'] = unweighted_daily_TIL_info['Ventilation'].rank(method='dense') - 1

# Save unweighted Daily TIL dataframe in new directory
unweighted_daily_TIL_info.to_csv('../formatted_data/formatted_unweighted_TIL_scores.csv',index=False)

### III. Load and prepare hourly changes in TIL
## Load and prepare formatted daily TIL dataframe
recalc_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Convert dates from string to date format
recalc_daily_TIL_info.TILDate = pd.to_datetime(recalc_daily_TIL_info.TILDate,format = '%Y-%m-%d')
recalc_daily_TIL_info.ICUAdmTimeStamp = pd.to_datetime(recalc_daily_TIL_info.ICUAdmTimeStamp,format = '%Y-%m-%d %H:%M:%S')
recalc_daily_TIL_info.ICUDischTimeStamp = pd.to_datetime(recalc_daily_TIL_info.ICUDischTimeStamp,format = '%Y-%m-%d %H:%M:%S')

## Load and prepare HourlyValues dataframe
# Load HourlyValues dataframe
daily_hourly_info = pd.read_csv('../CENTER-TBI/DailyHourlyValues/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients for whom TIL values exist
daily_hourly_info = daily_hourly_info[daily_hourly_info.GUPI.isin(recalc_daily_TIL_info.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Remove all entries without date or `HourlyValueTimePoint`
daily_hourly_info = daily_hourly_info[(daily_hourly_info.HourlyValueTimePoint!='None')|(~daily_hourly_info.HVDate.isna())].reset_index(drop=True)

# Remove all rows in which HVTIL is missing
daily_hourly_info = daily_hourly_info[~(daily_hourly_info.HVTIL.isna())].reset_index(drop=True)

# Convert dates from string to date format
daily_hourly_info.HVDate = pd.to_datetime(daily_hourly_info.HVDate,format = '%Y-%m-%d')

# Iterate through GUPIs and fix `HVDate` based on `HourlyValueTimePoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(daily_hourly_info.GUPI.unique(),'Fixing daily hourly dates if possible'):
    curr_GUPI_daily_hourly = daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_hourly.HVDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_hourly.HVDate.dt.day - curr_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)-1)],index=daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].index)
    daily_hourly_info.HVDate[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')] = fixed_date_vector    

# Change HVTIL to reflect decreasing intensity with negative value
daily_hourly_info['HVTIL'][daily_hourly_info['HVTIL']==2] = -1

# Sort dataframe and extract columns of interest
daily_hourly_info = daily_hourly_info[['GUPI','HVDate','HVHour','HVTime','HVTIL','HVTILChangeReason']].sort_values(by=['GUPI','HVDate','HVHour'],ignore_index=True)

# Create column designating magnitude of TIL change
daily_hourly_info['MagnitudeChange'] = daily_hourly_info.HVTIL.abs()

# Keep just one row per hour combination based on maximum change
daily_hourly_info = daily_hourly_info.loc[daily_hourly_info.groupby(['GUPI','HVDate','HVHour'])['MagnitudeChange'].idxmax()].reset_index(drop=True)

## Merge daily hourly and daily TIL change information and save
# Calculate daily differences in TIL from previously calculated values
recalc_daily_TIL_info['ChangeInTIL'] = recalc_daily_TIL_info.groupby(['GUPI'])['TotalSum'].diff()

# Remove NA change rows and select only pertinent rows
delta_daily_TIL_info = recalc_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TotalSum','ChangeInTIL']]

# Merge hourly changes in TIL to daily delta dataframe
delta_daily_TIL_info = delta_daily_TIL_info.merge(daily_hourly_info.rename(columns={'HVDate':'TILDate'}),how='inner')

# Save deltaTIL dataframe in new directory
delta_daily_TIL_info.to_csv('../formatted_data/formatted_delta_TIL_scores.csv',index=False)

### IV. Load and prepare low-resolution ICP and CPP information
## Load and prepare formatted daily TIL dataframe
recalc_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Convert dates from string to date format
recalc_daily_TIL_info.TILDate = pd.to_datetime(recalc_daily_TIL_info.TILDate,format = '%Y-%m-%d')
recalc_daily_TIL_info.ICUAdmTimeStamp = pd.to_datetime(recalc_daily_TIL_info.ICUAdmTimeStamp,format = '%Y-%m-%d %H:%M:%S')
recalc_daily_TIL_info.ICUDischTimeStamp = pd.to_datetime(recalc_daily_TIL_info.ICUDischTimeStamp,format = '%Y-%m-%d %H:%M:%S')

## Load and prepare HourlyValues dataframe
# Load HourlyValues dataframe
daily_hourly_info = pd.read_csv('../CENTER-TBI/DailyHourlyValues/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients for whom TIL values exist
daily_hourly_info = daily_hourly_info[daily_hourly_info.GUPI.isin(recalc_daily_TIL_info.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Remove all rows in which ICP or CPP is missing
daily_hourly_info = daily_hourly_info[~(daily_hourly_info.HVICP.isna() & daily_hourly_info.HVCPP.isna())].reset_index(drop=True)

# Remove all entries without date or `HourlyValueTimePoint`
daily_hourly_info = daily_hourly_info[(daily_hourly_info.HourlyValueTimePoint!='None')|(~daily_hourly_info.HVDate.isna())].reset_index(drop=True)

# Convert dates from string to date format
daily_hourly_info.HVDate = pd.to_datetime(daily_hourly_info.HVDate,format = '%Y-%m-%d')

# Iterate through GUPIs and fix `HVDate` based on `HourlyValueTimePoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(daily_hourly_info.GUPI.unique(),'Fixing daily hourly dates if possible'):
    curr_GUPI_daily_hourly = daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_hourly.HVDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_hourly.HVDate.dt.day - curr_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)-1)],index=daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].index)
    daily_hourly_info.HVDate[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')] = fixed_date_vector    

# Select relevant columns and rename column to match TIL dataframe
daily_hourly_info = daily_hourly_info[['GUPI','HVDate','HVHour','HVTime','HVICP','HVCPP']].rename(columns={'HVDate':'TILDate'})

# Calculate ICPmean and CPPmean from end-hour values
mean_daily_hourly_info = daily_hourly_info.melt(id_vars=['GUPI','TILDate','HVHour','HVTime']).groupby(['GUPI','TILDate','variable'],as_index=False)['value'].aggregate({'mean':'mean','n':'count'})

# Widen dataframe per mean and count
mean_lo_res_info = pd.pivot_table(mean_daily_hourly_info, values = 'mean', index=['GUPI','TILDate'], columns = 'variable').reset_index().rename(columns={'HVCPP':'CPPmean','HVICP':'ICPmean'})
n_lo_res_info = pd.pivot_table(mean_daily_hourly_info, values = 'n', index=['GUPI','TILDate'], columns = 'variable').reset_index().rename(columns={'HVCPP':'nCPP','HVICP':'nICP'})

# Combine mean and count dataframes
lo_res_info = mean_lo_res_info.merge(n_lo_res_info,how='inner')

# Merge hourly ICP and CPP values with daily TIL
lo_res_info = recalc_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TotalSum']].merge(lo_res_info,how='inner')

## Save modified Daily hourly value dataframes in new directory
lo_res_info.to_csv('../formatted_data/formatted_low_resolution_values.csv',index=False)

## Calculate means and maxes
# Filter to only keep values from the first week
first_week_lo_res_info = lo_res_info[lo_res_info.TILTimepoint<=7].reset_index(drop=True)

# Calculate ICPmean, CPPmean, ICPmax, and CPPmax
lo_res_maxes_means = first_week_lo_res_info.groupby('GUPI',as_index=False).ICPmean.aggregate({'ICPmean':'mean','ICPmax':'max'}).merge(first_week_lo_res_info.groupby('GUPI',as_index=False).CPPmean.aggregate({'CPPmean':'mean','CPPmax':'max'}),how='left')

# Save ICPmean, CPPmean, ICPmax, and CPPmax
lo_res_maxes_means.to_csv('../formatted_data/formatted_low_resolution_maxes_means.csv',index=False)

### V. Load and prepare high-resolution ICP and CPP information
## Load and prepare formatted daily TIL dataframe
recalc_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Convert dates from string to date format
recalc_daily_TIL_info.TILDate = pd.to_datetime(recalc_daily_TIL_info.TILDate,format = '%Y-%m-%d')
recalc_daily_TIL_info.ICUAdmTimeStamp = pd.to_datetime(recalc_daily_TIL_info.ICUAdmTimeStamp,format = '%Y-%m-%d %H:%M:%S')
recalc_daily_TIL_info.ICUDischTimeStamp = pd.to_datetime(recalc_daily_TIL_info.ICUDischTimeStamp,format = '%Y-%m-%d %H:%M:%S')

## Load and format high-resolution ICP/CPP summary values
# Load high-resolution ICP/CPP summary values of same day as TIL assessments
hi_res_info = pd.read_csv('../CENTER-TBI/HighResolution/til_same.csv',na_values = ["NA","NaN"," ", ""])

# Filter columns of interest
hi_res_info = hi_res_info[['GUPI','TimeStamp','TotalTIL','ICP_mean','CPP_mean','ICP_n','CPP_n']]

# Convert TimeStamp to proper format
hi_res_info['TimeStamp'] = pd.to_datetime(hi_res_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Remove missing ICP or CPP values
hi_res_info = hi_res_info[~(hi_res_info.ICP_mean.isna()&hi_res_info.CPP_mean.isna())].reset_index(drop=True)

## Add EVD indicator
# Load EVD indicator
evd_indicator = pd.read_csv('../CENTER-TBI/HighResolution/list_EVD.csv',na_values = ["NA","NaN"," ", ""])

# Add column for EVD indicator
hi_res_info['EVD'] = 0

# Fix EVD indicator
hi_res_info.EVD[hi_res_info.GUPI.isin(evd_indicator.GUPI)] = 1

# Add a TILDate column
hi_res_info['TILDate'] = pd.to_datetime(hi_res_info['TimeStamp'].dt.date)

# Drop duplicate rows
hi_res_info = hi_res_info[['GUPI','TILDate','ICP_mean','CPP_mean','ICP_n','CPP_n','EVD']].drop_duplicates(ignore_index=True)

# Merge hi-resolution ICP and CPP values with daily TIL
hi_res_info = recalc_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TotalSum']].merge(hi_res_info,how='inner').rename(columns={'ICP_mean':'ICPmean','CPP_mean':'CPPmean','ICP_n':'nICP','CPP_n':'nCPP'})

## Save modified hi-resolution value dataframes in new directory
hi_res_info.to_csv('../formatted_data/formatted_high_resolution_values.csv',index=False)

## Calculate means and maxes
# Filter to only keep values from the first week
first_week_hi_res_info = hi_res_info[hi_res_info.TILTimepoint<=7].reset_index(drop=True)

# Calculate ICPmean, CPPmean, ICPmax, and CPPmax
hi_res_maxes_means = first_week_hi_res_info.groupby('GUPI',as_index=False).ICPmean.aggregate({'ICPmean':'mean','ICPmax':'max'}).merge(first_week_hi_res_info.groupby('GUPI',as_index=False).CPPmean.aggregate({'CPPmean':'mean','CPPmax':'max'}),how='left')

# Save ICPmean, CPPmean, ICPmax, and CPPmax
hi_res_maxes_means.to_csv('../formatted_data/formatted_high_resolution_maxes_means.csv',index=False)

### VI. Load and prepare demographic information and baseline characteristics
## Load demographic and outcome scores of patients in TIL dataframe
# Load formatted daily TIL dataframe
recalc_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Load low-resolution value dataframe
lo_res_info = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Load high-resolution value dataframe
hi_res_info = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')

# Load CENTER-TBI dataset demographic information
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Select columns that indicate pertinent baseline and outcome information
CENTER_TBI_demo_info = CENTER_TBI_demo_info[['GUPI','PatientType','SiteCode','Age','Sex','Race','GCSScoreBaselineDerived','GOSE6monthEndpointDerived','ICURaisedICP','DecompressiveCranReason']].reset_index(drop=True)

# Filter GUPIs to daily TIL dataframe GUPIs
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.GUPI.isin(recalc_daily_TIL_info.GUPI)].reset_index(drop=True)

# Add marker of refractory IC hypertension
CENTER_TBI_demo_info['RefractoryICP'] = np.nan
CENTER_TBI_demo_info['RefractoryICP'][(~CENTER_TBI_demo_info.ICURaisedICP.isna())|(~CENTER_TBI_demo_info.DecompressiveCranReason.isna())] = ((CENTER_TBI_demo_info[(~CENTER_TBI_demo_info.ICURaisedICP.isna())|(~CENTER_TBI_demo_info.DecompressiveCranReason.isna())].ICURaisedICP==2)|(CENTER_TBI_demo_info[(~CENTER_TBI_demo_info.ICURaisedICP.isna())|(~CENTER_TBI_demo_info.DecompressiveCranReason.isna())].DecompressiveCranReason==2)).astype(int)

# Load and filter CENTER-TBI IMPACT dataframe
IMPACT_df = pd.read_csv('../CENTER-TBI/IMPACT/data.csv').rename(columns={'entity_id':'GUPI'})
IMPACT_df = IMPACT_df[['GUPI','marshall']].rename(columns={'marshall':'MarshallCT'}).reset_index(drop=True)

# Merge IMPACT-sourced information to outcome and demographic dataframe
CENTER_TBI_demo_info = CENTER_TBI_demo_info.merge(IMPACT_df,how='left')

# Add indicators for inclusion in low- and high-resolution subsets
CENTER_TBI_demo_info['LowResolutionSet'] = 0
CENTER_TBI_demo_info['HighResolutionSet'] = 0

# Fill in indicators
CENTER_TBI_demo_info.LowResolutionSet[CENTER_TBI_demo_info.GUPI.isin(lo_res_info.GUPI)] = 1
CENTER_TBI_demo_info.HighResolutionSet[CENTER_TBI_demo_info.GUPI.isin(hi_res_info.GUPI)] = 1

# Load and prepare ordinal prediction estimates
ordinal_prediction_estimates = pd.read_csv('../../ordinal_GOSE_prediction/APM_outputs/DEEP_v1-0/APM_deepMN_compiled_test_predictions.csv').drop(columns='Unnamed: 0')
ordinal_prediction_estimates = ordinal_prediction_estimates[(ordinal_prediction_estimates.GUPI.isin(CENTER_TBI_demo_info.GUPI))&(ordinal_prediction_estimates.TUNE_IDX==8)].reset_index(drop=True)
prob_cols = [col for col in ordinal_prediction_estimates if col.startswith('Pr(GOSE=')]
prob_matrix = ordinal_prediction_estimates[prob_cols]
thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
for thresh in range(1,len(prob_cols)):
    cols_gt = prob_cols[thresh:]
    prob_gt = ordinal_prediction_estimates[cols_gt].sum(1).values
    ordinal_prediction_estimates['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
ordinal_prediction_estimates = ordinal_prediction_estimates.drop(columns=prob_cols+['TrueLabel','TUNE_IDX'])
ordinal_prediction_estimates = ordinal_prediction_estimates.melt(id_vars=['GUPI'],var_name='Threshold',value_name='Probability').groupby(['GUPI','Threshold'],as_index=False).Probability.mean()
ordinal_prediction_estimates = pd.pivot_table(ordinal_prediction_estimates, values = 'Probability', index=['GUPI'], columns = 'Threshold').reset_index()

# Merge ordinal prognosis estimates to outcome and demographic dataframe
CENTER_TBI_demo_info = CENTER_TBI_demo_info.merge(ordinal_prediction_estimates,how='left')

# Save baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_info.to_csv('../formatted_data/formatted_outcome_and_demographics.csv',index=False)

### VII. Calculate TIL_1987, PILOT, and TIL_Basic
## Load and prepare unweighted TIL scores and components
# Load unweighted TIL scores and components
unweighted_daily_TIL_info = pd.read_csv('../formatted_data/formatted_unweighted_TIL_scores.csv')

# Convert ICU admission/discharge timestamps to datetime variables
unweighted_daily_TIL_info['TILDate'] = pd.to_datetime(unweighted_daily_TIL_info['TILDate'],format = '%Y-%m-%d')
unweighted_daily_TIL_info['ICUAdmTimeStamp'] = pd.to_datetime(unweighted_daily_TIL_info['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S')
unweighted_daily_TIL_info['ICUDischTimeStamp'] = pd.to_datetime(unweighted_daily_TIL_info['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

## Calculate TIL_1987
calc_daily_TIL_1987_info = calculate_TIL_1987(unweighted_daily_TIL_info)

# Save
calc_daily_TIL_1987_info.to_csv('../formatted_data/formatted_TIL_1987_scores.csv',index=False)

## Calculate PILOT
calc_daily_PILOT_info = calculate_PILOT(unweighted_daily_TIL_info)

# Save
calc_daily_PILOT_info.to_csv('../formatted_data/formatted_PILOT_scores.csv',index=False)

## Calculate TIL_Basic
calc_daily_TIL_Basic_info = calculate_TIL_Basic(unweighted_daily_TIL_info)

# Save
calc_daily_TIL_Basic_info.to_csv('../formatted_data/formatted_TIL_Basic_scores.csv',index=False)

### VIII. Calculate summarised TIL metrics
## TIL
# Load formatted daily TIL dataframe
recalc_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Filter to only keep TIL scores from the first week
first_week_daily_TIL_info = recalc_daily_TIL_info[recalc_daily_TIL_info.TILTimepoint<=7].reset_index(drop=True)

# Keep rows corresponding to TIL_max
til_max_info = first_week_daily_TIL_info.loc[first_week_daily_TIL_info.groupby(['GUPI'])['TotalSum'].idxmax()].reset_index(drop=True).rename(columns={'TotalSum':'TILmax'})

# Calculate TIL_mean info
til_mean_info = pd.pivot_table(first_week_daily_TIL_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TotalSum':'TILmean'})

## PILOT
# Load formatted daily PILOT dataframe
calc_daily_PILOT_info = pd.read_csv('../formatted_data/formatted_PILOT_scores.csv')

# Filter to only keep PILOT scores from the first week
first_week_daily_PILOT_info = calc_daily_PILOT_info[calc_daily_PILOT_info.TILTimepoint<=7].reset_index(drop=True)

# Keep rows corresponding to PILOT_max
pilot_max_info = first_week_daily_PILOT_info.loc[first_week_daily_PILOT_info.groupby(['GUPI'])['PILOTSum'].idxmax()].reset_index(drop=True).rename(columns={'PILOTSum':'PILOTmax'})

# Calculate PILOT_mean info
pilot_mean_info = pd.pivot_table(first_week_daily_PILOT_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'PILOTSum':'PILOTmean','TotalSum':'TILmean'})

## TIL_1987
# Load formatted daily TIL_1987 dataframe
calc_daily_TIL_1987_info = pd.read_csv('../formatted_data/formatted_TIL_1987_scores.csv')

# Filter to only keep TIL_1987 scores from the first week
first_week_daily_TIL_1987_info = calc_daily_TIL_1987_info[calc_daily_TIL_1987_info.TILTimepoint<=7].reset_index(drop=True)

# Keep rows corresponding to TIL_1987_max
til_1987_max_info = first_week_daily_TIL_1987_info.loc[first_week_daily_TIL_1987_info.groupby(['GUPI'])['TIL_1987Sum'].idxmax()].reset_index(drop=True).rename(columns={'TIL_1987Sum':'TIL_1987max'})

# Calculate TIL_1987_mean info
til_1987_mean_info = pd.pivot_table(first_week_daily_TIL_1987_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TIL_1987Sum':'TIL_1987mean','TotalSum':'TILmean'})

## TIL_Basic
# Load formatted daily TIL_Basic dataframe
calc_daily_TIL_Basic_info = pd.read_csv('../formatted_data/formatted_TIL_Basic_scores.csv')

# Filter to only keep TIL_Basic scores from the first week
first_week_daily_TIL_Basic_info = calc_daily_TIL_Basic_info[calc_daily_TIL_Basic_info.TILTimepoint<=7].reset_index(drop=True)

# Keep rows corresponding to TIL_Basic_max
til_basic_max_info = first_week_daily_TIL_Basic_info.loc[first_week_daily_TIL_Basic_info.groupby(['GUPI'])['TIL_Basic'].idxmax()].reset_index(drop=True).rename(columns={'TIL_Basic':'TIL_Basicmax'})

# Calculate TIL_Basic_mean info
til_basic_mean_info = pd.pivot_table(first_week_daily_TIL_Basic_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TIL_Basic':'TIL_Basicmean','TotalSum':'TILmean'})

## Save dataframes
# Max dataframes
til_max_info.to_csv('../formatted_data/formatted_TIL_max.csv',index=False)
pilot_max_info.to_csv('../formatted_data/formatted_PILOT_max.csv',index=False)
til_1987_max_info.to_csv('../formatted_data/formatted_TIL_1987_max.csv',index=False)
til_basic_max_info.to_csv('../formatted_data/formatted_TIL_Basic_max.csv',index=False)

# Mean dataframes
til_mean_info.to_csv('../formatted_data/formatted_TIL_mean.csv',index=False)
pilot_mean_info.to_csv('../formatted_data/formatted_PILOT_mean.csv',index=False)
til_1987_mean_info.to_csv('../formatted_data/formatted_TIL_1987_mean.csv',index=False)
til_basic_mean_info.to_csv('../formatted_data/formatted_TIL_Basic_mean.csv',index=False)

### VII. Load and prepare serum sodium values from CENTER-TBI
## Load and prepare sodium values
# Load sodium lab values
sodium_values = pd.read_csv('../CENTER-TBI/Labs/data.csv',na_values = ["NA","NaN"," ", ""])[['GUPI','DLDate','DLTime','DLSodiummmolL']].dropna(subset=['DLDate','DLSodiummmolL'],how='any').sort_values(by=['GUPI','DLDate']).reset_index(drop=True)

# Convert `DLDate` to timestamp format
sodium_values['TILDate'] = pd.to_datetime(sodium_values['DLDate'],format = '%Y-%m-%d')

# Calculate daily mean sodium
sodium_values = sodium_values.groupby(['GUPI','TILDate'],as_index=False).DLSodiummmolL.aggregate({'meanSodium':'mean','nSodium':'count'})

# Load formatted TIL values and add row index
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Convert ICU admission/discharge timestamps to datetime variables
formatted_TIL_scores['TILDate'] = pd.to_datetime(formatted_TIL_scores['TILDate'],format = '%Y-%m-%d')
formatted_TIL_scores['ICUAdmTimeStamp'] = pd.to_datetime(formatted_TIL_scores['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S')
formatted_TIL_scores['ICUDischTimeStamp'] = pd.to_datetime(formatted_TIL_scores['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# Merge sodium values to formatted TIL dataframe
sodium_TIL_dataframe = formatted_TIL_scores.merge(sodium_values,how='left')

# Remove rows with missing sodium values
sodium_TIL_dataframe = sodium_TIL_dataframe.dropna(subset='meanSodium').drop_duplicates(ignore_index=True)

## Save prepared sodium values
sodium_TIL_dataframe.to_csv('../formatted_data/formatted_daily_sodium_values.csv',index=False)

## Calculate means and maxes
# Filter to only keep values from the first week
first_week_sodium_info = sodium_TIL_dataframe[sodium_TIL_dataframe.TILTimepoint<=7].reset_index(drop=True)

# Calculate meanSodium and maxSodium
sodium_maxes_means = first_week_sodium_info.groupby('GUPI',as_index=False).meanSodium.aggregate({'meanSodium':'mean','maxSodium':'max'})

# Save meanSodium and maxSodium
sodium_maxes_means.to_csv('../formatted_data/formatted_sodium_maxes_means.csv',index=False)