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

# Custom methods
from functions.analysis import calculate_TILsum, calculate_TIL_1987, calculate_PILOT, calculate_TIL_Basic

## Create relevant directories
# Initialise directory to store formatted data
formatted_data_dir = '../formatted_data'

# Create formatted data directory
os.makedirs(formatted_data_dir,exist_ok=True)

# Initialise results directory
results_dir = '../results'

# Create results directory
os.makedirs(results_dir,exist_ok=True)

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

# Load list of ICP-monitored patients
icp_monitored_patients = pd.read_csv('../CENTER-TBI/ICP_monitored_patients.csv')

# Apply inclusion criteria no. 2: ICP monitored during ICU stay
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime.GUPI.isin(icp_monitored_patients.GUPI)].reset_index(drop=True)

## Load and prepare TIL information
# Load DailyTIL dataframe
daily_TIL_info = pd.read_csv('../CENTER-TBI/DailyTIL/data.csv',na_values = ["NA","NaN"," ", ""])

# Remove all entries without date or `TILTimepoint`
daily_TIL_info = daily_TIL_info[(daily_TIL_info.TILTimepoint!='None')|(~daily_TIL_info.TILDate.isna())].reset_index(drop=True)

# Remove all TIL entries with NA for all data variable columns
true_var_columns = daily_TIL_info.columns.difference(['GUPI', 'TILTimepoint', 'TILDate', 'TILTime','DailyTILCompleteStatus','TotalTIL','TILFluidCalcStartDate','TILFluidCalcStartTime','TILFluidCalcStopDate','TILFluidCalcStopTime'])
daily_TIL_info = daily_TIL_info.dropna(axis=1,how='all').dropna(subset=true_var_columns,how='all').reset_index(drop=True)

# Remove all TIL entries marked as "Not Performed" or "Not Started"
daily_TIL_info = daily_TIL_info[~daily_TIL_info.DailyTILCompleteStatus.isin(['INCPT','NOSTART'])]

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

# Apply inclusion criteria no. 3 and 4: Filter out columns in which WLST decision happened within 24 hours of ICU admission and filter out columns for whom TIL is unavailable
mod_daily_TIL_info = mod_daily_TIL_info[(mod_daily_TIL_info.TILDate<mod_daily_TIL_info.WLSTDateComponent)|(mod_daily_TIL_info.WLST==0)].reset_index(drop=True)

## First, fix missing instances of decompressive craniectomy documented elsewhere
# First, sort modified daily TIL dataframe
mod_daily_TIL_info = mod_daily_TIL_info.sort_values(by=['GUPI','TILDate'],ignore_index=True)

# Load cranial surgeries dataset information
cran_surgeries_info = pd.read_csv('../CENTER-TBI/SurgeriesCranial/data.csv',na_values = ["NA","NaN"," ", ""])[['GUPI','SurgeryStartDate','SurgeryStartTime','SurgeryEndDate','SurgeryEndTime','SurgeryDescCranial','SurgeryCranialReason']]

# Select columns that indicate pertinent decompresive craniectomy information
cran_surgeries_info = cran_surgeries_info[(cran_surgeries_info.GUPI.isin(mod_daily_TIL_info.GUPI))&(cran_surgeries_info.SurgeryDescCranial.isin([7,71,72]))].reset_index(drop=True)
cran_surgeries_info['TILDate'] = pd.to_datetime(cran_surgeries_info['SurgeryStartDate'],format = '%Y-%m-%d')
cran_surgeries_info = cran_surgeries_info[['GUPI','TILDate']].drop_duplicates(ignore_index=True).merge(mod_daily_TIL_info[['GUPI','TILDate']].drop_duplicates(ignore_index=True),how='inner')

# Iterate through cranial surgery dataframe to document instances of decompressive craniectomy if missing
for curr_cs_row in tqdm(range(cran_surgeries_info.shape[0]), 'Fixing decompressive craniectomy instances if missing'):
    # Extract current GUPI and date
    curr_GUPI = cran_surgeries_info.GUPI[curr_cs_row]
    curr_date = cran_surgeries_info.TILDate[curr_cs_row]

    # Current TIL row corresponding to the current date and GUPI
    curr_TIL_row = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)&(mod_daily_TIL_info.TILDate==curr_date)]

    # Ensure decompressive craniectomy markers if date and GUPI match
    if curr_TIL_row.shape[0] != 0:
        mod_daily_TIL_info.TILICPSurgeryDecomCranectomy[curr_TIL_row.index] = 1

## Ensure refractory decompressive craniectomy surgery markers carry over onto subsequent TIL assessments for each patient
# Load CENTER-TBI dataset demographic information
CENTER_TBI_DC_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])[['GUPI','DecompressiveCran','DecompressiveCranLocation','DecompressiveCranReason','DecompressiveCranType','DecompressiveSize']]

# Select columns that indicate pertinent decompresive craniectomy information
CENTER_TBI_DC_info = CENTER_TBI_DC_info[(CENTER_TBI_DC_info.GUPI.isin(mod_daily_TIL_info.GUPI))&(CENTER_TBI_DC_info.DecompressiveCran==1)].reset_index(drop=True)

# Identify GUPIs by the categorized type of decompressive craniectomy
gupis_with_DecomCranectomy = mod_daily_TIL_info[(mod_daily_TIL_info.TILICPSurgeryDecomCranectomy==1)].GUPI.unique()
gupis_with_initial_DecomCranectomy = mod_daily_TIL_info[(mod_daily_TIL_info.TILICPSurgeryDecomCranectomy==1)&(mod_daily_TIL_info.TILTimepoint==1)].GUPI.unique()
gupis_with_noninitial_DecomCranectomy = np.setdiff1d(gupis_with_DecomCranectomy, gupis_with_initial_DecomCranectomy)
gupis_with_refract_DecomCranectomy_1 = CENTER_TBI_DC_info[CENTER_TBI_DC_info.DecompressiveCranReason==2].GUPI.unique()
gupis_with_refract_DecomCranectomy_2 = np.intersect1d(np.setdiff1d(gupis_with_DecomCranectomy, CENTER_TBI_DC_info.GUPI.unique()),gupis_with_noninitial_DecomCranectomy)
gupis_with_refract_DecomCranectomy = np.union1d(gupis_with_refract_DecomCranectomy_1,gupis_with_refract_DecomCranectomy_2)
gupis_with_nonrefract_DecomCranectomy = np.setdiff1d(gupis_with_DecomCranectomy, gupis_with_refract_DecomCranectomy)

# Iterate through GUPIs with initial decompressive craniectomies and correct surgery indicators and the total TIL score
for curr_GUPI in tqdm(gupis_with_nonrefract_DecomCranectomy, 'Fixing TotalTIL and initial decompressive craniectomy indicators'):

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
    try:
        firstSurgInstance = curr_TILICPSurgeryDecomCranectomy.index[curr_TILICPSurgeryDecomCranectomy==1].tolist()[0]
    except:
        firstSurgInstance = 1
        
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
recalc_daily_TIL_info = calculate_TILsum(mod_daily_TIL_info).sort_values(['GUPI','TILTimepoint'],ignore_index=True)

# Filter ICU timestamp information to study population
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime.GUPI.isin(recalc_daily_TIL_info.GUPI)].reset_index(drop=True)

## Identify expected cases that are missing
# Calculate minimum number of expected TIL scores from ICU stay duration
CENTER_TBI_datetime['ExpectedCount'] = (CENTER_TBI_datetime['ICUDurationHours']/24).apply(lambda x: max(np.floor(x),1),7).astype(int)

# Create a dummy dataframe of TILTimepoints
dummy_timepoints = pd.DataFrame({'TILTimepoint':[i+1 for i in range(CENTER_TBI_datetime.ExpectedCount.max())],'key':1})

# Add dummy 'key' column to dataframe
CENTER_TBI_datetime['key'] = 1

# Create dataframe of expected GUPI-TILTimepoint combinations
expected_combinations = CENTER_TBI_datetime[['GUPI','ExpectedCount','key']].merge(dummy_timepoints).drop(columns='key')

# Filter to keep timepoints within expected range
expected_combinations = expected_combinations[expected_combinations.TILTimepoint<=expected_combinations.ExpectedCount].drop(columns='ExpectedCount').reset_index(drop=True)

# Merge expected combinations with available cases to identify missing instances
merged_expected_combinations = expected_combinations.merge(recalc_daily_TIL_info,how='outer')

## Fix dates in merged expected combinations dataframe
# Iterate through GUPIs and fix `TILDate` based on `TILTimepoint` information if possible
for curr_GUPI in tqdm(merged_expected_combinations.GUPI.unique(),'Fixing merged daily TIL dates if possible'):
    curr_GUPI_daily_TIL = merged_expected_combinations[merged_expected_combinations.GUPI==curr_GUPI].reset_index(drop=True)
    curr_date_diff = int((curr_GUPI_daily_TIL.TILDate.dt.day - curr_GUPI_daily_TIL.TILTimepoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_TIL.TILTimepoint.astype(float)-1)],index=merged_expected_combinations[(merged_expected_combinations.GUPI==curr_GUPI)&(merged_expected_combinations.TILTimepoint!='None')].index)
    merged_expected_combinations.TILDate[merged_expected_combinations.GUPI==curr_GUPI] = fixed_date_vector    

## Prepare and save formatted daily TIL scores (and expected days)
# Sort merged expected combination dataframe
merged_expected_combinations = merged_expected_combinations.sort_values(['GUPI','TILTimepoint'],ignore_index=True)

# Move TotalSum column to after TILTimepoint
merged_expected_combinations.insert(2, 'TotalSum', merged_expected_combinations.pop('TotalSum'))

# Save modified Daily TIL dataframes in formatted data directory
merged_expected_combinations.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_scores.csv'),index=False)

## Remove weighting from TIL components and save unweighted TIL scores
# Remove weighting from dataframe scores and re-save
unweighted_daily_TIL_info = merged_expected_combinations.copy()
unweighted_daily_TIL_info['CSFDrainage'] = unweighted_daily_TIL_info['CSFDrainage'].rank(method='dense') - 1
unweighted_daily_TIL_info['DecomCraniectomy'] = unweighted_daily_TIL_info['DecomCraniectomy'].rank(method='dense') - 1
unweighted_daily_TIL_info['Hypertonic'] = unweighted_daily_TIL_info['Hypertonic'].rank(method='dense') - 1
unweighted_daily_TIL_info['ICPSurgery'] = unweighted_daily_TIL_info['ICPSurgery'].rank(method='dense') - 1
unweighted_daily_TIL_info['Mannitol'] = unweighted_daily_TIL_info['Mannitol'].rank(method='dense') - 1
unweighted_daily_TIL_info['Neuromuscular'] = unweighted_daily_TIL_info['Neuromuscular'].rank(method='dense') - 1
unweighted_daily_TIL_info['Sedation'] = unweighted_daily_TIL_info['Sedation'].rank(method='dense') - 1
unweighted_daily_TIL_info['Temperature'] = unweighted_daily_TIL_info['Temperature'].rank(method='dense') - 1
unweighted_daily_TIL_info['Ventilation'] = unweighted_daily_TIL_info['Ventilation'].rank(method='dense') - 1
unweighted_daily_TIL_info['uwTILSum'] = unweighted_daily_TIL_info.CSFDrainage + unweighted_daily_TIL_info.DecomCraniectomy + unweighted_daily_TIL_info.FluidLoading + unweighted_daily_TIL_info.Hypertonic + unweighted_daily_TIL_info.ICPSurgery + unweighted_daily_TIL_info.Mannitol + unweighted_daily_TIL_info.Neuromuscular + unweighted_daily_TIL_info.Positioning + unweighted_daily_TIL_info.Sedation + unweighted_daily_TIL_info.Temperature + unweighted_daily_TIL_info.Vasopressor + unweighted_daily_TIL_info.Ventilation

# Move uwTILSum column to after TILTimepoint
unweighted_daily_TIL_info.insert(2, 'uwTILSum', unweighted_daily_TIL_info.pop('uwTILSum'))

# Save unweighted Daily TIL dataframe in new directory
unweighted_daily_TIL_info.to_csv(os.path.join(formatted_data_dir,'formatted_unweighted_TIL_scores.csv'),index=False)

### III. Load and prepare hourly changes in TIL
## Load and prepare formatted daily TIL dataframe
formatted_TIL_scores = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_scores.csv'))

# Convert dates from string to date format
formatted_TIL_scores.TILDate = pd.to_datetime(formatted_TIL_scores.TILDate,format = '%Y-%m-%d')
formatted_TIL_scores.ICUAdmTimeStamp = pd.to_datetime(formatted_TIL_scores.ICUAdmTimeStamp,format = '%Y-%m-%d %H:%M:%S')
formatted_TIL_scores.ICUDischTimeStamp = pd.to_datetime(formatted_TIL_scores.ICUDischTimeStamp,format = '%Y-%m-%d %H:%M:%S')

## Load and prepare HourlyValues dataframe
# Load HourlyValues dataframe
daily_hourly_info = pd.read_csv('../CENTER-TBI/DailyHourlyValues/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients for whom TIL values exist
daily_hourly_info = daily_hourly_info[daily_hourly_info.GUPI.isin(formatted_TIL_scores.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

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
formatted_TIL_scores['ChangeInTIL'] = formatted_TIL_scores.groupby(['GUPI'])['TotalSum'].diff()

# Remove NA change rows and select only pertinent rows
delta_daily_TIL_info = formatted_TIL_scores[['GUPI','TILTimepoint','TILDate','TotalSum','ChangeInTIL']]

# Merge hourly changes in TIL to daily delta dataframe
delta_daily_TIL_info = delta_daily_TIL_info.merge(daily_hourly_info.rename(columns={'HVDate':'TILDate'}),how='outer')

# Save deltaTIL dataframe in new directory
delta_daily_TIL_info.to_csv(os.path.join(formatted_data_dir,'formatted_delta_TIL_scores.csv'),index=False)

### IV. Load and prepare low-resolution ICP and CPP information
## Load and prepare formatted daily TIL dataframe
formatted_TIL_scores = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_scores.csv'))

# Convert dates from string to date format
formatted_TIL_scores.TILDate = pd.to_datetime(formatted_TIL_scores.TILDate,format = '%Y-%m-%d')
formatted_TIL_scores.ICUAdmTimeStamp = pd.to_datetime(formatted_TIL_scores.ICUAdmTimeStamp,format = '%Y-%m-%d %H:%M:%S')
formatted_TIL_scores.ICUDischTimeStamp = pd.to_datetime(formatted_TIL_scores.ICUDischTimeStamp,format = '%Y-%m-%d %H:%M:%S')

## Load and prepare HourlyValues dataframe
# Load HourlyValues dataframe
daily_hourly_info = pd.read_csv('../CENTER-TBI/DailyHourlyValues/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients for whom TIL values exist
daily_hourly_info = daily_hourly_info[daily_hourly_info.GUPI.isin(formatted_TIL_scores.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Remove all rows with NA for all data variable columns
true_var_columns = daily_hourly_info.columns.difference(['GUPI', 'HourlyValueTimePoint', 'HVHour', 'HVDate','HVTime','HourlyValuesCompleteStatus'])
daily_hourly_info = daily_hourly_info.dropna(axis=1,how='all').dropna(subset=true_var_columns,how='all').reset_index(drop=True)

# Remove all rows in which assessment was not started or not performed
daily_hourly_info = daily_hourly_info[(~daily_hourly_info.HourlyValuesCompleteStatus.isin(['INCPT','NOSTART']))|(~(daily_hourly_info.HVICP.isna() & daily_hourly_info.HVCPP.isna()))]
# daily_hourly_info = daily_hourly_info[~(daily_hourly_info.HVICP.isna() & daily_hourly_info.HVCPP.isna())].reset_index(drop=True)

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
daily_hourly_info = daily_hourly_info[['GUPI','HVDate','HVHour','HVTime','HVICP','HVCPP','HVTILChangeReason','HourlyValueLevelICP','HourlyValueLevelABP','HourlyValueICPDiscontinued']].rename(columns={'HVDate':'TILDate'})

# Calculate ICPmean and CPPmean from end-hour values
mean_daily_hourly_info = daily_hourly_info.melt(id_vars=['GUPI','TILDate','HVHour','HVTime'],value_vars=['HVICP','HVCPP']).groupby(['GUPI','TILDate','variable'],as_index=False)['value'].aggregate({'mean':'mean','n':'count'})

# Widen dataframe per mean and count
mean_lo_res_info = pd.pivot_table(mean_daily_hourly_info, values = 'mean', index=['GUPI','TILDate'], columns = 'variable').reset_index().rename(columns={'HVCPP':'CPPmean','HVICP':'ICPmean'})
n_lo_res_info = pd.pivot_table(mean_daily_hourly_info, values = 'n', index=['GUPI','TILDate'], columns = 'variable').reset_index().rename(columns={'HVCPP':'nCPP','HVICP':'nICP'})

# Take mode of meta-information and widen
mode_daily_hourly_info = daily_hourly_info.melt(id_vars=['GUPI','TILDate','HVHour','HVTime'],value_vars=['HVTILChangeReason','HourlyValueLevelICP','HourlyValueLevelABP','HourlyValueICPDiscontinued']).dropna(subset='value').groupby(['GUPI','TILDate','variable'],as_index=False)['value'].aggregate({'mode':lambda x: x.mode()[0],'n':'count'})
mode_lo_res_info = pd.pivot_table(mode_daily_hourly_info, values = 'mode', index=['GUPI','TILDate'], columns = 'variable').reset_index()

# Combine mean and count dataframes
lo_res_info = mean_lo_res_info.merge(n_lo_res_info,how='inner')

# Calculate day-by-day differences in ICP and CPP
lo_res_info['ChangeInCPP'] = lo_res_info.groupby(['GUPI'])['CPPmean'].diff()
lo_res_info['ChangeInICP'] = lo_res_info.groupby(['GUPI'])['ICPmean'].diff()

# Merge hourly ICP and CPP values with daily TIL
lo_res_info = formatted_TIL_scores[['GUPI','TILTimepoint','TILDate','TotalSum']].merge(lo_res_info,how='outer').merge(mode_lo_res_info,how='left')

# Add a marker to indicate whether TIL score is expected at each row
lo_res_info['TILExpected'] = lo_res_info['TILTimepoint'].notna().astype(int)

# Determine GUPIs with missing timepoints
none_GUPIs = lo_res_info[lo_res_info.TILTimepoint.isna()].GUPI.unique()

# Iterate through 'None' GUPIs and impute missing timepoint values
for curr_GUPI in none_GUPIs:
    curr_GUPI_lo_res = lo_res_info[lo_res_info.GUPI==curr_GUPI].reset_index(drop=True)
    non_missing_timepoint_mask = ~curr_GUPI_lo_res.TILTimepoint.isna()
    if non_missing_timepoint_mask.sum() != 1:
        curr_default_date = (curr_GUPI_lo_res.TILDate[non_missing_timepoint_mask] - pd.to_timedelta(curr_GUPI_lo_res.TILTimepoint.astype(float)[non_missing_timepoint_mask],unit='d')).mode()[0]
    else:
        curr_default_date = (curr_GUPI_lo_res.TILDate[non_missing_timepoint_mask] - timedelta(days=curr_GUPI_lo_res.TILTimepoint.astype(float)[non_missing_timepoint_mask].values[0])).mode()[0]
    fixed_timepoints_vector = ((curr_GUPI_lo_res.TILDate - curr_default_date)/np.timedelta64(1,'D')).astype(int)
    fixed_timepoints_vector.index=lo_res_info[lo_res_info.GUPI==curr_GUPI].index
    lo_res_info.TILTimepoint[lo_res_info.GUPI==curr_GUPI] = fixed_timepoints_vector

# Convert TILTimepoint variable from string to integer
lo_res_info.TILTimepoint = lo_res_info.TILTimepoint.astype(int)

# Sort dataframe
lo_res_info = lo_res_info.sort_values(['GUPI','TILTimepoint'],ignore_index=True)

## Add ICP/CPP monitoring stoppage information
# Load preformatted ICP/CPP monitoring stoppage information
icp_monitored_patients = pd.read_csv('../CENTER-TBI/ICP_monitored_patients.csv',na_values = ["NA","NaN"," ", ""])

# Extract 'DateComponent' from ICP monitoring stoppage timestamp values
icp_monitored_patients['ICPRemTimeStamp'] = pd.to_datetime(icp_monitored_patients['ICPRemTimeStamp'].str[:10],format = '%Y-%m-%d').dt.date
icp_monitored_patients['ICPRevisionTimeStamp'] = pd.to_datetime(icp_monitored_patients['ICPRevisionTimeStamp'].str[:10],format = '%Y-%m-%d').dt.date

# Merge ICP removal timestamp and ICP stoppage information to low-resolution neuromonitoring dataframe
lo_res_info = lo_res_info.merge(icp_monitored_patients[['GUPI','ICPRemTimeStamp','ICPRevisionTimeStamp','ICUReasonICP','ICPDevice','ICUReasonForTypeICPMont','ICUReasonForTypeICPMontPare','ICUProblemsICP','ICUProblemsICPYes','ICUCatheterICP','ICPMonitorStop','ICPStopReason']],how='left')

# Add indicators for ICP monitor removal and/or revision
lo_res_info['ICPRemovedIndicator'] = (lo_res_info.TILDate[~lo_res_info.ICPRemTimeStamp.isna()]>lo_res_info.ICPRemTimeStamp[~lo_res_info.ICPRemTimeStamp.isna()]).astype(int)
lo_res_info['ICPRevisedIndicator'] = (lo_res_info.TILDate[~lo_res_info.ICPRevisionTimeStamp.isna()]>lo_res_info.ICPRevisionTimeStamp[~lo_res_info.ICPRevisionTimeStamp.isna()]).astype(int)
lo_res_info = lo_res_info.drop(columns=['ICPRemTimeStamp','ICPRevisionTimeStamp'])

## Save modified Daily hourly value dataframes in new directory
lo_res_info.to_csv(os.path.join(formatted_data_dir,'formatted_low_resolution_values.csv'),index=False)

## Calculate means and maxes
# Filter to only keep values from the first week
first_week_lo_res_info = lo_res_info[(lo_res_info.TILTimepoint>=1)&(lo_res_info.TILTimepoint<=7)&(lo_res_info.TILExpected==1)].reset_index(drop=True).rename(columns={'ICPmean':'ICP','CPPmean':'CPP'})

# Calculate ICPmean, CPPmean, ICPmax, and CPPmax
lo_res_maxes_means = first_week_lo_res_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','TotalSum'],value_vars=['CPP','ICP','ChangeInCPP','ChangeInICP']).groupby(by=['GUPI','variable'],as_index=False).value.aggregate({'median':'median','mean':'mean','min':'min','max':'max'})
lo_res_maxes_means = pd.pivot_table(lo_res_maxes_means, values = ['mean','median','min','max'], index=['GUPI'], columns = 'variable').reset_index()
lo_res_maxes_means.columns = [col[1]+col[0] for col in lo_res_maxes_means.columns.values]

# Save ICPmean, CPPmean, ICPmax, and CPPmax
lo_res_maxes_means.to_csv(os.path.join(formatted_data_dir,'formatted_low_resolution_mins_maxes_medians_means.csv'),index=False)

### V. Load and prepare high-resolution ICP and CPP information
## Load and prepare formatted daily TIL dataframe
formatted_TIL_scores = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_scores.csv'))

# Convert dates from string to date format
formatted_TIL_scores.TILDate = pd.to_datetime(formatted_TIL_scores.TILDate,format = '%Y-%m-%d')
formatted_TIL_scores.ICUAdmTimeStamp = pd.to_datetime(formatted_TIL_scores.ICUAdmTimeStamp,format = '%Y-%m-%d %H:%M:%S')
formatted_TIL_scores.ICUDischTimeStamp = pd.to_datetime(formatted_TIL_scores.ICUDischTimeStamp,format = '%Y-%m-%d %H:%M:%S')

## Load and format high-resolution ICP/CPP summary values
# Load high-resolution ICP/CPP summary values of same day as TIL assessments
hi_res_info = pd.read_csv('../CENTER-TBI/HighResolution/til_same.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients for whom TIL values exist
hi_res_info = hi_res_info[hi_res_info.GUPI.isin(formatted_TIL_scores.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

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

# Calculate day-by-day differences in ICP and CPP
hi_res_info['ChangeInCPP'] = hi_res_info.groupby(['GUPI'])['CPP_mean'].diff()
hi_res_info['ChangeInICP'] = hi_res_info.groupby(['GUPI'])['ICP_mean'].diff()

# Merge hi-resolution ICP and CPP values with daily TIL
hi_res_info = formatted_TIL_scores[['GUPI','TILTimepoint','TILDate','TotalSum']].merge(hi_res_info,how='outer').rename(columns={'ICP_mean':'ICPmean','CPP_mean':'CPPmean','ICP_n':'nICP','CPP_n':'nCPP'})

# Add a marker to indicate whether TIL score is expected at each row
hi_res_info['TILExpected'] = hi_res_info['TILTimepoint'].notna().astype(int)

# Determine GUPIs with missing timepoints
none_GUPIs = hi_res_info[hi_res_info.TILTimepoint.isna()].GUPI.unique()

# Iterate through 'None' GUPIs and impute missing timepoint values
for curr_GUPI in none_GUPIs:
    curr_GUPI_hi_res = hi_res_info[hi_res_info.GUPI==curr_GUPI].reset_index(drop=True)
    non_missing_timepoint_mask = ~curr_GUPI_hi_res.TILTimepoint.isna()
    if non_missing_timepoint_mask.sum() != 1:
        curr_default_date = (curr_GUPI_hi_res.TILDate[non_missing_timepoint_mask] - pd.to_timedelta(curr_GUPI_hi_res.TILTimepoint.astype(float)[non_missing_timepoint_mask],unit='d')).mode()[0]
    else:
        curr_default_date = (curr_GUPI_hi_res.TILDate[non_missing_timepoint_mask] - timedelta(days=curr_GUPI_hi_res.TILTimepoint.astype(float)[non_missing_timepoint_mask].values[0])).mode()[0]
    fixed_timepoints_vector = ((curr_GUPI_hi_res.TILDate - curr_default_date)/np.timedelta64(1,'D')).astype(int)
    fixed_timepoints_vector.index=hi_res_info[hi_res_info.GUPI==curr_GUPI].index
    hi_res_info.TILTimepoint[hi_res_info.GUPI==curr_GUPI] = fixed_timepoints_vector

# Convert TILTimepoint variable from string to integer
hi_res_info.TILTimepoint = hi_res_info.TILTimepoint.astype(int)

# Sort dataframe
hi_res_info = hi_res_info.sort_values(['GUPI','TILTimepoint'],ignore_index=True)

## Add ICP/CPP monitoring stoppage information
# Load preformatted ICP/CPP monitoring stoppage information
icp_monitored_patients = pd.read_csv('../CENTER-TBI/ICP_monitored_patients.csv',na_values = ["NA","NaN"," ", ""])

# Extract 'DateComponent' from ICP monitoring stoppage timestamp values
icp_monitored_patients['ICPRemTimeStamp'] = pd.to_datetime(icp_monitored_patients['ICPRemTimeStamp'].str[:10],format = '%Y-%m-%d').dt.date
icp_monitored_patients['ICPRevisionTimeStamp'] = pd.to_datetime(icp_monitored_patients['ICPRevisionTimeStamp'].str[:10],format = '%Y-%m-%d').dt.date

# Merge ICP removal timestamp and ICP stoppage information to low-resolution neuromonitoring dataframe
hi_res_info = hi_res_info.merge(icp_monitored_patients[['GUPI','ICPRemTimeStamp','ICPRevisionTimeStamp','ICUReasonICP','ICPDevice','ICUReasonForTypeICPMont','ICUReasonForTypeICPMontPare','ICUProblemsICP','ICUProblemsICPYes','ICUCatheterICP','ICPMonitorStop','ICPStopReason']],how='left')

# Add indicators for ICP monitor removal and/or revision
hi_res_info['ICPRemovedIndicator'] = (hi_res_info.TILDate[~hi_res_info.ICPRemTimeStamp.isna()]>hi_res_info.ICPRemTimeStamp[~hi_res_info.ICPRemTimeStamp.isna()]).astype(int)
hi_res_info['ICPRevisedIndicator'] = (hi_res_info.TILDate[~hi_res_info.ICPRevisionTimeStamp.isna()]>hi_res_info.ICPRevisionTimeStamp[~hi_res_info.ICPRevisionTimeStamp.isna()]).astype(int)
hi_res_info = hi_res_info.drop(columns=['ICPRemTimeStamp','ICPRevisionTimeStamp'])

## Save modified Daily hourly value dataframes in new directory
hi_res_info.to_csv(os.path.join(formatted_data_dir,'formatted_high_resolution_values.csv'),index=False)

## Calculate means, medians, and maxes
# Filter to only keep values from the first week
first_week_hi_res_info = hi_res_info[(hi_res_info.TILTimepoint>=1)&(hi_res_info.TILTimepoint<=7)&(hi_res_info.TILExpected==1)].reset_index(drop=True).rename(columns={'ICPmean':'ICP','CPPmean':'CPP'})

# Calculate ICPmean, CPPmean, ICPmax, and CPPmax
hi_res_maxes_means = first_week_hi_res_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','TotalSum'],value_vars=['CPP','ICP','ChangeInCPP','ChangeInICP']).groupby(by=['GUPI','variable'],as_index=False).value.aggregate({'median':'median','mean':'mean','min':'min','max':'max'})
hi_res_maxes_means = pd.pivot_table(hi_res_maxes_means, values = ['mean','median','min','max'], index=['GUPI'], columns = 'variable').reset_index()
hi_res_maxes_means.columns = [col[1]+col[0] for col in hi_res_maxes_means.columns.values]

# Save ICPmean, CPPmean, ICPmax, and CPPmax
hi_res_maxes_means.to_csv(os.path.join(formatted_data_dir,'formatted_high_resolution_mins_maxes_medians_means.csv'),index=False)

### VI. Load and prepare demographic information and baseline characteristics
## Load demographic and outcome scores of patients in TIL dataframe
# Load formatted daily TIL dataframe
formatted_TIL_scores = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_scores.csv'))

# Load low-resolution value dataframe
lo_res_info = pd.read_csv(os.path.join(formatted_data_dir,'formatted_low_resolution_values.csv'))

# Load high-resolution value dataframe
hi_res_info = pd.read_csv(os.path.join(formatted_data_dir,'formatted_high_resolution_values.csv'))

# Load CENTER-TBI dataset demographic information
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Select columns that indicate pertinent baseline and outcome information
CENTER_TBI_demo_info = CENTER_TBI_demo_info[['GUPI','PatientType','SiteCode','Age','Sex','Race','GCSScoreBaselineDerived','GOSE6monthEndpointDerived','ICURaisedICP','DecompressiveCranReason','AssociatedStudy_1','AssociatedStudy_2','AssociatedStudy_3']].reset_index(drop=True)

# Filter GUPIs to daily TIL dataframe GUPIs
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.GUPI.isin(formatted_TIL_scores.GUPI)].reset_index(drop=True)

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
CENTER_TBI_demo_info.LowResolutionSet[CENTER_TBI_demo_info.GUPI.isin(lo_res_info[(~lo_res_info.ICPmean.isna())|(~lo_res_info.CPPmean.isna())].GUPI)] = 1
CENTER_TBI_demo_info.HighResolutionSet[CENTER_TBI_demo_info.GUPI.isin(hi_res_info[(~hi_res_info.ICPmean.isna())|(~hi_res_info.CPPmean.isna())].GUPI)] = 1

# Load and prepare ordinal prediction estimates
ordinal_prediction_estimates = pd.read_csv('../CENTER-TBI/APM_deepMN_compiled_test_predictions.csv').drop(columns='Unnamed: 0')
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
CENTER_TBI_demo_info.to_csv(os.path.join(formatted_data_dir,'formatted_outcome_and_demographics.csv'),index=False)

### VII. Calculate TIL_1987, PILOT, and TIL_Basic
## Load and prepare unweighted TIL scores and components
# Load unweighted TIL scores and components
unweighted_daily_TIL_info = pd.read_csv(os.path.join(formatted_data_dir,'formatted_unweighted_TIL_scores.csv'))

# Convert ICU admission/discharge timestamps to datetime variables
unweighted_daily_TIL_info['TILDate'] = pd.to_datetime(unweighted_daily_TIL_info['TILDate'],format = '%Y-%m-%d')
unweighted_daily_TIL_info['ICUAdmTimeStamp'] = pd.to_datetime(unweighted_daily_TIL_info['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S')
unweighted_daily_TIL_info['ICUDischTimeStamp'] = pd.to_datetime(unweighted_daily_TIL_info['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

## Calculate TIL_1987
calc_daily_TIL_1987_info = calculate_TIL_1987(unweighted_daily_TIL_info)

# Save
calc_daily_TIL_1987_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_1987_scores.csv'),index=False)

## Calculate PILOT
calc_daily_PILOT_info = calculate_PILOT(unweighted_daily_TIL_info)

# Save
calc_daily_PILOT_info.to_csv(os.path.join(formatted_data_dir,'formatted_PILOT_scores.csv'),index=False)

## Calculate TIL_Basic
calc_daily_TIL_Basic_info = calculate_TIL_Basic(unweighted_daily_TIL_info)

# Save
calc_daily_TIL_Basic_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_Basic_scores.csv'),index=False)

### VIII. Calculate summarised TIL metrics
## TIL
# Load formatted daily TIL dataframe
formatted_TIL_scores = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_scores.csv'))

# Filter to only keep TIL scores from the first week
first_week_daily_TIL_info = formatted_TIL_scores[(formatted_TIL_scores.TILTimepoint>=1)&(formatted_TIL_scores.TILTimepoint<=7)].reset_index(drop=True)

# Keep rows corresponding to TIL_max
til_max_info = first_week_daily_TIL_info.loc[first_week_daily_TIL_info.groupby(['GUPI'])['TotalSum'].idxmax()].reset_index(drop=True).rename(columns={'TotalSum':'TILmax'})

# Calculate TIL_median info
til_median_info = pd.pivot_table(first_week_daily_TIL_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].median(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TotalSum':'TILmedian'})

# Calculate TIL_mean info
til_mean_info = pd.pivot_table(first_week_daily_TIL_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TotalSum':'TILmean'})

## PILOT
# Load formatted daily PILOT dataframe
calc_daily_PILOT_info = pd.read_csv(os.path.join(formatted_data_dir,'formatted_PILOT_scores.csv'))

# Filter to only keep PILOT scores from the first week
first_week_daily_PILOT_info = calc_daily_PILOT_info[(calc_daily_PILOT_info.TILTimepoint>=1)&(calc_daily_PILOT_info.TILTimepoint<=7)].reset_index(drop=True)

# Keep rows corresponding to PILOT_max
pilot_max_info = first_week_daily_PILOT_info.loc[first_week_daily_PILOT_info.groupby(['GUPI'])['PILOTSum'].idxmax()].reset_index(drop=True).rename(columns={'PILOTSum':'PILOTmax'})

# Calculate PILOT_median info
pilot_median_info = pd.pivot_table(first_week_daily_PILOT_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].median(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'PILOTSum':'PILOTmedian','TotalSum':'TILmedian'})

# Calculate PILOT_mean info
pilot_mean_info = pd.pivot_table(first_week_daily_PILOT_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'PILOTSum':'PILOTmean','TotalSum':'TILmean'})

## TIL_1987
# Load formatted daily TIL_1987 dataframe
calc_daily_TIL_1987_info = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_1987_scores.csv'))

# Filter to only keep TIL_1987 scores from the first week
first_week_daily_TIL_1987_info = calc_daily_TIL_1987_info[(calc_daily_TIL_1987_info.TILTimepoint>=1)&(calc_daily_TIL_1987_info.TILTimepoint<=7)].reset_index(drop=True)

# Keep rows corresponding to TIL_1987_max
til_1987_max_info = first_week_daily_TIL_1987_info.loc[first_week_daily_TIL_1987_info.groupby(['GUPI'])['TIL_1987Sum'].idxmax()].reset_index(drop=True).rename(columns={'TIL_1987Sum':'TIL_1987max'})

# Calculate TIL_1987_median info
til_1987_median_info = pd.pivot_table(first_week_daily_TIL_1987_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].median(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TIL_1987Sum':'TIL_1987median','TotalSum':'TILmedian'})

# Calculate TIL_1987_mean info
til_1987_mean_info = pd.pivot_table(first_week_daily_TIL_1987_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TIL_1987Sum':'TIL_1987mean','TotalSum':'TILmean'})

## TIL_Basic
# Load formatted daily TIL_Basic dataframe
calc_daily_TIL_Basic_info = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_Basic_scores.csv'))

# Filter to only keep TIL_Basic scores from the first week
first_week_daily_TIL_Basic_info = calc_daily_TIL_Basic_info[(calc_daily_TIL_Basic_info.TILTimepoint>=1)&(calc_daily_TIL_Basic_info.TILTimepoint<=7)].reset_index(drop=True)

# Keep rows corresponding to TIL_Basic_max
til_basic_max_info = first_week_daily_TIL_Basic_info.loc[first_week_daily_TIL_Basic_info.groupby(['GUPI'])['TIL_Basic'].idxmax()].reset_index(drop=True).rename(columns={'TIL_Basic':'TIL_Basicmax'})

# Calculate TIL_Basic_median info
til_basic_median_info = pd.pivot_table(first_week_daily_TIL_Basic_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].median(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TIL_Basic':'TIL_Basicmedian','TotalSum':'TILmedian'})

# Calculate TIL_Basic_mean info
til_basic_mean_info = pd.pivot_table(first_week_daily_TIL_Basic_info.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp','DailyTILCompleteStatus']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'TIL_Basic':'TIL_Basicmean','TotalSum':'TILmean'})

## Save dataframes
# Max dataframes
til_max_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_max.csv'),index=False)
pilot_max_info.to_csv(os.path.join(formatted_data_dir,'formatted_PILOT_max.csv'),index=False)
til_1987_max_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_1987_max.csv'),index=False)
til_basic_max_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_Basic_max.csv'),index=False)

# Median dataframes
til_median_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_median.csv'),index=False)
pilot_median_info.to_csv(os.path.join(formatted_data_dir,'formatted_PILOT_median.csv'),index=False)
til_1987_median_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_1987_median.csv'),index=False)
til_basic_median_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_Basic_median.csv'),index=False)

# Mean dataframes
til_mean_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_mean.csv'),index=False)
pilot_mean_info.to_csv(os.path.join(formatted_data_dir,'formatted_PILOT_mean.csv'),index=False)
til_1987_mean_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_1987_mean.csv'),index=False)
til_basic_mean_info.to_csv(os.path.join(formatted_data_dir,'formatted_TIL_Basic_mean.csv'),index=False)

### IX. Load and prepare serum sodium values from CENTER-TBI
## Load and prepare sodium values
# Load sodium lab values
sodium_values = pd.read_csv('../CENTER-TBI/Labs/data.csv',na_values = ["NA","NaN"," ", ""])[['GUPI','DLDate','DLTime','DLSodiummmolL']].dropna(subset=['DLDate','DLSodiummmolL'],how='any').sort_values(by=['GUPI','DLDate']).reset_index(drop=True)

# Convert `DLDate` to timestamp format
sodium_values['TILDate'] = pd.to_datetime(sodium_values['DLDate'],format = '%Y-%m-%d')

# Calculate daily mean sodium
sodium_values = sodium_values.groupby(['GUPI','TILDate'],as_index=False).DLSodiummmolL.aggregate({'meanSodium':'mean','nSodium':'count'})

# Calculate daily difference in sodium
sodium_values['ChangeInSodium'] = sodium_values.groupby(['GUPI'])['meanSodium'].diff()

# Load formatted TIL values and add row index
formatted_TIL_scores = pd.read_csv(os.path.join(formatted_data_dir,'formatted_TIL_scores.csv'))

# Convert ICU admission/discharge timestamps to datetime variables
formatted_TIL_scores['TILDate'] = pd.to_datetime(formatted_TIL_scores['TILDate'],format = '%Y-%m-%d')
formatted_TIL_scores['ICUAdmTimeStamp'] = pd.to_datetime(formatted_TIL_scores['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S')
formatted_TIL_scores['ICUDischTimeStamp'] = pd.to_datetime(formatted_TIL_scores['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# Create a list of patients who received hypertonic saline or did not receive hyperosmolar therapy at all
HTS_GUPIs = formatted_TIL_scores[formatted_TIL_scores.Hypertonic!=0].GUPI.unique()
Mannitol_GUPIs = formatted_TIL_scores[formatted_TIL_scores.Mannitol!=0].GUPI.unique()
nonHOT_GUPIs = np.setdiff1d(formatted_TIL_scores.GUPI,np.union1d(HTS_GUPIs,Mannitol_GUPIs))

# Merge sodium values to formatted TIL dataframe
sodium_TIL_dataframe = formatted_TIL_scores.merge(sodium_values,how='left')

# Add columns to indicate hyperosmolar treatment stratification
sodium_TIL_dataframe['HTSPtInd'] = sodium_TIL_dataframe.GUPI.isin(HTS_GUPIs).astype(int)
sodium_TIL_dataframe['NoHyperosmolarPtInd'] = sodium_TIL_dataframe.GUPI.isin(nonHOT_GUPIs).astype(int)
sodium_TIL_dataframe['MannitolPtInd'] = sodium_TIL_dataframe.GUPI.isin(Mannitol_GUPIs).astype(int)

# Remove rows with missing sodium values
sodium_TIL_dataframe = sodium_TIL_dataframe.dropna(subset='meanSodium').drop_duplicates(ignore_index=True)

## Save prepared sodium values
sodium_TIL_dataframe.to_csv(os.path.join(formatted_data_dir,'formatted_daily_sodium_values.csv'),index=False)

## Calculate means and maxes
# Filter to only keep values from the first week
first_week_sodium_info = sodium_TIL_dataframe[(sodium_TIL_dataframe.TILTimepoint>=1)&(sodium_TIL_dataframe.TILTimepoint<=7)].reset_index(drop=True)

# Calculate meanSodium and maxSodium
sodium_maxes_means = first_week_sodium_info.groupby('GUPI',as_index=False).meanSodium.aggregate({'meanSodium':'mean','maxSodium':'max'})

# Calculate deltas meanSodium and maxSodium
deltas_sodium_maxes_means = first_week_sodium_info.groupby('GUPI',as_index=False).ChangeInSodium.aggregate({'meanChangeSodium':'mean','maxChangeSodium':'max'})

# Save meanSodium and maxSodium
sodium_maxes_means = sodium_maxes_means.merge(deltas_sodium_maxes_means,how='left')

# Add columns to indicate hyperosmolar treatment stratification
sodium_maxes_means['HTSPtInd'] = sodium_maxes_means.GUPI.isin(HTS_GUPIs).astype(int)
sodium_maxes_means['NoHyperosmolarPtInd'] = sodium_maxes_means.GUPI.isin(nonHOT_GUPIs).astype(int)
sodium_maxes_means['MannitolPtInd'] = sodium_maxes_means.GUPI.isin(Mannitol_GUPIs).astype(int)

# Save values
sodium_maxes_means.to_csv(os.path.join(formatted_data_dir,'formatted_sodium_maxes_means.csv'),index=False)