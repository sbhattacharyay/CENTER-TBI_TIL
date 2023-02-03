#### Master Script 1a: Extract and prepare study sample covariates from CENTER-TBI dataset ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and prepare TIL information
# III. Load and prepare low-resolution ICP and CPP information
# IV. Load and prepare six-month functional outcome scores
# V. Prepare dataframe of high-resolution sub-study patients

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

### II. Load and prepare TIL information
## Fix timestamp inaccuracies in DailyTIL dataframe based on TILTimepoint
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

## Load ICU or hospital admission timestamps of patients in TIL dataframe
# Load CENTER-TBI dataset demographic information
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.GUPI.isin(daily_TIL_info.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Filter patients over 16 years in age
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.Age>=16].reset_index(drop=True)

# Select columns that indicate pertinent admission and discharge times
CENTER_TBI_datetime = CENTER_TBI_demo_info[['GUPI','PatientType','ICUAdmDate','ICUAdmTime','ICUDischDate','ICUDischTime','WardAdmDate','WardAdmTime','HospDischDate','HospDischTime']].reset_index(drop=True)

# Compile date and time information and convert to datetime
CENTER_TBI_datetime['ICUAdmTimeStamp'] = CENTER_TBI_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUAdmTimeStamp'][CENTER_TBI_datetime.ICUAdmDate.isna() | CENTER_TBI_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_datetime['WardAdmTimeStamp'] = CENTER_TBI_datetime[['WardAdmDate', 'WardAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['WardAdmTimeStamp'][CENTER_TBI_datetime.WardAdmDate.isna() | CENTER_TBI_datetime.WardAdmTime.isna()] = np.nan
CENTER_TBI_datetime['WardAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['WardAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_datetime['HospDischTimeStamp'] = CENTER_TBI_datetime[['HospDischDate', 'HospDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['HospDischTimeStamp'][CENTER_TBI_datetime.HospDischDate.isna() | CENTER_TBI_datetime.HospDischTime.isna()] = np.nan
CENTER_TBI_datetime['HospDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['HospDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600
CENTER_TBI_datetime['WardDurationHours'] = (CENTER_TBI_datetime['HospDischTimeStamp'] - CENTER_TBI_datetime['WardAdmTimeStamp']).astype('timedelta64[s]')/3600

# For one patient with missing ICU admission date and time, impute with ward admission date and time
ward_impute_index = CENTER_TBI_datetime.ICUAdmDate.isna()&CENTER_TBI_datetime.ICUAdmTime.isna()
CENTER_TBI_datetime.ICUAdmDate[ward_impute_index] = CENTER_TBI_datetime.WardAdmDate[ward_impute_index]
CENTER_TBI_datetime.ICUAdmTime[ward_impute_index] = CENTER_TBI_datetime.WardAdmTime[ward_impute_index]
CENTER_TBI_datetime['ICUAdmTimeStamp'] = CENTER_TBI_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUAdmTimeStamp'][CENTER_TBI_datetime.ICUAdmDate.isna() | CENTER_TBI_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# To fill in missing timestamps, first check all available timestamps in nonrepeatable time info
non_repeatable_date_info = pd.read_csv('../CENTER-TBI/NonRepeatableDateInfo/data.csv',na_values = ["NA","NaN"," ", ""]).dropna(axis=1,how='all')
non_repeatable_date_info = non_repeatable_date_info[non_repeatable_date_info.GUPI.isin(CENTER_TBI_datetime.GUPI)].reset_index(drop=True)
non_repeatable_date_info = non_repeatable_date_info[['GUPI']+[col for col in non_repeatable_date_info.columns if 'Date' in col]].select_dtypes(exclude='number')
non_repeatable_date_info = non_repeatable_date_info.melt(id_vars='GUPI',var_name='desc',value_name='date').dropna(subset=['date']).sort_values(by=['GUPI','date']).reset_index(drop=True)
non_repeatable_date_info['desc'] = non_repeatable_date_info['desc'].str.replace('Date','')

non_repeatable_time_info = pd.read_csv('../CENTER-TBI/NonRepeatableTimeInfo/data.csv',na_values = ["NA","NaN"," ", ""]).dropna(axis=1,how='all')
non_repeatable_time_info = non_repeatable_time_info[non_repeatable_time_info.GUPI.isin(CENTER_TBI_datetime.GUPI)].reset_index(drop=True)
non_repeatable_time_info = non_repeatable_time_info[['GUPI']+[col for col in non_repeatable_time_info.columns if ('Time' in col)|('Hour' in col)]].select_dtypes(exclude='number')
non_repeatable_time_info = non_repeatable_time_info.melt(id_vars='GUPI',var_name='desc',value_name='time').dropna(subset=['time']).sort_values(by=['GUPI','time']).reset_index(drop=True)
non_repeatable_time_info['desc'] = non_repeatable_time_info['desc'].str.replace('Time','')

non_repeatable_datetime_info = pd.merge(non_repeatable_date_info,non_repeatable_time_info,how='outer',on=['GUPI','desc']).dropna(how='all',subset=['date','time'])
non_repeatable_datetime_info = non_repeatable_datetime_info.sort_values(by=['GUPI','date','time']).reset_index(drop=True)
os.makedirs('../timestamps/',exist_ok=True)
non_repeatable_datetime_info.to_csv('../timestamps/nonrepeatable_timestamps.csv',index=False)

# To fill in missing timestamps, second check all available timestamps in repeatable time info
repeatable_datetime_info = pd.read_csv('../CENTER-TBI/RepeatableDateTimeInfo/data.csv',na_values = ["NA","NaN"," ", ""]).dropna(axis=1,how='all')
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info.GUPI.isin(CENTER_TBI_datetime.GUPI)].reset_index(drop=True)
repeatable_datetime_info['RowID'] = list(range(1,repeatable_datetime_info.shape[0]+1))
repeatable_datetime_info = repeatable_datetime_info.merge(CENTER_TBI_datetime,how='left',on='GUPI')

# Fix cases in which a missing date or time can be inferred from another variable
missing_HV_date = repeatable_datetime_info[(repeatable_datetime_info.HVDate.isna())&(~repeatable_datetime_info.HourlyValueTimePoint.isna())&(repeatable_datetime_info.HourlyValueTimePoint != 'None')].reset_index(drop=True)
missing_HV_date = missing_HV_date[['RowID','PatientType','GUPI','ICUAdmDate','WardAdmDate','HourlyValueTimePoint','HVDate']]
missing_HV_date['ICUAdmDate'] = pd.to_datetime(missing_HV_date['ICUAdmDate'],format = '%Y-%m-%d')
missing_HV_date['HourlyValueTimePoint'] = missing_HV_date['HourlyValueTimePoint'].astype('int')
missing_HV_date['HVDate'] = (missing_HV_date['ICUAdmDate'] + pd.to_timedelta((missing_HV_date['HourlyValueTimePoint']-1),'days')).astype(str)
for curr_rowID in tqdm(missing_HV_date.RowID,'Fixing missing HVDate values'):
    curr_Date = missing_HV_date.HVDate[missing_HV_date.RowID == curr_rowID].values[0]
    repeatable_datetime_info.HVDate[repeatable_datetime_info.RowID == curr_rowID] = curr_Date
repeatable_datetime_info.HVTime[(repeatable_datetime_info.HVTime.isna())&(~repeatable_datetime_info.HVHour.isna())] = repeatable_datetime_info.HVHour[(repeatable_datetime_info.HVTime.isna())&(~repeatable_datetime_info.HVHour.isna())]

ward_daily_vitals = pd.read_csv('../CENTER-TBI/DailyVitals/data.csv',na_values = ["NA","NaN"," ", ""])
ward_daily_vitals = ward_daily_vitals[ward_daily_vitals.PatientLocation == 'Ward'].reset_index(drop=True)
ward_daily_vitals = ward_daily_vitals[['GUPI','DVTimepoint','DVDate']]
repeatable_datetime_info = pd.merge(repeatable_datetime_info,ward_daily_vitals,how='left',on=['GUPI','DVTimepoint','DVDate'],indicator=True)
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info._merge != 'both'].drop(columns=['_merge']).reset_index(drop=True)
missing_DV_date = repeatable_datetime_info[(repeatable_datetime_info.DVDate.isna())&(~repeatable_datetime_info.DVTimepoint.isna())&(repeatable_datetime_info.DVTimepoint != 'None')].reset_index(drop=True)
missing_DV_date = missing_DV_date[['RowID','GUPI','ICUAdmDate','DVTimepoint','DVDate']]
missing_DV_date['ICUAdmDate'] = pd.to_datetime(missing_DV_date['ICUAdmDate'],format = '%Y-%m-%d')
missing_DV_date['DVTimepoint'] = missing_DV_date['DVTimepoint'].astype('int')
missing_DV_date['DVDate'] = (missing_DV_date['ICUAdmDate'] + pd.to_timedelta((missing_DV_date['DVTimepoint']-1),'days')).astype(str)
for curr_rowID in tqdm(missing_DV_date.RowID,'Fixing missing DVDate values'):
    curr_Date = missing_DV_date.DVDate[missing_DV_date.RowID == curr_rowID].values[0]
    repeatable_datetime_info.DVDate[repeatable_datetime_info.RowID == curr_rowID] = curr_Date
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info.TimePoint.isna()].reset_index(drop=True)

non_baseline_outcomes = pd.read_csv('../CENTER-TBI/Outcomes/data.csv',na_values = ["NA","NaN"," ", ""])
non_baseline_outcomes = non_baseline_outcomes[non_baseline_outcomes != 'Base'].reset_index(drop=True)
non_baseline_outcomes = non_baseline_outcomes[['GUPI','Timepoint']]
repeatable_datetime_info = pd.merge(repeatable_datetime_info,non_baseline_outcomes,how='left',on=['GUPI','Timepoint'],indicator=True)
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info._merge != 'both'].drop(columns=['_merge']).reset_index(drop=True)

missing_TIL_date = repeatable_datetime_info[(repeatable_datetime_info.TILDate.isna())&(~repeatable_datetime_info.TILTimepoint.isna())&(repeatable_datetime_info.TILTimepoint != 'None')].reset_index(drop=True)
missing_TIL_date = missing_TIL_date[['RowID','GUPI','ICUAdmDate','TILTimepoint','TILDate']]
missing_TIL_date['ICUAdmDate'] = pd.to_datetime(missing_TIL_date['ICUAdmDate'],format = '%Y-%m-%d')
missing_TIL_date['TILTimepoint'] = missing_TIL_date['TILTimepoint'].astype('int')
missing_TIL_date['TILDate'] = (missing_TIL_date['ICUAdmDate'] + pd.to_timedelta((missing_TIL_date['TILTimepoint']-1),'days')).astype(str)
for curr_rowID in tqdm(missing_TIL_date.RowID,'Fixing missing TILDate values'):
    curr_Date = missing_TIL_date.TILDate[missing_TIL_date.RowID == curr_rowID].values[0]
    repeatable_datetime_info.TILDate[repeatable_datetime_info.RowID == curr_rowID] = curr_Date

repeatable_date_info = repeatable_datetime_info[['GUPI','RowID']+[col for col in repeatable_datetime_info.columns if 'Date' in col]].drop(columns=['ICUAdmDate']).dropna(axis=1,how='all')
repeatable_date_info['RowID'] = repeatable_date_info['RowID'].astype(str)
repeatable_date_info = repeatable_date_info.select_dtypes(exclude='number').melt(id_vars=['GUPI','RowID'],var_name='desc',value_name='date').dropna(subset=['date']).sort_values(by=['GUPI','date']).reset_index(drop=True)
repeatable_date_info['desc'] = repeatable_date_info['desc'].str.replace('Date','')
repeatable_time_info = repeatable_datetime_info[['GUPI','RowID']+[col for col in repeatable_datetime_info.columns if 'Time' in col]].dropna(axis=1,how='all')
repeatable_time_info['RowID'] = repeatable_time_info['RowID'].astype(str)
repeatable_time_info = repeatable_time_info.select_dtypes(exclude='number').melt(id_vars=['GUPI','RowID'],var_name='desc',value_name='time').dropna(subset=['time']).sort_values(by=['GUPI']).reset_index(drop=True)
repeatable_time_info['desc'] = repeatable_time_info['desc'].str.replace('Time','')
repeatable_datetime_info = pd.merge(repeatable_date_info,repeatable_time_info,how='outer',on=['RowID','GUPI','desc']).dropna(how='all',subset=['date','time'])
repeatable_datetime_info = repeatable_datetime_info.sort_values(by=['GUPI','date']).reset_index(drop=True)
repeatable_datetime_info.to_csv('../timestamps/repeatable_timestamps.csv',index=False)

# Plan A: replace missing ICU admission timestamp with ED discharge times
missing_ICU_adm_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)

ED_discharge_datetime = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_adm_timestamps.GUPI))&(non_repeatable_datetime_info.desc == 'EDDisch')].fillna('1970-01-01').reset_index(drop=True)
ED_discharge_datetime = ED_discharge_datetime.merge(missing_ICU_adm_timestamps[['GUPI','ICUAdmDate']].rename(columns={'ICUAdmDate':'date'}),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm(ED_discharge_datetime.GUPI,'Imputing missing ICU admission timestamps with matching ED discharge information'):
    curr_EDDisch_time = ED_discharge_datetime.time[ED_discharge_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_adm_timestamps.ICUAdmTime[missing_ICU_adm_timestamps.GUPI == curr_GUPI] = curr_EDDisch_time
    CENTER_TBI_datetime.ICUAdmTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_EDDisch_time

CENTER_TBI_datetime['ICUAdmTimeStamp'] = CENTER_TBI_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUAdmTimeStamp'][CENTER_TBI_datetime.ICUAdmDate.isna() | CENTER_TBI_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan B: replace missing ICU admission timestamp with time of arrival to study hospital
missing_ICU_adm_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)

hosp_arrival_datetime = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_adm_timestamps.GUPI))&(non_repeatable_datetime_info.desc == 'PresSTHosp')].reset_index(drop=True)
hosp_arrival_datetime = hosp_arrival_datetime.merge(missing_ICU_adm_timestamps[['GUPI','ICUAdmDate']].rename(columns={'ICUAdmDate':'date'}),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm(hosp_arrival_datetime.GUPI,'Imputing missing ICU admission timestamps with matching hosptial arrival information'):
    curr_hosp_arrival_time = hosp_arrival_datetime.time[hosp_arrival_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_adm_timestamps.ICUAdmTime[missing_ICU_adm_timestamps.GUPI == curr_GUPI] = curr_hosp_arrival_time
    CENTER_TBI_datetime.ICUAdmTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_hosp_arrival_time

CENTER_TBI_datetime['ICUAdmTimeStamp'] = CENTER_TBI_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUAdmTimeStamp'][CENTER_TBI_datetime.ICUAdmDate.isna() | CENTER_TBI_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan C: replace missing ICU admission timestamp with time of initial study consent
missing_ICU_adm_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)

consent_datetime = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_adm_timestamps.GUPI))&(non_repeatable_datetime_info.desc == 'InfConsInitial')].reset_index(drop=True)
consent_datetime = consent_datetime.merge(missing_ICU_adm_timestamps[['GUPI','ICUAdmDate']].rename(columns={'ICUAdmDate':'date'}),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm(consent_datetime.GUPI,'Imputing missing ICU admission timestamps with matching study consent information'):
    curr_consent_time = consent_datetime.time[consent_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_adm_timestamps.ICUAdmTime[missing_ICU_adm_timestamps.GUPI == curr_GUPI] = curr_consent_time
    CENTER_TBI_datetime.ICUAdmTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_consent_time

CENTER_TBI_datetime['ICUAdmTimeStamp'] = CENTER_TBI_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUAdmTimeStamp'][CENTER_TBI_datetime.ICUAdmDate.isna() | CENTER_TBI_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan D: replace missing ICU admission timestamp with midday (no other information available on that day)
missing_ICU_adm_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)
CENTER_TBI_datetime.ICUAdmTime[CENTER_TBI_datetime.GUPI.isin(missing_ICU_adm_timestamps.GUPI)] = '12:00:00'
CENTER_TBI_datetime['ICUAdmTimeStamp'] = CENTER_TBI_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUAdmTimeStamp'][CENTER_TBI_datetime.ICUAdmDate.isna() | CENTER_TBI_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan A: if patient died in the ICU, replace missing discharge information with death date
missing_ICU_disch_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

death_datetime = non_repeatable_datetime_info[non_repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI)&(non_repeatable_datetime_info.desc == 'Death')].reset_index(drop=True)
death_datetime = death_datetime.merge(missing_ICU_disch_timestamps[['GUPI','ICUDischDate']].rename(columns={'ICUDischDate':'date'}).dropna(subset=['date']),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm(death_datetime.GUPI,'Imputing missing ICU discharge timestamps with matching time of death information'):
    curr_death_time = death_datetime.time[death_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_death_time
    CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_death_time

CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan A_1: if patient died in the ICU, but either date time or date is unavailable, replace with manual check
missing_ICU_disch_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

death_datetime = non_repeatable_datetime_info[non_repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI)&(non_repeatable_datetime_info.desc == 'Death')].reset_index(drop=True)
death_datetime = death_datetime[death_datetime.GUPI.isin(missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.ICUDischTime.isna()]['GUPI'])].reset_index(drop=True)

for curr_GUPI in tqdm(death_datetime.GUPI,'Imputing missing ICU discharge timestamps with matching time of death information'):
    curr_death_date = death_datetime.date[death_datetime.GUPI == curr_GUPI].values[0]
    curr_death_time = death_datetime.time[death_datetime.GUPI == curr_GUPI].values[0]
    
    missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_death_date
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_death_time
    
    CENTER_TBI_datetime.ICUDischDate[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_death_date
    CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_death_time
    
CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan B: Find transfer-out-of-ICU date and time if available
missing_ICU_disch_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

transfer_datetime = pd.read_csv('../CENTER-TBI/TransitionsOfCare/data.csv',na_values = ["NA","NaN"," ", ""])
transfer_datetime = transfer_datetime[transfer_datetime.GUPI.isin(missing_ICU_disch_timestamps.GUPI)&(transfer_datetime.TransFrom == 'ICU')].dropna(subset=['DateEffectiveTransfer'])[['GUPI','DateEffectiveTransfer']].reset_index(drop=True)

for curr_GUPI in tqdm(transfer_datetime.GUPI,'Imputing missing ICU discharge timestamps with transfer out of ICU information'):
    curr_transfer_date = transfer_datetime.DateEffectiveTransfer[transfer_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_transfer_date
    CENTER_TBI_datetime.ICUDischDate[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_transfer_date

CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan C: For non-missing ICU discharge dates, replace time with latest time available among non-repeatable timestamps
missing_ICU_disch_timestamps = CENTER_TBI_datetime[(CENTER_TBI_datetime.ICUDischTime.isna())&(~CENTER_TBI_datetime.ICUDischDate.isna())].reset_index(drop=True)

viable_non_repeatables = non_repeatable_datetime_info.merge(missing_ICU_disch_timestamps.rename(columns={'ICUDischDate':'date'}),how='inner',on=['GUPI','date']).dropna(subset=['time']).reset_index(drop=True)
viable_non_repeatables = viable_non_repeatables.groupby(['GUPI','date'],as_index=False).time.aggregate(max)

for curr_GUPI in tqdm(viable_non_repeatables.GUPI,'Imputing missing ICU discharge times with last available timestamp on the day'):
    
    curr_disch_time = viable_non_repeatables.time[viable_non_repeatables.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
    CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_disch_time

CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan D: For non-missing ICU discharge dates, replace time with latest time available among repeatable timestamps
missing_ICU_disch_timestamps = CENTER_TBI_datetime[(CENTER_TBI_datetime.ICUDischTime.isna())&(~CENTER_TBI_datetime.ICUDischDate.isna())].reset_index(drop=True)

viable_repeatables = repeatable_datetime_info.merge(missing_ICU_disch_timestamps.rename(columns={'ICUDischDate':'date'}),how='inner',on=['GUPI','date']).dropna(subset=['time']).reset_index(drop=True)
viable_repeatables = viable_repeatables.groupby(['GUPI','date'],as_index=False).time.aggregate(max)

for curr_GUPI in tqdm(viable_repeatables.GUPI,'Imputing missing ICU discharge times with last available timestamp on the day'):
    curr_disch_time = viable_repeatables.time[viable_repeatables.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
    CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_disch_time

CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI.isin(missing_ICU_disch_timestamps.GUPI[missing_ICU_disch_timestamps.ICUDischTime.isna()])] = '23:59:00'
missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.ICUDischTime.isna()] = '23:59:00'

CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan E: Replace with the maximum timestamp within 30 days of admission
missing_ICU_disch_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

viable_non_repeatables = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI))&(~non_repeatable_datetime_info.date.isna())]
viable_non_repeatables = viable_non_repeatables[viable_non_repeatables.date <= '1970-01-30'].sort_values(by=['GUPI','date','time'],ascending=False).reset_index(drop=True)
viable_non_repeatables = viable_non_repeatables.groupby('GUPI',as_index=False).first()

viable_repeatables = repeatable_datetime_info[(repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI))&(~repeatable_datetime_info.date.isna())]
viable_repeatables = viable_repeatables[viable_repeatables.date <= '1970-01-30'].sort_values(by=['GUPI','date','time'],ascending=False).reset_index(drop=True)
viable_repeatables = viable_repeatables.groupby('GUPI',as_index=False).first()

viable_timepoints = pd.concat([viable_non_repeatables,viable_repeatables.drop(columns='RowID')],ignore_index=True)
viable_timepoints.time = viable_timepoints.time.str.replace('"','')

for curr_GUPI in tqdm(viable_timepoints.GUPI,'Imputing missing ICU discharge times with last available timestamp on the day'):
    
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischTime.isna().values[0]:
        
        curr_disch_time = viable_timepoints.time[viable_timepoints.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
        CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_disch_time
        
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischDate.isna().values[0]:
        
        curr_disch_date = viable_timepoints.date[viable_timepoints.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_date
        CENTER_TBI_datetime.ICUDischDate[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_disch_date
                                    
CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan F: Replace with hospital discharge timestamp if available
missing_ICU_disch_timestamps = CENTER_TBI_datetime[CENTER_TBI_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

hosp_disch_timestamps = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv')
hosp_disch_timestamps = hosp_disch_timestamps[['GUPI','ICUDischDate','ICUDischTime','HospDischTime','HospDischDate']]
hosp_disch_timestamps = hosp_disch_timestamps[hosp_disch_timestamps.GUPI.isin(missing_ICU_disch_timestamps.GUPI)].reset_index(drop=True)

for curr_GUPI in tqdm(hosp_disch_timestamps.GUPI,'Imputing missing ICU discharge times with hospital discharge'):
    
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischTime.isna().values[0]:
        
        curr_disch_time = hosp_disch_timestamps.HospDischTime[hosp_disch_timestamps.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
        CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_disch_time
        
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischDate.isna().values[0]:
        
        curr_disch_date = hosp_disch_timestamps.HospDischDate[hosp_disch_timestamps.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_date
        CENTER_TBI_datetime.ICUDischDate[CENTER_TBI_datetime.GUPI == curr_GUPI] = curr_disch_date

CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Manual corrections based on plausible types
CENTER_TBI_datetime.ICUAdmDate[CENTER_TBI_datetime.GUPI == '6URh589'] = '1970-01-01'
CENTER_TBI_datetime.ICUAdmTime[CENTER_TBI_datetime.GUPI == '8uBs474'] = '05:19:00'
CENTER_TBI_datetime.ICUAdmTime[CENTER_TBI_datetime.GUPI == '8MVh882'] = '06:41:00'
CENTER_TBI_datetime.ICUDischTime[CENTER_TBI_datetime.GUPI == '8MVh882'] = '08:48:00'

CENTER_TBI_datetime['ICUAdmTimeStamp'] = CENTER_TBI_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUAdmTimeStamp'][CENTER_TBI_datetime.ICUAdmDate.isna() | CENTER_TBI_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_datetime['ICUDischTimeStamp'] = CENTER_TBI_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_datetime['ICUDischTimeStamp'][CENTER_TBI_datetime.ICUDischDate.isna() | CENTER_TBI_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_datetime['ICUDurationHours'] = (CENTER_TBI_datetime['ICUDischTimeStamp'] - CENTER_TBI_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Save timestamps as CSV
CENTER_TBI_datetime.to_csv('../timestamps/adm_disch_timestamps.csv',index = False)

## Fix missing TILDates based on patients with similar admission time
# Load fixed CENTER-TBI timestamps
CENTER_TBI_datetime = pd.read_csv('../timestamps/adm_disch_timestamps.csv')
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['WardAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['WardAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['HospDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['HospDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Add `PatientType` information to the DailyTIL information dataframe
mod_daily_TIL_info = CENTER_TBI_datetime[['GUPI','PatientType','ICUAdmTimeStamp','ICUDischTimeStamp','WardAdmTimeStamp','HospDischTimeStamp']].merge(daily_TIL_info,how='left')

# For problem_GUPIs, for whom the TILDates are still missing, find a patient with the closest ICU admission time, and employ their date difference
non_problem_set = CENTER_TBI_datetime[(~CENTER_TBI_datetime.GUPI.isin(problem_GUPIs))&(CENTER_TBI_datetime.GUPI.isin(mod_daily_TIL_info.GUPI))].reset_index(drop=True)
for curr_GUPI in tqdm(problem_GUPIs, 'Fixing problem TIL dates'):
    
    # Extract current patient primary admission type
    curr_PatientType = CENTER_TBI_datetime.PatientType[CENTER_TBI_datetime.GUPI == curr_GUPI].values[0]

    # Extract current admission timestamp based on primary admission type
    if curr_PatientType == 3:
        curr_TimeStamp = CENTER_TBI_datetime.ICUAdmTimeStamp[CENTER_TBI_datetime.GUPI == curr_GUPI].values[0]
    else:
        curr_TimeStamp = CENTER_TBI_datetime.WardAdmTimeStamp[CENTER_TBI_datetime.GUPI == curr_GUPI].values[0]
    
    # Find a non-problem-GUPI patient with the closest ICU admission time
    closest_GUPI = non_problem_set.GUPI[(non_problem_set.ICUAdmTimeStamp - curr_TimeStamp).dt.total_seconds().abs().argmin()]
    
    # Calculate date difference on closest GUPI
    closest_GUPI_daily_TIL = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==closest_GUPI)&(mod_daily_TIL_info.TILTimepoint!='None')].reset_index(drop=True)
    curr_date_diff = int((closest_GUPI_daily_TIL.TILDate.dt.day - closest_GUPI_daily_TIL.TILTimepoint.astype(float)).mode()[0])
    
    # Calulcate fixed date vector for current problem GUPI
    curr_GUPI_daily_TIL = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)&(mod_daily_TIL_info.TILTimepoint!='None')].reset_index(drop=True)
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_TIL.TILTimepoint.astype(float)-1)],index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)&(mod_daily_TIL_info.TILTimepoint!='None')].index)

    # Fix problem GUPI dates in the original dataframe
    mod_daily_TIL_info.TILDate[(mod_daily_TIL_info.GUPI==curr_GUPI)&(mod_daily_TIL_info.TILTimepoint!='None')] = fixed_date_vector    

# Merge median TIL time to daily TIL dataframe
mod_daily_TIL_info = mod_daily_TIL_info.merge(median_TILTime,how='left',on='GUPI')

# If daily TIL assessment time is missing, first impute with patient-specific median time
mod_daily_TIL_info.TILTime[mod_daily_TIL_info.TILTime.isna()&~mod_daily_TIL_info.medianTILTime.isna()] = mod_daily_TIL_info.medianTILTime[mod_daily_TIL_info.TILTime.isna()&~mod_daily_TIL_info.medianTILTime.isna()]

# If daily TIL assessment time is still missing, then impute with overall-set median time
mod_daily_TIL_info.TILTime[mod_daily_TIL_info.TILTime.isna()] = overall_median_TILTime

# Construct daily TIL assessment timestamp
mod_daily_TIL_info['TimeStamp'] = mod_daily_TIL_info[['TILDate', 'TILTime']].astype(str).agg(' '.join, axis=1)
mod_daily_TIL_info['TimeStamp'] = pd.to_datetime(mod_daily_TIL_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# # If daily TIL Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
# mod_daily_TIL_info.TimeStamp[((mod_daily_TIL_info.TimeStamp > mod_daily_TIL_info.ICUDischTimeStamp)&(mod_daily_TIL_info.TILDate.dt.date == mod_daily_TIL_info.ICUDischTimeStamp.dt.date))|((mod_daily_TIL_info.TimeStamp > mod_daily_TIL_info.HospDischTimeStamp)&(mod_daily_TIL_info.TILDate.dt.date == mod_daily_TIL_info.HospDischTimeStamp.dt.date))] = mod_daily_TIL_info.ICUDischTimeStamp[((mod_daily_TIL_info.TimeStamp > mod_daily_TIL_info.ICUDischTimeStamp)&(mod_daily_TIL_info.TILDate.dt.date == mod_daily_TIL_info.ICUDischTimeStamp.dt.date))|((mod_daily_TIL_info.TimeStamp > mod_daily_TIL_info.HospDischTimeStamp)&(mod_daily_TIL_info.TILDate.dt.date == mod_daily_TIL_info.HospDischTimeStamp.dt.date))]

# Fix volume and dose variables if incorrectly casted as character types
fix_TIL_columns = [col for col, dt in mod_daily_TIL_info.dtypes.items() if (col.endswith('Dose')|('Volume' in col))&(dt == object)]
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].replace(to_replace='^\D*$', value=np.nan, regex=True)
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace(',','.',regex=False))
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('[^0-9\\.]','',regex=True))
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('\\.\\.','.',regex=True))
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(pd.to_numeric)

# Remove unnecessary variables from dataframe
mod_daily_TIL_info = mod_daily_TIL_info.drop(columns=['medianTILTime'])

# Rearrange columns
first_cols = ['GUPI','PatientType','TimeStamp','TotalTIL','ICUAdmTimeStamp','ICUDischTimeStamp','WardAdmTimeStamp','HospDischTimeStamp','TILTimepoint','TILDate','TILTime']
other_cols = mod_daily_TIL_info.columns.difference(first_cols).to_list()
mod_daily_TIL_info = mod_daily_TIL_info[first_cols+other_cols]

## Ensure surgery markers carry over onto subsequent TIL assessments for each patient
# First, sort modified daily TIL dataframe
mod_daily_TIL_info = mod_daily_TIL_info.sort_values(by=['GUPI','TimeStamp'],ignore_index=True)

# Identify GUPIs which contain markers for intracranial operation and decompressive craniectomy
gupis_with_ICPSurgery = mod_daily_TIL_info[mod_daily_TIL_info.TILICPSurgery==1].GUPI.unique()
gupis_with_DecomCranectomy = mod_daily_TIL_info[mod_daily_TIL_info.TILICPSurgeryDecomCranectomy==1].GUPI.unique()

# Iterate through GUPIs with intracranial operations and correct surgery indicators and the total TIL score
for curr_GUPI in tqdm(gupis_with_ICPSurgery, 'Fixing TotalTIL and intracranial operation indicators'):

    # Extract TIL assessments of current patient
    curr_GUPI_daily_TIL = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].reset_index(drop=True)

    # Extract total TIL scores of current patient
    curr_TotalTIL = curr_GUPI_daily_TIL.TotalTIL

    # Extract ICP surgery indicators of current patient
    curr_TILICPSurgery = curr_GUPI_daily_TIL.TILICPSurgery

    # Identify first TIL instance of surgery
    firstSurgInstance = curr_TILICPSurgery.index[curr_TILICPSurgery==1].tolist()[0]

    # Fix the ICP surgery indicators of current patient
    fix_TILICPSurgery = curr_TILICPSurgery.copy()
    if firstSurgInstance != (len(fix_TILICPSurgery)-1):
        fix_TILICPSurgery[range(firstSurgInstance+1,len(fix_TILICPSurgery))] = 1
    fix_TILICPSurgery.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Fix the total TIL score of current patient
    fix_TotalTIL = curr_TotalTIL.copy()
    if firstSurgInstance != (len(fix_TotalTIL)-1):
        fix_TotalTIL[(fix_TILICPSurgery.reset_index(drop=True) - curr_TILICPSurgery).astype('bool')] = fix_TotalTIL[(fix_TILICPSurgery.reset_index(drop=True) - curr_TILICPSurgery).astype('bool')]+4
    fix_TotalTIL.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Place fixed vectors into modified daily TIL dataframe
    mod_daily_TIL_info.TotalTIL[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TotalTIL    
    mod_daily_TIL_info.TILICPSurgery[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TILICPSurgery    

# Iterate through GUPIs with decompressive craniectomies and correct surgery indicators and the total TIL score
for curr_GUPI in tqdm(gupis_with_DecomCranectomy, 'Fixing TotalTIL and decompressive craniectomy indicators'):

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

## Save modified Daily TIL dataframes in new directory
os.makedirs('../formatted_data/',exist_ok=True)
mod_daily_TIL_info.to_csv('../formatted_data/formatted_TIL_scores.csv',index=False)

### III. Load and prepare low-resolution ICP and CPP information
## Fix timestamp inaccuracies in HourlyValues dataframe based on HourlyValueTimePoint
# Load modified Daily TIL dataframes
mod_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Load HourlyValues dataframe
daily_hourly_info = pd.read_csv('../CENTER-TBI/DailyHourlyValues/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients for whom TIL values exist
daily_hourly_info = daily_hourly_info[daily_hourly_info.GUPI.isin(mod_daily_TIL_info.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Remove all entries without date or `HourlyValueTimePoint`
daily_hourly_info = daily_hourly_info[(daily_hourly_info.HourlyValueTimePoint!='None')|(~daily_hourly_info.HVDate.isna())].reset_index(drop=True)

# Remove all rows in which ICP or CPP is missing
daily_hourly_info = daily_hourly_info[~(daily_hourly_info.HVICP.isna() & daily_hourly_info.HVCPP.isna())].reset_index(drop=True)

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

## Fix timestamp inaccuracies in HourlyValues dataframe based on HVHour
# If 'HVTime' is missing, replace with 'HVHour'
daily_hourly_info.HVTime[(daily_hourly_info.HVTime.isna())&(~daily_hourly_info.HVHour.isna())] = daily_hourly_info.HVHour[(daily_hourly_info.HVTime.isna())&(~daily_hourly_info.HVHour.isna())]

# Fix cases in which 'HVTime' equals '24:00:00'
daily_hourly_info.HVDate[daily_hourly_info.HVTime == '24:00:00'] = daily_hourly_info.HVDate[daily_hourly_info.HVTime == '24:00:00'] + timedelta(days=1)
daily_hourly_info.HVTime[daily_hourly_info.HVTime == '24:00:00'] = '00:00:00'

# Construct daily hourly assessment timestamp
daily_hourly_info['TimeStamp'] = daily_hourly_info[['HVDate', 'HVTime']].astype(str).agg(' '.join, axis=1)
daily_hourly_info['TimeStamp'] = pd.to_datetime(daily_hourly_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

## Construct final dataframe and save
# Load fixed CENTER-TBI timestamps
CENTER_TBI_datetime = pd.read_csv('../timestamps/adm_disch_timestamps.csv')
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['WardAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['WardAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['HospDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['HospDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Add `PatientType` information to the DailyHourlyValues information dataframe
mod_daily_hourly_info = CENTER_TBI_datetime[['GUPI','PatientType','ICUAdmTimeStamp','ICUDischTimeStamp','WardAdmTimeStamp','HospDischTimeStamp']].merge(daily_hourly_info,how='right')

# # If daily hourly value date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
# mod_daily_hourly_info.TimeStamp[((mod_daily_hourly_info.TimeStamp > mod_daily_hourly_info.ICUDischTimeStamp)&(mod_daily_hourly_info.HVDate.dt.date == mod_daily_hourly_info.ICUDischTimeStamp.dt.date))|((mod_daily_hourly_info.TimeStamp > mod_daily_hourly_info.HospDischTimeStamp)&(mod_daily_hourly_info.HVDate.dt.date == mod_daily_hourly_info.HospDischTimeStamp.dt.date))] = mod_daily_hourly_info.ICUDischTimeStamp[((mod_daily_hourly_info.TimeStamp > mod_daily_hourly_info.ICUDischTimeStamp)&(mod_daily_hourly_info.HVDate.dt.date == mod_daily_hourly_info.ICUDischTimeStamp.dt.date))|((mod_daily_hourly_info.TimeStamp > mod_daily_hourly_info.HospDischTimeStamp)&(mod_daily_hourly_info.HVDate.dt.date == mod_daily_hourly_info.HospDischTimeStamp.dt.date))]

# Rearrange columns
first_cols = ['GUPI','PatientType','TimeStamp','HVICP','ICUAdmTimeStamp','ICUDischTimeStamp','WardAdmTimeStamp','HospDischTimeStamp','HourlyValueTimePoint','HVDate','HVTime']
other_cols = mod_daily_hourly_info.columns.difference(first_cols).to_list()
mod_daily_hourly_info = mod_daily_hourly_info[first_cols+other_cols]

# Save modified Daily hourly value dataframes in new directory
os.makedirs('../formatted_data/',exist_ok=True)
mod_daily_hourly_info.to_csv('../formatted_data/formatted_daily_hourly_values.csv',index=False)

### IV. Load and prepare six-month functional outcome scores
## Load function outcome scores of patients in TIL dataframe
# Load modified Daily TIL dataframes
mod_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Load CENTER-TBI dataset demographic information
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.GUPI.isin(mod_daily_TIL_info.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Select columns that indicate pertinent baseline and outcome information
CENTER_TBI_demo_outcome = CENTER_TBI_demo_info[['GUPI','PatientType','Age','Sex','Race','GCSScoreBaselineDerived','GOSE6monthEndpointDerived']].reset_index(drop=True)

# Load and filter CENTER-TBI IMPACT dataframe
IMPACT_df = pd.read_csv('../CENTER-TBI/IMPACT/data.csv').rename(columns={'entity_id':'GUPI'})
IMPACT_df = IMPACT_df[IMPACT_df.GUPI.isin(CENTER_TBI_demo_outcome.GUPI)][['GUPI','SiteCode','marshall']].rename(columns={'marshall':'MarshallCT'}).reset_index(drop=True)

# Merge IMPACT-sourced information to outcome and demographic dataframe
CENTER_TBI_demo_outcome = CENTER_TBI_demo_outcome.merge(IMPACT_df,how='left')

# Load and prepare ordinal prediction estimates
ordinal_prediction_estimates = pd.read_csv('../../ordinal_GOSE_prediction/APM_outputs/DEEP_v1-0/APM_deepMN_compiled_test_predictions.csv').drop(columns='Unnamed: 0')
ordinal_prediction_estimates = ordinal_prediction_estimates[(ordinal_prediction_estimates.GUPI.isin(CENTER_TBI_demo_outcome.GUPI))&(ordinal_prediction_estimates.TUNE_IDX==8)].reset_index(drop=True)
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
CENTER_TBI_demo_outcome = CENTER_TBI_demo_outcome.merge(ordinal_prediction_estimates,how='left')

## Save baseline demographic and functional outcome score dataframe
# Create directory, if it does not exist, to store formatted dataframes
os.makedirs('../formatted_data/',exist_ok=True)

# Save dataframe
CENTER_TBI_demo_outcome.to_csv('../formatted_data/formatted_outcome_and_demographics.csv',index=False)

### V. Prepare dataframe of high-resolution sub-study patients
## Identify patients for whom TIL and high-resolution ICP information are available
# Load modified Daily TIL dataframes
mod_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Load dataframe of patients with high-resolution data
hi_res_patients = pd.read_excel('../high_res_patients.xlsx',na_values = ["NA","NaN"," ", ""])

# Filter modified daily TIL information to high-resolution sub-study
hi_res_daily_TIL_info = mod_daily_TIL_info[mod_daily_TIL_info.GUPI.isin(hi_res_patients.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Select pertinent columns of dataframe
hi_res_daily_TIL_info = hi_res_daily_TIL_info[['GUPI','PatientType','TotalTIL','TimeStamp']]

## Check available hi-res files and determine if EVD
# Identify directory of current hi-res data
hi_res_dir = '../CENTER-TBI/HighResolution/10sec_windows/'

# Create list of high resolution CSVs
curr_list_of_csvs = []
for path in Path(hi_res_dir).rglob('*.csv'):
    curr_list_of_csvs.append(str(path.resolve()))

# Characterise filed by GUPI and EVD-type
curr_csv_info_df = pd.DataFrame({'FILE':curr_list_of_csvs,'GUPI':[re.search('10sec_windows/(.*).csv', curr_file).group(1) for curr_file in curr_list_of_csvs]}).sort_values(by=['GUPI']).reset_index(drop=True)
curr_csv_info_df['EVD'] = curr_csv_info_df.GUPI.str.contains('EVD')
curr_csv_info_df.GUPI = curr_csv_info_df.GUPI.str.replace('EVD/','')

# Add EVD indicator to `hi_res_daily_TIL_info` dataframe
hi_res_daily_TIL_info = hi_res_daily_TIL_info.merge(curr_csv_info_df[['GUPI','EVD']],how='left')

## Prepare dataframe with empty columns representing target values
# Create a dataframe of all combinations of target columns
timespans = ['samedate','24hbefore','12hbefore12hafter','24hafter']
waveform = ['ICP','CPP']
metrics = ['n','mean','std','min','q1','median','q3','max']
target_columns = pd.DataFrame(list(itertools.product(timespans,waveform,metrics)), columns=['timespans','waveform','metrics'])
target_columns['label'] = target_columns.metrics + '_' + target_columns.waveform + '_' + target_columns.timespans

# Add empty columns of target values to dataframe
target_hi_res_daily_TIL_info = hi_res_daily_TIL_info.reindex(columns=hi_res_daily_TIL_info.columns.tolist()+target_columns.label.tolist())

## Save requisite dataframes
# High-resolution patient TIL timestamps
hi_res_daily_TIL_info.to_csv('../CENTER-TBI/HighResolution/high_res_TIL_timestamps.csv',index=False)

# List of target measures to calculate from high-resolution data
target_columns.to_csv('../CENTER-TBI/HighResolution/high_resolution_metrics.csv',index=False)

# High-resolution patient TIL timestamps with empty target columns
target_hi_res_daily_TIL_info.to_csv('../CENTER-TBI/HighResolution/sample_result_dataframe.csv',index=False)