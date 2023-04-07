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
# IV. Load and prepare high-resolution ICP and CPP information
# V. Load and prepare demographic information and baseline characteristics
# VI. Load and prepare information from prior study data
# VII. Load and prepare serum sodium values from CENTER-TBI
# VIII. Calculate TIL_1987 values

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

## Fix rows with 'None' timepoints
# Replace 'None' timepoints with NaN
mod_daily_TIL_info.TILTimepoint[mod_daily_TIL_info.TILTimepoint=='None'] = np.nan

# Determine GUPIs with 'None' timepoints
none_GUPIs = mod_daily_TIL_info[mod_daily_TIL_info.TILTimepoint.isna()].GUPI.unique()

# Iterate through 'None' GUPIs and impute missing timepoint values
for curr_GUPI in none_GUPIs:
    curr_GUPI_TIL_scores = mod_daily_TIL_info[mod_daily_TIL_info.GUPI==curr_GUPI].reset_index(drop=True)
    non_missing_timepoint_mask = ~curr_GUPI_TIL_scores.TILTimepoint.isna()
    if non_missing_timepoint_mask.sum() != 1:
        curr_default_date = (curr_GUPI_TIL_scores.TimeStamp.dt.date[non_missing_timepoint_mask] - pd.to_timedelta(curr_GUPI_TIL_scores.TILTimepoint.astype(float)[non_missing_timepoint_mask],unit='d')).mode()[0]
    else:
        curr_default_date = (curr_GUPI_TIL_scores.TimeStamp.dt.date[non_missing_timepoint_mask] - timedelta(days=curr_GUPI_TIL_scores.TILTimepoint.astype(float)[non_missing_timepoint_mask].values[0])).mode()[0]
    fixed_timepoints_vector = ((curr_GUPI_TIL_scores.TimeStamp.dt.date - curr_default_date)/np.timedelta64(1,'D')).astype(int).astype(str)
    fixed_timepoints_vector.index=mod_daily_TIL_info[mod_daily_TIL_info.GUPI==curr_GUPI].index
    mod_daily_TIL_info.TILTimepoint[mod_daily_TIL_info.GUPI==curr_GUPI] = fixed_timepoints_vector

# Convert TILTimepoint variable from string to integer
mod_daily_TIL_info.TILTimepoint = mod_daily_TIL_info.TILTimepoint.astype(int)

## Fix instances in which a patient has more than one TIL score per day
# Count number of TIL scores available per patient-Timepoint combination
patient_TIL_counts = mod_daily_TIL_info.groupby(['GUPI','TILTimepoint'],as_index=False).TotalTIL.count()

# Isolate patients with instances of more than 1 daily TIL per day
more_than_one_GUPIs = patient_TIL_counts[patient_TIL_counts.TotalTIL>1].GUPI.unique()

# Filter dataframe of more-than-one-instance patients to visually examine
more_than_one_TIL_scores = mod_daily_TIL_info[mod_daily_TIL_info.GUPI.isin(more_than_one_GUPIs)].reset_index(drop=True)

# Select the rows which correspond to the greatest TIL score per ICU stay day per patient
keep_idx = mod_daily_TIL_info.groupby(['GUPI','TILTimepoint'])['TotalTIL'].transform(max) == mod_daily_TIL_info['TotalTIL']

# Filter to keep selected rows only
mod_daily_TIL_info = mod_daily_TIL_info[keep_idx].reset_index(drop=True)

## Add a DateComponent column to dataframe
mod_daily_TIL_info['DateComponent'] = mod_daily_TIL_info.TimeStamp.dt.date

## Save modified Daily TIL dataframes in new directory
os.makedirs('../formatted_data/',exist_ok=True)
mod_daily_TIL_info.to_csv('../formatted_data/formatted_TIL_scores.csv',index=False)

### III. Load and prepare low-resolution ICP and CPP information
## Fix timestamp inaccuracies in HourlyValues dataframe based on HourlyValueTimePoint
# Load modified Daily TIL dataframes
mod_daily_TIL_info = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')

# Convert dates from string to date format
mod_daily_TIL_info.TimeStamp = pd.to_datetime(mod_daily_TIL_info.TimeStamp,format = '%Y-%m-%d %H:%M:%S')
mod_daily_TIL_info.TILDate = pd.to_datetime(mod_daily_TIL_info.TILDate,format = '%Y-%m-%d')
mod_daily_TIL_info.TILDate = mod_daily_TIL_info.TILDate.dt.date
mod_daily_TIL_info.DateComponent = pd.to_datetime(mod_daily_TIL_info.DateComponent,format = '%Y-%m-%d')
mod_daily_TIL_info.DateComponent = mod_daily_TIL_info.DateComponent.dt.date

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

## Merge TIL scores onto corresponding low-resolution ICP/CPP scores based on 'DateComponent'
# Add a column to designate day component in low-resolution dataframe
mod_daily_hourly_info['DateComponent'] = mod_daily_hourly_info.TimeStamp.dt.date

# Merge
lo_res_info = mod_daily_hourly_info.merge(mod_daily_TIL_info[['GUPI','DateComponent','TILTimepoint','TotalTIL']],how='left',on=['GUPI','DateComponent'])

# Filter out rows without TIL scores on the day and select relevant columns
lo_res_info = lo_res_info[~lo_res_info.TotalTIL.isna()][['GUPI','TimeStamp','DateComponent','HVICP','HVCPP','TILTimepoint','TotalTIL']]

# Melt out ICP and CPP values to long-form
lo_res_info = lo_res_info.melt(id_vars=['GUPI','TimeStamp','DateComponent','TILTimepoint','TotalTIL'])

# Remove missing CPP values
lo_res_info = lo_res_info[~lo_res_info.value.isna()].reset_index(drop=True)

## Filter out rows during/after WLST
# Find CENTER-TBI patients who experienced WLST
CENTER_TBI_WLST_patients = pd.read_csv('../CENTER-TBI/WLST_patients.csv',na_values = ["NA","NaN"," ", ""])

# Filter WLST patients in current set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients[CENTER_TBI_WLST_patients.GUPI.isin(lo_res_info.GUPI)].reset_index(drop=True)

# Find CENTER-TBI patients who died in ICU
CENTER_TBI_death_patients = pd.read_csv('../CENTER-TBI/death_patients.csv',na_values = ["NA","NaN"," ", ""])

# Add ICU death information to WLST set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.merge(CENTER_TBI_death_patients[['GUPI','ICUDischargeStatus']],how='left')

# Add ICU discharge information to WLST set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.merge(CENTER_TBI_datetime[['GUPI','ICUDischTimeStamp']],how='left')

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
lo_res_info = lo_res_info.merge(CENTER_TBI_WLST_patients.rename(columns={'value':'WLSTDateComponent'})[['GUPI','WLSTDateComponent','WLST']],how='left')

# Fill in missing dummy-WLST markers
lo_res_info.WLST = lo_res_info.WLST.fillna(0)

# Filter out columns in which DateComponent occurs during or after WLST decision
lo_res_info = lo_res_info[(lo_res_info.DateComponent<lo_res_info.WLSTDateComponent)|(lo_res_info.WLST==0)].reset_index(drop=True)

## Save modified Daily hourly value dataframes in new directory
os.makedirs('../formatted_data/',exist_ok=True)
lo_res_info.to_csv('../formatted_data/formatted_low_resolution_values.csv',index=False)

### IV. Load and prepare high-resolution ICP and CPP information
## Load and format high-resolution ICP/CPP summary values
# Load high-resolution ICP/CPP summary values of same day as TIL assessments
hi_res_info = pd.read_csv('../CENTER-TBI/HighResolution/til_same.csv',na_values = ["NA","NaN"," ", ""])

# Filter columns of interest
hi_res_info = hi_res_info[['GUPI','TimeStamp','TotalTIL','ICP_mean','CPP_mean']]

# Convert TimeStamp to proper format
hi_res_info['TimeStamp'] = pd.to_datetime(hi_res_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Melt out ICP and CPP values to long-form
hi_res_info = hi_res_info.melt(id_vars=['GUPI','TimeStamp','TotalTIL'])

# Remove missing ICP or CPP values
hi_res_info = hi_res_info[~hi_res_info.value.isna()].reset_index(drop=True)

## Add EVD indicator
# Load EVD indicator
evd_indicator = pd.read_csv('../CENTER-TBI/HighResolution/list_EVD.csv',na_values = ["NA","NaN"," ", ""])

# Add column for EVD indicator
hi_res_info['EVD'] = 0

# Fix EVD indicator
hi_res_info.EVD[hi_res_info.GUPI.isin(evd_indicator.GUPI)] = 1

## Filter out rows during/after WLST
# Find CENTER-TBI patients who experienced WLST
CENTER_TBI_WLST_patients = pd.read_csv('../CENTER-TBI/WLST_patients.csv',na_values = ["NA","NaN"," ", ""])

# Filter WLST patients in current set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients[CENTER_TBI_WLST_patients.GUPI.isin(hi_res_info.GUPI)].reset_index(drop=True)

# Find CENTER-TBI patients who died in ICU
CENTER_TBI_death_patients = pd.read_csv('../CENTER-TBI/death_patients.csv',na_values = ["NA","NaN"," ", ""])

# Add ICU death information to WLST set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.merge(CENTER_TBI_death_patients[['GUPI','ICUDischargeStatus']],how='left')

# Add ICU discharge information to WLST set
CENTER_TBI_WLST_patients = CENTER_TBI_WLST_patients.merge(CENTER_TBI_datetime[['GUPI','ICUDischTimeStamp']],how='left')

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
hi_res_info = hi_res_info.merge(CENTER_TBI_WLST_patients.rename(columns={'value':'WLSTDateComponent'})[['GUPI','WLSTDateComponent','WLST']],how='left')

# Fill in missing dummy-WLST markers
hi_res_info.WLST = hi_res_info.WLST.fillna(0)

# Filter out columns in which DateComponent occurs during or after WLST decision
hi_res_info = hi_res_info[(hi_res_info.TimeStamp.dt.date<hi_res_info.WLSTDateComponent)|(hi_res_info.WLST==0)].reset_index(drop=True)

## Save modified hi-resolution value dataframes in new directory
os.makedirs('../formatted_data/',exist_ok=True)
hi_res_info.to_csv('../formatted_data/formatted_high_resolution_values.csv',index=False)

### V. Load and prepare demographic information and baseline characteristics
## Load demographic and outcome scores of patients in TIL dataframe
# Load low-resolution value dataframe
lo_res_info = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Load high-resolution value dataframe
hi_res_info = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')

# Load CENTER-TBI dataset demographic information
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Select columns that indicate pertinent baseline and outcome information
CENTER_TBI_demo_info = CENTER_TBI_demo_info[['GUPI','PatientType','SiteCode','Age','Sex','Race','GCSScoreBaselineDerived','GOSE6monthEndpointDerived']].reset_index(drop=True)

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

# Filter study set patients
CENTER_TBI_demo_info = CENTER_TBI_demo_info[(CENTER_TBI_demo_info.LowResolutionSet==1)|(CENTER_TBI_demo_info.HighResolutionSet==1)].dropna(axis=1,how='all').reset_index(drop=True)

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

## Save baseline demographic and functional outcome score dataframe
# Create directory, if it does not exist, to store formatted dataframes
os.makedirs('../formatted_data/',exist_ok=True)

# Save dataframe
CENTER_TBI_demo_info.to_csv('../formatted_data/formatted_outcome_and_demographics.csv',index=False)

### VI. Load and prepare information from prior study data
## Load prior study dataframes
# Main database extraction
main_database_extraction = pd.read_excel('../prior_study_data/SPSS_Main_Database.xlsx',na_values = ["NA","NaN"," ", "","#NULL!"]).dropna(axis=1,how='all').dropna(subset=['Pt'],how='all').reset_index(drop=True)

# Create dataframe to visually inspect columns
main_database_columns = pd.DataFrame(main_database_extraction.columns,columns=['ColumnName'])

# ICP vs TIL dataframe
ICP_TIL_in_TBI_ICU_df = pd.read_excel('../prior_study_data/ICP_vs_TIL_in_TBI_ICU.xlsx',na_values = ["NA","NaN"," ", "","#NULL!"]).dropna(axis=1,how='all').dropna(subset=['CamGro_ID'],how='all').rename(columns={'CamGro_ID':'Pt'}).reset_index(drop=True)

# Full TIL dataframe
prior_TIL_df = pd.read_csv('../prior_study_data/TIL.csv',na_values = ["NA","NaN"," ", "","#NULL!"]).dropna(axis=1,how='all').dropna(subset=['CamGro_ID'],how='all').rename(columns={'CamGro_ID':'Pt'}).reset_index(drop=True)

## Extract demographic information
# First, identify TBI patients in ICU cohort
TBI_ICU_patients = np.sort(ICP_TIL_in_TBI_ICU_df.Pt.unique())

# Filter TBI-ICU patients from main database extraction
filt_main_database_extraction = main_database_extraction[main_database_extraction.Pt.isin(TBI_ICU_patients)].dropna(axis=1,how='all').reset_index(drop=True)

# Select demographic columns and rename columns to match CENTER-TBI format
filt_main_database_extraction = filt_main_database_extraction[['Pt','HospID','Age','Gender','GCS','GOS','Marshall_OLD']].drop_duplicates(ignore_index=True).rename(columns={'Pt':'GUPI','HospID':'SiteCode','Gender':'Sex','GCS':'GCSScoreBaselineDerived','GOS':'GOS6monthEndpointDerived','Marshall_OLD':'MarshallCT'})

# Save extracted demographic information from prior study
filt_main_database_extraction.to_csv('../formatted_data/prior_study_formatted_outcome_and_demographics.csv',index=False)

### VII. Load and prepare serum sodium values from CENTER-TBI
## Load and prepare sodium values
# Load sodium lab values
sodium_values = pd.read_csv('../CENTER-TBI/Labs/data.csv',na_values = ["NA","NaN"," ", ""])[['GUPI','DLDate','DLTime','DLSodiummmolL']].dropna(subset=['DLDate','DLSodiummmolL'],how='any').sort_values(by=['GUPI','DLDate']).reset_index(drop=True)

# Convert `DLDate` to timestamp format
sodium_values['DateComponent'] = pd.to_datetime(sodium_values['DLDate'],format = '%Y-%m-%d')

# Calculate daily mean sodium
sodium_values = sodium_values.groupby(['GUPI','DateComponent'],as_index=False).DLSodiummmolL.aggregate({'meanSodium':'mean','nSodium':'count'})

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

# Keep only columns from formatted TIL dataframe
formatted_TIL_scores = formatted_TIL_scores[['GUPI','LowResolutionSet','HighResolutionSet','DateComponent','TILTimepoint','TILHyperosmolarThearpy','TILMannitolDose','TILHyperosomolarTherapyMannitolGreater2g','TILHypertonicSalineDose','TILHyperosomolarTherapyHypertonicLow','TILHyperosomolarTherapyHigher','TotalTIL']]

# Add column marking mannitol use
formatted_TIL_scores['MannitolUse'] = (formatted_TIL_scores.TILHyperosmolarThearpy>0)|(formatted_TIL_scores.TILMannitolDose>0)|(formatted_TIL_scores.TILHyperosomolarTherapyMannitolGreater2g>0)

# Add column marking saline use
formatted_TIL_scores['SalineUse'] = (formatted_TIL_scores.TILHypertonicSalineDose>0)|(formatted_TIL_scores.TILHyperosomolarTherapyHypertonicLow>0)|(formatted_TIL_scores.TILHyperosomolarTherapyHigher>0)

# Remove rows in which mannitol use is unaccompanied by saline use
formatted_TIL_scores = formatted_TIL_scores[~((formatted_TIL_scores.MannitolUse)&(~formatted_TIL_scores.SalineUse))].reset_index(drop=True)

# Remove unneccessary columns
formatted_TIL_scores = formatted_TIL_scores[['GUPI','LowResolutionSet','HighResolutionSet','DateComponent','TILTimepoint','MannitolUse','SalineUse','TotalTIL']]

# Merge sodium values to formatted TIL dataframe
sodium_TIL_dataframe = formatted_TIL_scores.merge(sodium_values,how='left')

# Remove rows with missing sodium values
sodium_TIL_dataframe = sodium_TIL_dataframe.dropna(subset='meanSodium').drop_duplicates(ignore_index=True)

## Save prepared sodium values
sodium_TIL_dataframe.to_csv('../formatted_data/formatted_daily_sodium_values.csv',index=False)

### VIII. Calculate TIL_1987 values
## Load and prepare formatted TIL_1987 values from CENTER-TBI
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

# Create new dataframe to store TIL_1987 values for CENTER-TBI
TIL_1987_scores = formatted_TIL_scores[['GUPI','DateComponent','LowResolutionSet','HighResolutionSet','TimeStamp','TILTimepoint','TotalTIL']]

# Add marker for barbiturate administration
TIL_1987_scores['TIL_Barbiturate'] = 3*(formatted_TIL_scores.TILSedationMetabolic == 1).astype(int)

# Add marker for mannitol administration
TIL_1987_scores['TIL_Mannitol'] = 0
TIL_1987_scores['TIL_Mannitol'][formatted_TIL_scores.TILHyperosmolarThearpy == 1] = 3
TIL_1987_scores['TIL_Mannitol'][formatted_TIL_scores.TILHyperosomolarTherapyMannitolGreater2g == 1] = 6

# Add marker for ventricular drainage
TIL_1987_scores['TIL_Ventricular'] = 0
TIL_1987_scores['TIL_Ventricular'][formatted_TIL_scores.TILCSFDrainage == 1] = 1
TIL_1987_scores['TIL_Ventricular'][(formatted_TIL_scores.TILCCSFDrainageVolume>=120)|(formatted_TIL_scores.TILFluidOutCSFDrain>=120)] = 2

# Add marker for hyperventilation
TIL_1987_scores['TIL_Hyperventilation'] = 0
TIL_1987_scores['TIL_Hyperventilation'][(formatted_TIL_scores.TILHyperventilation == 1)|(formatted_TIL_scores.TILHyperventilationModerate == 1)] = 1
TIL_1987_scores['TIL_Hyperventilation'][formatted_TIL_scores.TILHyperventilationIntensive == 1] = 2

# Add marker for paralysis induction
TIL_1987_scores['TIL_Paralysis'] = 1*(formatted_TIL_scores.TILSedationNeuromuscular == 1).astype(int)

# Add marker for sedation
TIL_1987_scores['TIL_Sedation'] = 1*((formatted_TIL_scores.TILSedation == 1)|(formatted_TIL_scores.TILSedationHigher == 1)).astype(int)

# Calculate TIL_1987
TIL_1987_scores['TIL_1987'] = TIL_1987_scores.TIL_Barbiturate + TIL_1987_scores.TIL_Mannitol + TIL_1987_scores.TIL_Ventricular + TIL_1987_scores.TIL_Hyperventilation + TIL_1987_scores.TIL_Paralysis + TIL_1987_scores.TIL_Sedation

# Save calculated TIL_1987 values
TIL_1987_scores.to_csv('../formatted_data/formatted_TIL_1987_scores.csv',index=False)

## Load and prepare formatted TIL_1987 values from prior study
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

# Sort dataframe
prior_study_database_extraction = prior_study_database_extraction.sort_values(by=['GUPI','Period']).reset_index(drop=True)

# Create new dataframe to store TIL_1987 values for prior study
prior_study_TIL_1987_scores = prior_study_database_extraction[['GUPI','Start','End','Period','TIL_sum']]

# Add marker for barbiturate administration
prior_study_TIL_1987_scores['TIL_Barbiturate'] = 3*(prior_study_database_extraction.Barbiturates == 1).astype(int)

# Add marker for mannitol administration
prior_study_TIL_1987_scores['TIL_Mannitol'] = 0
prior_study_TIL_1987_scores['TIL_Mannitol'][(prior_study_database_extraction.Mannitol_perKG/4 <= 1)] = 3
prior_study_TIL_1987_scores['TIL_Mannitol'][(prior_study_database_extraction.Mannitol_perKG/4 > 1)] = 6

# Add marker for ventricular drainage
prior_study_TIL_1987_scores['TIL_Ventricular'] = 0
prior_study_TIL_1987_scores['TIL_Ventricular'][prior_study_database_extraction.TIL_CSF == 2] = 1
prior_study_TIL_1987_scores['TIL_Ventricular'][prior_study_database_extraction.TIL_CSF == 3] = 2

# Add marker for hyperventilation
prior_study_TIL_1987_scores['TIL_Hyperventilation'] = 0
prior_study_TIL_1987_scores['TIL_Hyperventilation'][(prior_study_database_extraction.CO2_TIL == 1)|(prior_study_database_extraction.CO2_TIL == 2)] = 1
prior_study_TIL_1987_scores['TIL_Hyperventilation'][prior_study_database_extraction.CO2_TIL == 4] = 2

# Add marker for paralysis induction
prior_study_TIL_1987_scores['TIL_Paralysis'] = 1*(prior_study_database_extraction.Paralysis == 1).astype(int)

# Add marker for sedation
prior_study_TIL_1987_scores['TIL_Sedation'] = 1*((prior_study_database_extraction.TIL_SED == 1)|(prior_study_database_extraction.TIL_SED == 2)).astype(int)

# Add column representing date component of end timestamp
prior_study_TIL_1987_scores['DateComponent'] = prior_study_TIL_1987_scores['End'].dt.date

# Calculate TIL24
prior_study_dailyTILs = prior_study_TIL_1987_scores.groupby(['GUPI','DateComponent'],as_index=False).TIL_sum.max().rename(columns={'TIL_sum':'TotalTIL'})

# Calculate TIL_1987_24
prior_study_dailyTIL_1987s = prior_study_TIL_1987_scores[['GUPI','DateComponent','TIL_Barbiturate','TIL_Mannitol', 'TIL_Ventricular', 'TIL_Hyperventilation','TIL_Paralysis', 'TIL_Sedation']].melt(id_vars=['GUPI','DateComponent']).groupby(['GUPI','DateComponent','variable'],as_index=False)['value'].max().groupby(['GUPI','DateComponent'],as_index=False)['value'].sum().rename(columns={'value':'TIL_1987'})

# Merge dataframes and save
prior_study_TIL_1987_scores = prior_study_dailyTILs.merge(prior_study_dailyTIL_1987s,how='left')

# Remove unneccessary columns
prior_study_TIL_1987_scores.to_csv('../formatted_data/prior_study_formatted_TIL_1987_scores.csv',index=False)