#### Master Script 2a: Prepare study resamples for bootstrapping TIL-based statistics ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and prepare correlation dataframes
# III. Draw resamples for bootstrapping

# III. Calculate population-level summary characteristics
# IV. Calculate population-level TIL correlations with overall characteristics
# V. Calculate TIL correlations with ICP measures

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
import multiprocessing
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.utils import resample

# Set number of resamples for bootstrapping-based inference
NUM_RESAMP = 1000

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Initialise directory for storing bootstrapping resamples
bs_dir = '../bootstrapping_results/resamples'
os.makedirs(bs_dir,exist_ok=True)

### II. Load and prepare correlation dataframes
## TIL_mean vs. ICP_mean
# Load and filter TIL_means
TIL_means = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_means = TIL_means[TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load manually recorded ICP values
formatted_lo_res_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Calculate low-resolution daily ICP means
lo_res_ICP_24 = formatted_lo_res_values[formatted_lo_res_values.variable=='HVICP'].groupby(['GUPI','DateComponent','TILTimepoint','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'ICPmean'}).drop(columns='variable')

# Calculate ICP_means
lo_res_ICP_means = lo_res_ICP_24.groupby(['GUPI'],as_index=False).ICPmean.mean()

# Merge TIL_mean information to low-resolution ICP_means
lo_res_ICP_TIL_means = lo_res_ICP_means.merge(TIL_means[TIL_means.Group=='LowResolutionSet'])

# Load high-resolution ICP values and add a `DateComponent`
formatted_hi_res_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')
formatted_hi_res_values['DateComponent'] = pd.to_datetime(formatted_hi_res_values.TimeStamp,format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate high-resolution daily ICP means
hi_res_ICP_24 = formatted_hi_res_values[formatted_hi_res_values.variable=='ICP_mean'].groupby(['GUPI','DateComponent','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'ICPmean'}).drop(columns='variable')

# Calculate ICP_means
hi_res_ICP_means = hi_res_ICP_24.groupby(['GUPI'],as_index=False).ICPmean.mean()

# Merge TIL_mean information to high-resolution ICP_means
hi_res_ICP_TIL_means = hi_res_ICP_means.merge(TIL_means[TIL_means.Group=='HighResolutionSet'])

# Load and filter prior study TIL_means
prior_study_TIL_means = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_means = prior_study_TIL_means[prior_study_TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load prior study high-resolution ICP values and add a `DateComponent`
prior_study_formatted_hi_res_values = pd.read_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv')
prior_study_formatted_hi_res_values['DateComponent'] = pd.to_datetime(prior_study_formatted_hi_res_values['End'].str[:19],format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate prior study high-resolution daily ICP means
prior_study_ICP_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['MeanICP'].mean().rename(columns={'MeanICP':'ICPmean'})

# Calculate ICP_means
prior_study_ICP_means = prior_study_ICP_24.groupby(['GUPI'],as_index=False).ICPmean.mean()

# Merge TIL_mean information to prior study high-resolution ICP_means
prior_study_ICP_TIL_means = prior_study_ICP_means.merge(prior_study_TIL_means)

## TIL_max vs. ICP_max
# Load and filter TIL_maxes
TIL_maxes = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_maxes = TIL_maxes[TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load manually recorded ICP values
formatted_lo_res_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Calculate low-resolution daily ICP maxes
lo_res_ICP_24 = formatted_lo_res_values[formatted_lo_res_values.variable=='HVICP'].groupby(['GUPI','DateComponent','TILTimepoint','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'ICPmax'}).drop(columns='variable')

# Calculate ICP_maxes
lo_res_ICP_maxes = lo_res_ICP_24.groupby(['GUPI'],as_index=False).ICPmax.max()

# Merge TIL_max information to low-resolution ICP_maxes
lo_res_ICP_TIL_maxes = lo_res_ICP_maxes.merge(TIL_maxes[TIL_maxes.Group=='LowResolutionSet'])

# Load high-resolution ICP values and add a `DateComponent`
formatted_hi_res_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')
formatted_hi_res_values['DateComponent'] = pd.to_datetime(formatted_hi_res_values.TimeStamp,format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate high-resolution daily ICP maxes
hi_res_ICP_24 = formatted_hi_res_values[formatted_hi_res_values.variable=='ICP_mean'].groupby(['GUPI','DateComponent','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'ICPmax'}).drop(columns='variable')

# Calculate ICP_maxes
hi_res_ICP_maxes = hi_res_ICP_24.groupby(['GUPI'],as_index=False).ICPmax.max()

# Merge TIL_max information to high-resolution ICP_maxes
hi_res_ICP_TIL_maxes = hi_res_ICP_maxes.merge(TIL_maxes[TIL_maxes.Group=='HighResolutionSet'])

# Load and filter prior study TIL_maxes
prior_study_TIL_maxes = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_maxes = prior_study_TIL_maxes[prior_study_TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load prior study high-resolution ICP values and add a `DateComponent`
prior_study_formatted_hi_res_values = pd.read_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv')
prior_study_formatted_hi_res_values['DateComponent'] = pd.to_datetime(prior_study_formatted_hi_res_values['End'].str[:19],format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate prior study high-resolution daily ICP maxes
prior_study_ICP_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['MeanICP'].mean().rename(columns={'MeanICP':'ICPmax'})

# Calculate ICP_maxes
prior_study_ICP_maxes = prior_study_ICP_24.groupby(['GUPI'],as_index=False).ICPmax.max()

# Merge TIL_max information to prior study high-resolution ICP_maxes
prior_study_ICP_TIL_maxes = prior_study_ICP_maxes.merge(prior_study_TIL_maxes)

## TIL_mean vs. CPP_mean
# Load and filter TIL_means
TIL_means = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_means = TIL_means[TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load manually recorded CPP values
formatted_lo_res_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Calculate low-resolution daily CPP means
lo_res_CPP_24 = formatted_lo_res_values[formatted_lo_res_values.variable=='HVCPP'].groupby(['GUPI','DateComponent','TILTimepoint','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'CPPmean'}).drop(columns='variable')

# Calculate CPP_means
lo_res_CPP_means = lo_res_CPP_24.groupby(['GUPI'],as_index=False).CPPmean.mean()

# Merge TIL_mean information to low-resolution CPP_means
lo_res_CPP_TIL_means = lo_res_CPP_means.merge(TIL_means[TIL_means.Group=='LowResolutionSet'])

# Load high-resolution CPP values and add a `DateComponent`
formatted_hi_res_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')
formatted_hi_res_values['DateComponent'] = pd.to_datetime(formatted_hi_res_values.TimeStamp,format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate high-resolution daily CPP means
hi_res_CPP_24 = formatted_hi_res_values[formatted_hi_res_values.variable=='CPP_mean'].groupby(['GUPI','DateComponent','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'CPPmean'}).drop(columns='variable')

# Calculate CPP_means
hi_res_CPP_means = hi_res_CPP_24.groupby(['GUPI'],as_index=False).CPPmean.mean()

# Merge TIL_mean information to high-resolution CPP_means
hi_res_CPP_TIL_means = hi_res_CPP_means.merge(TIL_means[TIL_means.Group=='HighResolutionSet'])

# Load and filter prior study TIL_means
prior_study_TIL_means = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_means = prior_study_TIL_means[prior_study_TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load prior study high-resolution CPP values and add a `DateComponent`
prior_study_formatted_hi_res_values = pd.read_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv')
prior_study_formatted_hi_res_values['DateComponent'] = pd.to_datetime(prior_study_formatted_hi_res_values['End'].str[:19],format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate prior study high-resolution daily CPP means
prior_study_CPP_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['MeanCPP'].mean().rename(columns={'MeanCPP':'CPPmean'})

# Calculate CPP_means
prior_study_CPP_means = prior_study_CPP_24.groupby(['GUPI'],as_index=False).CPPmean.mean()

# Merge TIL_mean information to prior study high-resolution CPP_means
prior_study_CPP_TIL_means = prior_study_CPP_means.merge(prior_study_TIL_means)

## TIL_max vs. CPP_max
# Load and filter TIL_maxes
TIL_maxes = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_maxes = TIL_maxes[TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load manually recorded CPP values
formatted_lo_res_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Calculate low-resolution daily CPP maxes
lo_res_CPP_24 = formatted_lo_res_values[formatted_lo_res_values.variable=='HVCPP'].groupby(['GUPI','DateComponent','TILTimepoint','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'CPPmax'}).drop(columns='variable')

# Calculate CPP_maxes
lo_res_CPP_maxes = lo_res_CPP_24.groupby(['GUPI'],as_index=False).CPPmax.max()

# Merge TIL_max information to low-resolution CPP_maxes
lo_res_CPP_TIL_maxes = lo_res_CPP_maxes.merge(TIL_maxes[TIL_maxes.Group=='LowResolutionSet'])

# Load high-resolution CPP values and add a `DateComponent`
formatted_hi_res_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')
formatted_hi_res_values['DateComponent'] = pd.to_datetime(formatted_hi_res_values.TimeStamp,format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate high-resolution daily CPP maxes
hi_res_CPP_24 = formatted_hi_res_values[formatted_hi_res_values.variable=='CPP_mean'].groupby(['GUPI','DateComponent','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'CPPmax'}).drop(columns='variable')

# Calculate CPP_maxes
hi_res_CPP_maxes = hi_res_CPP_24.groupby(['GUPI'],as_index=False).CPPmax.max()

# Merge TIL_max information to high-resolution CPP_maxes
hi_res_CPP_TIL_maxes = hi_res_CPP_maxes.merge(TIL_maxes[TIL_maxes.Group=='HighResolutionSet'])

# Load and filter prior study TIL_maxes
prior_study_TIL_maxes = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_maxes = prior_study_TIL_maxes[prior_study_TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load prior study high-resolution CPP values and add a `DateComponent`
prior_study_formatted_hi_res_values = pd.read_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv')
prior_study_formatted_hi_res_values['DateComponent'] = pd.to_datetime(prior_study_formatted_hi_res_values['End'].str[:19],format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate prior study high-resolution daily CPP maxes
prior_study_CPP_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['MeanCPP'].mean().rename(columns={'MeanCPP':'CPPmax'})

# Calculate CPP_maxes
prior_study_CPP_maxes = prior_study_CPP_24.groupby(['GUPI'],as_index=False).CPPmax.max()

# Merge TIL_max information to prior study high-resolution CPP_maxes
prior_study_CPP_TIL_maxes = prior_study_CPP_maxes.merge(prior_study_TIL_maxes)

## TIL_mean vs. sodium_mean
# Load and filter TIL_means
TIL_means = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_means = TIL_means[TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load daily sodium values
formatted_sodium_values = pd.read_csv('../formatted_data/formatted_daily_sodium_values.csv')

# Calculate sodium_means
sodium_means = formatted_sodium_values.groupby(['GUPI','LowResolutionSet','HighResolutionSet'],as_index=False).meanSodium.mean()

# Merge TIL_mean information to sodium_means
sodium_TIL_means = sodium_means.merge(TIL_means[['GUPI','TILmean']].drop_duplicates(ignore_index=True))

## TIL_max vs. sodium_max
# Load and filter TIL_maxes
TIL_maxes = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_maxes = TIL_maxes[TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load daily sodium values
formatted_sodium_values = pd.read_csv('../formatted_data/formatted_daily_sodium_values.csv')

# Calculate sodium_maxes
sodium_maxes = formatted_sodium_values.groupby(['GUPI','LowResolutionSet','HighResolutionSet'],as_index=False).meanSodium.max().rename(columns={'meanSodium':'maxSodium'})

# Merge TIL_max information to sodium_maxes
sodium_TIL_maxes = sodium_maxes.merge(TIL_maxes[['GUPI','TILmax']].drop_duplicates(ignore_index=True))

## TIL_mean vs. GCS
# Load and filter TIL_means
TIL_means = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_means = TIL_means[TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Prepare low-resolution GCS and TILmean dataframe
lo_res_GCS = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.LowResolutionSet==1)&(~CENTER_TBI_demo_outcome.GCSScoreBaselineDerived.isna())].reset_index(drop=True)[['GUPI','GCSScoreBaselineDerived']]

# Merge TIL means to low-resolution population GCS
lo_res_GCS_TIL_means = lo_res_GCS.merge(TIL_means[TIL_means.Group=='LowResolutionSet'])

# Prepare high-resolution GCS and TILmean dataframe
hi_res_GCS = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.HighResolutionSet==1)&(~CENTER_TBI_demo_outcome.GCSScoreBaselineDerived.isna())].reset_index(drop=True)[['GUPI','GCSScoreBaselineDerived']]

# Merge TIL means to high-resolution population GCS
hi_res_GCS_TIL_means = hi_res_GCS.merge(TIL_means[TIL_means.Group=='HighResolutionSet'])

# Load and filter prior study TIL_means
prior_study_TIL_means = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_means = prior_study_TIL_means[prior_study_TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load prior study demographic and functional outcome score dataframe
prior_study_demo_outcome = pd.read_csv('../formatted_data/prior_study_formatted_outcome_and_demographics.csv')

# Prepare prior study GCS and TILmean dataframe
prior_study_GCS = prior_study_demo_outcome[(~prior_study_demo_outcome.GCSScoreBaselineDerived.isna())].reset_index(drop=True)[['GUPI','GCSScoreBaselineDerived']]

# Merge TIL means to prior study population GCS
prior_study_GCS_TIL_means = prior_study_GCS.merge(prior_study_TIL_means)

## TIL_max vs. GCS
# Load and filter TIL_maxes
TIL_maxes = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_maxes = TIL_maxes[TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Prepare low-resolution GCS and TILmax dataframe
lo_res_GCS = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.LowResolutionSet==1)&(~CENTER_TBI_demo_outcome.GCSScoreBaselineDerived.isna())].reset_index(drop=True)[['GUPI','GCSScoreBaselineDerived']]

# Merge TIL maxes to low-resolution population GCS
lo_res_GCS_TIL_maxes = lo_res_GCS.merge(TIL_maxes[TIL_maxes.Group=='LowResolutionSet'])

# Prepare high-resolution GCS and TILmax dataframe
hi_res_GCS = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.HighResolutionSet==1)&(~CENTER_TBI_demo_outcome.GCSScoreBaselineDerived.isna())].reset_index(drop=True)[['GUPI','GCSScoreBaselineDerived']]

# Merge TIL maxes to high-resolution population GCS
hi_res_GCS_TIL_maxes = hi_res_GCS.merge(TIL_maxes[TIL_maxes.Group=='HighResolutionSet'])

# Load and filter prior study TIL_maxes
prior_study_TIL_maxes = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_maxes = prior_study_TIL_maxes[prior_study_TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load prior study demographic and functional outcome score dataframe
prior_study_demo_outcome = pd.read_csv('../formatted_data/prior_study_formatted_outcome_and_demographics.csv')

# Prepare prior study GCS and TILmax dataframe
prior_study_GCS = prior_study_demo_outcome[(~prior_study_demo_outcome.GCSScoreBaselineDerived.isna())].reset_index(drop=True)[['GUPI','GCSScoreBaselineDerived']]

# Merge TIL maxes to prior study population GCS
prior_study_GCS_TIL_maxes = prior_study_GCS.merge(prior_study_TIL_maxes)

## TIL_mean vs. GOSE
# Load and filter TIL_means
TIL_means = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_means = TIL_means[TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Prepare low-resolution GOSE and TILmean dataframe
lo_res_GOSE = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.LowResolutionSet==1)&(~CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived.isna())].reset_index(drop=True)[['GUPI','GOSE6monthEndpointDerived']]

# Merge TIL means to low-resolution population GOSE
lo_res_GOSE_TIL_means = lo_res_GOSE.merge(TIL_means[TIL_means.Group=='LowResolutionSet'])

# Prepare high-resolution GOSE and TILmean dataframe
hi_res_GOSE = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.HighResolutionSet==1)&(~CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived.isna())].reset_index(drop=True)[['GUPI','GOSE6monthEndpointDerived']]

# Merge TIL means to high-resolution population GOSE
hi_res_GOSE_TIL_means = hi_res_GOSE.merge(TIL_means[TIL_means.Group=='HighResolutionSet'])

## TIL_max vs. GOSE
# Load and filter TIL_maxes
TIL_maxes = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_maxes = TIL_maxes[TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Prepare low-resolution GOSE and TILmax dataframe
lo_res_GOSE = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.LowResolutionSet==1)&(~CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived.isna())].reset_index(drop=True)[['GUPI','GOSE6monthEndpointDerived']]

# Merge TIL maxes to low-resolution population GOSE
lo_res_GOSE_TIL_maxes = lo_res_GOSE.merge(TIL_maxes[TIL_maxes.Group=='LowResolutionSet'])

# Prepare high-resolution GOSE and TILmax dataframe
hi_res_GOSE = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.HighResolutionSet==1)&(~CENTER_TBI_demo_outcome.GOSE6monthEndpointDerived.isna())].reset_index(drop=True)[['GUPI','GOSE6monthEndpointDerived']]

# Merge TIL maxes to high-resolution population GOSE
hi_res_GOSE_TIL_maxes = hi_res_GOSE.merge(TIL_maxes[TIL_maxes.Group=='HighResolutionSet'])

## TIL_mean vs. GOS
# Convert GOS to GOS in low-resolution population
lo_res_GOSE_TIL_means['GOS6monthEndpointDerived'] = np.nan
lo_res_GOSE_TIL_means.GOS6monthEndpointDerived[lo_res_GOSE_TIL_means.GOSE6monthEndpointDerived=='1'] = '1'
lo_res_GOSE_TIL_means.GOS6monthEndpointDerived[lo_res_GOSE_TIL_means.GOSE6monthEndpointDerived.isin(['2_or_3','4'])] = '2_or_3'
lo_res_GOSE_TIL_means.GOS6monthEndpointDerived[lo_res_GOSE_TIL_means.GOSE6monthEndpointDerived.isin(['5','6'])] = '4'
lo_res_GOSE_TIL_means.GOS6monthEndpointDerived[lo_res_GOSE_TIL_means.GOSE6monthEndpointDerived.isin(['7','8'])] = '5'

# Convert GOSE to GOS in high-resolution population
hi_res_GOSE_TIL_means['GOS6monthEndpointDerived'] = np.nan
hi_res_GOSE_TIL_means.GOS6monthEndpointDerived[hi_res_GOSE_TIL_means.GOSE6monthEndpointDerived=='1'] = '1'
hi_res_GOSE_TIL_means.GOS6monthEndpointDerived[hi_res_GOSE_TIL_means.GOSE6monthEndpointDerived.isin(['2_or_3','4'])] = '2_or_3'
hi_res_GOSE_TIL_means.GOS6monthEndpointDerived[hi_res_GOSE_TIL_means.GOSE6monthEndpointDerived.isin(['5','6'])] = '4'
hi_res_GOSE_TIL_means.GOS6monthEndpointDerived[hi_res_GOSE_TIL_means.GOSE6monthEndpointDerived.isin(['7','8'])] = '5'

# Load and filter prior study TIL_means
prior_study_TIL_means = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_means = prior_study_TIL_means[prior_study_TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load prior study demographic and functional outcome score dataframe
prior_study_demo_outcome = pd.read_csv('../formatted_data/prior_study_formatted_outcome_and_demographics.csv')

# Prepare prior study GOS and TILmean dataframe
prior_study_GOS = prior_study_demo_outcome[(~prior_study_demo_outcome.GOS6monthEndpointDerived.isna())].reset_index(drop=True)[['GUPI','GOS6monthEndpointDerived']]
prior_study_GOS.GOS6monthEndpointDerived = prior_study_GOS.GOS6monthEndpointDerived.astype(int).astype(str)
prior_study_GOS.GOS6monthEndpointDerived[prior_study_GOS.GOS6monthEndpointDerived.isin(['2','3'])] = '2_or_3'

# Merge TIL means to prior study population GOS
prior_study_GOS_TIL_means = prior_study_GOS.merge(prior_study_TIL_means)

## TIL_max vs. GOS
# Convert GOS to GOS in low-resolution population
lo_res_GOSE_TIL_maxes['GOS6monthEndpointDerived'] = np.nan
lo_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[lo_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived=='1'] = '1'
lo_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[lo_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived.isin(['2_or_3','4'])] = '2_or_3'
lo_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[lo_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived.isin(['5','6'])] = '4'
lo_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[lo_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived.isin(['7','8'])] = '5'

# Convert GOSE to GOS in high-resolution population
hi_res_GOSE_TIL_maxes['GOS6monthEndpointDerived'] = np.nan
hi_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[hi_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived=='1'] = '1'
hi_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[hi_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived.isin(['2_or_3','4'])] = '2_or_3'
hi_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[hi_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived.isin(['5','6'])] = '4'
hi_res_GOSE_TIL_maxes.GOS6monthEndpointDerived[hi_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived.isin(['7','8'])] = '5'

# Load and filter prior study TIL_maxes
prior_study_TIL_maxes = pd.read_csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv')
prior_study_TIL_maxes = prior_study_TIL_maxes[prior_study_TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load prior study demographic and functional outcome score dataframe
prior_study_demo_outcome = pd.read_csv('../formatted_data/prior_study_formatted_outcome_and_demographics.csv')

# Prepare prior study GOS and TILmax dataframe
prior_study_GOS = prior_study_demo_outcome[(~prior_study_demo_outcome.GOS6monthEndpointDerived.isna())].reset_index(drop=True)[['GUPI','GOS6monthEndpointDerived']]
prior_study_GOS.GOS6monthEndpointDerived = prior_study_GOS.GOS6monthEndpointDerived.astype(int).astype(str)
prior_study_GOS.GOS6monthEndpointDerived[prior_study_GOS.GOS6monthEndpointDerived.isin(['2','3'])] = '2_or_3'

# Merge TIL maxes to prior study population GOS
prior_study_GOS_TIL_maxes = prior_study_GOS.merge(prior_study_TIL_maxes)

## TIL_mean vs. prognostic scores
# Load and filter TIL_means
TIL_means = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_means = TIL_means[TIL_means.TILmetric=='TILmean'].reset_index(drop=True).rename(columns={'value':'TILmean'}).drop(columns='TILmetric')

# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Extract names of ordinal prognosis columns
prog_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('Pr(GOSE>')]

# Prepare low-resolution prognostic scores and TILmean dataframe
lo_res_prognosis = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1].dropna(subset=prog_cols).reset_index(drop=True)[['GUPI']+prog_cols]

# Merge TIL means to low-resolution population prognosis
lo_res_prognosis_TIL_means = lo_res_prognosis.merge(TIL_means[TIL_means.Group=='LowResolutionSet'])

# Prepare high-resolution prognosis and TILmean dataframe
hi_res_prognosis = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1].dropna(subset=prog_cols).reset_index(drop=True)[['GUPI']+prog_cols]

# Merge TIL means to high-resolution population prognosis
hi_res_prognosis_TIL_means = hi_res_prognosis.merge(TIL_means[TIL_means.Group=='HighResolutionSet'])

## TIL_max vs. prognostic scores
# Load and filter TIL_maxes
TIL_maxes = pd.read_csv('../formatted_data/formatted_TIL_means_maxes.csv')
TIL_maxes = TIL_maxes[TIL_maxes.TILmetric=='TILmax'].reset_index(drop=True).rename(columns={'value':'TILmax'}).drop(columns='TILmetric')

# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Extract names of ordinal prognosis columns
prog_cols = [col for col in CENTER_TBI_demo_outcome if col.startswith('Pr(GOSE>')]

# Prepare low-resolution prognostic scores and TILmax dataframe
lo_res_prognosis = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1].dropna(subset=prog_cols).reset_index(drop=True)[['GUPI']+prog_cols]

# Merge TIL maxes to low-resolution population prognosis
lo_res_prognosis_TIL_maxes = lo_res_prognosis.merge(TIL_maxes[TIL_maxes.Group=='LowResolutionSet'])

# Prepare high-resolution prognosis and TILmax dataframe
hi_res_prognosis = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1].dropna(subset=prog_cols).reset_index(drop=True)[['GUPI']+prog_cols]

# Merge TIL maxes to high-resolution population prognosis
hi_res_prognosis_TIL_maxes = hi_res_prognosis.merge(TIL_maxes[TIL_maxes.Group=='HighResolutionSet'])

## TIL24 vs. ICP24
# Load manually recorded ICP values
formatted_lo_res_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Calculate low-resolution daily ICP means
lo_res_ICP_TIL_24 = formatted_lo_res_values[formatted_lo_res_values.variable=='HVICP'].groupby(['GUPI','DateComponent','TILTimepoint','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'ICPmean'}).drop(columns='variable')

# Load high-resolution ICP values and add a `DateComponent`
formatted_hi_res_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')
formatted_hi_res_values['DateComponent'] = pd.to_datetime(formatted_hi_res_values.TimeStamp,format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate high-resolution daily ICP means
hi_res_ICP_TIL_24 = formatted_hi_res_values[formatted_hi_res_values.variable=='ICP_mean'].groupby(['GUPI','DateComponent','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'ICPmean'}).drop(columns='variable')

# Load prior study high-resolution ICP values and add a `DateComponent`
prior_study_formatted_hi_res_values = pd.read_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv')
prior_study_formatted_hi_res_values['DateComponent'] = pd.to_datetime(prior_study_formatted_hi_res_values['End'].str[:19],format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate prior study high-resolution daily ICP means
prior_study_ICP_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['MeanICP'].mean().rename(columns={'MeanICP':'ICPmean'})

# Calculate prior study high-resolution daily TIL
prior_study_TIL_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['TIL_sum'].max().rename(columns={'TIL_sum':'TotalTIL'})

# Merge prior study dataframes
prior_study_ICP_TIL_24 = prior_study_ICP_24.merge(prior_study_TIL_24)

## TIL24 vs. CPP24
# Load manually recorded CPP values
formatted_lo_res_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')

# Calculate low-resolution daily CPP means
lo_res_CPP_TIL_24 = formatted_lo_res_values[formatted_lo_res_values.variable=='HVCPP'].groupby(['GUPI','DateComponent','TILTimepoint','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'CPPmean'}).drop(columns='variable')

# Load high-resolution CPP values and add a `DateComponent`
formatted_hi_res_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')
formatted_hi_res_values['DateComponent'] = pd.to_datetime(formatted_hi_res_values.TimeStamp,format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate high-resolution daily CPP means
hi_res_CPP_TIL_24 = formatted_hi_res_values[formatted_hi_res_values.variable=='CPP_mean'].groupby(['GUPI','DateComponent','TotalTIL','variable'],as_index=False)['value'].mean().rename(columns={'value':'CPPmean'}).drop(columns='variable')

# Load prior study high-resolution CPP values and add a `DateComponent`
prior_study_formatted_hi_res_values = pd.read_csv('../formatted_data/prior_study_formatted_high_resolution_values.csv')
prior_study_formatted_hi_res_values['DateComponent'] = pd.to_datetime(prior_study_formatted_hi_res_values['End'].str[:19],format = '%Y-%m-%d %H:%M:%S').dt.date

# Calculate prior study high-resolution daily CPP means
prior_study_CPP_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['MeanCPP'].mean().rename(columns={'MeanCPP':'CPPmean'})

# Calculate prior study high-resolution daily TIL
prior_study_TIL_24 = prior_study_formatted_hi_res_values.groupby(['GUPI','DateComponent'],as_index=False)['TIL_sum'].max().rename(columns={'TIL_sum':'TotalTIL'})

# Merge prior study dataframes
prior_study_CPP_TIL_24 = prior_study_CPP_24.merge(prior_study_TIL_24)

## TIL24 vs. Sodium24
# Load daily sodium values
formatted_sodium_values = pd.read_csv('../formatted_data/formatted_daily_sodium_values.csv')

# Isolate low-resolution TIL24 vs. Sodium24
lo_res_Sodium_TIL_24 = formatted_sodium_values[formatted_sodium_values.LowResolutionSet==1][['GUPI','DateComponent','TotalTIL','meanSodium']]

# Isolate high-resolution TIL24 vs. Sodium24
hi_res_Sodium_TIL_24 = formatted_sodium_values[formatted_sodium_values.HighResolutionSet==1][['GUPI','DateComponent','TotalTIL','meanSodium']]

### III. Draw resamples for bootstrapping
## Group 1: Manually-recording neuromonitoring population
# Create sub-directory for group 1
group1_dir = os.path.join(bs_dir,'group1')
os.makedirs(group1_dir,exist_ok=True)

# Extract manually-recording neuromonitoring population GUPIs
group1_GUPIs = lo_res_ICP_TIL_means.GUPI.unique()

# Make resamples for bootstrapping metrics
group1_bs_rs_GUPIs = [resample(group1_GUPIs,replace=True,n_samples=len(group1_GUPIs)) for _ in range(NUM_RESAMP)]
group1_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group1_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group1_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group1_bs_rs_GUPIs})

# Store group 1 resamples
group1_bs_resamples.to_pickle(os.path.join(group1_dir,'group1_resamples.pkl'))

# Save formatted group 1 correlation dataframes
lo_res_ICP_TIL_means.to_pickle(os.path.join(group1_dir,'lo_res_ICP_mean_TIL_mean.pkl'))
lo_res_ICP_TIL_maxes.to_pickle(os.path.join(group1_dir,'lo_res_ICP_max_TIL_max.pkl'))
lo_res_CPP_TIL_means.to_pickle(os.path.join(group1_dir,'lo_res_CPP_mean_TIL_mean.pkl'))
lo_res_CPP_TIL_maxes.to_pickle(os.path.join(group1_dir,'lo_res_CPP_max_TIL_max.pkl'))
lo_res_ICP_TIL_24.to_pickle(os.path.join(group1_dir,'lo_res_ICP_24_TIL_24.pkl'))
lo_res_CPP_TIL_24.to_pickle(os.path.join(group1_dir,'lo_res_CPP_24_TIL_24.pkl'))

## Group 2: High-resolution neuromonitoring population
# Create sub-directory for group 2
group2_dir = os.path.join(bs_dir,'group2')
os.makedirs(group2_dir,exist_ok=True)

# Extract high-resolution neuromonitoring population GUPIs
group2_GUPIs = hi_res_ICP_TIL_24.GUPI.unique()

# Make resamples for bootstrapping metrics
group2_bs_rs_GUPIs = [resample(group2_GUPIs,replace=True,n_samples=len(group2_GUPIs)) for _ in range(NUM_RESAMP)]
group2_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group2_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group2_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group2_bs_rs_GUPIs})

# Store group 2 resamples
group2_bs_resamples.to_pickle(os.path.join(group2_dir,'group2_resamples.pkl'))

# Save formatted group 2 correlation dataframes
hi_res_ICP_TIL_means.to_pickle(os.path.join(group2_dir,'hi_res_ICP_mean_TIL_mean.pkl'))
hi_res_ICP_TIL_maxes.to_pickle(os.path.join(group2_dir,'hi_res_ICP_max_TIL_max.pkl'))
hi_res_CPP_TIL_means.to_pickle(os.path.join(group2_dir,'hi_res_CPP_mean_TIL_mean.pkl'))
hi_res_CPP_TIL_maxes.to_pickle(os.path.join(group2_dir,'hi_res_CPP_max_TIL_max.pkl'))
hi_res_ICP_TIL_24.to_pickle(os.path.join(group2_dir,'hi_res_ICP_24_TIL_24.pkl'))
hi_res_CPP_TIL_24.to_pickle(os.path.join(group2_dir,'hi_res_CPP_24_TIL_24.pkl'))

## Group 3: Prior study population
# Create sub-directory for group 3
group3_dir = os.path.join(bs_dir,'group3')
os.makedirs(group3_dir,exist_ok=True)

# Extract prior study population GUPIs
group3_GUPIs = prior_study_ICP_TIL_means.GUPI.unique()

# Make resamples for bootstrapping metrics
group3_bs_rs_GUPIs = [resample(group3_GUPIs,replace=True,n_samples=len(group3_GUPIs)) for _ in range(NUM_RESAMP)]
group3_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group3_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group3_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group3_bs_rs_GUPIs})

# Store group 3 resamples
group3_bs_resamples.to_pickle(os.path.join(group3_dir,'group3_resamples.pkl'))

# Save formatted group 3 correlation dataframes
prior_study_ICP_TIL_means.to_pickle(os.path.join(group3_dir,'prior_study_ICP_mean_TIL_mean.pkl'))
prior_study_ICP_TIL_maxes.to_pickle(os.path.join(group3_dir,'prior_study_ICP_max_TIL_max.pkl'))
prior_study_CPP_TIL_means.to_pickle(os.path.join(group3_dir,'prior_study_CPP_mean_TIL_mean.pkl'))
prior_study_CPP_TIL_maxes.to_pickle(os.path.join(group3_dir,'prior_study_CPP_max_TIL_max.pkl'))
prior_study_GCS_TIL_means.to_pickle(os.path.join(group3_dir,'prior_study_GCS_TIL_mean.pkl'))
prior_study_GCS_TIL_maxes.to_pickle(os.path.join(group3_dir,'prior_study_GCS_TIL_max.pkl'))
prior_study_GOS_TIL_means.to_pickle(os.path.join(group3_dir,'prior_study_GOS_TIL_mean.pkl'))
prior_study_GOS_TIL_maxes.to_pickle(os.path.join(group3_dir,'prior_study_GOS_TIL_max.pkl'))
prior_study_ICP_TIL_24.to_pickle(os.path.join(group3_dir,'prior_study_ICP_24_TIL_24.pkl'))
prior_study_CPP_TIL_24.to_pickle(os.path.join(group3_dir,'prior_study_CPP_24_TIL_24.pkl'))

## Group 4: Manually recorded neuromonitoring + sodium population
# Create sub-directory for group 4
group4_dir = os.path.join(bs_dir,'group4')
os.makedirs(group4_dir,exist_ok=True)

# Extract manually recorded neuromonitoring + sodium population GUPIs
group4_GUPIs = lo_res_Sodium_TIL_24.GUPI.unique()

# Make resamples for bootstrapping metrics
group4_bs_rs_GUPIs = [resample(group4_GUPIs,replace=True,n_samples=len(group4_GUPIs)) for _ in range(NUM_RESAMP)]
group4_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group4_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group4_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group4_bs_rs_GUPIs})

# Store group 4 resamples
group4_bs_resamples.to_pickle(os.path.join(group4_dir,'group4_resamples.pkl'))

# Save formatted group 4 correlation dataframes
sodium_TIL_means[sodium_TIL_means.LowResolutionSet==1].reset_index(drop=True).to_pickle(os.path.join(group4_dir,'lo_res_sodium_mean_TIL_mean.pkl'))
sodium_TIL_maxes[sodium_TIL_maxes.LowResolutionSet==1].reset_index(drop=True).to_pickle(os.path.join(group4_dir,'lo_res_sodium_max_TIL_max.pkl'))
lo_res_Sodium_TIL_24.to_pickle(os.path.join(group4_dir,'lo_res_sodium_24_TIL_24.pkl'))

## Group 5: high-resolution neuromonitoring + sodium population
# Create sub-directory for group 5
group5_dir = os.path.join(bs_dir,'group5')
os.makedirs(group5_dir,exist_ok=True)

# Extract high-resolution neuromonitoring + sodium populationn GUPIs
group5_GUPIs = hi_res_Sodium_TIL_24.GUPI.unique()

# Make resamples for bootstrapping metrics
group5_bs_rs_GUPIs = [resample(group5_GUPIs,replace=True,n_samples=len(group5_GUPIs)) for _ in range(NUM_RESAMP)]
group5_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group5_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group5_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group5_bs_rs_GUPIs})

# Store group 5 resamples
group5_bs_resamples.to_pickle(os.path.join(group5_dir,'group5_resamples.pkl'))

# Save formatted group 5 correlation dataframes
sodium_TIL_means[sodium_TIL_means.HighResolutionSet==1].reset_index(drop=True).to_pickle(os.path.join(group5_dir,'hi_res_sodium_mean_TIL_mean.pkl'))
sodium_TIL_maxes[sodium_TIL_maxes.HighResolutionSet==1].reset_index(drop=True).to_pickle(os.path.join(group5_dir,'hi_res_sodium_max_TIL_max.pkl'))
hi_res_Sodium_TIL_24.to_pickle(os.path.join(group5_dir,'hi_res_sodium_24_TIL_24.pkl'))

## Group 6: manually-recorded neuromonitoring + GCS population
# Create sub-directory for group 6
group6_dir = os.path.join(bs_dir,'group6')
os.makedirs(group6_dir,exist_ok=True)

# Extract manually-recorded neuromonitoring + GCS populationn GUPIs
group6_GUPIs = lo_res_GCS_TIL_means.GUPI.unique()

# Make resamples for bootstrapping metrics
group6_bs_rs_GUPIs = [resample(group6_GUPIs,replace=True,n_samples=len(group6_GUPIs)) for _ in range(NUM_RESAMP)]
group6_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group6_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group6_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group6_bs_rs_GUPIs})

# Store group 6 resamples
group6_bs_resamples.to_pickle(os.path.join(group6_dir,'group6_resamples.pkl'))

# Save formatted group 6 correlation dataframes
lo_res_GCS_TIL_means.to_pickle(os.path.join(group6_dir,'lo_res_GCS_TIL_mean.pkl'))
lo_res_GCS_TIL_maxes.to_pickle(os.path.join(group6_dir,'lo_res_GCS_TIL_max.pkl'))

## Group 7: high-resolution neuromonitoring + GCS population
# Create sub-directory for group 7
group7_dir = os.path.join(bs_dir,'group7')
os.makedirs(group7_dir,exist_ok=True)

# Extract high-resolution neuromonitoring + GCS populationn GUPIs
group7_GUPIs = hi_res_GCS_TIL_means.GUPI.unique()

# Make resamples for bootstrapping metrics
group7_bs_rs_GUPIs = [resample(group7_GUPIs,replace=True,n_samples=len(group7_GUPIs)) for _ in range(NUM_RESAMP)]
group7_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group7_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group7_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group7_bs_rs_GUPIs})

# Store group 7 resamples
group7_bs_resamples.to_pickle(os.path.join(group7_dir,'group7_resamples.pkl'))

# Save formatted group 7 correlation dataframes
hi_res_GCS_TIL_means.to_pickle(os.path.join(group7_dir,'hi_res_GCS_TIL_mean.pkl'))
hi_res_GCS_TIL_maxes.to_pickle(os.path.join(group7_dir,'hi_res_GCS_TIL_max.pkl'))

## Group 8: manually-recorded neuromonitoring + GOSE population
# Create sub-directory for group 8
group8_dir = os.path.join(bs_dir,'group8')
os.makedirs(group8_dir,exist_ok=True)

# Extract manually-recorded neuromonitoring + GOSE populationn GUPIs
group8_GUPIs = lo_res_GOSE_TIL_means.GUPI.unique()

# Make resamples for bootstrapping metrics
group8_bs_rs_GUPIs = [resample(group8_GUPIs,replace=True,n_samples=len(group8_GUPIs)) for _ in range(NUM_RESAMP)]
group8_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group8_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group8_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group8_bs_rs_GUPIs})

# Store group 8 resamples
group8_bs_resamples.to_pickle(os.path.join(group8_dir,'group8_resamples.pkl'))

# Save formatted group 8 correlation dataframes
lo_res_GOSE_TIL_means.to_pickle(os.path.join(group8_dir,'lo_res_GOSE_TIL_mean.pkl'))
lo_res_GOSE_TIL_maxes.to_pickle(os.path.join(group8_dir,'lo_res_GOSE_TIL_max.pkl'))
lo_res_prognosis_TIL_means.to_pickle(os.path.join(group8_dir,'lo_res_prognosis_TIL_mean.pkl'))
lo_res_prognosis_TIL_maxes.to_pickle(os.path.join(group8_dir,'lo_res_prognosis_TIL_max.pkl'))

## Group 9: high-resolution neuromonitoring + GOSE population
# Create sub-directory for group 9
group9_dir = os.path.join(bs_dir,'group9')
os.makedirs(group9_dir,exist_ok=True)

# Extract high-resolution neuromonitoring + GOSE populationn GUPIs
group9_GUPIs = hi_res_GOSE_TIL_means.GUPI.unique()

# Make resamples for bootstrapping metrics
group9_bs_rs_GUPIs = [resample(group9_GUPIs,replace=True,n_samples=len(group9_GUPIs)) for _ in range(NUM_RESAMP)]
group9_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in group9_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
group9_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':group9_bs_rs_GUPIs})

# Store group 9 resamples
group9_bs_resamples.to_pickle(os.path.join(group9_dir,'group9_resamples.pkl'))

# Save formatted group 9 correlation dataframes
hi_res_GOSE_TIL_means.to_pickle(os.path.join(group9_dir,'hi_res_GOSE_TIL_mean.pkl'))
hi_res_GOSE_TIL_maxes.to_pickle(os.path.join(group9_dir,'hi_res_GOSE_TIL_max.pkl'))
hi_res_prognosis_TIL_means.to_pickle(os.path.join(group9_dir,'hi_res_prognosis_TIL_mean.pkl'))
hi_res_prognosis_TIL_maxes.to_pickle(os.path.join(group9_dir,'hi_res_prognosis_TIL_max.pkl'))