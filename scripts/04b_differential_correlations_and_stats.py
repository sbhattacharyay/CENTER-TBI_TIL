#### Master Script 4b: Calculate TIL correlations and statistics of different study sub-samples ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate correlation and statistics based on provided bootstrapping resample row index

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
import pingouin as pg
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import spearmanr
warnings.filterwarnings(action="ignore")

# StatsModels libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Custom methods
from functions.analysis import spearman_rho, melm_R2, calculate_spearman_rhos, calculate_rmcorr, calc_melm, calc_ICP_Na_melm, calc_HTS_Na_melm

# Initialise directory for storing bootstrapping resamples
bs_dir = '../bootstrapping_results/resamples'

# Initalise subdirectory to store individual resample results
bs_results_dir = '../bootstrapping_results/results'
os.makedirs(bs_results_dir,exist_ok=True)

### II. Calculate correlation and statistics based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):

    ## Initalise variables of validation population sub-directories
    # Designate sub-directory for TIL validation population
    TIL_validation_dir = os.path.join(bs_dir,'TIL_validation')

    # Designate sub-directory for TIL-ICP_EH validation population
    TIL_ICPEH_dir = os.path.join(bs_dir,'TIL_ICPEH')

    # Designate sub-directory for TIL-ICP_HR validation population
    TIL_ICPHR_dir = os.path.join(bs_dir,'TIL_ICPHR')

    # Designate sub-directory for TIL-Na+ validation populations
    TIL_Na_dir = os.path.join(bs_dir,'TIL_Na')

    # Designate current resample index
    curr_rs_idx = array_task_id+1

    ## Load bootstrapping resamples and select GUPIs of current resample_idx
    # Load and extract TIL validation resamples
    TIL_validation_bs_resamples = pd.read_pickle(os.path.join(TIL_validation_dir,'TIL_validation_resamples.pkl'))
    curr_TIL_validation_resamples = TIL_validation_bs_resamples[TIL_validation_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]

    # Load and extract TIL-ICP_EH validation resamples
    TIL_ICPEH_bs_resamples = pd.read_pickle(os.path.join(TIL_ICPEH_dir,'TIL_ICPEH_resamples.pkl'))
    curr_TIL_ICPEH_resamples = TIL_ICPEH_bs_resamples[TIL_ICPEH_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]

    # Load and extract TIL-ICP_HR validation resamples
    TIL_ICPHR_bs_resamples = pd.read_pickle(os.path.join(TIL_ICPHR_dir,'TIL_ICPHR_resamples.pkl'))
    curr_TIL_ICPHR_resamples = TIL_ICPHR_bs_resamples[TIL_ICPHR_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]

    # Load and extract TIL-Na validation resamples
    TIL_Na_bs_resamples = pd.read_pickle(os.path.join(TIL_Na_dir,'TIL_Na_resamples.pkl'))
    curr_TIL_Na_resamples = TIL_Na_bs_resamples[TIL_Na_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]

    # Load and extract TIL-ICP_EH validation resamples
    TIL_Na_ICPEH_bs_resamples = pd.read_pickle(os.path.join(TIL_Na_dir,'TIL_Na_ICPEH_resamples.pkl'))
    curr_TIL_Na_ICPEH_resamples = TIL_Na_ICPEH_bs_resamples[TIL_Na_ICPEH_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]

    # Load and extract TIL-ICP_HR validation resamples
    TIL_Na_ICPHR_bs_resamples = pd.read_pickle(os.path.join(TIL_Na_dir,'TIL_Na_ICPHR_resamples.pkl'))
    curr_TIL_Na_ICPHR_resamples = TIL_Na_ICPHR_bs_resamples[TIL_Na_ICPHR_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]

    ## Load and filter information dataframes
    # Formatted scale scores
    raw_formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv')
    raw_formatted_TIL_scores = raw_formatted_TIL_scores[raw_formatted_TIL_scores.TILTimepoint<=7].reset_index(drop=True)
    raw_formatted_TIL_1987_scores = pd.read_csv('../formatted_data/formatted_TIL_1987_scores.csv')
    raw_formatted_TIL_1987_scores = raw_formatted_TIL_1987_scores[raw_formatted_TIL_1987_scores.TILTimepoint<=7].reset_index(drop=True)
    raw_formatted_PILOT_scores = pd.read_csv('../formatted_data/formatted_PILOT_scores.csv')
    raw_formatted_PILOT_scores = raw_formatted_PILOT_scores[raw_formatted_PILOT_scores.TILTimepoint<=7].reset_index(drop=True)
    raw_formatted_TIL_Basic_scores = pd.read_csv('../formatted_data/formatted_TIL_Basic_scores.csv')
    raw_formatted_TIL_Basic_scores = raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.TILTimepoint<=7].reset_index(drop=True)
    raw_formatted_uwTIL_scores = pd.read_csv('../formatted_data/formatted_unweighted_TIL_scores.csv')
    raw_formatted_uwTIL_scores = raw_formatted_uwTIL_scores[raw_formatted_uwTIL_scores.TILTimepoint<=7].reset_index(drop=True)
    raw_formatted_uwTIL_scores['uwTILSum'] = raw_formatted_uwTIL_scores.CSFDrainage + raw_formatted_uwTIL_scores.DecomCraniectomy + raw_formatted_uwTIL_scores.FluidLoading + raw_formatted_uwTIL_scores.Hypertonic + raw_formatted_uwTIL_scores.ICPSurgery + raw_formatted_uwTIL_scores.Mannitol + raw_formatted_uwTIL_scores.Neuromuscular + raw_formatted_uwTIL_scores.Positioning + raw_formatted_uwTIL_scores.Sedation + raw_formatted_uwTIL_scores.Temperature + raw_formatted_uwTIL_scores.Vasopressor + raw_formatted_uwTIL_scores.Ventilation

    # Create a pre-filtered dataframe of all scale sum scores
    raw_all_total_scores = raw_formatted_uwTIL_scores[['GUPI','TILTimepoint','TILDate','TotalSum','uwTILSum']].merge(raw_formatted_TIL_1987_scores[['GUPI','TILTimepoint','TILDate','TIL_1987Sum']]).merge(raw_formatted_PILOT_scores[['GUPI','TILTimepoint','TILDate','PILOTSum']]).merge(raw_formatted_TIL_Basic_scores[['GUPI','TILTimepoint','TILDate','TIL_Basic']])

    # Formatted scale maxes
    raw_formatted_TIL_max = pd.read_csv('../formatted_data/formatted_TIL_max.csv')
    raw_formatted_TIL_1987_max = pd.read_csv('../formatted_data/formatted_TIL_1987_max.csv')
    raw_formatted_PILOT_max = pd.read_csv('../formatted_data/formatted_PILOT_max.csv')
    raw_formatted_TIL_Basic_max = pd.read_csv('../formatted_data/formatted_TIL_Basic_max.csv')
    raw_formatted_uwTIL_max = raw_formatted_uwTIL_scores.loc[raw_formatted_uwTIL_scores.groupby(['GUPI'])['uwTILSum'].idxmax()].reset_index(drop=True).rename(columns={'uwTILSum':'uwTILmax'})
    raw_combined_max_scores = raw_formatted_TIL_max[['GUPI','TILmax']].merge(raw_formatted_TIL_1987_max[['GUPI','TIL_1987max']]).merge(raw_formatted_PILOT_max[['GUPI','PILOTmax']]).merge(raw_formatted_TIL_Basic_max[['GUPI','TIL_Basicmax']]).merge(raw_formatted_uwTIL_max[['GUPI','uwTILmax']])

    # Formatted scale means
    raw_formatted_TIL_mean = pd.read_csv('../formatted_data/formatted_TIL_mean.csv')
    raw_formatted_TIL_1987_mean = pd.read_csv('../formatted_data/formatted_TIL_1987_mean.csv')
    raw_formatted_PILOT_mean = pd.read_csv('../formatted_data/formatted_PILOT_mean.csv')
    raw_formatted_TIL_Basic_mean = pd.read_csv('../formatted_data/formatted_TIL_Basic_mean.csv')
    raw_formatted_uwTIL_mean = pd.pivot_table(raw_formatted_uwTIL_scores.melt(id_vars=['GUPI','TILTimepoint','TILDate','ICUAdmTimeStamp','ICUDischTimeStamp']).groupby(['GUPI','variable'],as_index=False)['value'].mean(), values = 'value', index=['GUPI'], columns = 'variable').reset_index().rename(columns={'uwTILSum':'uwTILmean'})
    raw_combined_mean_scores = raw_formatted_TIL_mean[['GUPI','TILmean']].merge(raw_formatted_TIL_1987_mean[['GUPI','TIL_1987mean']]).merge(raw_formatted_PILOT_mean[['GUPI','PILOTmean']]).merge(raw_formatted_TIL_Basic_mean[['GUPI','TIL_Basicmean']]).merge(raw_formatted_uwTIL_mean[['GUPI','uwTILmean']])

    # Combine raw mean and max dataframes
    raw_combined_max_mean_scores = raw_combined_max_scores.merge(raw_combined_mean_scores,how='inner')

    # Demographic and outcome information
    raw_CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')
    raw_CENTER_TBI_demo_outcome = raw_CENTER_TBI_demo_outcome.reset_index(drop=True)

    # Formatted low-resolution values, means, and maxes 
    raw_formatted_low_resolution_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv')
    raw_formatted_low_resolution_values = raw_formatted_low_resolution_values[(raw_formatted_low_resolution_values.TILTimepoint<=7)].reset_index(drop=True)
    raw_formatted_low_resolution_maxes_means = pd.read_csv('../formatted_data/formatted_low_resolution_maxes_means.csv')
    raw_formatted_low_resolution_maxes_means = raw_formatted_low_resolution_maxes_means.reset_index(drop=True)

    # Formatted high-resolution values, means, and maxes 
    raw_formatted_high_resolution_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv')
    raw_formatted_high_resolution_values = raw_formatted_high_resolution_values[(raw_formatted_high_resolution_values.TILTimepoint<=7)].reset_index(drop=True)
    raw_formatted_high_resolution_maxes_means = pd.read_csv('../formatted_data/formatted_high_resolution_maxes_means.csv')
    raw_formatted_high_resolution_maxes_means = raw_formatted_high_resolution_maxes_means.reset_index(drop=True)

    # Formatted serum sodium concentration values, means, and maxes
    raw_formatted_sodium_values = pd.read_csv('../formatted_data/formatted_daily_sodium_values.csv')
    raw_formatted_sodium_values = raw_formatted_sodium_values[raw_formatted_sodium_values.TILTimepoint<=7].reset_index(drop=True)
    raw_formatted_sodium_maxes_means = pd.read_csv('../formatted_data/formatted_sodium_maxes_means.csv')
    raw_formatted_sodium_maxes_means = raw_formatted_sodium_maxes_means.reset_index(drop=True)

    ## Calculate TILmean/TILmax correlations
    # Calculate Spearman's rho between TILmean/max scores and Na+ values
    TIL_Na_spearmans = calculate_spearman_rhos(raw_combined_max_mean_scores[raw_combined_max_mean_scores.GUPI.isin(curr_TIL_Na_resamples)],raw_formatted_sodium_maxes_means[raw_formatted_sodium_maxes_means.GUPI.isin(curr_TIL_Na_resamples)].drop(columns=['HTSPtInd','NoHyperosmolarPtInd','MannitolPtInd']),'Spearman rho between TILmaxes/means and Na values')
    TIL_Na_spearmans['Population'] = 'TIL-Na'

    # Calculate Spearman's rho between TILmean/max scores and lo-res global values
    TIL_lo_res_spearmans = calculate_spearman_rhos(raw_combined_max_mean_scores[raw_combined_max_mean_scores.GUPI.isin(curr_TIL_ICPEH_resamples)],raw_formatted_low_resolution_maxes_means[raw_formatted_low_resolution_maxes_means.GUPI.isin(curr_TIL_ICPEH_resamples)],'Spearman rho between TILmaxes/means and lo-res neuromonitoring')
    TIL_lo_res_spearmans['Population'] = 'TIL-ICP_EH'

    # Calculate Spearman's rho between TILmean/max scores and hi-res global values
    TIL_hi_res_spearmans = calculate_spearman_rhos(raw_combined_max_mean_scores[raw_combined_max_mean_scores.GUPI.isin(curr_TIL_ICPHR_resamples)],raw_formatted_high_resolution_maxes_means[raw_formatted_high_resolution_maxes_means.GUPI.isin(curr_TIL_ICPHR_resamples)],'Spearman rho between TILmaxes/means and hi-res neuromonitoring')
    TIL_hi_res_spearmans['Population'] = 'TIL-ICP_HR'

    # Concatenate Spearman's rho dataframes
    compiled_spearmans_dataframe = pd.concat([TIL_Na_spearmans,TIL_lo_res_spearmans,TIL_hi_res_spearmans],ignore_index=True)
    compiled_spearmans_dataframe['resample_idx'] = curr_rs_idx

    # Save concatenated dataframe
    compiled_spearmans_dataframe.to_pickle(os.path.join(bs_results_dir,'differential_compiled_spearman_rhos_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    ## Calculate TIL correlations
    # Define vectors of particular columns columns
    timestamp_columns = ['ICUAdmTimeStamp','ICUDischTimeStamp']
    physician_impression_columns = ['TILPhysicianConcernsCPP', 'TILPhysicianConcernsICP','TILPhysicianOverallSatisfaction','TILPhysicianOverallSatisfactionSurvival', 'TILPhysicianSatICP']

    # Define physician impression columns
    within_TIL_Basic_rms = calculate_rmcorr(raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_validation_resamples)].drop(columns=timestamp_columns+physician_impression_columns+['TotalSum']),raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_validation_resamples)].drop(columns=timestamp_columns+['TotalSum','TIL_Basic']),'rmcorr within TIL_Basic')
    within_TIL_Basic_rms['Scale'] = 'TIL_Basic'
    compiled_within_rms = within_TIL_Basic_rms
    compiled_within_rms['Population'] = 'TIL'

    # Calculate correlation between TIL scores and low-resolution neuromonitoring
    TIL_lo_res_rms = calculate_rmcorr(raw_formatted_TIL_scores[raw_formatted_TIL_scores.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=timestamp_columns),raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'rmcorr between TIL and ICP_EH')
    TIL_lo_res_rms['Scale'] = 'TIL'
    PILOT_lo_res_rms = calculate_rmcorr(raw_formatted_PILOT_scores[raw_formatted_PILOT_scores.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'rmcorr between PILOT and ICP_EH')
    PILOT_lo_res_rms['Scale'] = 'PILOT'
    TIL_1987_lo_res_rms = calculate_rmcorr(raw_formatted_TIL_1987_scores[raw_formatted_TIL_1987_scores.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'rmcorr between TIL_1987 and ICP_EH')
    TIL_1987_lo_res_rms['Scale'] = 'TIL_1987'
    uwTIL_lo_res_rms = calculate_rmcorr(raw_formatted_uwTIL_scores[raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'rmcorr between uwTIL and ICP_EH')
    uwTIL_lo_res_rms['Scale'] = 'uwTIL'
    TIL_Basic_lo_res_rms = calculate_rmcorr(raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'rmcorr betweenTIL_Basic and ICP_EH')
    TIL_Basic_lo_res_rms['Scale'] = 'TIL_Basic'
    compiled_lo_res_rms = pd.concat([TIL_lo_res_rms,PILOT_lo_res_rms,TIL_1987_lo_res_rms,uwTIL_lo_res_rms,TIL_Basic_lo_res_rms],ignore_index=True)
    compiled_lo_res_rms['Population'] = 'TIL-ICP_EH'

    # Calculate correlation between TIL scores and high-resolution neuromonitoring
    TIL_hi_res_rms = calculate_rmcorr(raw_formatted_TIL_scores[raw_formatted_TIL_scores.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=timestamp_columns),raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'rmcorr between TIL and ICP_HR')
    TIL_hi_res_rms['Scale'] = 'TIL'
    PILOT_hi_res_rms = calculate_rmcorr(raw_formatted_PILOT_scores[raw_formatted_PILOT_scores.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'rmcorr between PILOT and ICP_HR')
    PILOT_hi_res_rms['Scale'] = 'PILOT'
    TIL_1987_hi_res_rms = calculate_rmcorr(raw_formatted_TIL_1987_scores[raw_formatted_TIL_1987_scores.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'rmcorr between TIL_1987 and ICP_HR')
    TIL_1987_hi_res_rms['Scale'] = 'TIL_1987'
    uwTIL_hi_res_rms = calculate_rmcorr(raw_formatted_uwTIL_scores[raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'rmcorr between uwTIL and ICP_HR')
    uwTIL_hi_res_rms['Scale'] = 'uwTIL'
    TIL_Basic_hi_res_rms = calculate_rmcorr(raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'rmcorr between TIL_Basic and ICP_HR')
    TIL_Basic_hi_res_rms['Scale'] = 'TIL_Basic'
    compiled_hi_res_rms = pd.concat([TIL_hi_res_rms,PILOT_hi_res_rms,TIL_1987_hi_res_rms,uwTIL_hi_res_rms,TIL_Basic_hi_res_rms],ignore_index=True)
    compiled_hi_res_rms['Population'] = 'TIL-ICP_HR'
    
    # Calculate correlation between TIL scores and serum sodium values
    TIL_Na_rms = calculate_rmcorr(raw_formatted_TIL_scores[raw_formatted_TIL_scores.GUPI.isin(curr_TIL_Na_resamples)].drop(columns=timestamp_columns),raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'rmcorr between TIL and Na+')
    TIL_Na_rms['Scale'] = 'TIL'
    PILOT_Na_rms = calculate_rmcorr(raw_formatted_PILOT_scores[raw_formatted_PILOT_scores.GUPI.isin(curr_TIL_Na_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'rmcorr between PILOT and Na+')
    PILOT_Na_rms['Scale'] = 'PILOT'
    TIL_1987_Na_rms = calculate_rmcorr(raw_formatted_TIL_1987_scores[raw_formatted_TIL_1987_scores.GUPI.isin(curr_TIL_Na_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'rmcorr between TIL_1987 and Na+')
    TIL_1987_Na_rms['Scale'] = 'TIL_1987'
    uwTIL_Na_rms = calculate_rmcorr(raw_formatted_uwTIL_scores[raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_Na_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'rmcorr between uwTIL and Na+')
    uwTIL_Na_rms['Scale'] = 'uwTIL'
    TIL_Basic_Na_rms = calculate_rmcorr(raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_Na_resamples)].drop(columns=timestamp_columns+['TotalSum']+physician_impression_columns),raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'rmcorr between TIL_Basic and Na+')
    TIL_Basic_Na_rms['Scale'] = 'TIL_Basic'
    compiled_Na_rms = pd.concat([TIL_Na_rms,PILOT_Na_rms,TIL_1987_Na_rms,uwTIL_Na_rms,TIL_Basic_Na_rms],ignore_index=True)
    compiled_Na_rms['Population'] = 'TIL-Na'

    # Calculate correlation between ICP/CPP and serum sodium values
    Na_lo_res_rms = calculate_rmcorr(raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_ICPEH_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_Na_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'rmcorr between Na+ and ICP_EH')
    Na_lo_res_rms['Population'] = 'TIL-Na-ICP_EH'
    Na_hi_res_rms = calculate_rmcorr(raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_ICPHR_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_Na_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'rmcorr between Na+ and ICP_HR')
    Na_hi_res_rms['Population'] = 'TIL-Na-ICP_HR'
    compiled_Na_res_rms = pd.concat([Na_lo_res_rms,Na_hi_res_rms],ignore_index=True)
    compiled_Na_res_rms['Scale'] = 'All'

    ## Concatenate all the repeated-measures correlation results and save
    # Concatenate
    compiled_rms_df = pd.concat([compiled_within_rms,compiled_lo_res_rms,compiled_hi_res_rms,compiled_Na_rms,compiled_Na_res_rms],ignore_index=True)
    compiled_rms_df['resample_idx'] = curr_rs_idx

    # Save concatenated dataframe
    compiled_rms_df.to_pickle(os.path.join(bs_results_dir,'differential_compiled_rmcorr_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    ## Mixed-effects regression
    # Define component columns for each scale
    TIL_components = ['CSFDrainage', 'DecomCraniectomy','FluidLoading', 'Hypertonic', 'ICPSurgery', 'Mannitol', 'Neuromuscular','Positioning', 'Sedation', 'Temperature','Vasopressor','Ventilation']
    PILOT_components = ['Temperature', 'Sedation', 'Neuromuscular', 'Ventilation', 'Mannitol','Hypertonic', 'CSFDrainage', 'ICPSurgery', 'DecomCraniectomy','Vasopressor']
    TIL_1987_components = ['Sedation', 'Mannitol', 'Ventricular', 'Hyperventilation', 'Paralysis']

    # Calculate low-resolution mixed effect models
    TIL_lo_res_mlm = calc_melm(raw_formatted_TIL_scores[raw_formatted_TIL_scores.GUPI.isin(curr_TIL_ICPEH_resamples)],raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'TotalSum',False,TIL_components,'Calculating ICP_EH ~ TIL')
    TIL_lo_res_mlm['Scale'] = 'TIL'
    PILOT_lo_res_mlm = calc_melm(raw_formatted_PILOT_scores[raw_formatted_PILOT_scores.GUPI.isin(curr_TIL_ICPEH_resamples)],raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'PILOTSum',False,PILOT_components,'Calculating ICP_EH ~ PILOT')
    PILOT_lo_res_mlm['Scale'] = 'PILOT'
    TIL_1987_lo_res_mlm = calc_melm(raw_formatted_TIL_1987_scores[raw_formatted_TIL_1987_scores.GUPI.isin(curr_TIL_ICPEH_resamples)],raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'TIL_1987Sum',False,TIL_1987_components,'Calculating ICP_EH ~ TIL_1987')
    TIL_1987_lo_res_mlm['Scale'] = 'TIL_1987'
    TIL_Basic_lo_res_mlm = calc_melm(raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_ICPEH_resamples)],raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'TIL_Basic',False,[],'Calculating ICP_EH ~ TIL_Basic')
    TIL_Basic_lo_res_mlm['Scale'] = 'TIL_Basic'
    uwTIL_lo_res_mlm = calc_melm(raw_formatted_uwTIL_scores[raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_ICPEH_resamples)],raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_ICPEH_resamples)].drop(columns=['TotalSum','nCPP','nICP']),'uwTILSum',True,TIL_components,'Calculating ICP_EH ~ uwTIL')
    uwTIL_lo_res_mlm['Scale'] = 'uwTIL'
    compiled_lo_res_mlm = pd.concat([TIL_lo_res_mlm,PILOT_lo_res_mlm,TIL_1987_lo_res_mlm,TIL_Basic_lo_res_mlm,uwTIL_lo_res_mlm],ignore_index=True)
    compiled_lo_res_mlm['Population'] = 'TIL-ICP_EH'
                 
    # Calculate high-resolution mixed effect models
    TIL_hi_res_mlm = calc_melm(raw_formatted_TIL_scores[raw_formatted_TIL_scores.GUPI.isin(curr_TIL_ICPHR_resamples)],raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'TotalSum',False,TIL_components,'Calculating ICP_HR ~ TIL')
    TIL_hi_res_mlm['Scale'] = 'TIL'
    PILOT_hi_res_mlm = calc_melm(raw_formatted_PILOT_scores[raw_formatted_PILOT_scores.GUPI.isin(curr_TIL_ICPHR_resamples)],raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'PILOTSum',False,PILOT_components,'Calculating ICP_HR ~ PILOT')
    PILOT_hi_res_mlm['Scale'] = 'PILOT'
    TIL_1987_hi_res_mlm = calc_melm(raw_formatted_TIL_1987_scores[raw_formatted_TIL_1987_scores.GUPI.isin(curr_TIL_ICPHR_resamples)],raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'TIL_1987Sum',False,TIL_1987_components,'Calculating ICP_HR ~ TIL_1987')
    TIL_1987_hi_res_mlm['Scale'] = 'TIL_1987'
    TIL_Basic_hi_res_mlm = calc_melm(raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_ICPHR_resamples)],raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'TIL_Basic',False,[],'Calculating ICP_HR ~ TIL_Basic')
    TIL_Basic_hi_res_mlm['Scale'] = 'TIL_Basic'
    uwTIL_hi_res_mlm = calc_melm(raw_formatted_uwTIL_scores[raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_ICPHR_resamples)],raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_ICPHR_resamples)].drop(columns=['TotalSum','nCPP','nICP','EVD']),'uwTILSum',True,TIL_components,'Calculating ICP_HR ~ uwTIL')
    uwTIL_hi_res_mlm['Scale'] = 'uwTIL'
    compiled_hi_res_mlm = pd.concat([TIL_hi_res_mlm,PILOT_hi_res_mlm,TIL_1987_hi_res_mlm,TIL_Basic_hi_res_mlm,uwTIL_hi_res_mlm],ignore_index=True)
    compiled_hi_res_mlm['Population'] = 'TIL-ICP_HR'

    # Calculate sodium mixed effect models
    TIL_Na_mlm = calc_melm(raw_formatted_TIL_scores[raw_formatted_TIL_scores.GUPI.isin(curr_TIL_Na_resamples)],raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'TotalSum',False,TIL_components,'Calculating Na+ ~ TIL')
    TIL_Na_mlm['Scale'] = 'TIL'
    PILOT_Na_mlm = calc_melm(raw_formatted_PILOT_scores[raw_formatted_PILOT_scores.GUPI.isin(curr_TIL_Na_resamples)],raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'PILOTSum',False,PILOT_components,'Calculating Na+ ~ PILOT')
    PILOT_Na_mlm['Scale'] = 'PILOT'
    TIL_1987_Na_mlm = calc_melm(raw_formatted_TIL_1987_scores[raw_formatted_TIL_1987_scores.GUPI.isin(curr_TIL_Na_resamples)],raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'TIL_1987Sum',False,TIL_1987_components,'Calculating Na+ ~ TIL_1987')
    TIL_1987_Na_mlm['Scale'] = 'TIL_1987'
    TIL_Basic_Na_mlm = calc_melm(raw_formatted_TIL_Basic_scores[raw_formatted_TIL_Basic_scores.GUPI.isin(curr_TIL_Na_resamples)],raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'TIL_Basic',False,[],'Calculating Na+ ~ TIL_Basic')
    TIL_Basic_Na_mlm['Scale'] = 'TIL_Basic'
    uwTIL_Na_mlm = calc_melm(raw_formatted_uwTIL_scores[raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_Na_resamples)],raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']],'uwTILSum',True,TIL_components,'Calculating Na+ ~ uwTIL')
    uwTIL_Na_mlm['Scale'] = 'uwTIL'
    compiled_Na_mlm = pd.concat([TIL_Na_mlm,PILOT_Na_mlm,TIL_1987_Na_mlm,TIL_Basic_Na_mlm,uwTIL_Na_mlm],ignore_index=True)
    compiled_Na_mlm['Population'] = 'TIL-Na'

    # Calculate bespoke TIL ~ Na + ICP mixed effect models
    TIL_lo_res_Na_df = raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_ICPEH_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']].merge(raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_Na_ICPEH_resamples)].drop(columns=['nCPP','nICP','CPPmean']))
    TIL_lo_res_Na_mlm = calc_ICP_Na_melm(TIL_lo_res_Na_df,'TotalSum')
    TIL_lo_res_Na_mlm['Scale'] = 'TIL'
    TIL_lo_res_Na_mlm['Population'] = 'TIL-Na-ICP_EH'
    TIL_hi_res_Na_df = raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_ICPHR_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']].merge(raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_Na_ICPHR_resamples)].drop(columns=['nCPP','nICP','CPPmean','EVD']))
    TIL_hi_res_Na_mlm = calc_ICP_Na_melm(TIL_hi_res_Na_df,'TotalSum')
    TIL_hi_res_Na_mlm['Scale'] = 'TIL'
    TIL_hi_res_Na_mlm['Population'] = 'TIL-Na-ICP_HR'
    TIL_ICP_Na_mlm = pd.concat([TIL_lo_res_Na_mlm,TIL_hi_res_Na_mlm],ignore_index=True)

    # Calculate bespoke (d)Na ~ HTS + (d)ICP mixed effect models
    HTS_lo_res_Na_df = raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_ICPEH_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']].merge(raw_formatted_low_resolution_values[raw_formatted_low_resolution_values.GUPI.isin(curr_TIL_Na_ICPEH_resamples)].drop(columns=['nCPP','nICP','CPPmean'])).merge(raw_formatted_uwTIL_scores[['GUPI','TILTimepoint','TILDate','Hypertonic']][raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_Na_ICPEH_resamples)])
    HTS_hi_res_Na_df = raw_formatted_sodium_values[raw_formatted_sodium_values.GUPI.isin(curr_TIL_Na_ICPHR_resamples)][['GUPI','TILTimepoint','TILDate','meanSodium','ChangeInSodium']].merge(raw_formatted_high_resolution_values[raw_formatted_high_resolution_values.GUPI.isin(curr_TIL_Na_ICPHR_resamples)].drop(columns=['nCPP','nICP','CPPmean'])).merge(raw_formatted_uwTIL_scores[['GUPI','TILTimepoint','TILDate','Hypertonic']][raw_formatted_uwTIL_scores.GUPI.isin(curr_TIL_Na_ICPHR_resamples)])
    HTS_lo_res_Na_mlm = calc_HTS_Na_melm(HTS_lo_res_Na_df)
    HTS_lo_res_Na_mlm['Scale'] = 'TIL'
    HTS_lo_res_Na_mlm['Population'] = 'TIL-Na-ICP_EH'
    HTS_hi_res_Na_mlm = calc_HTS_Na_melm(HTS_hi_res_Na_df)
    HTS_hi_res_Na_mlm['Scale'] = 'TIL'
    HTS_hi_res_Na_mlm['Population'] = 'TIL-Na-ICP_HR'
    HTS_ICP_Na_mlm = pd.concat([HTS_lo_res_Na_mlm,HTS_hi_res_Na_mlm],ignore_index=True)

    # Concatenate all mlm model information
    compiled_mlm_df = pd.concat([compiled_lo_res_mlm,compiled_hi_res_mlm,compiled_Na_mlm,TIL_ICP_Na_mlm,HTS_ICP_Na_mlm],ignore_index=True)
    compiled_mlm_df['resample_idx'] = curr_rs_idx

    # Save concatenated dataframe
    compiled_mlm_df.to_pickle(os.path.join(bs_results_dir,'differential_compiled_mixed_effects_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)