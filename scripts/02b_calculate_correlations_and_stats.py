#### Master Script 2b: Calculate TIL correlations and statistics of different study sub-samples ####
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

# SciKit-Learn methods
from sklearn.utils import resample
from sklearn.metrics import mutual_info_score, roc_curve, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.feature_selection._mutual_info import mutual_info_regression, _estimate_mi

# StatsModels libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Custom methods
from functions.analysis import calculate_spearman_rhos, calculate_rmcorr, calc_melm, calculate_dynamic_spearman_rhos, calc_ROC

# Define results directory
results_dir = '../results'

# Define formatted directory
formatted_data_dir = '../formatted_data'

# Define directory for storing bootstrapping resamples within results directory
bs_dir = os.path.join(results_dir,'bootstrapping_results','resamples')

# Initalise subdirectory to store individual resample results
bs_results_dir = os.path.join(results_dir,'bootstrapping_results','results')
os.makedirs(bs_results_dir,exist_ok=True)

### II. Calculate correlation and statistics based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):

    ## Save current resampling index value
    curr_rs_idx = array_task_id+1

    ## Initalise variables of validation population sub-directories
    # Designate sub-directory for TIL validation population
    TIL_validation_dir = os.path.join(bs_dir,'TIL_validation')

    # Designate sub-directory for TIL-ICP_EH validation population
    TIL_ICPEH_dir = os.path.join(bs_dir,'TIL_ICPEH')

    # Designate sub-directory for TIL-ICP_HR validation population
    TIL_ICPHR_dir = os.path.join(bs_dir,'TIL_ICPHR')

    ## Load bootstrapping resamples and select GUPIs and imputation index of current resample_idx
    # Load and extract TIL validation resamples
    TIL_validation_bs_resamples = pd.read_pickle(os.path.join(TIL_validation_dir,'TIL_validation_resamples.pkl'))
    curr_TIL_validation_resamples = TIL_validation_bs_resamples[TIL_validation_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]
    curr_imp_index = TIL_validation_bs_resamples.IMPUTATION_IDX[TIL_validation_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].values[0]
    # TIL_validation_bs_resamples = pd.read_pickle(os.path.join(TIL_validation_dir,'remaining_TIL_validation_resamples.pkl'))
    # curr_rs_idx = TIL_validation_bs_resamples.RESAMPLE_IDX[array_task_id]
    # curr_TIL_validation_resamples = TIL_validation_bs_resamples[TIL_validation_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]

    # Load and extract TIL-ICP_EH validation resamples
    TIL_ICPEH_bs_resamples = pd.read_pickle(os.path.join(TIL_ICPEH_dir,'TIL_ICPEH_resamples.pkl'))
    curr_TIL_ICPEH_resamples = TIL_ICPEH_bs_resamples[TIL_ICPEH_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]
    # TIL_ICPEH_bs_resamples = pd.read_pickle(os.path.join(TIL_ICPEH_dir,'remaining_TIL_ICPEH_resamples.pkl'))
    # curr_rs_idx = TIL_ICPEH_bs_resamples.RESAMPLE_IDX[array_task_id]
    # curr_TIL_ICPEH_resamples = TIL_ICPEH_bs_resamples[TIL_ICPEH_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]
    
    # Load and extract TIL-ICP_HR validation resamples
    TIL_ICPHR_bs_resamples = pd.read_pickle(os.path.join(TIL_ICPHR_dir,'TIL_ICPHR_resamples.pkl'))
    curr_TIL_ICPHR_resamples = TIL_ICPHR_bs_resamples[TIL_ICPHR_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]
    # TIL_ICPHR_bs_resamples = pd.read_pickle(os.path.join(TIL_ICPHR_dir,'remaining_TIL_ICPHR_resamples.pkl'))
    # curr_rs_idx = TIL_ICPHR_bs_resamples.RESAMPLE_IDX[array_task_id]
    # curr_TIL_ICPHR_resamples = TIL_ICPHR_bs_resamples[TIL_ICPHR_bs_resamples.RESAMPLE_IDX==(curr_rs_idx)].GUPIs.values[0]
    
    ## Load and filter information dataframes
    # Define directory of current imputation
    imp_dir = os.path.join(formatted_data_dir,'imputed_sets','imp'+str(curr_imp_index).zfill(3))
    
    # Formatted dynamic variable set
    raw_dynamic_var_set = pd.read_csv(os.path.join(imp_dir,'dynamic_var_set.csv'))

    # Filter dynamic variable set by (sub-)population
    global_dynamic_var_set = raw_dynamic_var_set[raw_dynamic_var_set.GUPI.isin(curr_TIL_validation_resamples)].reset_index(drop=True)
    lores_dynamic_var_set = raw_dynamic_var_set[raw_dynamic_var_set.GUPI.isin(curr_TIL_ICPEH_resamples)].reset_index(drop=True)
    hires_dynamic_var_set = raw_dynamic_var_set[raw_dynamic_var_set.GUPI.isin(curr_TIL_ICPHR_resamples)].reset_index(drop=True)

    # Formatted static variable set
    raw_static_var_set = pd.read_csv(os.path.join(imp_dir,'static_var_set.csv'))

    # Modify baseline prognosis score column names
    og_prog_names = ['Pr.GOSE.1.', 'Pr.GOSE.3.', 'Pr.GOSE.4.', 'Pr.GOSE.5.', 'Pr.GOSE.6.','Pr.GOSE.7.']
    new_prog_names = ['Pr(GOSE>1)', 'Pr(GOSE>3)', 'Pr(GOSE>4)', 'Pr(GOSE>5)', 'Pr(GOSE>6)','Pr(GOSE>7)']
    raw_static_var_set = raw_static_var_set.rename(columns=dict(zip(og_prog_names, new_prog_names)))

    # Filter static variable set by (sub-)population
    global_static_var_set = raw_static_var_set[raw_static_var_set.GUPI.isin(curr_TIL_validation_resamples)].reset_index(drop=True)
    lores_static_var_set = raw_static_var_set[raw_static_var_set.GUPI.isin(curr_TIL_ICPEH_resamples)].reset_index(drop=True)
    hires_static_var_set = raw_static_var_set[raw_static_var_set.GUPI.isin(curr_TIL_ICPHR_resamples)].reset_index(drop=True)

    # ## Calculate Spearman's rhos
    # # Calculate Spearman's rhos among global static values
    # global_static_spearmans = calculate_spearman_rhos(global_static_var_set,global_static_var_set,'Global static Spearman rhos')
    # global_static_spearmans['Population'] = 'TIL'

    # # Calculate Spearman's rhos among ICP_EH static values
    # lores_static_spearmans = calculate_spearman_rhos(lores_static_var_set,lores_static_var_set,'ICP_EH static Spearman rhos')
    # lores_static_spearmans['Population'] = 'TIL-ICP_EH'

    # # Calculate Spearman's rhos among ICP_HR static values
    # hires_static_spearmans = calculate_spearman_rhos(hires_static_var_set,hires_static_var_set,'ICP_HR static Spearman rhos')
    # hires_static_spearmans['Population'] = 'TIL-ICP_HR'

    # # Compile static Spearman's rhos and format
    # compiled_static_spearmans = pd.concat([global_static_spearmans,lores_static_spearmans,hires_static_spearmans],ignore_index=True)
    # compiled_static_spearmans.insert(2,'TILTimepoint','Static')

    # # Calculate Spearman's rhos among global dynamic values
    # global_dynamic_spearmans = calculate_dynamic_spearman_rhos(global_dynamic_var_set,global_dynamic_var_set,'Global dynamic Spearman rhos')
    # global_dynamic_spearmans['Population'] = 'TIL'

    # # Calculate Spearman's rhos among ICP_EH dynamic values
    # lores_dynamic_spearmans = calculate_dynamic_spearman_rhos(lores_dynamic_var_set,lores_dynamic_var_set,'ICP_EH dynamic Spearman rhos')
    # lores_dynamic_spearmans['Population'] = 'TIL-ICP_EH'

    # # Calculate Spearman's rhos among ICP_HR dynamic values
    # hires_dynamic_spearmans = calculate_dynamic_spearman_rhos(hires_dynamic_var_set,hires_dynamic_var_set,'ICP_HR dynamic Spearman rhos')
    # hires_dynamic_spearmans['Population'] = 'TIL-ICP_HR'

    # # Concatenate all spearman dataframes into one and format
    # compiled_spearmans = pd.concat([compiled_static_spearmans,global_dynamic_spearmans,lores_dynamic_spearmans,hires_dynamic_spearmans],ignore_index=True)
    # compiled_spearmans['resample_idx'] = curr_rs_idx

    # # Save concatenated dataframe
    # compiled_spearmans.to_pickle(os.path.join(bs_results_dir,'compiled_spearman_rhos_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # ## Calculate repeated-measures correlations
    # # Calculate rmcorrs among TIL scores
    # TIL_scores_list = ['TotalSum','TIL_Basic','uwTILSum','PILOTSum','TIL_1987Sum']
    # across_TIL_rmcorrs = calculate_rmcorr(global_dynamic_var_set[['GUPI','TILTimepoint']+TIL_scores_list],global_dynamic_var_set[['GUPI','TILTimepoint']+TIL_scores_list],'Across-TIL rmcorrs')
    # across_TIL_rmcorrs['Population'] = 'TIL'

    # # Calculate rmcorrs between TIL scores and physician concerns
    # physician_concerns_list = ['TILPhysicianConcernsCPP','TILPhysicianConcernsICP']
    # TIL_concerns_rmcorrs = calculate_rmcorr(global_dynamic_var_set[['GUPI','TILTimepoint']+TIL_scores_list],global_dynamic_var_set[['GUPI','TILTimepoint']+physician_concerns_list],'TIL-concerns rmcorrs')
    # TIL_concerns_rmcorrs['Population'] = 'TIL'

    # # Calculate rmcorrs within-TIL scores
    # TIL_components = ['CSFDrainage', 'DecomCraniectomy', 'FluidLoading', 'Hypertonic','ICPSurgery', 'Mannitol', 'Neuromuscular', 'Positioning', 'Sedation','Temperature', 'Vasopressor', 'Ventilation']
    # within_TIL_rmcorrs = calculate_rmcorr(global_dynamic_var_set[['GUPI','TILTimepoint','TotalSum']+TIL_components],global_dynamic_var_set[['GUPI','TILTimepoint','TotalSum']+TIL_components],'Within-TIL rmcorrs')
    # within_TIL_rmcorrs['Population'] = 'TIL'

    # # Calculate rmcorrs within-uwTIL scores
    # uwTIL_components = ['uw'+comp for comp in TIL_components]
    # within_uwTIL_rmcorrs = calculate_rmcorr(global_dynamic_var_set[['GUPI','TILTimepoint','uwTILSum']+uwTIL_components],global_dynamic_var_set[['GUPI','TILTimepoint','uwTILSum']+uwTIL_components],'Within-uwTIL rmcorrs')
    # within_uwTIL_rmcorrs['Population'] = 'TIL'

    # # Calculate component correlations with physician concerns
    # concern_component_rmcorrs = calculate_rmcorr(global_dynamic_var_set[['GUPI','TILTimepoint']+TIL_components],global_dynamic_var_set[['GUPI','TILTimepoint']+physician_concerns_list],'Component-Concern rmcorrs')
    # concern_component_rmcorrs['Population'] = 'TIL'

    # # Calculate unweighted component correlations with physician concerns
    # concern_uwcomponent_rmcorrs = calculate_rmcorr(global_dynamic_var_set[['GUPI','TILTimepoint']+uwTIL_components],global_dynamic_var_set[['GUPI','TILTimepoint']+physician_concerns_list],'uwComponent-Concern rmcorrs')
    # concern_uwcomponent_rmcorrs['Population'] = 'TIL'

    # # Calculate correlation between TIL scores and low-resolution neuromonitoring
    # lores_rmcorrs = calculate_rmcorr(lores_dynamic_var_set[['GUPI','TILTimepoint']+TIL_scores_list+TIL_components+uwTIL_components],lores_dynamic_var_set[['GUPI','TILTimepoint','CPP24EH', 'ICP24EH']],'Low-resolution rmcorrs')
    # lores_rmcorrs['Population'] = 'TIL-ICP_EH'

    # # Calculate correlation between TIL scores and high-resolution neuromonitoring
    # hires_rmcorrs = calculate_rmcorr(hires_dynamic_var_set[['GUPI','TILTimepoint']+TIL_scores_list+TIL_components+uwTIL_components],hires_dynamic_var_set[['GUPI','TILTimepoint','CPP24HR', 'ICP24HR']],'High-resolution rmcorrs')
    # hires_rmcorrs['Population'] = 'TIL-ICP_HR'

    # # Concatenate all repeated-measures correlation dataframes
    # compiled_rmcorrs = pd.concat([across_TIL_rmcorrs,TIL_concerns_rmcorrs,within_TIL_rmcorrs,within_uwTIL_rmcorrs,concern_component_rmcorrs,concern_uwcomponent_rmcorrs,lores_rmcorrs,hires_rmcorrs],ignore_index=True)
    # compiled_rmcorrs['resample_idx'] = curr_rs_idx

    # # Save concatenated dataframe
    # compiled_rmcorrs.to_pickle(os.path.join(bs_results_dir,'compiled_rmcorr_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # ## Calculate mixed-effects regression coefficients
    # # Regression model of ICP_EH on TIL scale sums
    # TIL_lores_mlm = calc_melm(lores_dynamic_var_set[['GUPI','TILTimepoint']+TIL_scores_list],lores_dynamic_var_set[['GUPI','TILTimepoint','CPP24EH', 'ICP24EH']],TIL_scores_list,False,uwTIL_components,'Calculating ICP_EH ~ TotalSumScores')

    # # Regression model of ICP_HR on TIL scale sums
    # TIL_hires_mlm = calc_melm(hires_dynamic_var_set[['GUPI','TILTimepoint']+TIL_scores_list],hires_dynamic_var_set[['GUPI','TILTimepoint','CPP24HR', 'ICP24HR']],TIL_scores_list,False,uwTIL_components,'Calculating ICP_HR ~ TotalSumScores')

    # # Regression model of ICP_EH on TIL scale components
    # component_lores_mlm = calc_melm(lores_dynamic_var_set[['GUPI','TILTimepoint','uwTILSum']+uwTIL_components],lores_dynamic_var_set[['GUPI','TILTimepoint','CPP24EH', 'ICP24EH']],['uwTILSum'],True,uwTIL_components,'Calculating ICP_EH ~ TILComponents')

    # # Regression model of ICP_HR on TIL scale components
    # component_hires_mlm = calc_melm(hires_dynamic_var_set[['GUPI','TILTimepoint','uwTILSum']+uwTIL_components],hires_dynamic_var_set[['GUPI','TILTimepoint','CPP24HR', 'ICP24HR']],['uwTILSum'],True,uwTIL_components,'Calculating ICP_HR ~ TILComponents')
    
    # # Concatenate all mlm model information
    # compiled_mlm_df = pd.concat([TIL_lores_mlm,TIL_hires_mlm,component_lores_mlm,component_hires_mlm],ignore_index=True)
    # compiled_mlm_df['resample_idx'] = curr_rs_idx

    # # Save concatenated dataframe
    # compiled_mlm_df.to_pickle(os.path.join(bs_results_dir,'compiled_mixed_effects_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    ## Calculate ROC curves for refractory intracranial hypertension detection
    # Designate TIL score maximum columns
    TIL_max_list = ['TILmax','TIL_Basicmax','uwTILmax','PILOTmax','TIL_1987max']

    # Designate TIL score median columns
    TIL_median_list = ['TILmedian','TIL_Basicmedian','uwTILmedian','PILOTmedian','TIL_1987median']

    # Calculate ROC curves associated with refractory intracranial hypertension detection
    compiled_ROC_df = calc_ROC(global_static_var_set,TIL_max_list+TIL_median_list,['RefractoryICP'],'ROC curves for refractory intracranial pressure detection').rename(columns={'Predictor':'Scale'})
    lores_ROC_df = calc_ROC(lores_static_var_set,['ICPmaxEH','ICPmedianEH','CPPminEH','CPPmedianEH'],['RefractoryICP'],'ROC curves for refractory intracranial pressure detection').rename(columns={'Predictor':'Scale'})
    hires_ROC_df = calc_ROC(hires_static_var_set,['ICPmaxHR','ICPmedianHR','CPPminHR','CPPmedianHR'],['RefractoryICP'],'ROC curves for refractory intracranial pressure detection').rename(columns={'Predictor':'Scale'})
    compiled_ROC_df = pd.concat([compiled_ROC_df,lores_ROC_df,hires_ROC_df],ignore_index=True)

    # Calculate AUC associated with each ROC and merge
    refract_ICP_AUCs = global_static_var_set[['GUPI','RefractoryICP']+TIL_max_list+TIL_median_list].melt(id_vars=['GUPI','RefractoryICP'],value_vars=TIL_max_list+TIL_median_list).groupby(['variable'],as_index=False).apply(lambda dfx: roc_auc_score(dfx.RefractoryICP,dfx['value'])).rename(columns={None:'AUC','variable':'Scale'})
    refract_lores_AUCs = lores_static_var_set[['GUPI','RefractoryICP']+['ICPmaxEH','ICPmedianEH','CPPminEH','CPPmedianEH']].melt(id_vars=['GUPI','RefractoryICP'],value_vars=['ICPmaxEH','ICPmedianEH','CPPminEH','CPPmedianEH']).groupby(['variable'],as_index=False).apply(lambda dfx: roc_auc_score(dfx.RefractoryICP,dfx['value'])).rename(columns={None:'AUC','variable':'Scale'})
    refract_hires_AUCs = lores_static_var_set[['GUPI','RefractoryICP']+['ICPmaxHR','ICPmedianHR','CPPminHR','CPPmedianHR']].melt(id_vars=['GUPI','RefractoryICP'],value_vars=['ICPmaxHR','ICPmedianHR','CPPminHR','CPPmedianHR']).groupby(['variable'],as_index=False).apply(lambda dfx: roc_auc_score(dfx.RefractoryICP,dfx['value'])).rename(columns={None:'AUC','variable':'Scale'})
    refract_ICP_AUCs = pd.concat([refract_ICP_AUCs,refract_lores_AUCs,refract_hires_AUCs],ignore_index=True)
    compiled_ROC_df = compiled_ROC_df.merge(refract_ICP_AUCs,how='left')
    
    # Add resampling index to ROC dataframe
    compiled_ROC_df['resample_idx'] = curr_rs_idx

    # Save concatenated dataframe
    compiled_ROC_df.to_pickle(os.path.join(bs_results_dir,'compiled_ROCs_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # ## Calculate mutual entropy between TIL and TIL_Basic
    # # Calculate mutual entropy between TILmax scores and TIL_Basicmax
    # max_mi = pd.DataFrame([_estimate_mi(global_static_var_set[TIL_max_list].values,global_static_var_set['TIL_Basicmax'].values,True,True)],columns=[nm.replace('max','') for nm in TIL_max_list])
    # max_mi.insert(0,'TILTimepoint','Max')
    # max_mi.insert(1,'METRIC','MutualInfo')
    # max_entropy = pd.DataFrame([global_static_var_set[TIL_max_list].apply(lambda x: stats.entropy(x.value_counts().values/x.count())).values],columns=[nm.replace('max','') for nm in TIL_max_list])
    # max_entropy.insert(0,'TILTimepoint','Max')
    # max_entropy.insert(1,'METRIC','Entropy')

    # # Calculate mutual information and entropy for daily TIL scores
    # s = global_dynamic_var_set.groupby('TILTimepoint',as_index=True).apply(lambda dfx: _estimate_mi(dfx[TIL_scores_list].values,dfx['TIL_Basic'].values,True,True)) 
    # daily_mi = pd.DataFrame.from_dict(dict(zip(s.index, s.values)),orient='index',columns=TIL_scores_list).rename(columns={'TotalSum':'TIL','uwTILSum':'uwTIL','PILOTSum':'PILOT','TIL_1987Sum':'TIL_1987'})
    # daily_mi.insert(0,'TILTimepoint',daily_mi.index.values)
    # daily_mi.insert(1,'METRIC','MutualInfo')
    # daily_entropy = global_dynamic_var_set.groupby('TILTimepoint',as_index=False).apply(lambda dfx: dfx[TIL_scores_list].apply(lambda x: stats.entropy(x.value_counts().values/x.count()))).rename(columns={'TotalSum':'TIL','uwTILSum':'uwTIL','PILOTSum':'PILOT','TIL_1987Sum':'TIL_1987'})
    # daily_entropy.insert(1,'METRIC','Entropy')

    # # Concatenate current daily/max mutual information and entropy to running list
    # compiled_MI_entropy_df = pd.concat([daily_mi,daily_entropy,max_mi,max_entropy],ignore_index=True)

    # # Add resampling index to mutual information dataframe
    # compiled_MI_entropy_df['resample_idx'] = curr_rs_idx

    # # Save concatenated dataframe
    # compiled_MI_entropy_df.to_pickle(os.path.join(bs_results_dir,'compiled_mutual_info_resample_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)