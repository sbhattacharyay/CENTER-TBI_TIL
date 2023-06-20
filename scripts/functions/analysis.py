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
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
from scipy.special import logit
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.utils import resample
from sklearn.metrics import mutual_info_score, roc_curve, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.feature_selection._mutual_info import mutual_info_regression, _estimate_mi

# StatsModels libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Function to calculate Spearman's Rho and p-value per pandas dataframe group
def spearman_rho(x,column_name):
    d = {}
    curr_sr = stats.spearmanr(x['value'],x[column_name])
    d['rho'] = curr_sr.statistic
    d['p_val'] = curr_sr.pvalue
    d['count'] = x.shape[0]
    return pd.Series(d, index=['rho', 'p_val', 'count'])

# Define function to calculate marginal and conditional R2 from mixed effect linear models
def melm_R2(fitted_lmer):
    var_resid = fitted_lmer.scale
    var_random_effect = float(fitted_lmer.cov_re.iloc[0])
    var_fixed_effect = fitted_lmer.fittedvalues.var()
    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var
    return (marginal_r2,conditional_r2)

# Define function to load statistics
def load_statistics(info_df, progress_bar=True, progress_bar_desc=''):
    
    compiled_statistics = []
        
    if progress_bar:
        iterator = tqdm(range(info_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(info_df.shape[0])
        
    # Load each validation statistics file
    for curr_row in iterator:
        compiled_statistics.append(pd.read_pickle(info_df.FILE[curr_row]))      
    return pd.concat(compiled_statistics,ignore_index=True)

## Carefully recalculate TIL by component
def calculate_TILsum(mod_daily_TIL_info):
    # Positioning (max = 1)
    TIL_positioning = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILPosition','TILPositionNursedFlat']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_positioning = TIL_positioning[TIL_positioning['value']==1].drop_duplicates(ignore_index=True)
    TIL_positioning = TIL_positioning[['GUPI','TILTimepoint','TILDate']].drop_duplicates(ignore_index=True)
    TIL_positioning['Item'] = 'Positioning'
    TIL_positioning['Score'] = 1

    # Sedation (max = 8)
    TIL_sedation = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILSedation','TILSedationHigher','TILSedationMetabolic','TILSedationNeuromuscular']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_sedation = TIL_sedation[TIL_sedation['value']==1].drop_duplicates(ignore_index=True)
    TIL_sedation['Item'] = TIL_sedation['variable'].map({'TILSedation':'Sedation','TILSedationHigher':'Sedation','TILSedationMetabolic':'Sedation','TILSedationNeuromuscular':'Neuromuscular'})
    TIL_sedation['score'] = TIL_sedation['variable'].map({'TILSedation':1,'TILSedationHigher':2,'TILSedationMetabolic':5,'TILSedationNeuromuscular':3})
    TIL_sedation = TIL_sedation.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['score'].max().rename(columns={'score':'Score'})

    # CSF Drainage (max = 3)
    TIL_csf_drainage = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILCSFDrainage','TILFluidOutCSFDrain','TILCCSFDrainageVolume']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_csf_drainage = TIL_csf_drainage[(TIL_csf_drainage['value']!=0)&(~TIL_csf_drainage['value'].isna())].drop_duplicates(ignore_index=True)
    TIL_csf_drainage['score'] = ((TIL_csf_drainage['variable'] == 'TILCSFDrainage')|(TIL_csf_drainage['value']<120)).astype(int)*2
    TIL_csf_drainage['score'][TIL_csf_drainage['value']>=120] = 3
    TIL_csf_drainage['Item'] = 'CSFDrainage'
    TIL_csf_drainage = TIL_csf_drainage.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['score'].max().rename(columns={'score':'Score'})

    # CPP Management (max = 2)
    TIL_cpp_management = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILFluidLoading','TILFluidLoadingVasopressor']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_cpp_management = TIL_cpp_management[TIL_cpp_management['value']==1].drop_duplicates(ignore_index=True)
    TIL_cpp_management['Item'] = TIL_cpp_management['variable'].map({'TILFluidLoading':'FluidLoading','TILFluidLoadingVasopressor':'Vasopressor'})
    TIL_cpp_management['score'] = TIL_cpp_management['value'].astype(int)
    TIL_cpp_management = TIL_cpp_management.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['score'].max().rename(columns={'score':'Score'})

    # Ventilatory management (Max = 4)
    TIL_ventilation = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILHyperventilation','TILHyperventilationModerate','TILHyperventilationIntensive']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_ventilation = TIL_ventilation[TIL_ventilation['value']==1].drop_duplicates(ignore_index=True)
    TIL_ventilation['Item'] = 'Ventilation'
    TIL_ventilation['score'] = TIL_ventilation['variable'].map({'TILHyperventilation':1,'TILHyperventilationModerate':2,'TILHyperventilationIntensive':4})
    TIL_ventilation = TIL_ventilation.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['score'].max().rename(columns={'score':'Score'})

    # Hyperosmolar (max = 6)
    TIL_hyperosmolar = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILHyperosmolarThearpy','TILHyperosomolarTherapyMannitolGreater2g','TILMannitolDose','TILHyperosomolarTherapyHypertonicLow','TILHyperosomolarTherapyHigher','TILHypertonicSalineDose']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_hyperosmolar = TIL_hyperosmolar[(TIL_hyperosmolar['value']!=0)&(~TIL_hyperosmolar['value'].isna())].drop_duplicates(ignore_index=True)
    TIL_hyperosmolar['Item'] = TIL_hyperosmolar['variable'].map({'TILHyperosmolarThearpy':'Mannitol','TILHyperosomolarTherapyMannitolGreater2g':'Mannitol','TILMannitolDose':'Mannitol','TILHyperosomolarTherapyHypertonicLow':'Hypertonic','TILHyperosomolarTherapyHigher':'Hypertonic','TILHypertonicSalineDose':'Hypertonic'})
    TIL_hyperosmolar['score'] = TIL_hyperosmolar['variable'].map({'TILHyperosmolarThearpy':2,'TILHyperosomolarTherapyMannitolGreater2g':3,'TILMannitolDose':2,'TILHyperosomolarTherapyHypertonicLow':2,'TILHyperosomolarTherapyHigher':3,'TILHypertonicSalineDose':2})
    TIL_hyperosmolar = TIL_hyperosmolar.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['score'].max().rename(columns={'score':'Score'})

    # Temperature control (max = 5)
    TIL_temperature = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILFever','TILFeverMildHypothermia','TILFeverHypothermia']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_temperature = TIL_temperature[TIL_temperature['value']==1].drop_duplicates(ignore_index=True)
    TIL_temperature['Item'] = 'Temperature'
    TIL_temperature['score'] = TIL_temperature['variable'].map({'TILFever':1,'TILFeverMildHypothermia':2,'TILFeverHypothermia':5})
    TIL_temperature = TIL_temperature.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['score'].max().rename(columns={'score':'Score'})

    # Surgery (max = 9)
    TIL_surgery = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILICPSurgery','TILICPSurgeryDecomCranectomy']].melt(id_vars=['GUPI','TILTimepoint','TILDate'])
    TIL_surgery = TIL_surgery[TIL_surgery['value']==1].drop_duplicates(ignore_index=True)
    TIL_surgery['Item'] = TIL_surgery['variable'].map({'TILICPSurgery':'ICPSurgery','TILICPSurgeryDecomCranectomy':'DecomCraniectomy'})
    TIL_surgery['score'] = TIL_surgery['variable'].map({'TILICPSurgery':4,'TILICPSurgeryDecomCranectomy':5})
    TIL_surgery = TIL_surgery.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['score'].max().rename(columns={'score':'Score'})

    # Extract physician impressions
    TIL_physician_impressions = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','TILPhysicianSatICP','TILPhysicianConcernsICP','TILPhysicianConcernsCPP','TILReasonForChange','TILPhysicianOverallSatisfaction','TILPhysicianOverallSatisfactionSurvival']].melt(id_vars=['GUPI','TILTimepoint','TILDate'],var_name='Item',value_name='Score')
    TIL_physician_impressions = TIL_physician_impressions[(TIL_physician_impressions['Score']!=77)&(~TIL_physician_impressions['Score'].isna())].drop_duplicates(ignore_index=True)
    TIL_physician_impressions = TIL_physician_impressions.groupby(['GUPI','TILTimepoint','TILDate','Item'],as_index=False)['Score'].mean()

    # Combine TIL Item dataframes
    TIL_subscores = pd.concat([TIL_positioning,TIL_sedation,TIL_csf_drainage,TIL_cpp_management,TIL_ventilation,TIL_hyperosmolar,TIL_temperature,TIL_surgery],ignore_index=True)

    # Calculate TILsum and append to dataframe
    TIL_sums = TIL_subscores.groupby(['GUPI','TILTimepoint','TILDate'],as_index=False).Score.sum()
    TIL_sums['Item'] = 'TotalSum'
    TIL_subscores = pd.concat([TIL_subscores,TIL_sums],ignore_index=True).sort_values(by=['GUPI','TILTimepoint','Item'],ignore_index=False)

    # Widen dataframe
    TIL_subscores = pd.pivot_table(TIL_subscores, values = 'Score', index=['GUPI','TILTimepoint','TILDate'], columns = 'Item').reset_index().fillna(0)
    TIL_physician_impressions = pd.pivot_table(TIL_physician_impressions, values = 'Score', index=['GUPI','TILTimepoint','TILDate'], columns = 'Item').reset_index()

    # Merge dataframe to full set of TIL assessments and return
    fixed_daily_TIL_info = mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate','DailyTILCompleteStatus','ICUAdmTimeStamp','ICUDischTimeStamp']].drop_duplicates(ignore_index=True).merge(TIL_subscores,how='left').sort_values(by=['GUPI','TILTimepoint'],ignore_index=False).fillna(0).merge(TIL_physician_impressions,how='left')
    return(fixed_daily_TIL_info)

## Calculate TIL_1987 from unweighted TIL component dataframe
def calculate_TIL_1987(unweighted_daily_TIL_info):
    # Create new dataframe for TIL_1987 scores
    fixed_daily_TIL_1987_info = unweighted_daily_TIL_info[['GUPI', 'TILTimepoint', 'TILDate', 'DailyTILCompleteStatus','ICUAdmTimeStamp','ICUDischTimeStamp','TotalSum','TILPhysicianConcernsCPP', 'TILPhysicianConcernsICP','TILPhysicianOverallSatisfaction','TILPhysicianOverallSatisfactionSurvival', 'TILPhysicianSatICP','TILReasonForChange']]

    # Barbiturate and sedation administration
    fixed_daily_TIL_1987_info['Sedation'] = 3*(unweighted_daily_TIL_info.Sedation == 3).astype(int) + (unweighted_daily_TIL_info.Sedation != 0).astype(int)

    # Mannitol administration
    fixed_daily_TIL_1987_info['Mannitol'] = 3*unweighted_daily_TIL_info.Mannitol.astype(int)

    # Ventricular drainage
    fixed_daily_TIL_1987_info['Ventricular'] = unweighted_daily_TIL_info.CSFDrainage.astype(int)

    # Hyperventilation
    fixed_daily_TIL_1987_info['Hyperventilation'] = 0
    fixed_daily_TIL_1987_info['Hyperventilation'][unweighted_daily_TIL_info.Ventilation==3] = 2
    fixed_daily_TIL_1987_info['Hyperventilation'][unweighted_daily_TIL_info.Ventilation.isin([1,2])] = 1

    # Paralysis
    fixed_daily_TIL_1987_info['Paralysis'] = unweighted_daily_TIL_info.Neuromuscular.astype(int)

    # Calculate summed TIL_1987
    fixed_daily_TIL_1987_info['TIL_1987Sum'] = fixed_daily_TIL_1987_info.Mannitol + fixed_daily_TIL_1987_info.Ventricular + fixed_daily_TIL_1987_info.Hyperventilation + fixed_daily_TIL_1987_info.Paralysis + fixed_daily_TIL_1987_info.Sedation
    return(fixed_daily_TIL_1987_info)

## Calculate PILOT from unweighted TIL component dataframe
def calculate_PILOT(unweighted_daily_TIL_info):
    # Create new dataframe for PILOT scores
    fixed_daily_PILOT_info = unweighted_daily_TIL_info[['GUPI', 'TILTimepoint', 'TILDate', 'DailyTILCompleteStatus','ICUAdmTimeStamp','ICUDischTimeStamp','TotalSum','TILPhysicianConcernsCPP', 'TILPhysicianConcernsICP','TILPhysicianOverallSatisfaction','TILPhysicianOverallSatisfactionSurvival', 'TILPhysicianSatICP','TILReasonForChange']]

    # Fever treatment and hypothermia (max = 5)
    fixed_daily_PILOT_info['Temperature'] = (unweighted_daily_TIL_info.Temperature != 0).astype(int)
    fixed_daily_PILOT_info.Temperature[unweighted_daily_TIL_info.Temperature==2] = 3
    fixed_daily_PILOT_info.Temperature[unweighted_daily_TIL_info.Temperature==3] = 5

    # Sedation (max = 5)
    fixed_daily_PILOT_info['Sedation'] = (unweighted_daily_TIL_info.Sedation != 0).astype(int) + 4*(unweighted_daily_TIL_info.Sedation == 3).astype(int)

    # Neuromuscular blockade (max = 2)
    fixed_daily_PILOT_info['Neuromuscular'] = 2*unweighted_daily_TIL_info.Neuromuscular.astype(int)

    # Ventilation (max = 4)
    fixed_daily_PILOT_info['Ventilation'] = unweighted_daily_TIL_info.Ventilation
    fixed_daily_PILOT_info['Ventilation'][fixed_daily_PILOT_info.Ventilation==3] = 4

    # Mannitol (max = 3)
    fixed_daily_PILOT_info['Mannitol'] = unweighted_daily_TIL_info.Mannitol
    fixed_daily_PILOT_info.Mannitol[fixed_daily_PILOT_info.Mannitol!=0] = fixed_daily_PILOT_info.Mannitol[fixed_daily_PILOT_info.Mannitol!=0]+1

    # Hypertonic saline (max = 3)
    fixed_daily_PILOT_info['Hypertonic'] = 3*(unweighted_daily_TIL_info.Hypertonic != 0).astype(int)

    # CSF drainage (max = 5)
    fixed_daily_PILOT_info['CSFDrainage'] = unweighted_daily_TIL_info.CSFDrainage
    fixed_daily_PILOT_info.CSFDrainage[fixed_daily_PILOT_info.CSFDrainage!=0] = fixed_daily_PILOT_info.CSFDrainage[fixed_daily_PILOT_info.CSFDrainage!=0]+3

    # Hematoma evacuation (max = 4)
    fixed_daily_PILOT_info['ICPSurgery'] = 4*unweighted_daily_TIL_info.ICPSurgery.astype(int)

    # Decompressive craniectomy (max = 5)
    fixed_daily_PILOT_info['DecomCraniectomy'] = 5*unweighted_daily_TIL_info.DecomCraniectomy.astype(int)

    # Induced hypertension (max = 2)
    fixed_daily_PILOT_info['Vasopressor'] = 2*unweighted_daily_TIL_info.Vasopressor.astype(int)

    # Calculate summed PILOT
    fixed_daily_PILOT_info['PILOTSum'] = fixed_daily_PILOT_info.Temperature + fixed_daily_PILOT_info.Sedation + fixed_daily_PILOT_info.Neuromuscular + fixed_daily_PILOT_info.Ventilation + fixed_daily_PILOT_info.Mannitol + fixed_daily_PILOT_info.Hypertonic + fixed_daily_PILOT_info.CSFDrainage + fixed_daily_PILOT_info.ICPSurgery + fixed_daily_PILOT_info.DecomCraniectomy + fixed_daily_PILOT_info.Vasopressor
    return(fixed_daily_PILOT_info)

## Calculate TIL_Basic from unweighted TIL component dataframe
def calculate_TIL_Basic(unweighted_daily_TIL_info):
    # Create new dataframe for TIL_Basic scores
    fixed_daily_TIL_Basic_info = unweighted_daily_TIL_info[['GUPI', 'TILTimepoint', 'TILDate', 'DailyTILCompleteStatus','ICUAdmTimeStamp','ICUDischTimeStamp','TotalSum','TILPhysicianConcernsCPP', 'TILPhysicianConcernsICP','TILPhysicianOverallSatisfaction','TILPhysicianOverallSatisfactionSurvival', 'TILPhysicianSatICP','TILReasonForChange']]

    # Create new column for TIL_Basic
    fixed_daily_TIL_Basic_info['TIL_Basic'] = 0

    # Mark all TIL_Basic 1 instances
    fixed_daily_TIL_Basic_info.TIL_Basic[(unweighted_daily_TIL_info.Sedation==1)|(unweighted_daily_TIL_info.Positioning==1)] = 1

    # Mark all TIL_Basic 2 instances
    fixed_daily_TIL_Basic_info.TIL_Basic[(unweighted_daily_TIL_info.Sedation==2)|(unweighted_daily_TIL_info.Vasopressor==1)|(unweighted_daily_TIL_info.FluidLoading==1)|(unweighted_daily_TIL_info.Mannitol==1)|(unweighted_daily_TIL_info.Hypertonic==1)|(unweighted_daily_TIL_info.Ventilation==1)|(unweighted_daily_TIL_info.CSFDrainage==1)] = 2

    # Mark all TIL_Basic 3 instances
    fixed_daily_TIL_Basic_info.TIL_Basic[(unweighted_daily_TIL_info.Mannitol==2)|(unweighted_daily_TIL_info.Hypertonic==2)|(unweighted_daily_TIL_info.Ventilation==2)|(unweighted_daily_TIL_info.Temperature==2)|(unweighted_daily_TIL_info.CSFDrainage==2)] = 3

    # Mark all TIL_Basic 4 instances
    fixed_daily_TIL_Basic_info.TIL_Basic[(unweighted_daily_TIL_info.Sedation==3)|(unweighted_daily_TIL_info.Ventilation==3)|(unweighted_daily_TIL_info.Temperature==3)|((unweighted_daily_TIL_info.DecomCraniectomy==1)&(unweighted_daily_TIL_info.TILTimepoint>1))|(unweighted_daily_TIL_info.ICPSurgery==1)] = 4

    # Calculate summed TIL_Basic
    return(fixed_daily_TIL_Basic_info)

# Function to calculate Spearman's Rhos between 2 dataframes
def calculate_spearman_rhos(x,y,message):
    
    # Initialise lists to store correlation values and metadata for each pair
    col_1, col_2, rhos, pvals, counts = [], [], [], [], []

    # Find all unique pairs of columns for which to calculate correlation
    pairs = list(set(tuple(sorted(pair)) for pair in itertools.product(x.drop('GUPI', axis=1).columns.tolist(), y.drop('GUPI', axis=1).columns.tolist()) if pair[0] != pair[1]))

    # Iterate through column pairs:
    for curr_pair in tqdm(pairs,message):
        
        # Create a dataframe of non-missing rows common to both columns
        try:
            curr_pair_df = x[['GUPI',curr_pair[0]]].merge(y[['GUPI',curr_pair[1]]],how='inner').dropna().reset_index(drop=True)
        except:
            curr_pair_df = y[['GUPI',curr_pair[0]]].merge(x[['GUPI',curr_pair[1]]],how='inner').dropna().reset_index(drop=True)

        # Calculate Spearman's correlation
        curr_sr = stats.spearmanr(curr_pair_df[curr_pair[0]],curr_pair_df[curr_pair[1]])

        # Append information to running lists
        col_1.append(curr_pair[0])
        col_2.append(curr_pair[1])
        rhos.append(curr_sr.statistic)
        pvals.append(curr_sr.pvalue)
        counts.append(curr_pair_df.shape[0])

    # Construct dataframe from running lists
    corr_df = pd.DataFrame({'first':col_1,'second':col_2,'rho':rhos,'pval':pvals,'count':counts})

    # Return correlation datframe
    return(corr_df)

# Function to calculate repeated-measures correlation
def calculate_rmcorr(x,y,message):

    # Initialise lists to store correlation values and metadata for each pair
    col_1, col_2, rs, pvals, patients, counts = [], [], [], [], [], []

    # Find all unique pairs of columns for which to calculate correlation
    pairs = list(set(tuple(sorted(pair)) for pair in itertools.product(x.drop(['GUPI','TILTimepoint','TILDate'], axis=1).columns.tolist(), y.drop(['GUPI','TILTimepoint','TILDate'], axis=1).columns.tolist()) if pair[0] != pair[1]))

    # Iterate through column pairs:
    for curr_pair in tqdm(pairs,message):
        
        # Create a dataframe of non-missing rows common to both columns
        try:
            curr_pair_df = x[['GUPI','TILTimepoint','TILDate',curr_pair[0]]].merge(y[['GUPI','TILTimepoint','TILDate',curr_pair[1]]],how='inner').dropna().reset_index(drop=True)
        except:
            curr_pair_df = y[['GUPI','TILTimepoint','TILDate',curr_pair[0]]].merge(x[['GUPI','TILTimepoint','TILDate',curr_pair[1]]],how='inner').dropna().reset_index(drop=True)
        
        try:
            # Calculate repeated-measures correlation
            curr_rm_corr = pg.rm_corr(data=curr_pair_df, x=curr_pair[0], y=curr_pair[1], subject='GUPI')

            # Append information to running lists
            col_1.append(curr_pair[0])
            col_2.append(curr_pair[1])
            rs.append(curr_rm_corr.r[0])
            pvals.append(curr_rm_corr.pval[0])
            counts.append(curr_pair_df.shape[0])
            patients.append(curr_pair_df.GUPI.nunique())
            
        except:
            pass

    # Construct dataframe from running lists
    corr_df = pd.DataFrame({'first':col_1,'second':col_2,'rmcorr':rs,'pval':pvals,'count':counts,'patient_count':patients})

    # Return correlation datframe
    return(corr_df)

# Function to calculate mixed effect linear models regressed on TIL score and components
def calc_melm(x,y,total_name,component_tf,component_list,message):
    # Initialise lists to store running information dataframes
    mlm_outputs = []

    # Extract potential target names from second dataframe
    target_names = y.drop(columns=['GUPI','TILTimepoint','TILDate']).columns

    # Iterate through target names
    for curr_target in tqdm(target_names,message):
        # Define MELM regression formaulae
        total_sum_formula = curr_target+' ~ '+total_name

        # Create dataframe of non-missing rows common to both dataframes
        curr_common_df = x[['GUPI','TILTimepoint','TILDate']+component_list+[total_name]].merge(y[['GUPI','TILTimepoint','TILDate']+[curr_target]],how='inner').dropna(subset=[total_name,curr_target])

        # Regression on summed score
        total_score_mlm = smf.mixedlm(total_sum_formula, curr_common_df, groups=curr_common_df["GUPI"]).fit()
        
        # Create dataframe to store relevant information
        curr_total_df = pd.DataFrame({'Type':'TotalScore',
                                        'Formula':total_sum_formula,
                                        'Name':total_score_mlm.params.index.tolist(),
                                        'Coefficient':total_score_mlm.params.tolist(),
                                        'pvalues':total_score_mlm.pvalues.tolist(),
                                        'ResidVar':total_score_mlm.scale,
                                        'RandomEffectVar':float(total_score_mlm.cov_re.iloc[0]),
                                        'PredictedValueVar':total_score_mlm.predict(curr_common_df).var(),
                                        'FittedValueVar':total_score_mlm.fittedvalues.var(),
                                        'LogLikelihood':total_score_mlm.llf,
                                        'count':curr_common_df.shape[0],
                                        'patient_count':curr_common_df.GUPI.nunique()})
        
        # Append currently constructed dataframe to running list
        mlm_outputs.append(curr_total_df)

        if component_tf:
            # Define component formula
            component_formula = curr_target+' ~ '+' + '.join(['C('+var+',Treatment)' for var in component_list])

            # Regression on components
            component_score_mlm = smf.mixedlm(component_formula, curr_common_df, groups=curr_common_df["GUPI"]).fit()

            # Create dataframe to store relevant information
            curr_component_df = pd.DataFrame({'Type':'Component',
                                                'Formula':component_formula,
                                                'Name':component_score_mlm.params.index.tolist(),
                                                'Coefficient':component_score_mlm.params.tolist(),
                                                'pvalues':component_score_mlm.pvalues.tolist(),
                                                'ResidVar':component_score_mlm.scale,
                                                'RandomEffectVar':float(component_score_mlm.cov_re.iloc[0]),
                                                'PredictedValueVar':component_score_mlm.predict(curr_common_df).var(),
                                                'FittedValueVar':component_score_mlm.fittedvalues.var(),
                                                'LogLikelihood':component_score_mlm.llf,
                                                'count':curr_common_df.shape[0],
                                                'patient_count':curr_common_df.GUPI.nunique()})
            
            # Append currently constructed dataframe to running list
            mlm_outputs.append(curr_component_df)
        
    # Concatenate list of mlm outputs and return
    return(pd.concat(mlm_outputs,ignore_index=True))

# Function to calculate unique mixed effect models for TIL regressed on ICP and sodium
def calc_ICP_Na_melm(scale_ICP_Na_df,total_name):

    # Calculate current melm
    curr_mlm1 = smf.mixedlm(total_name+' ~ meanSodium + ICPmean', scale_ICP_Na_df, groups=scale_ICP_Na_df["GUPI"]).fit()
    curr_mlm2 = smf.mixedlm(total_name+' ~ meanSodium + ChangeInICP', scale_ICP_Na_df.dropna(subset=['meanSodium','ChangeInICP']), groups=scale_ICP_Na_df.dropna(subset=['meanSodium','ChangeInICP'])["GUPI"]).fit()
    curr_mlm3 = smf.mixedlm(total_name+' ~ ChangeInSodium + ICPmean', scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ICPmean']), groups=scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ICPmean'])["GUPI"]).fit()
    curr_mlm4 = smf.mixedlm(total_name+' ~ ChangeInSodium + ChangeInICP', scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ChangeInICP']), groups=scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ChangeInICP'])["GUPI"]).fit()

    # Compile model information into dataframes
    curr_mlm1_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':total_name+' ~ meanSodium + ICPmean',
                                 'Name':curr_mlm1.params.index.tolist(),
                                 'Coefficient':curr_mlm1.params.tolist(),
                                 'pvalues':curr_mlm1.pvalues.tolist(),
                                 'ResidVar':curr_mlm1.scale,
                                 'RandomEffectVar':float(curr_mlm1.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm1.predict(scale_ICP_Na_df).var(),
                                 'FittedValueVar':curr_mlm1.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm1.llf,
                                 'count':scale_ICP_Na_df.shape[0],
                                 'patient_count':scale_ICP_Na_df.GUPI.nunique()})
    
    curr_mlm2_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':total_name+' ~ meanSodium + ChangeInICP',
                                 'Name':curr_mlm2.params.index.tolist(),
                                 'Coefficient':curr_mlm2.params.tolist(),
                                 'pvalues':curr_mlm2.pvalues.tolist(),
                                 'ResidVar':curr_mlm2.scale,
                                 'RandomEffectVar':float(curr_mlm2.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm2.predict(scale_ICP_Na_df).var(),
                                 'FittedValueVar':curr_mlm2.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm2.llf,
                                 'count':scale_ICP_Na_df.dropna(subset=['meanSodium','ChangeInICP']).shape[0],
                                 'patient_count':scale_ICP_Na_df.dropna(subset=['meanSodium','ChangeInICP']).GUPI.nunique()})
    
    curr_mlm3_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':total_name+' ~ ChangeInSodium + ICPmean',
                                 'Name':curr_mlm3.params.index.tolist(),
                                 'Coefficient':curr_mlm3.params.tolist(),
                                 'pvalues':curr_mlm3.pvalues.tolist(),
                                 'ResidVar':curr_mlm3.scale,
                                 'RandomEffectVar':float(curr_mlm3.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm3.predict(scale_ICP_Na_df).var(),
                                 'FittedValueVar':curr_mlm3.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm3.llf,
                                 'count':scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ICPmean']).shape[0],
                                 'patient_count':scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ICPmean']).GUPI.nunique()})

    curr_mlm4_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':total_name+' ~ ChangeInSodium + ChangeInICP',
                                 'Name':curr_mlm4.params.index.tolist(),
                                 'Coefficient':curr_mlm4.params.tolist(),
                                 'pvalues':curr_mlm4.pvalues.tolist(),
                                 'ResidVar':curr_mlm4.scale,
                                 'RandomEffectVar':float(curr_mlm4.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm4.predict(scale_ICP_Na_df).var(),
                                 'FittedValueVar':curr_mlm4.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm4.llf,
                                 'count':scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ChangeInICP']).shape[0],
                                 'patient_count':scale_ICP_Na_df.dropna(subset=['ChangeInSodium','ChangeInICP']).GUPI.nunique()})
    
    # Return MELM information dataframe
    return(pd.concat([curr_mlm1_df,curr_mlm2_df,curr_mlm3_df,curr_mlm4_df],ignore_index=True))

# Function to calculate unique mixed effect models for sodium regressed on HTS and ICP
def calc_HTS_Na_melm(scale_HTS_Na_df):

    # Calculate current melm
    curr_mlm1 = smf.mixedlm('meanSodium ~ ICPmean + C(Hypertonic,Treatment)', scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ICPmean']), groups=scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ICPmean'])["GUPI"]).fit()
    curr_mlm2 = smf.mixedlm('meanSodium ~ ChangeInICP + C(Hypertonic,Treatment)', scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ChangeInICP']), groups=scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ChangeInICP'])["GUPI"]).fit()
    curr_mlm3 = smf.mixedlm('ChangeInSodium ~ ICPmean + C(Hypertonic,Treatment)', scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ICPmean']), groups=scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ICPmean'])["GUPI"]).fit()
    curr_mlm4 = smf.mixedlm('ChangeInSodium ~ ChangeInICP + C(Hypertonic,Treatment)', scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ChangeInICP']), groups=scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ChangeInICP'])["GUPI"]).fit()

    # Compile model information into dataframes
    curr_mlm1_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':'meanSodium ~ ICPmean + C(Hypertonic,Treatment)',
                                 'Name':curr_mlm1.params.index.tolist(),
                                 'Coefficient':curr_mlm1.params.tolist(),
                                 'pvalues':curr_mlm1.pvalues.tolist(),
                                 'ResidVar':curr_mlm1.scale,
                                 'RandomEffectVar':float(curr_mlm1.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm1.predict(scale_HTS_Na_df).var(),
                                 'FittedValueVar':curr_mlm1.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm1.llf,
                                 'count':scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ICPmean']).shape[0],
                                 'patient_count':scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ICPmean']).GUPI.nunique()})
    
    curr_mlm2_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':'meanSodium ~ ChangeInICP + C(Hypertonic,Treatment)',
                                 'Name':curr_mlm2.params.index.tolist(),
                                 'Coefficient':curr_mlm2.params.tolist(),
                                 'pvalues':curr_mlm2.pvalues.tolist(),
                                 'ResidVar':curr_mlm2.scale,
                                 'RandomEffectVar':float(curr_mlm2.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm2.predict(scale_HTS_Na_df).var(),
                                 'FittedValueVar':curr_mlm2.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm2.llf,
                                 'count':scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ChangeInICP']).shape[0],
                                 'patient_count':scale_HTS_Na_df.dropna(subset=['Hypertonic','meanSodium','ChangeInICP']).GUPI.nunique()})
    
    curr_mlm3_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':'ChangeInSodium ~ ICPmean + C(Hypertonic,Treatment)',
                                 'Name':curr_mlm3.params.index.tolist(),
                                 'Coefficient':curr_mlm3.params.tolist(),
                                 'pvalues':curr_mlm3.pvalues.tolist(),
                                 'ResidVar':curr_mlm3.scale,
                                 'RandomEffectVar':float(curr_mlm3.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm3.predict(scale_HTS_Na_df).var(),
                                 'FittedValueVar':curr_mlm3.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm3.llf,
                                 'count':scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ICPmean']).shape[0],
                                 'patient_count':scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ICPmean']).GUPI.nunique()})

    curr_mlm4_df = pd.DataFrame({'Type':'TotalScore',
                                 'Formula':'ChangeInSodium ~ ChangeInICP + C(Hypertonic,Treatment)',
                                 'Name':curr_mlm4.params.index.tolist(),
                                 'Coefficient':curr_mlm4.params.tolist(),
                                 'pvalues':curr_mlm4.pvalues.tolist(),
                                 'ResidVar':curr_mlm4.scale,
                                 'RandomEffectVar':float(curr_mlm4.cov_re.iloc[0]),
                                 'PredictedValueVar':curr_mlm4.predict(scale_HTS_Na_df).var(),
                                 'FittedValueVar':curr_mlm4.fittedvalues.var(),
                                 'LogLikelihood':curr_mlm4.llf,
                                 'count':scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ChangeInICP']).shape[0],
                                 'patient_count':scale_HTS_Na_df.dropna(subset=['Hypertonic','ChangeInSodium','ChangeInICP']).GUPI.nunique()})
    
    # Return MELM information dataframe
    return(pd.concat([curr_mlm1_df,curr_mlm2_df,curr_mlm3_df,curr_mlm4_df],ignore_index=True))

# Function to calculate mutual information and entropy between TIL scores and TILBasic
def calc_MI_entropy(curr_rs,scores_df,max_df,progress_bar=True,progress_bar_desc=''):

    compiled_MI_entropy_df = []
        
    if progress_bar:
        iterator = tqdm(curr_rs,desc=progress_bar_desc)
    else:
        iterator = curr_rs
        
    # Iterate through resamples
    for rs in iterator:

        # Filter each dataframe by current resample
        curr_rs_scores = scores_df[scores_df.GUPI.isin(rs)].reset_index(drop=True)
        curr_rs_max = max_df[max_df.GUPI.isin(rs)].reset_index(drop=True)

        # Calculate mutual information and entropy for daily TIL scores
        s = curr_rs_scores.groupby('TILTimepoint',as_index=True).apply(lambda dfx: _estimate_mi(dfx[['TotalSum','uwTILSum','PILOTSum','TIL_1987Sum','TIL_Basic']].values,dfx['TIL_Basic'].values,True,True)) 
        daily_mi = pd.DataFrame.from_dict(dict(zip(s.index, s.values)),orient='index',columns=['TotalSum','uwTILSum','PILOTSum','TIL_1987Sum','TIL_Basic'])
        daily_mi.insert(0,'TILTimepoint',daily_mi.index.values)
        daily_mi.insert(1,'METRIC','MutualInfo')
        daily_entropy = curr_rs_scores.groupby('TILTimepoint',as_index=False).apply(lambda dfx: dfx[['TotalSum','uwTILSum','PILOTSum','TIL_1987Sum','TIL_Basic']].apply(lambda x: stats.entropy(x.value_counts().values/x.count()))) 
        daily_entropy.insert(1,'METRIC','Entropy')

        # Calculate mutual information and entropy for max TIL scores
        max_mi = pd.DataFrame([_estimate_mi(curr_rs_max[['TILmax','uwTILmax','PILOTmax','TIL_1987max','TIL_Basicmax']].values,curr_rs_max['TIL_Basicmax'].values,True,True)],columns=['TotalSum','uwTILSum','PILOTSum','TIL_1987Sum','TIL_Basic'])
        max_mi.insert(0,'TILTimepoint','Max')
        max_mi.insert(1,'METRIC','MutualInfo')
        max_entropy = pd.DataFrame([curr_rs_max[['TILmax','uwTILmax','PILOTmax','TIL_1987max','TIL_Basicmax']].apply(lambda x: stats.entropy(x.value_counts().values/x.count())).values],columns=['TotalSum','uwTILSum','PILOTSum','TIL_1987Sum','TIL_Basic'])
        max_entropy.insert(0,'TILTimepoint','Max')
        max_entropy.insert(1,'METRIC','Entropy')

        # Append current daily/max mutual information and entropy to running list
        compiled_MI_entropy_df.append(pd.concat([daily_mi,daily_entropy,max_mi,max_entropy],ignore_index=True))

    # Return concatenated list of mutual information and entropy dataframes
    return(pd.concat(compiled_MI_entropy_df,ignore_index=True))

# Function to calculate AUC for each score-label combination
def calc_AUC(curr_rs,max_df,score_cols,label_cols,progress_bar=True,progress_bar_desc=''):

    calc_AUC_df = []
    
    if progress_bar:
        iterator = tqdm(curr_rs,desc=progress_bar_desc)
    else:
        iterator = curr_rs

    # Convert dataframe to long form
    long_max_df = max_df.melt(id_vars=['GUPI']+label_cols,value_vars=score_cols,var_name='Scale',value_name='ScaleScore')

    # Convert labels to long form
    long_max_df = long_max_df.melt(id_vars=['GUPI','Scale','ScaleScore'],value_vars=label_cols,var_name='Label',value_name='LabelScore')

    # Iterate through resamples
    for rs in iterator:

        # Filter each dataframe by current resample
        curr_rs_max = long_max_df[long_max_df.GUPI.isin(rs)].reset_index(drop=True)

        # Calculate AUC for every scale-label combination
        curr_AUCs = curr_rs_max.groupby(['Scale','Label'],as_index=False).apply(lambda dfx: roc_auc_score(dfx.LabelScore,dfx.ScaleScore)).rename(columns={None:'AUC'})

        # Append to running list
        calc_AUC_df.append(curr_AUCs)
    
    # Concatenate dataframe and return
    return(pd.concat(calc_AUC_df,ignore_index=True))

# Define function to calculate MCC and accuracy for daily-4-hourly concordance check
def calc_MCC_accuracy(curr_rs,delta_df,progress_bar=True,progress_bar_desc=''):

    compiled_MCC_accuracy_df = []
        
    if progress_bar:
        iterator = tqdm(curr_rs,desc=progress_bar_desc)
    else:
        iterator = curr_rs
        
    # Iterate through resamples
    for rs in iterator:

        # Filter each dataframe by current resample
        curr_rs_deltas = delta_df[(delta_df.GUPI.isin(rs))&(delta_df.Ambiguous==0)].reset_index(drop=True)

        # Calculate increase MCC and accuracy
        increase_mcc = matthews_corrcoef(curr_rs_deltas['DailyIncreaseTIL'],curr_rs_deltas['4HourlyIncreaseTIL'])
        increase_accuracy = accuracy_score(curr_rs_deltas['DailyIncreaseTIL'],curr_rs_deltas['4HourlyIncreaseTIL'])

        # Calculate decrease MCC and accuracy
        decrease_mcc = matthews_corrcoef(curr_rs_deltas['DailyDecreaseTIL'],curr_rs_deltas['4HourlyDecreaseTIL'])
        decrease_accuracy = accuracy_score(curr_rs_deltas['DailyDecreaseTIL'],curr_rs_deltas['4HourlyDecreaseTIL'])

        # Compile scores into a single dataframe
        curr_df = pd.DataFrame({'TILChange':['Increase','Increase','Decrease','Decrease'],'Metric':['MCC','Accuracy','MCC','Accuracy'],'Value':[increase_mcc,increase_accuracy,decrease_mcc,decrease_accuracy]})

        # Append current dataframe to running list
        compiled_MCC_accuracy_df.append(curr_df)

    # Return compiled dataframe
    return (pd.concat(compiled_MCC_accuracy_df,ignore_index=True))