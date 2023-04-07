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
from tqdm import tqdm
import seaborn as sns
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
from functions.analysis import spearman_rho, melm_R2

# Initialise directory for storing bootstrapping resamples
bs_dir = '../bootstrapping_results/resamples'

# Initalise subdirectory to store individual resample results
bs_results_dir = '../bootstrapping_results/results'
os.makedirs(bs_results_dir,exist_ok=True)

### II. Calculate correlation and statistics based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):

    ## Group 1: manually-recorded neuromonitoring population
    # Create sub-directory for group 1
    group1_dir = os.path.join(bs_dir,'group1')

    # Load group 1 resamples
    group1_bs_resamples = pd.read_pickle(os.path.join(group1_dir,'group1_resamples.pkl'))

    # Extract current group 1 resamples
    curr_group1_resamples = group1_bs_resamples[group1_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]
    
    # Load current correlation dataframes
    lo_res_ICP_TIL_means = pd.read_pickle(os.path.join(group1_dir,'lo_res_ICP_mean_TIL_mean.pkl'))
    lo_res_ICP_TIL_maxes = pd.read_pickle(os.path.join(group1_dir,'lo_res_ICP_max_TIL_max.pkl'))
    lo_res_CPP_TIL_means = pd.read_pickle(os.path.join(group1_dir,'lo_res_CPP_mean_TIL_mean.pkl'))
    lo_res_CPP_TIL_maxes = pd.read_pickle(os.path.join(group1_dir,'lo_res_CPP_max_TIL_max.pkl'))
    lo_res_ICP_TIL_24 = pd.read_pickle(os.path.join(group1_dir,'lo_res_ICP_24_TIL_24.pkl'))
    lo_res_CPP_TIL_24 = pd.read_pickle(os.path.join(group1_dir,'lo_res_CPP_24_TIL_24.pkl'))

    # Filter current correlation dataframes
    lo_res_ICP_TIL_means = lo_res_ICP_TIL_means[lo_res_ICP_TIL_means.GUPI.isin(curr_group1_resamples)].reset_index(drop=True)
    lo_res_ICP_TIL_maxes = lo_res_ICP_TIL_maxes[lo_res_ICP_TIL_maxes.GUPI.isin(curr_group1_resamples)].reset_index(drop=True)
    lo_res_CPP_TIL_means = lo_res_CPP_TIL_means[lo_res_CPP_TIL_means.GUPI.isin(curr_group1_resamples)].reset_index(drop=True)
    lo_res_CPP_TIL_maxes = lo_res_CPP_TIL_maxes[lo_res_CPP_TIL_maxes.GUPI.isin(curr_group1_resamples)].reset_index(drop=True)
    lo_res_ICP_TIL_24 = lo_res_ICP_TIL_24[lo_res_ICP_TIL_24.GUPI.isin(curr_group1_resamples)].reset_index(drop=True)
    lo_res_CPP_TIL_24 = lo_res_CPP_TIL_24[lo_res_CPP_TIL_24.GUPI.isin(curr_group1_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group1_rhos = [spearmanr(lo_res_ICP_TIL_means.ICPmean,lo_res_ICP_TIL_means.TILmean).statistic,
     spearmanr(lo_res_ICP_TIL_maxes.ICPmax,lo_res_ICP_TIL_maxes.TILmax).statistic,
     spearmanr(lo_res_CPP_TIL_means.CPPmean,lo_res_CPP_TIL_means.TILmean).statistic,
     spearmanr(lo_res_CPP_TIL_maxes.CPPmax,lo_res_CPP_TIL_maxes.TILmax).statistic]
    group1_p_vals = [spearmanr(lo_res_ICP_TIL_means.ICPmean,lo_res_ICP_TIL_means.TILmean).pvalue,
     spearmanr(lo_res_ICP_TIL_maxes.ICPmax,lo_res_ICP_TIL_maxes.TILmax).pvalue,
     spearmanr(lo_res_CPP_TIL_means.CPPmean,lo_res_CPP_TIL_means.TILmean).pvalue,
     spearmanr(lo_res_CPP_TIL_maxes.CPPmax,lo_res_CPP_TIL_maxes.TILmax).pvalue]
    group1_firsts = ['TILmean','TILmax','TILmean','TILmax']
    group1_seconds = ['ICPmean','ICPmax','CPPmean','CPPmax']

    # Construct dataframe of Spearman Rho correlations
    group1_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'LowResolution','first':group1_firsts,'second':group1_seconds,'rho':group1_rhos,'pval':group1_p_vals})

    # Calculate mixed-effect model coefficients of dependent measures
    lo_res_mlm_ICP_TIL_24 = smf.mixedlm("ICPmean ~ TotalTIL", lo_res_ICP_TIL_24, groups=lo_res_ICP_TIL_24["GUPI"])
    lo_res_mlmf_ICP_TIL_24 = lo_res_mlm_ICP_TIL_24.fit()
    lo_res_mlm_CPP_TIL_24 = smf.mixedlm("CPPmean ~ TotalTIL", lo_res_CPP_TIL_24, groups=lo_res_CPP_TIL_24["GUPI"])
    lo_res_mlmf_CPP_TIL_24 = lo_res_mlm_CPP_TIL_24.fit()

    ## Group 2: high-resolution neuromonitoring population
    # Create sub-directory for group 2
    group2_dir = os.path.join(bs_dir,'group2')

    # Load group 2 resamples
    group2_bs_resamples = pd.read_pickle(os.path.join(group2_dir,'group2_resamples.pkl'))

    # Extract current group 2 resamples
    curr_group2_resamples = group2_bs_resamples[group2_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]
    
    # Load current correlation dataframes
    hi_res_ICP_TIL_means = pd.read_pickle(os.path.join(group2_dir,'hi_res_ICP_mean_TIL_mean.pkl'))
    hi_res_ICP_TIL_maxes = pd.read_pickle(os.path.join(group2_dir,'hi_res_ICP_max_TIL_max.pkl'))
    hi_res_CPP_TIL_means = pd.read_pickle(os.path.join(group2_dir,'hi_res_CPP_mean_TIL_mean.pkl'))
    hi_res_CPP_TIL_maxes = pd.read_pickle(os.path.join(group2_dir,'hi_res_CPP_max_TIL_max.pkl'))
    hi_res_ICP_TIL_24 = pd.read_pickle(os.path.join(group2_dir,'hi_res_ICP_24_TIL_24.pkl'))
    hi_res_CPP_TIL_24 = pd.read_pickle(os.path.join(group2_dir,'hi_res_CPP_24_TIL_24.pkl'))

    # Filter current correlation dataframes
    hi_res_ICP_TIL_means = hi_res_ICP_TIL_means[hi_res_ICP_TIL_means.GUPI.isin(curr_group2_resamples)].reset_index(drop=True)
    hi_res_ICP_TIL_maxes = hi_res_ICP_TIL_maxes[hi_res_ICP_TIL_maxes.GUPI.isin(curr_group2_resamples)].reset_index(drop=True)
    hi_res_CPP_TIL_means = hi_res_CPP_TIL_means[hi_res_CPP_TIL_means.GUPI.isin(curr_group2_resamples)].reset_index(drop=True)
    hi_res_CPP_TIL_maxes = hi_res_CPP_TIL_maxes[hi_res_CPP_TIL_maxes.GUPI.isin(curr_group2_resamples)].reset_index(drop=True)
    hi_res_ICP_TIL_24 = hi_res_ICP_TIL_24[hi_res_ICP_TIL_24.GUPI.isin(curr_group2_resamples)].reset_index(drop=True)
    hi_res_CPP_TIL_24 = hi_res_CPP_TIL_24[hi_res_CPP_TIL_24.GUPI.isin(curr_group2_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group2_rhos = [spearmanr(hi_res_ICP_TIL_means.ICPmean,hi_res_ICP_TIL_means.TILmean).statistic,
     spearmanr(hi_res_ICP_TIL_maxes.ICPmax,hi_res_ICP_TIL_maxes.TILmax).statistic,
     spearmanr(hi_res_CPP_TIL_means.CPPmean,hi_res_CPP_TIL_means.TILmean).statistic,
     spearmanr(hi_res_CPP_TIL_maxes.CPPmax,hi_res_CPP_TIL_maxes.TILmax).statistic]
    group2_p_vals = [spearmanr(hi_res_ICP_TIL_means.ICPmean,hi_res_ICP_TIL_means.TILmean).pvalue,
     spearmanr(hi_res_ICP_TIL_maxes.ICPmax,hi_res_ICP_TIL_maxes.TILmax).pvalue,
     spearmanr(hi_res_CPP_TIL_means.CPPmean,hi_res_CPP_TIL_means.TILmean).pvalue,
     spearmanr(hi_res_CPP_TIL_maxes.CPPmax,hi_res_CPP_TIL_maxes.TILmax).pvalue]
    group2_firsts = ['TILmean','TILmax','TILmean','TILmax']
    group2_seconds = ['ICPmean','ICPmax','CPPmean','CPPmax']

    # Construct dataframe of Spearman Rho correlations
    group2_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'HighResolution','first':group2_firsts,'second':group2_seconds,'rho':group2_rhos,'pval':group2_p_vals})

    # Calculate mixed-effect model coefficients of dependent measures
    hi_res_mlm_ICP_TIL_24 = smf.mixedlm("ICPmean ~ TotalTIL", hi_res_ICP_TIL_24, groups=hi_res_ICP_TIL_24["GUPI"])
    hi_res_mlmf_ICP_TIL_24 = hi_res_mlm_ICP_TIL_24.fit()
    hi_res_mlm_CPP_TIL_24 = smf.mixedlm("CPPmean ~ TotalTIL", hi_res_CPP_TIL_24, groups=hi_res_CPP_TIL_24["GUPI"])
    hi_res_mlmf_CPP_TIL_24 = hi_res_mlm_CPP_TIL_24.fit()

    ## Group 3: Prior study population
    # Create sub-directory for group 3
    group3_dir = os.path.join(bs_dir,'group3')

    # Load group 3 resamples
    group3_bs_resamples = pd.read_pickle(os.path.join(group3_dir,'group3_resamples.pkl'))

    # Extract current group 3 resamples
    curr_group3_resamples = group3_bs_resamples[group3_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]

    # Load current correlation dataframes
    prior_study_ICP_TIL_means = pd.read_pickle(os.path.join(group3_dir,'prior_study_ICP_mean_TIL_mean.pkl'))
    prior_study_ICP_TIL_maxes = pd.read_pickle(os.path.join(group3_dir,'prior_study_ICP_max_TIL_max.pkl'))
    prior_study_CPP_TIL_means = pd.read_pickle(os.path.join(group3_dir,'prior_study_CPP_mean_TIL_mean.pkl'))
    prior_study_CPP_TIL_maxes = pd.read_pickle(os.path.join(group3_dir,'prior_study_CPP_max_TIL_max.pkl'))
    prior_study_GCS_TIL_means = pd.read_pickle(os.path.join(group3_dir,'prior_study_GCS_TIL_mean.pkl'))
    prior_study_GCS_TIL_maxes = pd.read_pickle(os.path.join(group3_dir,'prior_study_GCS_TIL_max.pkl'))
    prior_study_GOS_TIL_means = pd.read_pickle(os.path.join(group3_dir,'prior_study_GOS_TIL_mean.pkl'))
    prior_study_GOS_TIL_maxes = pd.read_pickle(os.path.join(group3_dir,'prior_study_GOS_TIL_max.pkl'))
    prior_study_ICP_TIL_24 = pd.read_pickle(os.path.join(group3_dir,'prior_study_ICP_24_TIL_24.pkl'))
    prior_study_CPP_TIL_24 = pd.read_pickle(os.path.join(group3_dir,'prior_study_CPP_24_TIL_24.pkl'))

    # Filter current correlation dataframes
    prior_study_ICP_TIL_means = prior_study_ICP_TIL_means[prior_study_ICP_TIL_means.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_ICP_TIL_maxes = prior_study_ICP_TIL_maxes[prior_study_ICP_TIL_maxes.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_CPP_TIL_means = prior_study_CPP_TIL_means[prior_study_CPP_TIL_means.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_CPP_TIL_maxes = prior_study_CPP_TIL_maxes[prior_study_CPP_TIL_maxes.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_GCS_TIL_means = prior_study_GCS_TIL_means[prior_study_GCS_TIL_means.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_GCS_TIL_maxes = prior_study_GCS_TIL_maxes[prior_study_GCS_TIL_maxes.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_GOS_TIL_means = prior_study_GOS_TIL_means[prior_study_GOS_TIL_means.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_GOS_TIL_maxes = prior_study_GOS_TIL_maxes[prior_study_GOS_TIL_maxes.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_ICP_TIL_24 = prior_study_ICP_TIL_24[prior_study_ICP_TIL_24.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)
    prior_study_CPP_TIL_24 = prior_study_CPP_TIL_24[prior_study_CPP_TIL_24.GUPI.isin(curr_group3_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group3_rhos = [spearmanr(prior_study_ICP_TIL_means.ICPmean,prior_study_ICP_TIL_means.TILmean).statistic,
     spearmanr(prior_study_ICP_TIL_maxes.ICPmax,prior_study_ICP_TIL_maxes.TILmax).statistic,
     spearmanr(prior_study_CPP_TIL_means.CPPmean,prior_study_CPP_TIL_means.TILmean).statistic,
     spearmanr(prior_study_CPP_TIL_maxes.CPPmax,prior_study_CPP_TIL_maxes.TILmax).statistic,
     spearmanr(prior_study_GCS_TIL_means.GCSScoreBaselineDerived,prior_study_GCS_TIL_means.TILmean).statistic,
     spearmanr(prior_study_GCS_TIL_maxes.GCSScoreBaselineDerived,prior_study_GCS_TIL_maxes.TILmax).statistic,
     spearmanr(prior_study_GOS_TIL_means.GOS6monthEndpointDerived,prior_study_GOS_TIL_means.TILmean).statistic,
     spearmanr(prior_study_GOS_TIL_maxes.GOS6monthEndpointDerived,prior_study_GOS_TIL_maxes.TILmax).statistic]
    group3_p_vals = [spearmanr(prior_study_ICP_TIL_means.ICPmean,prior_study_ICP_TIL_means.TILmean).pvalue,
     spearmanr(prior_study_ICP_TIL_maxes.ICPmax,prior_study_ICP_TIL_maxes.TILmax).pvalue,
     spearmanr(prior_study_CPP_TIL_means.CPPmean,prior_study_CPP_TIL_means.TILmean).pvalue,
     spearmanr(prior_study_CPP_TIL_maxes.CPPmax,prior_study_CPP_TIL_maxes.TILmax).pvalue,
     spearmanr(prior_study_GCS_TIL_means.GCSScoreBaselineDerived,prior_study_GCS_TIL_means.TILmean).pvalue,
     spearmanr(prior_study_GCS_TIL_maxes.GCSScoreBaselineDerived,prior_study_GCS_TIL_maxes.TILmax).pvalue,
     spearmanr(prior_study_GOS_TIL_means.GOS6monthEndpointDerived,prior_study_GOS_TIL_means.TILmean).pvalue,
     spearmanr(prior_study_GOS_TIL_maxes.GOS6monthEndpointDerived,prior_study_GOS_TIL_maxes.TILmax).pvalue]
    group3_firsts = ['TILmean','TILmax','TILmean','TILmax','TILmean','TILmax','TILmean','TILmax']
    group3_seconds = ['ICPmean','ICPmax','CPPmean','CPPmax','GCS','GCS','GOS','GOS']

    # Construct dataframe of Spearman Rho correlations
    group3_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'PriorStudy','first':group3_firsts,'second':group3_seconds,'rho':group3_rhos,'pval':group3_p_vals})

    # Calculate mixed-effect model coefficients of dependent measures
    prior_study_ICP_TIL_24 = prior_study_ICP_TIL_24.dropna().reset_index(drop=True)
    prior_study_CPP_TIL_24 = prior_study_CPP_TIL_24.dropna().reset_index(drop=True)
    prior_study_mlm_ICP_TIL_24 = smf.mixedlm("ICPmean ~ TotalTIL", prior_study_ICP_TIL_24, groups=prior_study_ICP_TIL_24["GUPI"])
    prior_study_mlmf_ICP_TIL_24 = prior_study_mlm_ICP_TIL_24.fit()
    prior_study_mlm_CPP_TIL_24 = smf.mixedlm("CPPmean ~ TotalTIL", prior_study_CPP_TIL_24, groups=prior_study_CPP_TIL_24["GUPI"])
    prior_study_mlmf_CPP_TIL_24 = prior_study_mlm_CPP_TIL_24.fit()

    ## Group 4: Manually recorded neuromonitoring + sodium population
    # Create sub-directory for group 4
    group4_dir = os.path.join(bs_dir,'group4')

    # Load group 4 resamples
    group4_bs_resamples = pd.read_pickle(os.path.join(group4_dir,'group4_resamples.pkl'))

    # Extract current group 4 resamples
    curr_group4_resamples = group4_bs_resamples[group4_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]

    # Load current correlation dataframes
    lo_res_Sodium_TIL_means = pd.read_pickle(os.path.join(group4_dir,'lo_res_sodium_mean_TIL_mean.pkl'))
    lo_res_Sodium_TIL_maxes = pd.read_pickle(os.path.join(group4_dir,'lo_res_sodium_max_TIL_max.pkl'))
    lo_res_Sodium_TIL_24 = pd.read_pickle(os.path.join(group4_dir,'lo_res_sodium_24_TIL_24.pkl'))

    # Filter current correlation dataframes
    lo_res_Sodium_TIL_means = lo_res_Sodium_TIL_means[lo_res_Sodium_TIL_means.GUPI.isin(curr_group4_resamples)].reset_index(drop=True)
    lo_res_Sodium_TIL_maxes = lo_res_Sodium_TIL_maxes[lo_res_Sodium_TIL_maxes.GUPI.isin(curr_group4_resamples)].reset_index(drop=True)
    lo_res_Sodium_TIL_24 = lo_res_Sodium_TIL_24[lo_res_Sodium_TIL_24.GUPI.isin(curr_group4_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group4_rhos = [spearmanr(lo_res_Sodium_TIL_means.meanSodium,lo_res_Sodium_TIL_means.TILmean).statistic,
     spearmanr(lo_res_Sodium_TIL_maxes.maxSodium,lo_res_Sodium_TIL_maxes.TILmax).statistic]
    group4_p_vals = [spearmanr(lo_res_Sodium_TIL_means.meanSodium,lo_res_Sodium_TIL_means.TILmean).pvalue,
     spearmanr(lo_res_Sodium_TIL_maxes.maxSodium,lo_res_Sodium_TIL_maxes.TILmax).pvalue]
    group4_firsts = ['TILmean','TILmax']
    group4_seconds = ['NAmean','NAmax']

    # Construct dataframe of Spearman Rho correlations
    group4_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'LowResolution','first':group4_firsts,'second':group4_seconds,'rho':group4_rhos,'pval':group4_p_vals})

    # Calculate mixed-effect model coefficients of dependent measures
    lo_res_mlm_Sodium_TIL_24 = smf.mixedlm("meanSodium ~ TotalTIL", lo_res_Sodium_TIL_24, groups=lo_res_Sodium_TIL_24["GUPI"])
    lo_res_mlmf_Sodium_TIL_24 = lo_res_mlm_Sodium_TIL_24.fit()

    ## Group 5: High-resolution neuromonitoring + sodium population
    # Create sub-directory for group 5
    group5_dir = os.path.join(bs_dir,'group5')

    # Load group 5 resamples
    group5_bs_resamples = pd.read_pickle(os.path.join(group5_dir,'group5_resamples.pkl'))

    # Extract current group 5 resamples
    curr_group5_resamples = group5_bs_resamples[group5_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]

    # Load current correlation dataframes
    hi_res_Sodium_TIL_means = pd.read_pickle(os.path.join(group5_dir,'hi_res_sodium_mean_TIL_mean.pkl'))
    hi_res_Sodium_TIL_maxes = pd.read_pickle(os.path.join(group5_dir,'hi_res_sodium_max_TIL_max.pkl'))
    hi_res_Sodium_TIL_24 = pd.read_pickle(os.path.join(group5_dir,'hi_res_sodium_24_TIL_24.pkl'))

    # Filter current correlation dataframes
    hi_res_Sodium_TIL_means = hi_res_Sodium_TIL_means[hi_res_Sodium_TIL_means.GUPI.isin(curr_group5_resamples)].reset_index(drop=True)
    hi_res_Sodium_TIL_maxes = hi_res_Sodium_TIL_maxes[hi_res_Sodium_TIL_maxes.GUPI.isin(curr_group5_resamples)].reset_index(drop=True)
    hi_res_Sodium_TIL_24 = hi_res_Sodium_TIL_24[hi_res_Sodium_TIL_24.GUPI.isin(curr_group5_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group5_rhos = [spearmanr(hi_res_Sodium_TIL_means.meanSodium,hi_res_Sodium_TIL_means.TILmean).statistic,
     spearmanr(hi_res_Sodium_TIL_maxes.maxSodium,hi_res_Sodium_TIL_maxes.TILmax).statistic]
    group5_p_vals = [spearmanr(hi_res_Sodium_TIL_means.meanSodium,hi_res_Sodium_TIL_means.TILmean).pvalue,
     spearmanr(hi_res_Sodium_TIL_maxes.maxSodium,hi_res_Sodium_TIL_maxes.TILmax).pvalue]
    group5_firsts = ['TILmean','TILmax']
    group5_seconds = ['NAmean','NAmax']

    # Construct dataframe of Spearman Rho correlations
    group5_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'HighResolution','first':group5_firsts,'second':group5_seconds,'rho':group5_rhos,'pval':group5_p_vals})
    
    # Calculate mixed-effect model coefficients of dependent measures
    hi_res_mlm_Sodium_TIL_24 = smf.mixedlm("meanSodium ~ TotalTIL", hi_res_Sodium_TIL_24, groups=hi_res_Sodium_TIL_24["GUPI"])
    hi_res_mlmf_Sodium_TIL_24 = hi_res_mlm_Sodium_TIL_24.fit()

    ## Group 6: Manually-recorded neuromonitoring + GCS population
    # Create sub-directory for group 6
    group6_dir = os.path.join(bs_dir,'group6')

    # Load group 6 resamples
    group6_bs_resamples = pd.read_pickle(os.path.join(group6_dir,'group6_resamples.pkl'))

    # Extract current group 6 resamples
    curr_group6_resamples = group6_bs_resamples[group6_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]

    # Load current correlation dataframes
    lo_res_GCS_TIL_means = pd.read_pickle(os.path.join(group6_dir,'lo_res_GCS_TIL_mean.pkl'))
    lo_res_GCS_TIL_maxes = pd.read_pickle(os.path.join(group6_dir,'lo_res_GCS_TIL_max.pkl'))
    
    # Filter current correlation dataframes
    lo_res_GCS_TIL_means = lo_res_GCS_TIL_means[lo_res_GCS_TIL_means.GUPI.isin(curr_group6_resamples)].reset_index(drop=True)
    lo_res_GCS_TIL_maxes = lo_res_GCS_TIL_maxes[lo_res_GCS_TIL_maxes.GUPI.isin(curr_group6_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group6_rhos = [spearmanr(lo_res_GCS_TIL_means.GCSScoreBaselineDerived,lo_res_GCS_TIL_means.TILmean).statistic,
     spearmanr(lo_res_GCS_TIL_maxes.GCSScoreBaselineDerived,lo_res_GCS_TIL_maxes.TILmax).statistic]
    group6_p_vals = [spearmanr(lo_res_GCS_TIL_means.GCSScoreBaselineDerived,lo_res_GCS_TIL_means.TILmean).pvalue,
     spearmanr(lo_res_GCS_TIL_maxes.GCSScoreBaselineDerived,lo_res_GCS_TIL_maxes.TILmax).pvalue]
    group6_firsts = ['TILmean','TILmax']
    group6_seconds = ['GCS','GCS']

    # Construct dataframe of Spearman Rho correlations
    group6_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'LowResolution','first':group6_firsts,'second':group6_seconds,'rho':group6_rhos,'pval':group6_p_vals})

    ## Group 7: high-resolution neuromonitoring + GCS population
    # Create sub-directory for group 7
    group7_dir = os.path.join(bs_dir,'group7')

    # Load group 7 resamples
    group7_bs_resamples = pd.read_pickle(os.path.join(group7_dir,'group7_resamples.pkl'))

    # Extract current group 7 resamples
    curr_group7_resamples = group7_bs_resamples[group7_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]

    # Load current correlation dataframes
    hi_res_GCS_TIL_means = pd.read_pickle(os.path.join(group7_dir,'hi_res_GCS_TIL_mean.pkl'))
    hi_res_GCS_TIL_maxes = pd.read_pickle(os.path.join(group7_dir,'hi_res_GCS_TIL_max.pkl'))
    
    # Filter current correlation dataframes
    hi_res_GCS_TIL_means = hi_res_GCS_TIL_means[hi_res_GCS_TIL_means.GUPI.isin(curr_group7_resamples)].reset_index(drop=True)
    hi_res_GCS_TIL_maxes = hi_res_GCS_TIL_maxes[hi_res_GCS_TIL_maxes.GUPI.isin(curr_group7_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group7_rhos = [spearmanr(hi_res_GCS_TIL_means.GCSScoreBaselineDerived,hi_res_GCS_TIL_means.TILmean).statistic,
     spearmanr(hi_res_GCS_TIL_maxes.GCSScoreBaselineDerived,hi_res_GCS_TIL_maxes.TILmax).statistic]
    group7_p_vals = [spearmanr(hi_res_GCS_TIL_means.GCSScoreBaselineDerived,hi_res_GCS_TIL_means.TILmean).pvalue,
     spearmanr(hi_res_GCS_TIL_maxes.GCSScoreBaselineDerived,hi_res_GCS_TIL_maxes.TILmax).pvalue]
    group7_firsts = ['TILmean','TILmax']
    group7_seconds = ['GCS','GCS']

    # Construct dataframe of Spearman Rho correlations
    group7_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'HighResolution','first':group7_firsts,'second':group7_seconds,'rho':group7_rhos,'pval':group7_p_vals})

    ## Group 8: manually-recorded neuromonitoring + GOSE population
    # Create sub-directory for group 8
    group8_dir = os.path.join(bs_dir,'group8')

    # Load group 8 resamples
    group8_bs_resamples = pd.read_pickle(os.path.join(group8_dir,'group8_resamples.pkl'))

    # Extract current group 8 resamples
    curr_group8_resamples = group8_bs_resamples[group8_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]

    # Load current correlation dataframes
    lo_res_GOSE_TIL_means = pd.read_pickle(os.path.join(group8_dir,'lo_res_GOSE_TIL_mean.pkl'))
    lo_res_GOSE_TIL_maxes = pd.read_pickle(os.path.join(group8_dir,'lo_res_GOSE_TIL_max.pkl'))
    lo_res_prognosis_TIL_means = pd.read_pickle(os.path.join(group8_dir,'lo_res_prognosis_TIL_mean.pkl'))
    lo_res_prognosis_TIL_maxes = pd.read_pickle(os.path.join(group8_dir,'lo_res_prognosis_TIL_max.pkl'))
    
    # Filter current correlation dataframes
    lo_res_GOSE_TIL_means = lo_res_GOSE_TIL_means[lo_res_GOSE_TIL_means.GUPI.isin(curr_group8_resamples)].reset_index(drop=True)
    lo_res_GOSE_TIL_maxes = lo_res_GOSE_TIL_maxes[lo_res_GOSE_TIL_maxes.GUPI.isin(curr_group8_resamples)].reset_index(drop=True)
    lo_res_prognosis_TIL_means = lo_res_prognosis_TIL_means[lo_res_prognosis_TIL_means.GUPI.isin(curr_group8_resamples)].reset_index(drop=True)
    lo_res_prognosis_TIL_maxes = lo_res_prognosis_TIL_maxes[lo_res_prognosis_TIL_maxes.GUPI.isin(curr_group8_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group8_rhos = [spearmanr(lo_res_GOSE_TIL_means.GOSE6monthEndpointDerived,lo_res_GOSE_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_GOSE_TIL_means.GOS6monthEndpointDerived,lo_res_GOSE_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived,lo_res_GOSE_TIL_maxes.TILmax).statistic,
                   spearmanr(lo_res_GOSE_TIL_maxes.GOS6monthEndpointDerived,lo_res_GOSE_TIL_maxes.TILmax).statistic,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>1)'],lo_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>3)'],lo_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>4)'],lo_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>5)'],lo_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>6)'],lo_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>7)'],lo_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>1)'],lo_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>3)'],lo_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>4)'],lo_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>5)'],lo_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>6)'],lo_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>7)'],lo_res_prognosis_TIL_maxes.TILmax).statistic]
    group8_p_vals = [spearmanr(lo_res_GOSE_TIL_means.GOSE6monthEndpointDerived,lo_res_GOSE_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_GOSE_TIL_means.GOS6monthEndpointDerived,lo_res_GOSE_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived,lo_res_GOSE_TIL_maxes.TILmax).pvalue,
                   spearmanr(lo_res_GOSE_TIL_maxes.GOS6monthEndpointDerived,lo_res_GOSE_TIL_maxes.TILmax).pvalue,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>1)'],lo_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>3)'],lo_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>4)'],lo_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>5)'],lo_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>6)'],lo_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_prognosis_TIL_means['Pr(GOSE>7)'],lo_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>1)'],lo_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>3)'],lo_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>4)'],lo_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>5)'],lo_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>6)'],lo_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(lo_res_prognosis_TIL_maxes['Pr(GOSE>7)'],lo_res_prognosis_TIL_maxes.TILmax).pvalue]
    group8_firsts = ['TILmean','TILmean','TILmax','TILmax','TILmean','TILmean','TILmean','TILmean','TILmean','TILmean','TILmax','TILmax','TILmax','TILmax','TILmax','TILmax']
    group8_seconds = ['GOSE','GOS','GOSE','GOS','Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)','Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)']

    # Construct dataframe of Spearman Rho correlations
    group8_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'LowResolution','first':group8_firsts,'second':group8_seconds,'rho':group8_rhos,'pval':group8_p_vals})

    ## Group 9: high-resolution neuromonitoring + GOSE population
    # Create sub-directory for group 9
    group9_dir = os.path.join(bs_dir,'group9')

    # Load group 9 resamples
    group9_bs_resamples = pd.read_pickle(os.path.join(group9_dir,'group9_resamples.pkl'))

    # Extract current group 9 resamples
    curr_group9_resamples = group9_bs_resamples[group9_bs_resamples.RESAMPLE_IDX==(array_task_id+1)].GUPIs.values[0]

    # Load current correlation dataframes
    hi_res_GOSE_TIL_means = pd.read_pickle(os.path.join(group9_dir,'hi_res_GOSE_TIL_mean.pkl'))
    hi_res_GOSE_TIL_maxes = pd.read_pickle(os.path.join(group9_dir,'hi_res_GOSE_TIL_max.pkl'))
    hi_res_prognosis_TIL_means = pd.read_pickle(os.path.join(group9_dir,'hi_res_prognosis_TIL_mean.pkl'))
    hi_res_prognosis_TIL_maxes = pd.read_pickle(os.path.join(group9_dir,'hi_res_prognosis_TIL_max.pkl'))
    
    # Filter current correlation dataframes
    hi_res_GOSE_TIL_means = hi_res_GOSE_TIL_means[hi_res_GOSE_TIL_means.GUPI.isin(curr_group9_resamples)].reset_index(drop=True)
    hi_res_GOSE_TIL_maxes = hi_res_GOSE_TIL_maxes[hi_res_GOSE_TIL_maxes.GUPI.isin(curr_group9_resamples)].reset_index(drop=True)
    hi_res_prognosis_TIL_means = hi_res_prognosis_TIL_means[hi_res_prognosis_TIL_means.GUPI.isin(curr_group9_resamples)].reset_index(drop=True)
    hi_res_prognosis_TIL_maxes = hi_res_prognosis_TIL_maxes[hi_res_prognosis_TIL_maxes.GUPI.isin(curr_group9_resamples)].reset_index(drop=True)

    # Calculate Spearman Rho correlation of independent measures
    group9_rhos = [spearmanr(hi_res_GOSE_TIL_means.GOSE6monthEndpointDerived,hi_res_GOSE_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_GOSE_TIL_means.GOS6monthEndpointDerived,hi_res_GOSE_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived,hi_res_GOSE_TIL_maxes.TILmax).statistic,
                   spearmanr(hi_res_GOSE_TIL_maxes.GOS6monthEndpointDerived,hi_res_GOSE_TIL_maxes.TILmax).statistic,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>1)'],hi_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>3)'],hi_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>4)'],hi_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>5)'],hi_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>6)'],hi_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>7)'],hi_res_prognosis_TIL_means.TILmean).statistic,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>1)'],hi_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>3)'],hi_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>4)'],hi_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>5)'],hi_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>6)'],hi_res_prognosis_TIL_maxes.TILmax).statistic,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>7)'],hi_res_prognosis_TIL_maxes.TILmax).statistic]
    group9_p_vals = [spearmanr(hi_res_GOSE_TIL_means.GOSE6monthEndpointDerived,hi_res_GOSE_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_GOSE_TIL_means.GOS6monthEndpointDerived,hi_res_GOSE_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_GOSE_TIL_maxes.GOSE6monthEndpointDerived,hi_res_GOSE_TIL_maxes.TILmax).pvalue,
                   spearmanr(hi_res_GOSE_TIL_maxes.GOS6monthEndpointDerived,hi_res_GOSE_TIL_maxes.TILmax).pvalue,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>1)'],hi_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>3)'],hi_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>4)'],hi_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>5)'],hi_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>6)'],hi_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_prognosis_TIL_means['Pr(GOSE>7)'],hi_res_prognosis_TIL_means.TILmean).pvalue,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>1)'],hi_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>3)'],hi_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>4)'],hi_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>5)'],hi_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>6)'],hi_res_prognosis_TIL_maxes.TILmax).pvalue,
                   spearmanr(hi_res_prognosis_TIL_maxes['Pr(GOSE>7)'],hi_res_prognosis_TIL_maxes.TILmax).pvalue]
    group9_firsts = ['TILmean','TILmean','TILmax','TILmax','TILmean','TILmean','TILmean','TILmean','TILmean','TILmean','TILmax','TILmax','TILmax','TILmax','TILmax','TILmax']
    group9_seconds = ['GOSE','GOS','GOSE','GOS','Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)','Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)']

    # Construct dataframe of Spearman Rho correlations
    group9_spearmans = pd.DataFrame({'resample_idx':(array_task_id+1),'population':'HighResolution','first':group9_firsts,'second':group9_seconds,'rho':group9_rhos,'pval':group9_p_vals})

    ## Compile results and save
    # Compiled Spearman's Rho results
    compiled_spearmans = pd.concat([group1_spearmans,group2_spearmans,group3_spearmans,group4_spearmans,group5_spearmans,group6_spearmans,group7_spearmans,group8_spearmans,group9_spearmans],ignore_index=True)

    # Compile mixed-effect model parameters into single dataframe
    melm_firsts = ['TIL24','TIL24','TIL24','TIL24','TIL24','TIL24','TIL24','TIL24']
    melm_seconds = ['ICP24','CPP24','ICP24','CPP24','ICP24','CPP24','NA24','NA24']
    melm_populations = ['LowResolution','LowResolution','HighResolution','HighResolution','PriorStudy','PriorStudy','LowResolution','HighResolution']
    melm_intercepts = [lo_res_mlmf_ICP_TIL_24.params.Intercept,
                       lo_res_mlmf_CPP_TIL_24.params.Intercept,
                       hi_res_mlmf_ICP_TIL_24.params.Intercept,
                       hi_res_mlmf_CPP_TIL_24.params.Intercept,
                       prior_study_mlmf_ICP_TIL_24.params.Intercept,
                       prior_study_mlmf_CPP_TIL_24.params.Intercept,
                       lo_res_mlmf_Sodium_TIL_24.params.Intercept,
                       hi_res_mlmf_Sodium_TIL_24.params.Intercept]
    melm_TIL_coeffs = [lo_res_mlmf_ICP_TIL_24.params.TotalTIL,
                       lo_res_mlmf_CPP_TIL_24.params.TotalTIL,
                       hi_res_mlmf_ICP_TIL_24.params.TotalTIL,
                       hi_res_mlmf_CPP_TIL_24.params.TotalTIL,
                       prior_study_mlmf_ICP_TIL_24.params.TotalTIL,
                       prior_study_mlmf_CPP_TIL_24.params.TotalTIL,
                       lo_res_mlmf_Sodium_TIL_24.params.TotalTIL,
                       hi_res_mlmf_Sodium_TIL_24.params.TotalTIL]
    melm_RE_variance = [lo_res_mlmf_ICP_TIL_24.cov_re.values[0][0],
                       lo_res_mlmf_CPP_TIL_24.cov_re.values[0][0],
                       hi_res_mlmf_ICP_TIL_24.cov_re.values[0][0],
                       hi_res_mlmf_CPP_TIL_24.cov_re.values[0][0],
                       prior_study_mlmf_ICP_TIL_24.cov_re.values[0][0],
                       prior_study_mlmf_CPP_TIL_24.cov_re.values[0][0],
                       lo_res_mlmf_Sodium_TIL_24.cov_re.values[0][0],
                       hi_res_mlmf_Sodium_TIL_24.cov_re.values[0][0]]
    melm_resid_variance = [lo_res_mlmf_ICP_TIL_24.resid.var(),
                           lo_res_mlmf_CPP_TIL_24.resid.var(),
                           hi_res_mlmf_ICP_TIL_24.resid.var(),
                           hi_res_mlmf_CPP_TIL_24.resid.var(),
                           prior_study_mlmf_ICP_TIL_24.resid.var(),
                           prior_study_mlmf_CPP_TIL_24.resid.var(),
                           lo_res_mlmf_Sodium_TIL_24.resid.var(),
                           hi_res_mlmf_Sodium_TIL_24.resid.var()]
    compiled_mixed_effects = pd.DataFrame({'resample_idx':(array_task_id+1),
                                           'population':melm_populations,
                                           'first':melm_firsts,
                                           'second':melm_seconds,
                                           'TIL_coefficients':melm_TIL_coeffs,
                                           'TIL_intercepts':melm_intercepts,
                                           'RE_variance':melm_RE_variance,
                                           'Residual_variance':melm_resid_variance})
    
    # Save compiled Spearman's Rho results
    compiled_spearmans.to_pickle(os.path.join(bs_results_dir,'compiled_spearman_rhos_resample_'+str(array_task_id+1).zfill(4)+'.pkl'))

    # Save compiled mixed-effect model parameters
    compiled_mixed_effects.to_pickle(os.path.join(bs_results_dir,'compiled_mixed_effects_resample_'+str(array_task_id+1).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)