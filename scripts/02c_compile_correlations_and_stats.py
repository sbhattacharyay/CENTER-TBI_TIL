#### Master Script 2c: Compile TIL correlations and statistics of different study sub-samples ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save bootstrapped TIL correlations and statistics
# III. Calculate 95% confidence intervals

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
import scipy.stats
import numpy as np
import pandas as pd
import pickle as cp
from tqdm import tqdm
import seaborn as sns
import multiprocessing
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import spearmanr
warnings.filterwarnings(action="ignore")

# Custom methods
from functions.analysis import load_statistics

# Initialise directory for results
results_dir = '../results'

# Initialise directory for storing bootstrapping resamples
bs_dir = os.path.join(results_dir,'bootstrapping_results','resamples')

# Initalise subdirectory to store individual resample results
bs_results_dir = os.path.join(results_dir,'bootstrapping_results','results')

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count() - 2

# Set number of resamples for bootstrapping-based testing set performance
NUM_RESAMP = 1000

## II. Compile and save bootstrapped TIL correlations and statistics
# Search for all statistics files
stats_files = []
for path in Path(bs_results_dir).rglob('compiled_*.pkl'):
    stats_files.append(str(path.resolve()))

# Characterise the statistics files found
stats_file_info_df = pd.DataFrame({'FILE':stats_files,
                                   'METRIC':[re.search('/compiled_(.*)_resample_', curr_file).group(1) for curr_file in stats_files],
                                   'RESAMPLE_IDX':[int(re.search('_resample_(.*).pkl', curr_file).group(1)) for curr_file in stats_files],
                                  }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate dataframes by calculated metric
ROCs_info_df = stats_file_info_df[stats_file_info_df.METRIC == 'ROCs'].reset_index(drop=True)
mixed_effects_info_df = stats_file_info_df[stats_file_info_df.METRIC == 'mixed_effects'].reset_index(drop=True)
mutual_info_info_df = stats_file_info_df[stats_file_info_df.METRIC == 'mutual_info'].reset_index(drop=True)
rmcorr_info_df = stats_file_info_df[stats_file_info_df.METRIC == 'rmcorr'].reset_index(drop=True)
spearman_rhos_info_df = stats_file_info_df[stats_file_info_df.METRIC == 'spearman_rhos'].reset_index(drop=True)

# Partition stats files across available cores
s = [mixed_effects_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(mixed_effects_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(mixed_effects_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
mixed_effects_files_per_core = [(mixed_effects_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling mixed effect statistics') for idx in range(len(start_idx))]
rmcorr_files_per_core = [(rmcorr_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling repeated-measures correlation statistics') for idx in range(len(start_idx))]
spearman_rhos_files_per_core = [(spearman_rhos_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling Spearman rhos') for idx in range(len(start_idx))]
ROCs_files_per_core = [(ROCs_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling ROCs') for idx in range(len(start_idx))]
mutual_info_files_per_core = [(mutual_info_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling mutual info') for idx in range(len(start_idx))]

# Load statistics dataframes in parallel
# with multiprocessing.Pool(NUM_CORES) as pool:
#     compiled_mixed_effects = pd.concat(pool.starmap(load_statistics, mixed_effects_files_per_core),ignore_index=True)
# with multiprocessing.Pool(NUM_CORES) as pool:
#     compiled_rmcorr = pd.concat(pool.starmap(load_statistics, rmcorr_files_per_core),ignore_index=True)
# with multiprocessing.Pool(NUM_CORES) as pool:
#     compiled_spearman_rhos = pd.concat(pool.starmap(load_statistics, spearman_rhos_files_per_core),ignore_index=True)
# with multiprocessing.Pool(NUM_CORES) as pool:
#     compiled_ROCs = pd.concat(pool.starmap(load_statistics, ROCs_files_per_core),ignore_index=True)
# with multiprocessing.Pool(NUM_CORES) as pool:
#     compiled_mutual_info = pd.concat(pool.starmap(load_statistics, mutual_info_files_per_core),ignore_index=True)
compiled_ROCs = pd.concat([pd.read_pickle(f) for f in tqdm(ROCs_info_df.FILE,'Loading and compiling ROCs')],ignore_index=True)
compiled_mutual_info = pd.concat([pd.read_pickle(f) for f in tqdm(mutual_info_info_df.FILE,'Loading and compiling mutual_info')],ignore_index=True)
compiled_spearman_rhos = pd.concat([pd.read_pickle(f) for f in tqdm(spearman_rhos_info_df.FILE,'Loading and compiling spearman_rhos')],ignore_index=True)
compiled_rmcorr = pd.concat([pd.read_pickle(f) for f in tqdm(rmcorr_info_df.FILE,'Loading and compiling rmcorr')],ignore_index=True)
compiled_mixed_effects = pd.concat([pd.read_pickle(f) for f in tqdm(mixed_effects_info_df.FILE,'Loading and compiling mixed_effects')],ignore_index=True)

# Save compiled statistics
compiled_mixed_effects.to_csv(os.path.join(results_dir,'bootstrapping_results','compiled_mixed_effects_results.csv'),index=False)
compiled_rmcorr.to_csv(os.path.join(results_dir,'bootstrapping_results','compiled_rmcorr_results.csv'),index=False)
compiled_spearman_rhos.to_csv(os.path.join(results_dir,'bootstrapping_results','compiled_spearman_rhos_results.csv'),index=False)
compiled_ROCs.to_csv(os.path.join(results_dir,'bootstrapping_results','compiled_ROCs_results.csv'),index=False)
compiled_mutual_info.to_csv(os.path.join(results_dir,'bootstrapping_results','compiled_mutual_info_results.csv'),index=False)

### III. Calculate 95% confidence intervals
## Load compiled statistics
# Load mixed effects values
compiled_mixed_effects = pd.read_csv(os.path.join(results_dir,'bootstrapping_results','compiled_mixed_effects_results.csv'))

# Load repeated-measures correlation values
compiled_rmcorr = pd.read_csv(os.path.join(results_dir,'bootstrapping_results','compiled_rmcorr_results.csv'))

# Load Spearman's rho values
compiled_spearman_rhos = pd.read_csv(os.path.join(results_dir,'bootstrapping_results','compiled_spearman_rhos_results.csv'))

# Load ROC values
compiled_ROCs = pd.read_csv(os.path.join(results_dir,'bootstrapping_results','compiled_ROCs_results.csv'))

# Load mutual information values
compiled_mutual_info = pd.read_csv(os.path.join(results_dir,'bootstrapping_results','compiled_mutual_info_results.csv'))

## Calculate and format 95% confidence intervals
# Calculate and format 95% confidence intervals
CI_spearman_rhos = compiled_spearman_rhos.melt(id_vars=['first','second','Population','TILTimepoint','count','resample_idx'],var_name='metric').groupby(['Population','TILTimepoint','first','second','metric'],as_index=False)['value'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)
CI_mixed_effects = compiled_mixed_effects.melt(id_vars=['Type','Formula','Name','count','patient_count','resample_idx'],var_name='metric').drop_duplicates(ignore_index=True).groupby(['Type','Formula','Name','metric'],as_index=False)['value'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)
CI_rmcorr = compiled_rmcorr.melt(id_vars=['first','second','Population','count','patient_count','resample_idx'],var_name='metric').groupby(['Population','first','second','metric'],as_index=False)['value'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)
CI_ROCs = compiled_ROCs.melt(id_vars=['Scale','Threshold','resample_idx'],var_name='metric').groupby(['Scale','Threshold','metric'],as_index=False)['value'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)
CI_mutual_info = compiled_mutual_info.melt(id_vars=['TILTimepoint','METRIC','resample_idx'],var_name='Scale').groupby(['Scale','METRIC','TILTimepoint'],as_index=False)['value'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)

# Add formatting confidence interval 
CI_spearman_rhos['FormattedCI'] = CI_spearman_rhos['median'].round(2).astype(str)+' ('+CI_spearman_rhos.lo.round(2).astype(str)+'–'+CI_spearman_rhos.hi.round(2).astype(str)+')'
CI_mixed_effects['FormattedCI'] = CI_mixed_effects['median'].round(2).astype(str)+' ('+CI_mixed_effects.lo.round(2).astype(str)+'–'+CI_mixed_effects.hi.round(2).astype(str)+')'
CI_rmcorr['FormattedCI'] = CI_rmcorr['median'].round(2).astype(str)+' ('+CI_rmcorr.lo.round(2).astype(str)+'–'+CI_rmcorr.hi.round(2).astype(str)+')'
CI_ROCs['FormattedCI'] = CI_ROCs['median'].round(2).astype(str)+' ('+CI_ROCs.lo.round(2).astype(str)+'–'+CI_ROCs.hi.round(2).astype(str)+')'
CI_mutual_info['FormattedCI'] = CI_mutual_info['median'].round(2).astype(str)+' ('+CI_mutual_info.lo.round(2).astype(str)+'–'+CI_mutual_info.hi.round(2).astype(str)+')'

# Save formatted confidence intervals
CI_mixed_effects.to_csv(os.path.join(results_dir,'bootstrapping_results','CI_mixed_effects_results.csv'),index=False)
CI_spearman_rhos.to_csv(os.path.join(results_dir,'bootstrapping_results','CI_spearman_rhos_results.csv'),index=False)
CI_rmcorr.to_csv(os.path.join(results_dir,'bootstrapping_results','CI_rmcorr_results.csv'),index=False)
CI_ROCs.to_csv(os.path.join(results_dir,'bootstrapping_results','CI_ROCs_results.csv'),index=False)
CI_mutual_info.to_csv(os.path.join(results_dir,'bootstrapping_results','CI_mutual_info_results.csv'),index=False)