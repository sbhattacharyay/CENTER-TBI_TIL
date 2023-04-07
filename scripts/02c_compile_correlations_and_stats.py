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

# Initialise directory for storing bootstrapping resamples
bs_dir = '../bootstrapping_results/resamples'

# Initalise subdirectory to store individual resample results
bs_results_dir = '../bootstrapping_results/results'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

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

# Separate mixed effects and spearman's rho dataframes
mixed_effects_info_df = stats_file_info_df[stats_file_info_df.METRIC == 'mixed_effects'].reset_index(drop=True)
spearman_rhos_info_df = stats_file_info_df[stats_file_info_df.METRIC == 'spearman_rhos'].reset_index(drop=True)

# Partition stats files across available cores
s = [mixed_effects_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(mixed_effects_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(mixed_effects_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
mixed_effects_files_per_core = [(mixed_effects_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling mixed effect statistics') for idx in range(len(start_idx))]
spearman_rhos_files_per_core = [(spearman_rhos_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling Spearman rhos') for idx in range(len(start_idx))]

# Load statistics dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_mixed_effects = pd.concat(pool.starmap(load_statistics, mixed_effects_files_per_core),ignore_index=True)
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_spearman_rhos = pd.concat(pool.starmap(load_statistics, spearman_rhos_files_per_core),ignore_index=True)
    
# Save compiled statistics
compiled_mixed_effects.to_csv('../bootstrapping_results/compiled_mixed_effects_results.csv',index=False)
compiled_spearman_rhos.to_csv('../bootstrapping_results/compiled_spearman_rhos_results.csv',index=False)

### III. Calculate 95% confidence intervals
## Load compiled statistics
# Load mixed effects values
compiled_mixed_effects = pd.read_csv('../bootstrapping_results/compiled_mixed_effects_results.csv')

# Load Spearman's rho values
compiled_spearman_rhos = pd.read_csv('../bootstrapping_results/compiled_spearman_rhos_results.csv')

## Caclulate and format 95% confidence intervals
# Caclulate and format 95% confidence intervals
CI_spearman_rhos = compiled_spearman_rhos.groupby(['population','first','second'],as_index=False)['rho'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)
CI_mixed_effects = compiled_mixed_effects.groupby(['population','first','second'],as_index=False)['TIL_coefficients'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'min':np.min,'max':np.max,'resamples':'count'}).reset_index(drop=True)

# Add formatting confidence interval 
CI_spearman_rhos['FormattedCI'] = CI_spearman_rhos['median'].round(2).astype(str)+' ('+CI_spearman_rhos.lo.round(2).astype(str)+'–'+CI_spearman_rhos.hi.round(2).astype(str)+')'
CI_mixed_effects['FormattedCI'] = CI_mixed_effects['median'].round(2).astype(str)+' ('+CI_mixed_effects.lo.round(2).astype(str)+'–'+CI_mixed_effects.hi.round(2).astype(str)+')'

# Save formatted confidence intervals
CI_mixed_effects.to_csv('../bootstrapping_results/CI_mixed_effects_results.csv',index=False)
CI_spearman_rhos.to_csv('../bootstrapping_results/CI_spearman_rhos_results.csv',index=False)