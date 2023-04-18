#### Master Script 2a: Prepare study resamples for bootstrapping TIL-based statistics ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Draw resamples for bootstrapping

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

### II. Draw resamples for bootstrapping
## Full TIL validation population
# Load demographics and outcome information dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Create sub-directory for TIL population bootstrapping results
TIL_validation_dir = os.path.join(bs_dir,'TIL_validation')
os.makedirs(TIL_validation_dir,exist_ok=True)

# Extract population GUPIs
TIL_validation_GUPIs = CENTER_TBI_demo_outcome.GUPI.unique()

# Make resamples for bootstrapping metrics
TIL_validation_bs_rs_GUPIs = [resample(TIL_validation_GUPIs,replace=True,n_samples=len(TIL_validation_GUPIs)) for _ in range(NUM_RESAMP)]
TIL_validation_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in TIL_validation_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
TIL_validation_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':TIL_validation_bs_rs_GUPIs})

# Store TIL validation resamples
TIL_validation_bs_resamples.to_pickle(os.path.join(TIL_validation_dir,'TIL_validation_resamples.pkl'))

## TIL-ICP_EH validation population
# Create sub-directory for TIL population bootstrapping results
TIL_ICPEH_dir = os.path.join(bs_dir,'TIL_ICPEH')
os.makedirs(TIL_ICPEH_dir,exist_ok=True)

# Extract population GUPIs
TIL_ICPEH_GUPIs = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.LowResolutionSet==1].GUPI.unique()

# Make resamples for bootstrapping metrics
TIL_ICPEH_bs_rs_GUPIs = [resample(TIL_ICPEH_GUPIs,replace=True,n_samples=len(TIL_ICPEH_GUPIs)) for _ in range(NUM_RESAMP)]
TIL_ICPEH_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in TIL_ICPEH_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
TIL_ICPEH_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':TIL_ICPEH_bs_rs_GUPIs})

# Store TIL validation resamples
TIL_ICPEH_bs_resamples.to_pickle(os.path.join(TIL_ICPEH_dir,'TIL_ICPEH_resamples.pkl'))

## TIL-ICP_EH validation population
# Create sub-directory for TIL population bootstrapping results
TIL_ICPHR_dir = os.path.join(bs_dir,'TIL_ICPHR')
os.makedirs(TIL_ICPHR_dir,exist_ok=True)

# Extract population GUPIs
TIL_ICPHR_GUPIs = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.HighResolutionSet==1].GUPI.unique()

# Make resamples for bootstrapping metrics
TIL_ICPHR_bs_rs_GUPIs = [resample(TIL_ICPHR_GUPIs,replace=True,n_samples=len(TIL_ICPHR_GUPIs)) for _ in range(NUM_RESAMP)]
TIL_ICPHR_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in TIL_ICPHR_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
TIL_ICPHR_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':TIL_ICPHR_bs_rs_GUPIs})

# Store TIL validation resamples
TIL_ICPHR_bs_resamples.to_pickle(os.path.join(TIL_ICPHR_dir,'TIL_ICPHR_resamples.pkl'))