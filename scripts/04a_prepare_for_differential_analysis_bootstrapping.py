#### Master Script 4a: Prepare study resamples for bootstrapping TIL-based statistics ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Identify centres which use HTS
# III. Draw resamples for bootstrapping

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

# Initialise sub-directory for TIL population bootstrapping results
TIL_validation_dir = os.path.join(bs_dir,'TIL_validation')

### II. Identify centres which use HTS
# Load demographics and outcome information dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv')

# Load daily sodium values dataframe to determine HTS use status
sodium_TIL_dataframe = pd.read_csv('../formatted_data/formatted_daily_sodium_values.csv')

# Isolate list of patients belonging to HTS or non-HOT populations
ineligible_GUPIs = sodium_TIL_dataframe.GUPI[(sodium_TIL_dataframe.HTSPtInd==0)&(sodium_TIL_dataframe.MannitolPtInd==1)].unique()
eligible_centres = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.GUPI.isin(sodium_TIL_dataframe.GUPI[sodium_TIL_dataframe.HTSPtInd==1].unique())].SiteCode.unique()

# Add marker for HTS centres in the study population dataframe
CENTER_TBI_demo_outcome['InHTSSite'] = ((CENTER_TBI_demo_outcome.SiteCode.isin(eligible_centres)) & ~(CENTER_TBI_demo_outcome.GUPI.isin(ineligible_GUPIs))).astype(int)

### III. Draw resamples for bootstrapping
## TIL-Na+ validation population
# Create sub-directory for TIL population bootstrapping results
TIL_Na_dir = os.path.join(bs_dir,'TIL_Na')
os.makedirs(TIL_Na_dir,exist_ok=True)

# Extract population GUPIs
TIL_Na_GUPIs = CENTER_TBI_demo_outcome[CENTER_TBI_demo_outcome.InHTSSite==1].GUPI.unique()

# Make resamples for bootstrapping metrics
TIL_Na_bs_rs_GUPIs = [resample(TIL_Na_GUPIs,replace=True,n_samples=len(TIL_Na_GUPIs)) for _ in range(NUM_RESAMP)]
TIL_Na_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in TIL_Na_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
TIL_Na_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':TIL_Na_bs_rs_GUPIs})

# Store TIL validation resamples
TIL_Na_bs_resamples.to_pickle(os.path.join(TIL_Na_dir,'TIL_Na_resamples.pkl'))

## TIL-Na+-x-TIL-ICP_EH validation population
# Extract population GUPIs
TIL_Na_ICPEH_GUPIs = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.InHTSSite==1)&(CENTER_TBI_demo_outcome.LowResolutionSet==1)].GUPI.unique()

# Make resamples for bootstrapping metrics
TIL_Na_ICPEH_bs_rs_GUPIs = [resample(TIL_Na_ICPEH_GUPIs,replace=True,n_samples=len(TIL_Na_ICPEH_GUPIs)) for _ in range(NUM_RESAMP)]
TIL_Na_ICPEH_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in TIL_Na_ICPEH_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
TIL_Na_ICPEH_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':TIL_Na_ICPEH_bs_rs_GUPIs})

# Store TIL validation resamples
TIL_Na_ICPEH_bs_resamples.to_pickle(os.path.join(TIL_Na_dir,'TIL_Na_ICPEH_resamples.pkl'))

## TIL-Na+-x-TIL-ICP_HR validation population
# Extract population GUPIs
TIL_Na_ICPHR_GUPIs = CENTER_TBI_demo_outcome[(CENTER_TBI_demo_outcome.InHTSSite==1)&(CENTER_TBI_demo_outcome.HighResolutionSet==1)].GUPI.unique()

# Make resamples for bootstrapping metrics
TIL_Na_ICPHR_bs_rs_GUPIs = [resample(TIL_Na_ICPHR_GUPIs,replace=True,n_samples=len(TIL_Na_ICPHR_GUPIs)) for _ in range(NUM_RESAMP)]
TIL_Na_ICPHR_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in TIL_Na_ICPHR_bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
TIL_Na_ICPHR_bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':TIL_Na_ICPHR_bs_rs_GUPIs})

# Store TIL validation resamples
TIL_Na_ICPHR_bs_resamples.to_pickle(os.path.join(TIL_Na_dir,'TIL_Na_ICPHR_resamples.pkl'))