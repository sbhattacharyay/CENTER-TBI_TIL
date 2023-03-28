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
from ast import literal_eval
from scipy.special import logit
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# Function to calculate Spearman's Rho and p-value per pandas dataframe group
def spearman_rho(x,column_name):
    d = {}
    curr_sr = stats.spearmanr(x['value'],x[column_name])
    d['rho'] = curr_sr.statistic
    d['p_val'] = curr_sr.pvalue
    d['count'] = x.shape[0]
    return pd.Series(d, index=['rho', 'p_val', 'count'])

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