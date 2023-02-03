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