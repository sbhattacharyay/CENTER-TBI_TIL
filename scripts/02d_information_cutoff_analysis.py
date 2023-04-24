#### Master Script 2d: Perform analyses to quantify information content stored across TIL scales ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate Shannon's Entropy and mutual information between TILBasic and other TIL scores
# III. Calculate cutoffs for each TIL scale to designate refractory IC hypertension status
# IV. Calculate cutoffs for each TIL scale to map onto TILBasic
# V. Calculate detection capabilities of daily TIL for 4-hourly TIL

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
from sklearn.feature_selection import mutual_info_regression

### II. Calculate Shannon's Entropy and mutual information between TILBasic and other TIL scores
## Load different TIL scale dataframes
