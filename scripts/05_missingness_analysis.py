#### Master Script 5: Calculate metrics of missingness for analysis of study population ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Prepare missingness reports in static measures
# III. Prepare missingness report in longitudinal measures

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
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

### II. Prepare missingness reports in static measures
## Load and prepare static measures used in analysis
# Load baseline demographic and functional outcome score dataframe
CENTER_TBI_demo_outcome = pd.read_csv('../formatted_data/formatted_outcome_and_demographics.csv',na_values = ["NA","NaN","NaT"," ", ""])

### III. Prepare missingness report in longitudinal measures
## Load and prepare longitudinal measures used in analysis
# Load and prepare study admission/discharge timestamps
CENTER_TBI_datetime = pd.read_csv('../timestamps/adm_disch_timestamps.csv')
CENTER_TBI_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_datetime['ICUDurationDays'] = CENTER_TBI_datetime['ICUDurationHours']/24
CENTER_TBI_datetime = CENTER_TBI_datetime.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']],how='left')
CENTER_TBI_datetime['FullSet'] = 1

# Load and prepare formatted TIL scores over time
formatted_TIL_scores = pd.read_csv('../formatted_data/formatted_TIL_scores.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_TIL_scores = formatted_TIL_scores[(formatted_TIL_scores.TILTimepoint<=7)&(formatted_TIL_scores.TILTimepoint>=1)].reset_index(drop=True)
formatted_TIL_scores = formatted_TIL_scores.merge(CENTER_TBI_demo_outcome[['GUPI','LowResolutionSet','HighResolutionSet']],how='left')
formatted_TIL_scores['FullSet'] = 1

# Format substudy assignment for TIL dataframe
formatted_TIL_scores = formatted_TIL_scores.melt(id_vars=['GUPI','TILTimepoint','TotalSum','TILPhysicianConcernsCPP','TILPhysicianConcernsICP'], value_vars=['LowResolutionSet','HighResolutionSet','FullSet'],var_name='Substudy')
formatted_TIL_scores = formatted_TIL_scores[formatted_TIL_scores['value']==1].drop(columns='value').reset_index(drop=True)

# Filter timestamp dataframe to patients in study set
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime.GUPI.isin(formatted_TIL_scores.GUPI)].reset_index(drop=True)
CENTER_TBI_datetime = CENTER_TBI_datetime.melt(id_vars=['GUPI','ICUDurationDays'], value_vars=['LowResolutionSet','HighResolutionSet','FullSet'],var_name='Substudy')
CENTER_TBI_datetime = CENTER_TBI_datetime[CENTER_TBI_datetime['value']==1].drop(columns='value').reset_index(drop=True)

# Load and prepare formatted low-resolution neuromonitoring values over time
formatted_low_resolution_values = pd.read_csv('../formatted_data/formatted_low_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_low_resolution_values = formatted_low_resolution_values[formatted_low_resolution_values.TILTimepoint<=7].reset_index(drop=True)

# Load and prepare formatted high-resolution neuromonitoring values over time
formatted_high_resolution_values = pd.read_csv('../formatted_data/formatted_high_resolution_values.csv',na_values = ["NA","NaN","NaT"," ", ""])
formatted_high_resolution_values = formatted_high_resolution_values[formatted_high_resolution_values.TILTimepoint<=7].reset_index(drop=True)

## Calculate differences in TIL and baseline characteristics in each longitudinal population gap
# Load TILmax and TILmean values
formatted_TIL_max = pd.read_csv('../formatted_data/formatted_TIL_max.csv')
formatted_TIL_mean = pd.read_csv('../formatted_data/formatted_TIL_mean.csv')

char_set = CENTER_TBI_demo_outcome.merge(formatted_TIL_max[['GUPI','TILmax']],how='left').merge(formatted_TIL_mean[['GUPI','TILmean']],how='left')
char_set = char_set[['GUPI', 'SiteCode', 'Age', 'Sex','GCSSeverity', 'GOSE6monthEndpointDerived','RefractoryICP','MarshallCT','Pr(GOSE>1)', 'Pr(GOSE>3)','Pr(GOSE>4)', 'Pr(GOSE>5)', 'Pr(GOSE>6)', 'Pr(GOSE>7)','TILmax','TILmean']]
data_set = formatted_TIL_scores[formatted_TIL_scores.Substudy=='LowResolutionSet'].reset_index(drop=True).merge(formatted_low_resolution_values[['GUPI','TILTimepoint','TotalSum','ICPmean']],how='left')
timepoints = [1,2,3,4,5,6,7]
chosen_cols = ['ICPmean','TILPhysicianConcernsICP']

a_1, a_2 = long_missingness_analysis(char_set,data_set,timepoints,chosen_cols)

char_set = CENTER_TBI_demo_outcome.merge(formatted_TIL_max[['GUPI','TILmax']],how='left').merge(formatted_TIL_mean[['GUPI','TILmean']],how='left')
char_set = char_set[['GUPI', 'SiteCode', 'Age', 'Sex','GCSSeverity', 'GOSE6monthEndpointDerived','RefractoryICP','MarshallCT','Pr(GOSE>1)', 'Pr(GOSE>3)','Pr(GOSE>4)', 'Pr(GOSE>5)', 'Pr(GOSE>6)', 'Pr(GOSE>7)','TILmax','TILmean']]
data_set = formatted_TIL_scores[formatted_TIL_scores.Substudy=='HighResolutionSet'].reset_index(drop=True).merge(formatted_high_resolution_values[['GUPI','TILTimepoint','TotalSum','ICPmean']],how='left')
timepoints = [1,2,3,4,5,6,7]
chosen_cols = ['ICPmean','TILPhysicianConcernsICP']

b_1, b_2 = long_missingness_analysis(char_set,data_set,timepoints,chosen_cols)

