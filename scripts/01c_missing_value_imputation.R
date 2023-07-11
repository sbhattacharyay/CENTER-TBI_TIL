#### Master Script 01c: Perform multiple imputation of missing study values before analysis ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Perform multiple imputation of study measures

### I. Initialisation
## Import libraries and prepare environment
# Import necessary libraries
library(tidyverse)
library(readxl)
library(naniar)
library(mice)
library(Amelia)
library(doParallel)
library(foreach)

# Import custom plotting functions
source('functions/analysis.R')

# Create directory placeholders
formatted.data.dir <- '../formatted_data'
results.dir <- '../results'
imputed.dir <- file.path(formatted.data.dir,'imputed_sets')

# Create directory for storing imputed datasets
dir.create(imputed.dir, showWarnings = FALSE)

## Establish multiple imputation parameters
# Define number of imputations
NUM.IMPUTATIONS <- 100

# Set the number of parallel cores for imputation
NUM.CORES <- detectCores() - 2

### II. Perform multiple imputation of study measures
## Load and prepare study measure dataframe
# Load study set assignments
study.set.assignments <- read.csv('../formatted_data/formatted_outcome_and_demographics.csv',
                                  na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,LowResolutionSet,HighResolutionSet)
  
# Load formatted study demographic and outcome information
study.static.values <- read.csv('../formatted_data/formatted_outcome_and_demographics.csv',
                                na.strings = c("NA","NaN","", " ")) %>%
  select(-c(PatientType,starts_with('AssociatedStudy'),ends_with('Set'))) %>%
  mutate(across(-c('GUPI','Age',starts_with('Pr.GOSE.')),factor))

# Load TILmax and TILmedian values to append
TILmax.TILmedian.scores <- read.csv('../formatted_data/formatted_TIL_max.csv',
                                    na.strings = c("NA","NaN","", " ")) %>%
  filter(GUPI%in%study.static.values$GUPI) %>%
  select(GUPI,TILmax) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_median.csv',
                     na.strings = c("NA","NaN","", " ")) %>%
              select(GUPI,TILmedian))

# Load ICP/CPP summaries
ICP.CPP.summary.scores <- read.csv('../formatted_data/formatted_low_resolution_mins_maxes_medians_means.csv') %>%
  filter(GUPI%in%study.static.values$GUPI) %>%
  select(GUPI,ICPmax,ICPmedian,CPPmin,CPPmedian) %>%
  rename_with(~paste0(.x,'EH'), contains("CP")) %>%
  full_join(read.csv('../formatted_data/formatted_high_resolution_mins_maxes_medians_means.csv') %>%
              select(GUPI,ICPmax,ICPmedian,CPPmin,CPPmedian) %>%
              rename_with(~paste0(.x,'HR'), contains("CP")))

# Load ICP monitoring metadata
icp.monitored.patients <- read.csv('../CENTER-TBI/ICP_monitored_patients.csv',
                                   na.strings = c("NA","NaN","", " ")) %>%
  filter(GUPI%in%study.static.values$GUPI) %>%
  select(GUPI,ICUReasonICP,ICUProblemsICP,ICURaisedICP,ICPDevice,ICPMonitorStop,ICUCatheterICP,ICUReasonForTypeICPMontPare,ICPStopReason,ICUReasonForTypeICPMont,ICUProblemsICPYes)%>%
  mutate(across(-c('GUPI'),factor))

# Load formatted study TIL information over time
study.TIL.values <- read.csv('../formatted_data/formatted_TIL_scores.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  filter(GUPI %in% study.static.values$GUPI) %>%
  select(-c(TILDate,DailyTILCompleteStatus,ICUAdmTimeStamp,ICUDischTimeStamp)) %>%
  mutate(TILTimepoint = factor(TILTimepoint)) %>%
  .[which(rowSums(!is.na(.[,3:21])) != 0),]
TIL.component.names <- names(study.TIL.values)[!names(study.TIL.values) %in% c("GUPI","TILTimepoint","TotalSum","TILPhysicianConcernsCPP","TILPhysicianConcernsICP","TILPhysicianOverallSatisfaction","TILPhysicianOverallSatisfactionSurvival","TILPhysicianSatICP","TILReasonForChange")]
study.TIL.values <- study.TIL.values %>%
  pivot_wider(names_from="TILTimepoint",
              values_from=c("TotalSum","CSFDrainage","DecomCraniectomy","FluidLoading","Hypertonic","ICPSurgery","Mannitol","Neuromuscular","Positioning","Sedation","Temperature","Vasopressor","Ventilation","TILPhysicianConcernsCPP","TILPhysicianConcernsICP","TILPhysicianOverallSatisfaction","TILPhysicianOverallSatisfactionSurvival","TILPhysicianSatICP","TILReasonForChange"))

# Load low-resolution neuro-monitoring information
study.lores.values <- read.csv('../formatted_data/formatted_low_resolution_values.csv') %>%
  filter(GUPI %in% study.static.values$GUPI) %>%
  select(GUPI,TILTimepoint,CPPmean,ICPmean,nCPP,nICP,HVTILChangeReason,HourlyValueICPDiscontinued,HourlyValueLevelABP,HourlyValueLevelICP,ICPRemovedIndicator,ICPRevisedIndicator) %>%
  rename(CPP24EH=CPPmean,
         ICP24EH=ICPmean,
         nCPP24EH=nCPP,
         nICP24EH=nICP) %>%
  .[which(rowSums(!is.na(.[,3:10])) != 0),] %>%
  pivot_wider(names_from="TILTimepoint",
              values_from=c("CPP24EH","ICP24EH","nCPP24EH","nICP24EH","HVTILChangeReason","HourlyValueICPDiscontinued","HourlyValueLevelABP","HourlyValueLevelICP","ICPRemovedIndicator","ICPRevisedIndicator"))

# Load high-resolution neuro-monitoring information
study.hires.values <- read.csv('../formatted_data/formatted_high_resolution_values.csv') %>%
  filter(GUPI %in% study.static.values$GUPI) %>%
  select(GUPI,TILTimepoint,CPPmean,ICPmean,nCPP,nICP,EVD) %>%
  rename(CPP24HR=CPPmean,
         ICP24HR=ICPmean,
         nCPP24HR=nCPP,
         nICP24HR=nICP) %>%
  .[which(rowSums(!is.na(.[,3:7])) != 0),] %>%
  pivot_wider(names_from="TILTimepoint",
              values_from=c("CPP24HR","ICP24HR","nCPP24HR","nICP24HR","EVD"))

# Add extracted variables to static study measure dataframe
compiled.study.values <- study.static.values %>%
  left_join(TILmax.TILmedian.scores) %>%
  left_join(ICP.CPP.summary.scores) %>%
  left_join(icp.monitored.patients) %>%
  left_join(study.TIL.values) %>%
  left_join(study.lores.values) %>%
  left_join(study.hires.values)

# Create a list of static variables to impute
static.vars <- c("GCSScoreBaselineDerived", "GOSE6monthEndpointDerived","RefractoryICP","Pr.GOSE.1.","Pr.GOSE.3.","Pr.GOSE.4.","Pr.GOSE.5.","Pr.GOSE.6.","Pr.GOSE.7.")

# Create a cross of dynamic variables and timepoints 1 - 7
dynamic.vars <- expand.grid(c('TotalSum',TIL.component.names,"CPP24EH","ICP24EH","CPP24HR","ICP24HR","TILPhysicianConcernsCPP","TILPhysicianConcernsICP"), c(1,2,3,4,5,6,7)) %>%
  mutate(label = paste0(Var1,'_',Var2))
dynamic.vars <- dynamic.vars$label

# Extract study-plausible combinations of TIL
TIL.timepoint.combos <- read.csv('../formatted_data/formatted_high_resolution_values.csv') %>%
  filter(GUPI %in% study.static.values$GUPI,
         TILTimepoint<=7,TILTimepoint>=1) %>%
  select(GUPI,TILTimepoint) %>%
  unique()

# Format dataframe for multiple imputation function
compiled.study.values <- compiled.study.values %>%
  mutate(across(-c('GUPI','Age',starts_with('Pr.GOSE.'),starts_with('TILm'),ends_with('EH',ignore.case=F),ends_with('HR',ignore.case=F),contains('TotalSum_',ignore.case=F),contains('EH_',ignore.case=F),contains('HR_',ignore.case=F)),factor))

# Create "predictor" sets for each imputation variable
study.quick.pred <- quickpred(compiled.study.values,minpuc = .5,mincor = .2)
study.quick.pred[!row.names(study.quick.pred) %in% c(static.vars,dynamic.vars),] <- 0

# Initialize local cluster for parallel processing
registerDoParallel(cores = NUM.CORES)

# Iterate through cross-validation repeats
foreach(curr_imputation = (1:NUM.IMPUTATIONS), .inorder = F) %dopar% {
  
  # Create directory to store imputed static variables for current repeat
  curr.imp.dir <- file.path(imputed.dir,paste0('imp',sprintf('%03d',curr_imputation)))
  dir.create(curr.imp.dir,showWarnings = F)
  
  # Train multiple imputation object with defined predictor set
  mi.study.vars <- mice(data = compiled.study.values,
                        m = 1,
                        seed = curr_imputation,
                        maxit = 20,
                        pred=study.quick.pred,
                        method = 'pmm',
                        printFlag = TRUE)
  
  # Save multiple imputation object
  saveRDS(mi.study.vars,file.path(curr.imp.dir,'study_var_mice_object.rds'))
  
  # Extract current imputed variable set
  curr.imp <- complete(mi.study.vars, action = 1) %>%
    select(GUPI,c(static.vars,dynamic.vars))
  
  # Extract dynamic numeric variables from imputed dataframe
  curr.dynamic.num.set <- curr.imp %>%
    select(GUPI,dynamic.vars) %>%
    select(GUPI,where(is.numeric)) %>%
    pivot_longer(cols = where(is.numeric)) %>%
    mutate(Variable=sub("\\_.*", "",name),
           TILTimepoint=as.integer(sub('.+_(.+)', '\\1', name))) %>%
    pivot_wider(id_cols = c('GUPI','TILTimepoint'),names_from = 'Variable', values_from = 'value') %>%
    inner_join(TIL.timepoint.combos)
  
  # Extract dynamic categorical variables from imputed dataframe
  curr.dynamic.cat.set <- curr.imp %>%
    select(GUPI,dynamic.vars) %>%
    select(GUPI,!where(is.numeric)) %>%
    pivot_longer(cols = -GUPI) %>%
    mutate(Variable=sub("\\_.*", "",name),
           TILTimepoint=as.integer(sub('.+_(.+)', '\\1', name))) %>%
    pivot_wider(id_cols = c('GUPI','TILTimepoint'),names_from = 'Variable', values_from = 'value') %>%
    inner_join(TIL.timepoint.combos) %>%
    mutate(across(-c('GUPI','TILTimepoint'),factor),
           across(TIL.component.names, ~ as.integer(as.character(.x))))
  
  # Combine numeric and categorical dynamic variable sets
  curr.dynamic.set <- curr.dynamic.cat.set %>%
    left_join(curr.dynamic.num.set)
  
  # Process current dynamic set and extract summarised variables of dynamic set
  prep.output <- prep.imp.dynamic.set(curr.dynamic.set,TIL.component.names)
  
  # Extract finalised dynamic set dataframe, format, and save
  final.dynamic.set <- prep.output$pds %>%
    left_join(study.set.assignments) %>%
    relocate(GUPI,LowResolutionSet,HighResolutionSet) %>%
    arrange(GUPI,TILTimepoint)
  write.csv(final.dynamic.set,
            file.path(curr.imp.dir,'dynamic_var_set.csv'),
            row.names = F)
  
  # Extract summarised dynamic values
  summ.dynamic.values <- prep.output$sds

  # Extract static variables from imputed dataframe and add summarised dynamic values
  curr.static.set <- curr.imp %>%
    select(GUPI,static.vars) %>%
    left_join(summ.dynamic.values) %>%
    left_join(study.set.assignments) %>%
    relocate(GUPI,LowResolutionSet,HighResolutionSet) %>%
    arrange(GUPI)
  
  # Save current imputed static variable set
  write.csv(curr.static.set,
            file.path(curr.imp.dir,'static_var_set.csv'),
            row.names = F)
}