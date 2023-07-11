# Import necessary libraries
library(tidyverse)
library(readxl)
library(plotly)
library(ggbeeswarm)
library(cowplot)
library(rvg)
library(svglite)
library(openxlsx)
library(gridExtra)
library(extrafont)

# Function to finish calculation/formatted of imputed dynamic variable set
prep.imp.dynamic.set <- function(curr.dynamic.set,TIL.component.names) {
  
  # Recalculate sum of TIL components
  post.dynamic.set <- curr.dynamic.set %>%
    mutate(TotalSum = rowSums(across(all_of(TIL.component.names))),
           uwPositioning = Positioning,
           uwSedation = case_when((Sedation==5)~3,
                                  (Sedation==2)~2,
                                  (Sedation==1)~1,
                                  T~0),
           uwNeuromuscular = case_when((Neuromuscular==3)~1,
                                       T~0),
           uwCSFDrainage = case_when((CSFDrainage==3)~2,
                                     (CSFDrainage==2)~1,
                                     T~0),
           uwFluidLoading = FluidLoading,
           uwVasopressor = Vasopressor,
           uwVentilation = case_when((Ventilation==4)~3,
                                     (Ventilation==2)~2,
                                     (Ventilation==1)~1,
                                     T~0),
           uwMannitol = case_when((Mannitol==3)~2,
                                  (Mannitol==2)~1,
                                  T~0),
           uwHypertonic = case_when((Hypertonic==3)~2,
                                    (Hypertonic==2)~1,
                                    T~0),
           uwTemperature = case_when((Temperature==5)~3,
                                     (Temperature==2)~2,
                                     (Temperature==1)~1,
                                     T~0),
           uwICPSurgery = case_when((ICPSurgery==4)~1,
                                    T~0),
           uwDecomCraniectomy = case_when((DecomCraniectomy==5)~1,
                                          T~0),
           TIL_Basic = case_when(((Sedation==5)|(Ventilation==4)|(Temperature==5)|(DecomCraniectomy==5)|(ICPSurgery==4))~4,
                                 ((Mannitol==3)|(Hypertonic==3)|(Ventilation==2)|(Temperature==2)|(CSFDrainage==3))~3,
                                 ((Sedation==2)|(Vasopressor==1)|(FluidLoading==1)|(Mannitol==2)|(Hypertonic==2)|(Ventilation==1)|(CSFDrainage==2))~2,
                                 ((Sedation==1)|(Positioning==1))~1,
                                 T~0),
           pilotPositioning = 0,
           pilotSedation = case_when((Sedation==5)~5,
                                  (Sedation==2)~1,
                                  (Sedation==1)~1,
                                  T~0),
           pilotNeuromuscular = case_when((Neuromuscular==3)~2,
                                       T~0),
           pilotCSFDrainage = case_when((CSFDrainage==3)~5,
                                     (CSFDrainage==2)~4,
                                     T~0),
           pilotFluidLoading = 0,
           pilotVasopressor = 2*Vasopressor,
           pilotVentilation = case_when((Ventilation==4)~4,
                                     (Ventilation==2)~2,
                                     (Ventilation==1)~1,
                                     T~0),
           pilotMannitol = case_when((Mannitol==3)~3,
                                  (Mannitol==2)~2,
                                  T~0),
           pilotHypertonic = case_when((Hypertonic==3)~3,
                                    (Hypertonic==2)~3,
                                    T~0),
           pilotTemperature = case_when((Temperature==5)~5,
                                     (Temperature==2)~3,
                                     (Temperature==1)~1,
                                     T~0),
           pilotICPSurgery = case_when((ICPSurgery==4)~4,
                                    T~0),
           pilotDecomCraniectomy = case_when((DecomCraniectomy==5)~5,
                                          T~0),
           oldtilPositioning = 0,
           oldtilSedation = case_when((Sedation==5)~4,
                                     (Sedation==2)~1,
                                     (Sedation==1)~1,
                                     T~0),
           oldtilNeuromuscular = case_when((Neuromuscular==3)~1,
                                          T~0),
           oldtilCSFDrainage = case_when((CSFDrainage==3)~2,
                                        (CSFDrainage==2)~1,
                                        T~0),
           oldtilFluidLoading = 0,
           oldtilVasopressor = 0,
           oldtilVentilation = case_when((Ventilation==4)~2,
                                        (Ventilation==2)~1,
                                        (Ventilation==1)~1,
                                        T~0),
           oldtilMannitol = case_when((Mannitol==3)~6,
                                     (Mannitol==2)~3,
                                     T~0),
           oldtilHypertonic = 0,
           oldtilTemperature = 0,
           oldtilICPSurgery = 0,
           oldtilDecomCraniectomy = 0) %>%
    mutate(uwTILSum = rowSums(across(starts_with('uw',ignore.case=F))),
           PILOTSum = rowSums(across(starts_with('pilot',ignore.case=F))),
           TIL_1987Sum = rowSums(across(starts_with('oldtil',ignore.case=F)))) %>%
    select(-starts_with('pilot',ignore.case=F),-starts_with('oldtil',ignore.case=F))
  
  # Calculate summarised dynamic variables
  summ.dynamic.set <- post.dynamic.set %>%
    select(GUPI,TILTimepoint,CPP24EH,ICP24EH,CPP24HR,ICP24HR,TotalSum,TIL_Basic,uwTILSum,PILOTSum,TIL_1987Sum) %>%
    pivot_longer(cols=-c(GUPI,TILTimepoint)) %>%
    group_by(GUPI,name) %>%
    summarise(mean = mean(value),
              median = median(value),
              min=min(value),
              max=max(value)) %>%
    mutate(name = str_replace(name,'Sum',''),
           name = str_replace(name,'Total','TIL')) %>%
    pivot_longer(cols=-c(GUPI,name),names_to = 'Metric') %>%
    mutate(FormattedVarName = case_when(str_detect(name,'24')~str_replace(name,'24',Metric),
                                        T~paste0(name,Metric))) %>%
    pivot_wider(id_cols = GUPI,names_from = c(FormattedVarName),values_from = value)
  
  # Return the processed dynamic set and the summarised dynamic variables
  return(list(pds=post.dynamic.set,sds=summ.dynamic.set))
}