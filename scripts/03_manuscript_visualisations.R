#### Master Script 03: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Figure 2: Distributions of TIL and alternative scales.
# III. Figure 3: Associations of TIL and alternative scales with other clinical measures.
# IV. Figure 4: Distributions of daily intracranial pressure and cerebral perfusion pressure means per daily TIL score.
# V. Figure 5: Discrimination of refractory intracranial hypertension status by TILmax and alternative scale maximum scores.
# VI. Figure 6: Association of TIL component items with TIL24 and other study measures.
# VII. Figure 7: Relationship between TIL and TIL(Basic).
# VIII. Supplementary Figure S1: Missingness of static study measures
# IX. Supplementary Figure S2: Missingness of longitudinal study measures
# X. Supplementary Figure S3: Correlation matrices between total scores of TIL and alternative scales.
# XI. Supplementary Figure S4: Inter-item correlation matrices for daily scores of TIL and alternative scales.
# XII. Table 3: Optimised ranges for TIL categorisation
# XIII. Supplementary Table 4: TIL thresholds for detection of day of surgery
# XIV. Supplementary Figure S5: Internal reliability of TIL and uwTIL

### I. Initialisation
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
library(rmcorr)
library(lme4)
library(forcats)
library(yardstick)
library(naniar)
library(ggplotify)
library(reticulate)
library(ltm)

# Import custom plotting functions
source('functions/plotting.R')
use_python("/usr/local/bin/python3")
source_python("functions/pickle_reader.py")

### II. Figure 2: Distributions of TIL and alternative scales.
## Load and prepare formatted TILsummaries scores dataframe
# Load formatted TILsummaries scores dataframe and select relevant columns
formatted.uwTIL.max.mean <- read.csv('../formatted_data/formatted_unweighted_TIL_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7,TILTimepoint>=1) %>%
  group_by(GUPI) %>%
  summarise(uwTILmax = max(uwTILSum,na.rm=T),
            uwTILmean = mean(uwTILSum,na.rm=T),
            uwTILmedian = median(uwTILSum,na.rm=T))

formatted.TIL.max <- read.csv('../formatted_data/formatted_TIL_max.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(GUPI,TILmax) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_max.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_Basicmax)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_max.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,PILOTmax)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_max.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_1987max)) %>%
  left_join(formatted.uwTIL.max.mean%>%dplyr::select(GUPI,uwTILmax)) %>%
  pivot_longer(cols=-GUPI,names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = str_remove(Scale,'max'),
         MedianMax='Max over first week in ICU')

formatted.TIL.median <- read.csv('../formatted_data/formatted_TIL_median.csv',
                                 na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(GUPI,TILmedian) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_median.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_Basicmedian)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_median.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,PILOTmedian)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_median.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_1987median)) %>%
  left_join(formatted.uwTIL.max.mean%>%dplyr::select(GUPI,uwTILmedian)) %>%
  pivot_longer(cols=-GUPI,names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = str_remove(Scale,'median'),
         MedianMax='Median over first week in ICU')

formatted.TIL.maxes.medians <- rbind(formatted.TIL.max,formatted.TIL.median) %>%
  mutate(Scale = factor(Scale,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')))

## Create and save TILmedians and TILmaxes violin plots
# Create ggplot object for plot
TIL.medians.maxes.violin.plot <- formatted.TIL.maxes.medians %>%
  ggplot(aes(x = Scale, y = Score)) +
  geom_violin(aes(fill=Scale),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  geom_boxplot(aes(color=Scale),width=0.1,outlier.shape = NA,lwd=1.3/.pt) +
  coord_cartesian(ylim = c(0,31)) +
  scale_y_continuous(breaks = seq(0,31,5),minor_breaks = seq(0,38,1)) +
  scale_fill_manual(values=c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  scale_color_manual(values=c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  facet_wrap(~MedianMax,
             nrow = 1,
             scales = 'free',
             strip.position = "left") +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(0.05, "lines"),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    strip.placement = "outside",
    legend.position = 'none'
  )

# Create directory for current date and save plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_median_maxes.svg'),TIL.medians.maxes.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2)

## Load formatted TIL scores over first week of ICU stay
# Load formatted TIL24 scores dataframe and select relevant columns
formatted.TIL.scores <- read.csv('../formatted_data/formatted_TIL_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(GUPI,TILTimepoint,TotalSum) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_scores.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TILTimepoint,TIL_Basic)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_scores.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TILTimepoint,PILOTSum)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_scores.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TILTimepoint,TIL_1987Sum)) %>%
  left_join(read.csv('../formatted_data/formatted_unweighted_TIL_scores.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TILTimepoint,uwTILSum)) %>%
  rename(TIL24=TotalSum,
         TILBasic24 = TIL_Basic,
         PILOT24 = PILOTSum,
         TIL198724 = TIL_1987Sum,
         uwTIL24 = uwTILSum) %>%
  filter(TILTimepoint<=7,TILTimepoint>0) %>%
  pivot_longer(cols=-c(GUPI,TILTimepoint),names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = factor(Scale,levels=c('TIL24','uwTIL24','TILBasic24','PILOT24','TIL198724')),
         TILTimepoint = paste('Day',TILTimepoint))

## Create and save TIL24 violin plots
# Create ggplot object for plot
TIL.24s.violin.plot <- formatted.TIL.scores %>%
  ggplot(aes(x = factor(TILTimepoint), y = Score)) +
  geom_violin(aes(fill=Scale),width=.75,scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5,position = position_dodge(width = .75)) +
  geom_quasirandom(aes(color=Scale),width=.125,varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5,dodge.width =.75) +
  geom_boxplot(aes(color=Scale),width=0.2,outlier.shape = NA,lwd=1.3/.pt,position = position_dodge(width = .75)) +
  coord_cartesian(ylim = c(0,31)) +
  scale_y_continuous(breaks = seq(0,31,5),minor_breaks = seq(0,31,1)) +
  scale_fill_manual(values=c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  scale_color_manual(values=c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(0.05, "lines"),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    strip.placement = "outside",
    legend.position = 'bottom',
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_24s.png'),TIL.24s.violin.plot,units='in',dpi=600,width=7.5,height = 2.15)

### III. Figure 3: Associations of TIL and alternative scales with other clinical measures.
## Load and prepare formatted confidence intervals of Spearmans
# Extract names of scales for Spearman's correlation plot
TIL_spearman_names <- read.csv('../results/bootstrapping_results/CI_spearman_rhos_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(first,second) %>%
  unique() %>%
  pivot_longer(cols=c(first,second)) %>%
  filter((grepl('TIL',value))|(grepl('PILOT',value)),
         (grepl('max',value))|(grepl('median',value))) %>%
  dplyr::select(value) %>%
  unique() %>%
  .$value

# Load and format Spearman's correlation confidence interval dataframe
CI.spearman.rhos <- read.csv('../results/bootstrapping_results/CI_spearman_rhos_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((first %in% TIL_spearman_names)&!(second %in% TIL_spearman_names))|(!(first %in% TIL_spearman_names)&(second %in% TIL_spearman_names)),
         TILTimepoint=='Static') %>%
  mutate(FirstMax = grepl('max',first),
         SecondMax = grepl('max',second),
         FirstMean = grepl('mean',first),
         SecondMean = grepl('mean',second),
         FirstMedian = grepl('median',first),
         SecondMedian = grepl('median',second)) %>%
  filter(!(FirstMax&SecondMedian),
         !(FirstMedian&SecondMax),
         metric == 'rho') %>%
  mutate(MaxOrMedian = case_when(FirstMax|SecondMax~'Max',
                                 FirstMedian|SecondMedian~'Median'),
         TILScore = case_when(first %in% TIL_spearman_names ~ first,
                              second %in% TIL_spearman_names ~ second),
         OtherScore = case_when(!(first %in% TIL_spearman_names) ~ first,
                                !(second %in% TIL_spearman_names) ~ second)) %>%
  filter(OtherScore %in% c('GCSScoreBaselineDerived',
                           'GOSE6monthEndpointDerived',
                           'Pr(GOSE>1)',
                           'Pr(GOSE>3)',
                           'Pr(GOSE>4)',
                           'Pr(GOSE>5)',
                           'Pr(GOSE>6)',
                           'Pr(GOSE>7)',
                           'ICPmaxEH',
                           'ICPmaxHR',
                           'CPPminEH',
                           'CPPminHR',
                           'ICPmedianEH',
                           'ICPmedianHR',
                           'CPPmedianEH',
                           'CPPmedianHR'),
         !is.na(MaxOrMedian),
         ((Population=='TIL-ICP_EH')&(str_ends(OtherScore,'EH')))|((Population=='TIL-ICP_HR')&(str_ends(OtherScore,'HR')))|((Population=='TIL')&(!str_ends(OtherScore,'HR')&!str_ends(OtherScore,'EH'))),
         !((OtherScore %in% c('CPPminEH','CPPminHR'))&(MaxOrMedian=='Median'))) %>%
  mutate(BaseTILScore = case_when(MaxOrMedian=='Max'~str_remove(TILScore,'max'),
                                  MaxOrMedian=='Median'~str_remove(TILScore,'median')),
         OtherScore = case_when(OtherScore == "GCSScoreBaselineDerived" ~ "GCS",
                                OtherScore == "GOSE6monthEndpointDerived" ~ "GOSE",
                                TRUE ~ OtherScore)) %>%
  mutate(MaxOrMedian = plyr::mapvalues(MaxOrMedian,
                                       from=c('Max','Median'),
                                       to=c('Max score correlations','Median score correlations')),
         BaseTILScore = factor(BaseTILScore,levels=rev(c("TIL",
                                                         "uwTIL",
                                                         "TIL_Basic",
                                                         "PILOT",
                                                         "TIL_1987"))),
         OtherScore = factor(OtherScore,levels=c("ICPmaxEH",
                                                 "ICPmedianEH",
                                                 "ICPmaxHR",
                                                 "ICPmedianHR",
                                                 "CPPminEH",
                                                 "CPPmedianEH",
                                                 "CPPminHR",
                                                 "CPPmedianHR",
                                                 "DNa+max",
                                                 "DNa+mean",
                                                 "GCS",
                                                 "GOSE",
                                                 "Pr(GOSE>1)",
                                                 "Pr(GOSE>3)",
                                                 "Pr(GOSE>4)",
                                                 "Pr(GOSE>5)",
                                                 "Pr(GOSE>6)",
                                                 "Pr(GOSE>7)")))

# # Load and format Spearman's correlation confidence interval dataframe from differential calculation run
# CI.diff.spearman.rhos <- read.csv('../results/bootstrapping_results/archive/differential_CI_spearman_rhos_results.csv',
#                                   na.strings = c("NA","NaN","", " ")) %>%
#   filter(Population=='TIL-ICP_HR',
#          first=='ICPmax',
#          metric=='rho',
#          str_ends(second,'max')) %>%
#   mutate(TILTimepoint='Static',
#          FirstMax = grepl('max',first),
#          SecondMax = grepl('max',second),
#          FirstMean = grepl('mean',first),
#          SecondMean = grepl('mean',second),
#          FirstMedian = grepl('median',first),
#          SecondMedian = grepl('median',second),
#          MaxOrMedian = case_when(FirstMax|SecondMax~'Max',
#                                  FirstMedian|SecondMedian~'Median'),
#          TILScore = case_when(first %in% TIL_spearman_names ~ first,
#                               second %in% TIL_spearman_names ~ second),
#          OtherScore = case_when(!(first %in% TIL_spearman_names) ~ first,
#                                 !(second %in% TIL_spearman_names) ~ second),
#          BaseTILScore = case_when(MaxOrMedian=='Max'~str_remove(TILScore,'max'),
#                                   MaxOrMedian=='Median'~str_remove(TILScore,'median')),
#          OtherScore = case_when(!(Population %in% c('TIL','TIL-Na'))~paste0(OtherScore,sub(".*_","", Population)),
#                                 TRUE ~ OtherScore)) %>%
#   mutate(MaxOrMedian = plyr::mapvalues(MaxOrMedian,
#                                        from=c('Max','Median'),
#                                        to=c('Max score correlations','Median score correlations')))
# 
# # Combine both dataframes
# CI.spearman.rhos <- rbind(CI.spearman.rhos,CI.diff.spearman.rhos) %>%
#   mutate(BaseTILScore = factor(BaseTILScore,levels=rev(c("TIL",
#                                                          "uwTIL",
#                                                          "TIL_Basic",
#                                                          "PILOT",
#                                                          "TIL_1987"))),
#          OtherScore = factor(OtherScore,levels=c("ICPmaxEH",
#                                                  "ICPmedianEH",
#                                                  "ICPmaxHR",
#                                                  "ICPmedianHR",
#                                                  "CPPminEH",
#                                                  "CPPmedianEH",
#                                                  "CPPminHR",
#                                                  "CPPmedianHR",
#                                                  "DNa+max",
#                                                  "DNa+mean",
#                                                  "GCS",
#                                                  "GOSE",
#                                                  "Pr(GOSE>1)",
#                                                  "Pr(GOSE>3)",
#                                                  "Pr(GOSE>4)",
#                                                  "Pr(GOSE>5)",
#                                                  "Pr(GOSE>6)",
#                                                  "Pr(GOSE>7)")))

## Create and save Spearman's correlation plot
# Create ggplot object for plot
spearmans_correlation_plot <- CI.spearman.rhos %>%
  filter(!str_starts(OtherScore,'DNa+')) %>%
  ggplot() +
  coord_cartesian(xlim = c(-.51,.51)) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_errorbarh(aes(y = OtherScore, xmin = lo, xmax = hi, color = BaseTILScore),position=position_dodge(width=.675), height=.5)+
  geom_point(aes(y = OtherScore, x = median, color = BaseTILScore),position=position_dodge(width=.675),size=1)+
  scale_y_discrete(limits=rev) +
  xlab("Spearman's correlation coefficient (p)")+
  scale_color_manual(values=rev(c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600')),guide=guide_legend(title = 'Scale',reverse = TRUE))+
  facet_wrap(~MaxOrMedian,
             scales = 'free',
             nrow = 1) +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_text(size = 7, color = 'black',angle = 30, hjust=1, margin = margin(r=0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save Spearman's correlation plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'spearmans_correlation.svg'),spearmans_correlation_plot,device= svglite,units='in',dpi=600,width=7.5,height = 4.5)

## Load and prepare formatted confidence intervals of repeated-measures correlation
# Extract names of scales for repeated-measures correlation plot
TIL_rmcorrs_names <- read.csv('../results/bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(first,second) %>%
  unique() %>%
  pivot_longer(cols=c(first,second)) %>%
  filter((grepl('Sum',value))|(grepl('TIL_Basic',value))) %>%
  dplyr::select(value) %>%
  unique() %>%
  .$value

# Load and format repeated-measures correlation confidence interval dataframe
CI.rmcorrs <- read.csv('../results/bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((first %in% TIL_rmcorrs_names)&!(second %in% TIL_rmcorrs_names))|(!(first %in% TIL_rmcorrs_names)&(second %in% TIL_rmcorrs_names))) %>%
  filter(metric == 'rmcorr') %>%
  mutate(OtherScore = case_when(!(first %in% TIL_rmcorrs_names) ~ first,
                                !(second %in% TIL_rmcorrs_names) ~ second),
         Scale = case_when(first %in% TIL_rmcorrs_names ~ first,
                           second %in% TIL_rmcorrs_names ~ second),
         BaseTILScore = case_when(Scale=='TotalSum'~'TIL',
                                  Scale=='TIL_Basic'~'TIL_Basic',
                                  T~str_replace(Scale,'Sum',''))) %>%
  filter(OtherScore %in% c('TILPhysicianConcernsICP','TILPhysicianConcernsCPP','CPP24EH','ICP24EH','CPP24HR','ICP24HR')) %>%
  mutate(OtherScore = case_when(OtherScore == "TILPhysicianConcernsICP" ~ "Physician concern of ICP",
                                OtherScore == "TILPhysicianConcernsCPP" ~ "Physician concern of CPP",
                                TRUE ~ OtherScore),
         BaseTILScore = factor(Scale,levels=rev(c("TIL",
                                                  "uwTIL",
                                                  "TIL_Basic",
                                                  "PILOT",
                                                  "TIL_1987"))),
         OtherScore = factor(OtherScore,levels=c("ICP24EH",
                                                 "ICP24HR",
                                                 "CPP24EH",
                                                 "CPP24HR",
                                                 "DNa+24",
                                                 "Physician concern of ICP",
                                                 "Physician concern of CPP")))

# Load and format repeated-measures correlation confidence interval dataframe from differential calculation run
# CI.diff.rmcorrs <- read.csv('../results/bootstrapping_results/differential_CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
#   filter(((first %in% TIL_rmcorrs_names)&!(second %in% TIL_rmcorrs_names))|(!(first %in% TIL_rmcorrs_names)&(second %in% TIL_rmcorrs_names))) %>%
#   filter(metric == 'rmcorr') %>%
#   mutate(OtherScore = case_when(!(first %in% TIL_rmcorrs_names) ~ first,
#                                 !(second %in% TIL_rmcorrs_names) ~ second)) %>%
#   filter(OtherScore %in% c('ICPmean','CPPmean','ChangeInSodium','TILPhysicianConcernsICP','TILPhysicianConcernsCPP')) %>%
#   mutate(OtherScore = case_when(str_detect(OtherScore, 'mean') ~ str_replace(OtherScore, 'mean', '24'),
#                                 OtherScore == "ChangeInSodium" ~ "DNa+24",
#                                 OtherScore == "TILPhysicianConcernsICP" ~ "Physician concern of ICP",
#                                 OtherScore == "TILPhysicianConcernsCPP" ~ "Physician concern of CPP",
#                                 TRUE ~ OtherScore)) %>%
#   mutate(OtherScore = case_when(!(Population %in% c('TIL','TIL-Na'))~paste(sub(".*_","", Population),OtherScore),
#                                 TRUE ~ OtherScore))
# 
# # Combine both dataframes
# CI.rmcorrs <- rbind(CI.rmcorrs,CI.diff.rmcorrs)

## Create and save repeated-measures correlation plot
# Create ggplot object for plot
rm_correlation_plot <- CI.rmcorrs %>%
  filter(!str_starts(OtherScore,'DNa+')) %>%
  ggplot() +
  coord_cartesian(xlim = c(-.38,.38)) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_errorbarh(aes(y = OtherScore, xmin = lo, xmax = hi, color = Scale),position=position_dodge(width=.675), height=.5)+
  geom_point(aes(y = OtherScore, x = median, color = Scale),position=position_dodge(width=.675),size=1)+
  scale_y_discrete(limits=rev) +
  xlab("Repeated measures correlation coefficient (rrm)")+
  scale_color_manual(values=rev(c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600')),guide=guide_legend(title = 'Scale',reverse = TRUE))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_text(size = 7, color = 'black',angle = 30, hjust=1, margin = margin(r=0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save repeated-measures correlation plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'rm_correlation.svg'),rm_correlation_plot,device= svglite,units='in',dpi=600,width=3.75,height = 2.6)

## Load and prepare formatted confidence intervals of mixed effects coefficients
# Extract names of scales for mixed effects coefficients plot
TIL_mixed_effects_names <- read.csv('../results/bootstrapping_results/CI_mixed_effects_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Type=='TotalScore',
         metric=='Coefficient',
         !(Name%in%c('Intercept','Group Var','TILTimepoint'))) %>%
  dplyr::select(Formula,Name) %>%
  unique() %>%
  mutate(Target=sub(" ~.*","", Formula)) %>%
  filter(((grepl('Sum',Name))|(grepl('TIL_Basic',Name)))&(Target!='TotalSum')) %>%
  dplyr::select(Formula) %>%
  unique() %>%
  .$Formula

# Load and format mixed effects coefficients confidence interval dataframe
CI.mixed.effects <- read.csv('../results/bootstrapping_results/CI_mixed_effects_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Formula %in% TIL_mixed_effects_names,
         Type=='TotalScore',
         metric=='Coefficient',
         !(Name%in%c('Intercept','Group Var','TILTimepoint'))) %>%
  mutate(Target=sub(" ~.*", "", Formula),
         Scale = case_when(Name=='TotalSum'~'TIL',
                           Name=='TIL_Basic'~'TIL_Basic',
                           T~str_replace(Name,'Sum',''))) %>%
  # filter(Target %in% c('CPPmean','ICPmean','ChangeInSodium')) %>%
  # mutate(Target = case_when(Target=='ICPmean'~'ICP24',
  #                           Target=='CPPmean'~'CPP24',
  #                           Target=='ChangeInSodium'~'DNa+24')) %>%
  # mutate(Target = case_when(!(Population %in% c('TIL','TIL-Na'))~paste(sub(".*_","", Population),Target),
  #                           TRUE ~ Target)) %>%
  mutate(FormattedFormula = paste0(Target,'~DayICU+Scale')) %>%
  mutate(Scale = factor(Scale,levels=rev(c("TIL",
                                           "uwTIL",
                                           "TIL_Basic",
                                           "PILOT",
                                           "TIL_1987"))),
         FormattedFormula = factor(FormattedFormula,levels=c("ICP24EH~DayICU+Scale",
                                                             "ICP24HR~DayICU+Scale",
                                                             "CPP24EH~DayICU+Scale",
                                                             "CPP24HR~DayICU+Scale",
                                                             "DNa+24~DayICU+Scale")))

## Create and save mixed effects coefficients plot
# Create ggplot object for plot
mixed_effect_correlation_plot <- CI.mixed.effects %>%
  filter(!str_starts(FormattedFormula,'DNa+')) %>%
  ggplot() +
  coord_cartesian(xlim = c(-1.47,1.47)) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_errorbarh(aes(y = FormattedFormula, xmin = lo, xmax = hi, color = Scale),position=position_dodge(width=.675), height=.5)+
  geom_point(aes(y = FormattedFormula, x = median, color = Scale),position=position_dodge(width=.675),size=1)+
  scale_y_discrete(limits=rev) +
  xlab("Linear mixed effects model coefficient")+
  scale_color_manual(values=rev(c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600')),guide=guide_legend(title = 'Scale',reverse = TRUE))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_text(size = 7, color = 'black',angle = 30, hjust=1, margin = margin(r=0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save mixed-effects coefficient plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'mixed_effect_coefficients.svg'),mixed_effect_correlation_plot,device= svglite,units='in',dpi=600,width=3.75,height = 2.6)

### IV. Figure 4: Distributions of daily intracranial pressure and cerebral perfusion pressure means per daily TIL score.
## Load and prepare ICP information
# Load and filter dataframe of low-resolution ICP information
ICP24_lores.df <- read.csv('../formatted_data/formatted_low_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7,
         TILTimepoint>=1,
         TILExpected==1,
         !is.na(ICPmean),
         !is.na(TotalSum)) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'LowResolution') %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(ICPmean))

# Load and filter dataframe of high-resolution ICP information
ICP24_hires.df <- read.csv('../formatted_data/formatted_high_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7,
         TILTimepoint>=1,
         TILExpected==1,
         !is.na(ICPmean),
         !is.na(TotalSum)) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'HighResolution') %>%
  filter(TILTimepoint<=7) %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(ICPmean))

# Combine low- and high-resolution ICP values into single dataframe
combined.ICP24.TIL24.df <- rbind(ICP24_lores.df,ICP24_hires.df) %>%
  mutate(population = factor(population,levels=c('LowResolution','HighResolution')))

## Create and save TIL24 vs. ICP24 violin plots
# Create ggplot object for plot
ICP24.TIL24.violin.plot <- ggplot() +
  geom_split_violin(data=combined.ICP24.TIL24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = ICPmean, fill=population),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(data=combined.ICP24.TIL24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = ICPmean, color=population),varwidth = TRUE,alpha = 0.35,stroke = 0,size=.5,dodge.width =.5) +
  geom_quasirandom(data=combined.ICP24.TIL24.df %>% filter(!NotOutlier),mapping = aes(x = TotalTIL, y = ICPmean, color=population),varwidth = TRUE,alpha = 1,stroke = .2,size=.5,dodge.width =.5) +
  stat_summary(data=combined.ICP24.TIL24.df,
               mapping = aes(x = TotalTIL, y = ICPmean, color=population),
               fun = median,
               fun.min = function(x) quantile(x,.25),
               fun.max = function(x) quantile(x,.75),
               geom = "crossbar",
               width = 0.5,
               show.legend = FALSE,
               position = position_dodge(width = .5),
               fill='white',
               lwd=1.3/.pt) +
  coord_cartesian(ylim = c(-4,30)) +
  scale_fill_manual(values=c('#003f5c','#bc5090')) +
  scale_color_manual(values=c('#003f5c','#bc5090')) +
  ylab('ICP24') +
  xlab('TIL24') +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Create directory for current date and save TIL24 vs. ICP24 violin plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'icp24_til24.png'),ICP24.TIL24.violin.plot,units='in',dpi=600,width=7.5,height = 2.3)

## Load and prepare CPP information
# Load and filter dataframe of low-resolution CPP information
CPP24_lores.df <- read.csv('../formatted_data/formatted_low_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'LowResolution') %>%
  filter(TILTimepoint<=7) %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(CPPmean))

# Load and filter dataframe of high-resolution CPP information
CPP24_hires.df <- read.csv('../formatted_data/formatted_high_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'HighResolution') %>%
  filter(TILTimepoint<=7) %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(CPPmean))

# Combine low- and high-resolution CPP values into single dataframe
combined.CPP24.TIL24.df <- rbind(CPP24_lores.df,CPP24_hires.df) %>%
  mutate(population = factor(population,levels=c('LowResolution','HighResolution')))

## Create and save TIL24 vs. CPP24 violin plots
# Create ggplot object for plot
CPP24.TIL24.violin.plot <- ggplot() +
  geom_split_violin(data=combined.CPP24.TIL24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = CPPmean, fill=population),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(data=combined.CPP24.TIL24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = CPPmean, color=population),varwidth = TRUE,alpha = 0.35,stroke = 0,size=.5,dodge.width =.5) +
  geom_quasirandom(data=combined.CPP24.TIL24.df %>% filter(!NotOutlier),mapping = aes(x = TotalTIL, y = CPPmean, color=population),varwidth = TRUE,alpha = 1,stroke = .2,size=.5,dodge.width =.5) +
  stat_summary(data=combined.CPP24.TIL24.df,
               mapping = aes(x = TotalTIL, y = CPPmean, color=population),
               fun = median,
               fun.min = function(x) quantile(x,.25),
               fun.max = function(x) quantile(x,.75),
               geom = "crossbar",
               width = 0.5,
               show.legend = FALSE,
               position = position_dodge(width = .5),
               fill='white',
               lwd=1.3/.pt) +
  coord_cartesian(ylim = c(44,105.5)) +
  scale_fill_manual(values=c('#003f5c','#bc5090')) +
  scale_color_manual(values=c('#003f5c','#bc5090')) +
  ylab('CPP24') +
  xlab('TIL24') +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Create directory for current date and save CPP24 vs. TIL24 violin plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'cpp24_til24.png'),CPP24.TIL24.violin.plot,units='in',dpi=600,width=7.5,height = 2.3)

### V. Figure 5: Discrimination of refractory intracranial hypertension status by TILmax and alternative scale maximum scores.
## Load and prepare refractory status dataframes
# Calculate optimal cutpoints for refractory intracranial hypertension status detection
refractory.ROC.cutpoints <- read.csv('../results/bootstrapping_results/compiled_ROCs_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(YoudensJ = TPR-FPR) %>%
  group_by(Scale,Threshold) %>%
  summarise(lo=quantile(YoudensJ,.025),
            median=median(YoudensJ),
            hi=quantile(YoudensJ,.975)) %>%
  group_by(Scale) %>%
  slice(which.max(median)) %>%
  mutate(MedianMax = case_when(grepl('max',Scale)~'Max over first week in ICU',
                               grepl('median',Scale)~'Median over first week in ICU'),
         Scale = case_when(MedianMax=='Max over first week in ICU'~str_replace(Scale,'max',''),
                           MedianMax=='Median over first week in ICU'~str_replace(Scale,'median',''),
                           T~Scale))

# Load AUC confidence intervals for refractory intracranial hypertension status detection
refractory.AUCs <- read.csv('../results/bootstrapping_results/compiled_ROCs_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(Scale,AUC,resample_idx) %>%
  unique() %>%
  group_by(Scale) %>%
  summarise(lo=quantile(AUC,.025),
            median=median(AUC),
            hi=quantile(AUC,.975),
            resamples = n())

# Calculate ROC-based metrics for refractory intracranial hypertension detection
CI.ROCs.results <- read.csv('../results/bootstrapping_results/compiled_ROCs_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(Sensitivity = TPR,
         Specificity = 1-FPR,
         YoudensJ = TPR-FPR) %>%
  dplyr::select(Scale,Threshold,Sensitivity,Specificity,YoudensJ) %>%
  pivot_longer(cols = c(Sensitivity,Specificity,YoudensJ),names_to ='METRIC',values_to ='VALUES') %>%
  group_by(Scale,Threshold,METRIC) %>%
  summarise(lo = 100*quantile(VALUES,.025),
            median = 100*quantile(VALUES,.5),
            hi = 100*quantile(VALUES,.975),
            count = n()) %>%
  mutate(FormattedCI=sprintf('%.0f (%.0f–%.0f)',median,lo,hi)) %>%
  filter(Scale %in% c("PILOTmax","PILOTmedian","TIL_1987max","TIL_1987median","TIL_Basicmax","TIL_Basicmedian","TILmax","TILmedian","uwTILmax","uwTILmedian"))

# Format and save for table
table.CI.ROCs.results <- CI.ROCs.results %>%
  filter(Scale %in% c("TILmax","TILmedian","TIL_Basicmax","TIL_Basicmedian"),
         Threshold<=31) %>%
  dplyr::select(Scale,Threshold,METRIC,FormattedCI) %>%
  pivot_wider(names_from = 'METRIC', values_from = 'FormattedCI') %>%
  mutate(Scale = factor(Scale,levels=c("TILmax","TILmedian","TIL_Basicmax","TIL_Basicmedian"))) %>%
  arrange(Scale,Threshold)
write.xlsx(table.CI.ROCs.results,'../results/threshold_AUCs_refractory.xlsx')

# Load TILmax and TILmedian
formatted.uwTIL.max.mean <- read.csv('../formatted_data/formatted_unweighted_TIL_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7,TILTimepoint>=1) %>%
  group_by(GUPI) %>%
  summarise(uwTILmax = max(uwTILSum,na.rm=T),
            uwTILmean = mean(uwTILSum,na.rm=T),
            uwTILmedian = median(uwTILSum,na.rm=T))

formatted.TIL.max <- read.csv('../formatted_data/formatted_TIL_max.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(GUPI,TILmax) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_max.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_Basicmax)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_max.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,PILOTmax)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_max.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_1987max)) %>%
  left_join(formatted.uwTIL.max.mean%>%dplyr::select(GUPI,uwTILmax)) %>%
  pivot_longer(cols=-GUPI,names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = str_remove(Scale,'max'),
         MedianMax='Max over first week in ICU')

formatted.TIL.median <- read.csv('../formatted_data/formatted_TIL_median.csv',
                                 na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(GUPI,TILmedian) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_median.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_Basicmedian)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_median.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,PILOTmedian)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_median.csv',na.strings = c("NA","NaN","", " "))%>%dplyr::select(GUPI,TIL_1987median)) %>%
  left_join(formatted.uwTIL.max.mean%>%dplyr::select(GUPI,uwTILmedian)) %>%
  pivot_longer(cols=-GUPI,names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = str_remove(Scale,'median'),
         MedianMax='Median over first week in ICU')

formatted.TIL.maxes.medians <- rbind(formatted.TIL.max,formatted.TIL.median) %>%
  mutate(Scale = factor(Scale,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')))

# Load maximum scale scores stratified by refractory intracranial hypertension status
refractory.TIL.maxes <- formatted.TIL.maxes.medians %>%
  left_join(read.csv('../formatted_data/formatted_outcome_and_demographics.csv',na.strings = c("NA","NaN","", " ")) %>% dplyr::select(GUPI,RefractoryICP)) %>%
  filter(!is.na(RefractoryICP)) %>%
  left_join(refractory.ROC.cutpoints %>% dplyr::select(Scale,MedianMax,Threshold)) %>%
  mutate(Scale = factor(Scale,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')),
         RefractoryICP = plyr::mapvalues(RefractoryICP,c(0,1),c('No','Yes')))

## Create and save plot of maximum score distributions by refractory ICH status
# Create ggplot object
refractory.TIL.maxes.plot <- ggplot() +
  facet_wrap(~MedianMax,
             ncol = 2,
             strip.position = "left",
             scales = 'free') +
  geom_split_violin(data=refractory.TIL.maxes,mapping = aes(x = Scale, y = Score, fill=factor(RefractoryICP)),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  stat_summary(data=refractory.TIL.maxes,
               mapping = aes(x = Scale, y = Score, color=factor(RefractoryICP)),
               fun = median,
               fun.min = function(x) quantile(x,.25),
               fun.max = function(x) quantile(x,.75),
               geom = "crossbar",
               width = 0.3,
               show.legend = FALSE,
               position = position_dodge(width = .3),
               fill='white',
               lwd=1.3/.pt) +
  geom_errorbar(data = refractory.ROC.cutpoints%>%filter(str_starts(Scale,'ICP',negate=T),str_starts(Scale,'CPP',negate=T)), aes(x = Scale, ymin = Threshold, ymax = Threshold),lwd=1.3/.pt) +
  coord_cartesian(ylim = c(0,31)) +
  scale_y_continuous(breaks = seq(0,31,5),minor_breaks = seq(0,31,1)) +
  scale_fill_manual(values=c('#003f5c','#de425b')) +
  scale_color_manual(values=c('#003f5c','#de425b')) +
  guides(fill=guide_legend(title="Refractory intracranial hypertension"),
         color=guide_legend(title="Refractory intracranial hypertension")) +
  ylab('Max over first week in ICU') +
  theme_minimal(base_family = 'Roboto Condensed')+
  theme(
    panel.grid.minor.x = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(0.05, "lines"),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    strip.placement = "outside",
    legend.position = 'bottom',
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save plots of maximum score distributions by refractory ICH status
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_maxes_refractory.svg'),refractory.TIL.maxes.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

## Load and prepare ROC curves for changing maximum score thresholds
refractory.ROC.curves <- read.csv('../results/bootstrapping_results/compiled_ROC_refractory_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(YoudensJ = TPR-FPR,
         Scale = str_remove(Scale,'max'))
CI.refractory.ROC.curves <- read.csv('../results/bootstrapping_results/CI_ROCs_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Scale=='TILmedian',
         metric %in% c('FPR','TPR')) %>%
  dplyr::select(Scale,Threshold,metric,median) %>%
  pivot_wider(values_from = 'median',names_from = 'metric')

## Create and save plot of ROC curve for changing maximum score thresholds
# Create ggplot object
refractory.TIL.max.ROC <- refractory.ROC.curves %>%
  filter(Scale == 'TIL') %>%
  ggplot() +
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",lwd=.75, color = 'gray')+
  geom_line(data=refractory.ROC.curves %>% filter(Scale %in% c('ICP_EH','ICP_HR')),aes(x=FPR,y=TPR,linetype=Scale),color='lightgray',lwd=.75) +
  geom_line(aes(x=FPR,y=TPR,color=Threshold),lwd=1.3) +
  scale_color_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=15.5,limits = c(0,31),breaks=seq(0,31,by=5)) + 
  scale_linetype_manual(values = c('twodash','dotted')) +
  geom_point(x=0.207,y=0.675,fill=NA, color="darkred", size=5, shape = 1)+
  xlab("False positive rate") +
  ylab("True positive rate") +
  guides(color = guide_colourbar(title="TILmax threshold for refractory ICP (>=)",title.position = "top",title.hjust=.5,barwidth = .5, barheight = 7.5,ticks = FALSE))+
  theme_classic(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    aspect.ratio = 1,
    panel.border = element_rect(colour = 'black', fill=NA, linewidth = .75),
    legend.position = 'right',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line"),
    legend.margin=margin(0,0,0,0),
    plot.margin=margin(0,0,0,0)
  )

refractory.TIL.median.ROC <- CI.refractory.ROC.curves %>%
  ggplot() +
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",lwd=.75, color = 'gray')+
  geom_line(aes(x=FPR,y=TPR,color=Threshold),lwd=1.3) +
  scale_color_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=15.5,limits = c(0,31),breaks=seq(0,31,by=5)) + 
  scale_linetype_manual(values = c('twodash','dotted')) +
  geom_point(x=0.277056277,y=0.8144330,fill=NA, color="darkred", size=5, shape = 1)+
  xlab("False positive rate") +
  ylab("True positive rate") +
  guides(color = guide_colourbar(title="TILmax threshold for refractory ICP (>=)",title.position = "top",title.hjust=.5,barwidth = .5, barheight = 7.5,ticks = FALSE))+
  theme_classic(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    aspect.ratio = 1,
    panel.border = element_rect(colour = 'black', fill=NA, linewidth = .75),
    legend.position = 'right',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line"),
    legend.margin=margin(0,0,0,0),
    plot.margin=margin(0,0,0,0)
  )

# Create directory for current date and save ROC curve
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_maxes_ROC.svg'),refractory.TIL.max.ROC,device= svglite,units='in',dpi=600,width=3.75,height = 2.3)
ggsave(file.path('../plots',Sys.Date(),'til_medians_ROC.svg'),refractory.TIL.median.ROC,device= svglite,units='in',dpi=600,width=3.75,height = 2.3)

### VI. Figure 6: Association of TIL component items with TIL24 and other study measures.
## Calculate median TIL component item scores per total TIL24 score
TIL.24.components <- read.csv('../formatted_data/formatted_TIL_scores.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7) %>%
  dplyr::select(-starts_with('TILPhysician'),-DailyTILCompleteStatus) %>%
  pivot_longer(cols=-c(GUPI,TILTimepoint,TILDate,ICUAdmTimeStamp,ICUDischTimeStamp,TotalSum),
               names_to = 'Item',
               values_to = 'Score') %>%
  mutate(Category = plyr::mapvalues(Item,
                                    from=c("CSFDrainage","DecomCraniectomy","FluidLoading","Hypertonic","ICPSurgery","Mannitol","Neuromuscular","Positioning","Sedation","Temperature","Vasopressor","Ventilation"),
                                    to=c('CSF drainage','Surgery for ICP','CPP management','Hyperosmolar therapy','Surgery for ICP','Hyperosmolar therapy','Sedation and paralysis','Positioning','Sedation and paralysis','Temperature control','CPP management','Ventilatory management'))) %>%
  group_by(GUPI,TILTimepoint,TotalSum,Category) %>%
  summarise(Score = sum(Score)) %>%
  group_by(TotalSum,Category) %>%
  summarise(MedianScore = median(Score)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_scores.csv',na.strings = c("NA","NaN","", " "))%>%
              filter(TILTimepoint<=7)%>%
              group_by(TotalSum)%>%
              summarise(SumInstanceCount=n())) %>%
  mutate(Category = fct_rev(factor(Category,levels=c('Positioning',
                                                     'Sedation and paralysis',
                                                     'CPP management',
                                                     'Ventilatory management',
                                                     'Hyperosmolar therapy',
                                                     'Temperature control',
                                                     'Surgery for ICP',
                                                     'CSF drainage')))) %>%
  arrange(fct_relevel(Category, rev(levels(Category))))

## Create and save plot of median TIL component item scores per TIL24
# Create ggplot object
TIL.24.components.plot <- TIL.24.components %>%
  ggplot(aes(x=TotalSum)) +
  geom_segment(x = 0, y = 0, xend = 31, yend = 31,alpha = 0.5,linetype = "dashed",lwd=.75, color = 'gray')+
  geom_col(aes(fill=Category,y=MedianScore)) +
  geom_text(data=TIL.24.components %>% filter(MedianScore>0),
            aes(label = MedianScore,y=MedianScore),
            position = position_stack(vjust = .5),
            color='white',
            size=5/.pt) +
  geom_col(data=TIL.24.components %>% dplyr::select(TotalSum,SumInstanceCount) %>% unique() %>% mutate(ModCount = -SumInstanceCount*(31/5)/547),
           aes(y=ModCount),
           fill='lightgrey') +
  geom_text(data=TIL.24.components %>% dplyr::select(TotalSum,SumInstanceCount) %>% unique() %>% mutate(ModCount = -SumInstanceCount*(31/5)/547),
            aes(label = SumInstanceCount,y=ModCount),
            vjust = 1.5,
            color='black',
            size=5/.pt) +
  geom_hline(yintercept = 0,lwd=.75) + 
  scale_y_continuous(breaks = seq(0,31,5),expand = expansion(mult = c(.0275, 0))) +
  scale_x_continuous(breaks = seq(0,31,1),expand = c(0,0)) +
  scale_fill_manual(values=rev(c('#003f5c','#a05195','#ff7c43','#2f4b7c','#d45087','#ffa600','#665191','#f95d6a'))) +
  guides(fill = guide_legend(title='ICP-treatment modality',reverse = T,nrow=T)) +
  xlab("TIL24 score") +
  ylab("Median component score") +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(panel.grid.minor.x = element_blank(),
        axis.title.x = element_text(size = 7, color = 'black',face='bold'),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_text(size = 7, color = 'black',face='bold'),
        legend.position = 'bottom',
        legend.key.size = unit(1/.pt,"line"),
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save median TIL component item scores plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TIL_24_components_plot.svg'),TIL.24.components.plot,device=svglite,units='in',dpi=600,width=7.5,height = 3.15)

## Load and prepare repeated-measures correlations of TIL component items
# Explicitly define TIL item and score labels
item.names <- c('Positioning','Sedation','Neuromuscular','CSFDrainage','FluidLoading','Vasopressor','Ventilation','Mannitol','Hypertonic','Temperature','ICPSurgery','DecomCraniectomy')
item.labels <- c('Positioning','Sedation','Paralysis','CSF drainage','Fluid loading','Vasopressors','Ventilation','Mannitol','Hypertonic saline','Temperature control','Intracranial surgery','Decompressive craniectomy')
other.score.names <- c('TotalSum','ICP24EH','ICP24HR','CPP24EH','CPP24HR','TILPhysicianConcernsICP','TILPhysicianConcernsCPP')
other.score.proto.labels <- c('TIL24','ICP24','CPP24','Physician concern of ICP','Physician concern of CPP')
other.score.labels <- c('TIL24','ICP24EH','ICP24HR','CPP24EH','CPP24HR','Physician concern of ICP','Physician concern of CPP')

# Load and format confidence intervals of repeated-measures correlations results involving TIL item correlations
CI.rmcorr.results <- read.csv('../results/bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(metric == 'rmcorr') %>%
  filter(((first %in% item.names)&(second %in% c('TILPhysicianConcernsICP','TILPhysicianConcernsCPP','TotalSum','ICP24EH','ICP24HR','CPP24EH','CPP24HR')))|((second %in% item.names)&(first %in% c('TILPhysicianConcernsICP','TILPhysicianConcernsCPP','TotalSum','ICP24EH','ICP24HR','CPP24EH','CPP24HR')))) %>%
  mutate(TILComponent = case_when(first %in% item.names ~ first,
                                  second %in% item.names ~ second),
         OtherScore = case_when(first %in% other.score.names ~ first,
                                second %in% other.score.names ~ second)) %>%
  mutate(TILComponent = factor(plyr::mapvalues(TILComponent,
                                               from=item.names,
                                               to=item.labels),
                               levels=item.labels),
         OtherScore = plyr::mapvalues(OtherScore,
                                      from=other.score.names,
                                      to=other.score.labels)) %>%
  mutate(OtherScore = factor(OtherScore,levels=other.score.labels),
         BoxScore = case_when(lo<0&hi>0~NA,
                              T~median),
         LabelColor = case_when(is.na(BoxScore)|abs(median)<.46~'black',
                                T~'white'))
# 
# # Load and format confidence intervals of repeated-measures correlations differential analysis results involving TIL item correlations
# CI.diff.rmcorr.results <- read.csv('../results/bootstrapping_results/differential_CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
#   filter(Scale == 'TIL',
#          metric == 'rmcorr') %>%
#   filter(((first %in% item.names)&(second %in% other.score.names))|((second %in% item.names)&(first %in% other.score.names))) %>%
#   mutate(TILComponent = case_when(first %in% item.names ~ first,
#                                   second %in% item.names ~ second),
#          OtherScore = case_when(first %in% other.score.names ~ first,
#                                 second %in% other.score.names ~ second))

# # Combine dataframes and complete formatting
# CI.rmcorr.results <- rbind(CI.rmcorr.results,CI.diff.rmcorr.results) %>%
#   mutate(TILComponent = factor(plyr::mapvalues(TILComponent,
#                                                from=item.names,
#                                                to=item.labels),
#                                levels=item.labels),
#          OtherScore = plyr::mapvalues(OtherScore,
#                                       from=other.score.names,
#                                       to=other.score.proto.labels)) %>%
#   mutate(OtherScore = factor(case_when((OtherScore=='ICP24')|(OtherScore=='CPP24')~paste0(OtherScore,sub(".*_","", Population)),
#                                        TRUE ~ OtherScore),
#                              levels=other.score.labels),
#          BoxScore = case_when(lo<0&hi>0~NA,
#                               T~median),
#          LabelColor = case_when(is.na(BoxScore)|abs(median)<.46~'black',
#                                 T~'white'))

## Create and save plot of TIL component items repeated-measures correlation
# Create ggplot object
component.corrs <- CI.rmcorr.results %>%
  filter(!str_starts(OtherScore,'DNa+')) %>%
  ggplot(aes(x=TILComponent,y=OtherScore)) +
  geom_tile(aes(fill=BoxScore)) + 
  scale_fill_gradient2(na.value='gray90',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-0.6141,0.6141),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(CI.rmcorr.results$OtherScore))) +
  geom_text(aes(label=FormattedCI,color = LabelColor),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('black','white'),breaks = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(fill = guide_colourbar(title="Repeated measures correlation coefficient (rrm)",title.position = "top",title.hjust=.5,barwidth = 10, barheight = .5,ticks = FALSE))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0),angle = 30,hjust=1),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0),angle = 30,hjust=1),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_blank(),
        legend.position = 'bottom',
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save plot of TIL component items repeated-measures correlation
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'component_corrs.svg'),component.corrs,device=svglite,units='in',dpi=600,width=7.5,height = 3.15)

## Load and prepare mixed effect coefficients of TIL component items
# Load manually prepared labels
lmer.coeff.labels <- read_xlsx('../results/bootstrapping_results/coefficient_labels.xlsx')

# Load and format mixed effect coefficients involving TIL component items
CI.lmer.results <- read.csv('../results/bootstrapping_results/CI_mixed_effects_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Type == 'Component',
         !(Name %in% c('Group Var','Intercept','TILTimepoint')),
         metric == 'Coefficient',
         str_starts(Formula,'ICP24')|str_starts(Formula,'CPP24')|str_starts(Formula,'ChangeInSodium')) %>%
  left_join(lmer.coeff.labels) %>%
  mutate(DepVar = sub("\\ ~.*", "", Formula),
         DepVar = case_when(DepVar=='ICP24EH'~'EH ICP24',
                            DepVar=='ICP24HR'~'HR ICP24',
                            DepVar=='CPP24EH'~'EH CPP24',
                            DepVar=='CPP24HR'~'HR CPP24')) %>%
  mutate(Label = fct_reorder(Label, Order),
         DepVar = factor(DepVar,
                         levels=c('EH ICP24','HR ICP24','EH CPP24','HR CPP24','DNa+24')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi),
         BoxScore = case_when(lo<0&hi>0~NA,
                              T~median),
         LabelColor = case_when(is.na(BoxScore)|abs(median)<3.375~'black',
                                T~'white'))

## Create and save plot of TIL component items mixed effects coefficients
# Create ggplot object
lmer.coeffs <- CI.lmer.results %>%
  ggplot(aes(x=Label,y=DepVar)) +
  geom_tile(aes(fill=BoxScore)) + 
  scale_fill_gradient2(na.value='gray90',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-4.5,4.5),breaks=seq(-4.5,4.5,by=2.25)) +
  scale_y_discrete(limits = rev(levels(CI.lmer.results$DepVar))) +
  geom_text(aes(label=FormattedCI,color = LabelColor),family = 'Roboto Condensed',size=4/.pt) +
  scale_color_manual(values = c('black','white'),breaks = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  xlab('TIL sub-item')+
  guides(fill = guide_colourbar(title="Linear mixed effects model coefficient (BLMER)",title.position = "left",title.hjust=.5,barwidth = 10, barheight = .5,ticks = FALSE))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0),angle = 30,hjust=1),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0),angle = 30,hjust=1),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.x = element_text(size = 7, color = 'black',face = 'bold',margin = margin(r = 0)),
        legend.position = 'bottom',
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save plot of TIL component items mixed effects coefficients
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'lmer_coeffs.svg'),lmer.coeffs,device=svglite,units='in',dpi=600,width=7.5,height = 2.8)

### VII. Figure 7: Relationship between TIL and TIL(Basic).
## Load and prepare distribution of daily scale scores per daily TIL(Basic) scores
formatted.TIL.Basic.scores <- read.csv('../formatted_data/formatted_TIL_Basic_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(TIL_Basic = factor(TIL_Basic)) %>%
  filter(!((TIL_Basic==0)&(TotalSum==5)),
         TILTimepoint<=7,
         TILTimepoint>0) %>%
  mutate(TILCuts = cut(TotalSum, breaks=c(-Inf,.5,2.5,6.5,8.5,Inf), labels=c(0,1,2,3,4)))

# Create a custom dataframe designating derived ranges
TILBasic.TIL.ranges <- data.frame(TIL_Basic=seq(0,4,by=1),
                                  RangeMin=c(0,.5,2.5,6.5,8.5),
                                  RangeMax=c(0,2.5,6.5,8.5,38)) %>%
  mutate(TIL_Basic = factor(TIL_Basic))

## Create and save plots of distributions of daily scale scores per daily TIL(Basic) scores
# Create ggplot object
TIL.Basic.violin.plot <- formatted.TIL.Basic.scores %>%
  mutate(TIL_Basic = factor(TIL_Basic)) %>%
  ggplot(aes(x = TIL_Basic)) +
  geom_crossbar(data=TILBasic.TIL.ranges,
                aes(y=RangeMin,ymin=RangeMin,ymax=RangeMax),
                fill='gray',
                alpha=1,
                linetype = 0) +
  geom_violin(aes(y = TotalSum),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5,fill='#7a5195') +
  geom_quasirandom(aes(y = TotalSum),varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  geom_boxplot(aes(y = TotalSum),width=0.1,outlier.shape = NA,lwd=1.3/.pt,color='#7a5195') +
  coord_cartesian(ylim = c(0,31)) +
  xlab('TIL(Basic)24 score')+
  ylab('TIL24 score')+
  scale_y_continuous(breaks = seq(0,31,5),minor_breaks = seq(0,38,1)) +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(0.05, "lines"),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Create directory for current date and save plots of distributions of daily scale scores per daily TIL(Basic) scores
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TIL_Basic_violin.svg'),TIL.Basic.violin.plot,device= svglite,units='in',dpi=600,width=3.75,height = 2.15)
ggsave(file.path('../plots',Sys.Date(),'TIL_Basic_violin.png'),TIL.Basic.violin.plot,units='in',dpi=600,width=3.75,height = 2.15)

## Calculate distributions of daily TIL(Basic) scores per daily TIL scores
# Calculate distributions
TILBasic.TIL.distributions <- formatted.TIL.Basic.scores %>%
  group_by(TotalSum,TIL_Basic) %>%
  summarise(count=n()) %>%
  group_by(TotalSum) %>%
  mutate(totalCount = sum(count),
         percent = 100*count/totalCount) %>%
  rowwise() %>%
  mutate(FormattedPercent = paste0(as.character(signif(percent, 2)),'%'))

# Create dataframe of empty distributions
empty.distributions <- TILBasic.TIL.distributions %>%
  tidyr::expand(TotalSum,TIL_Basic) %>%
  mutate(count=0,
         percent=0,
         FormattedPercent='0%') %>%
  anti_join(TILBasic.TIL.distributions %>% dplyr::select(TotalSum,TIL_Basic)) %>%
  left_join(TILBasic.TIL.distributions %>% dplyr::select(TotalSum,totalCount) %>% unique())

# Combine two dataframes
TILBasic.TIL.distributions <- rbind(TILBasic.TIL.distributions,empty.distributions)

## Create and save frequency tables of daily TIL(Basic) scores per daily TIL scores 
# Create ggplot object
TILBasic.TIL.freq.table <- TILBasic.TIL.distributions %>%
  ggplot(aes(x=factor(TotalSum),y=factor(TIL_Basic))) +
  geom_tile(aes(fill=percent)) + 
  scale_fill_gradient(na.value = 'white',low='white',high='#33213e',limits = c(0,100),breaks=seq(0,100,by=25)) +
  scale_y_discrete(limits = c('4','3','2','1','0')) +
  geom_text(aes(label=FormattedPercent,color = as.factor(as.integer(abs(percent)>50))),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  xlab('TIL24 score')+
  ylab('TIL(Basic)24 score')+
  guides(fill = guide_colourbar(title="Spearman's correlation coefficient (p)",title.position = "top",title.hjust=.5,barwidth = 10, barheight = .5,ticks = FALSE))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.x = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_text(size = 7, color = 'black',face = 'bold'),
        legend.position = 'none',
        aspect.ratio = 5/32,
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save frequency tables of daily TIL(Basic) scores per daily TIL scores
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TIL_Basic_TIL_freqs.svg'),TILBasic.TIL.freq.table,device= svglite,units='in',dpi=600,width=7.5,height = 1.78)

## Load and prepare dataframe of information coverage of TIL(Basic) over ICU stay
TIL.Basic.TIL.information <- read.csv('../results/bootstrapping_results/compiled_mutual_info_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  group_by(TILTimepoint,METRIC) %>%
  mutate(group_idx = row_number()) %>%
  dplyr::select(group_idx,TILTimepoint,METRIC,TIL,TIL_Basic) %>%
  pivot_longer(cols=c(TIL,TIL_Basic),
               names_to = 'Scale') %>%
  filter(!((METRIC=='MutualInfo')&(Scale=='TIL_Basic'))) %>%
  mutate(METRIC = paste0(METRIC,'_',Scale)) %>%
  pivot_wider(names_from = METRIC, values_from = value, id_cols = c(TILTimepoint,group_idx)) %>%
  mutate(percent_coverage = (MutualInfo_TIL/Entropy_TIL)*100) %>%
  pivot_longer(cols=-c(TILTimepoint,group_idx),names_to = 'METRIC') %>%
  group_by(TILTimepoint,METRIC) %>%
  summarise(lo = quantile(value,.025),
            median = quantile(value,.5),
            hi = quantile(value,.975),
            mean = mean(value),
            count = n()) %>%
  mutate(FormattedCI = sprintf('%.2f (%.2f–%.2f)',median,lo,hi),
         TILTimepoint = case_when(!is.na(as.numeric(TILTimepoint))~paste0('Day ',TILTimepoint),
                                  TRUE~TILTimepoint)) %>%
  filter(METRIC=='percent_coverage')

## Create and save plots of information coverage of TIL(Basic) over ICU stay
# Create ggplot object
information_coverage <- ggplot() +
  geom_line(data=TIL.Basic.TIL.information%>%filter(!TILTimepoint%in%c('Max','Median')),
            mapping=aes(x=TILTimepoint, y=median, group=1),
            color='#7a5195',
            lwd=1.3) +
  geom_point(data=TIL.Basic.TIL.information,
             mapping=aes(x=TILTimepoint, y=median),
             color='#7a5195',
             size=2) +
  geom_errorbar(data=TIL.Basic.TIL.information,
                mapping=aes(x=TILTimepoint, ymin=lo, ymax=hi),
                width=.35,
                color='#7a5195') +
  coord_cartesian(ylim = c(0,35)) +
  xlab("Timepoint during ICU stay")+
  ylab("Information coverage of TIL(Basic) (%)")+
  scale_y_continuous(breaks = seq(0,35,5)) + 
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    axis.text.x = element_text(size = 6, color = 'black'),
    axis.text.y = element_text(size = 6, color = 'black'),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save plots of information coverage of TIL(Basic) over ICU stay
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'info_coverage.svg'),information_coverage,device= svglite,units='in',dpi=600,width=3.75,height = 2.05)

## Calculate 95% confidence interval for Matthews Correlation Coefficient between TIL(Basic) and TIL categorisation
# Iterate through 1000 resamples
datalist = vector("list", length = 1000)
for (x in 1:1000){
  curr.GUPIs <- unique(sample(unique(formatted.TIL.Basic.scores$GUPI), length(unique(formatted.TIL.Basic.scores$GUPI)), replace = TRUE))
  curr.MCCs <- formatted.TIL.Basic.scores %>%
    filter(GUPI %in% curr.GUPIs) %>%
    group_by(TILTimepoint) %>%
    mcc(TILCuts, TIL_Basic) %>%
    mutate(RESAMPLE = x)
  datalist[[x]] <- curr.MCCs
}
compiled.MCCs <- do.call(rbind, datalist) %>%
  group_by(TILTimepoint) ##

### VIII. Supplementary Figure S1: Missingness of static study measures
## Load and prepare static measures used in analysis
# Load baseline demographic and functional outcome score dataframe
demo.outcome <- read.csv('../formatted_data/formatted_outcome_and_demographics.csv',na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(-c(Race,ICURaisedICP,DecompressiveCranReason,starts_with('AssociatedStudy'))) %>%
  rename(GCS=GCSScoreBaselineDerived, GOSE=GOSE6monthEndpointDerived) %>%
  rename_with(~gsub("Pr.GOSE.","Pr_GOSE_gt_", .x, fixed = TRUE))

## Create UpSet plots of static feature missingness
overall.upset.plot <- gg_miss_upset(demo.outcome %>% dplyr::select(-c(ends_with('Set'))), nsets = n_var_miss(demo.outcome),nintersects = NA)
lores.upset.plot <- gg_miss_upset(demo.outcome %>% filter(LowResolutionSet==1) %>% dplyr::select(-c(ends_with('Set'))), nsets = n_var_miss(demo.outcome),nintersects = NA)
hires.upset.plot <- gg_miss_upset(demo.outcome %>% filter(HighResolutionSet==1) %>% dplyr::select(-c(ends_with('Set'))), nsets = n_var_miss(demo.outcome),nintersects = NA)
plot_grid(as.ggplot(overall.upset.plot), as.ggplot(lores.upset.plot), as.ggplot(hires.upset.plot), labels=c("A", "B", "C"), ncol = 3, nrow = 1)

### IX. Supplementary Figure S2: Missingness of longitudinal study measures
## Prepare longitudinal
long.avail.counts <- read.csv('../results/longitudinal_data_availability.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(Combination = case_when(Combination=='TotalSumMiss;TILPhysicianConcernsICPMiss;ICPMiss'~'TIL, ICP/CPP, and physician concerns of ICP/CPP',
                                 Combination=='TotalSumMiss;TILPhysicianConcernsICPMiss;ICPNonMiss'~'TIL and physician concerns of ICP/CPP',
                                 Combination=='TotalSumNonMiss;TILPhysicianConcernsICPMiss;ICPMiss'~'ICP/CPP and physician concerns of ICP/CPP',
                                 Combination=='TotalSumNonMiss;TILPhysicianConcernsICPMiss;ICPNonMiss'~'Physician concerns of ICP/CPP',
                                 Combination=='TotalSumNonMiss;TILPhysicianConcernsICPNonMiss;ICPMiss'~'ICP/CPP',
                                 Combination=='TotalSumNonMiss;TILPhysicianConcernsICPNonMiss;ICPNonMiss'~'None'),
         Substudy = case_when(Set=='OverallSet'~'TIL validation population',
                              Set=='LowResolutionSet'~'TIL-ICPEH population',
                              Set=='HighResolutionSet'~'TIL-ICPHR population'),
         TILTimepoint = paste0('Day ',TILTimepoint),
         PropLabel = sprintf('%.0f%%',100*Proportion)) %>%
  mutate(Combination = fct_rev(factor(Combination,levels=c('None',
                                                           'Physician concerns of ICP/CPP',
                                                           'ICP/CPP',
                                                           'ICP/CPP and physician concerns of ICP/CPP',
                                                           'TIL and physician concerns of ICP/CPP',
                                                           'TIL, ICP/CPP, and physician concerns of ICP/CPP')))) %>%
  arrange(fct_relevel(Combination, rev(levels(Combination))))

## Create and save plot of missing value combinations per day
# Create ggplot object
long.missingness.plot <- long.avail.counts %>%
  ggplot(aes(x=TILTimepoint)) +
  geom_col(aes(fill=Combination,y=Count),
           width = 0.85) +
  facet_wrap(~Substudy,
             ncol=3,
             scales = 'free') +
  coord_cartesian(ylim = c(0,900)) +
  scale_y_continuous(breaks = seq(0,875,125),expand = expansion(mult = c(0,.0275))) +
  geom_text(data=long.avail.counts,
            aes(label = PropLabel,y=Count),
            position = position_stack(vjust = .5),
            color='white',
            size=6/.pt) +
  geom_text(data=long.avail.counts %>% dplyr::select(TILTimepoint,Substudy,TotalCount) %>% unique(),
            aes(label = TotalCount,y=TotalCount),
            vjust = -.5,
            color='black',
            size=6/.pt) +
  scale_fill_manual(values=rev(c('#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600'))) +
  guides(fill = guide_legend(title='Missing value combination',reverse = T,nrow=2,byrow = F,)) +
  ylab("Count (n)") +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(panel.grid.minor.x = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7, color = 'black',face='bold',margin = margin(r = 0,t = 0,b = 0)),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 6, color = 'black',margin = margin(0,0,0,0)),
        legend.text = element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_text(size = 7, color = 'black',face='bold'),
        legend.position = 'bottom',
        legend.key.size = unit(1/.pt,"line"),
        strip.text = element_text(size = 7, color = "black",face = 'bold'),
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save missing value combinations per day plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'longitudinal_missingness_plot.svg'),long.missingness.plot,device=svglite,units='in',dpi=600,width=7.5,height = 3.15)

## Load and format missingness analysis table for supplementary tables
long.miss.table <- read.csv('../results/longitudinal_missingness_analysis.csv') %>%
  mutate(Substudy = case_when(Substudy=='OverallSet'~'TIL validation population',
                              Substudy=='LowResolutionSet'~'TIL-ICPEH population',
                              Substudy=='HighResolutionSet'~'TIL-ICPHR population'),
         Substudy = factor(Substudy,
                           levels = c('TIL validation population',
                                      'TIL-ICPEH population',
                                      'TIL-ICPHR population')),
         MissingVariable = case_when(MissingVariable=='ICPmean'~'ICP/CPP',
                                     MissingVariable=='TILPhysicianConcernsICP'~'Physician concerns of ICP/CPP',
                                     MissingVariable=='TotalSum'~'TIL24'),
         MissingVariable = factor(MissingVariable,
                                  levels = c('TIL24',
                                             'ICP/CPP',
                                             'Physician concerns of ICP/CPP')),
         value = case_when(str_starts(variable,'Pr')~variable,
                           variable=='RefractoryICP'~'',
                           T~value),
         variable = case_when(variable=='GCSSeverity'~'Baseline GCS',
                              variable=='MarshallCT'~'Marshall CT',
                              variable=='GOSE6monthEndpointDerived'~'Six-month GOSE',
                              variable=='SiteCode'~'Centre distribution*',
                              variable=='RefractoryICP'~'Refractory intracranial hypertension',
                              str_starts(variable,'Pr')~'Baseline functional prognosis',
                              T~variable),
         variable = factor(variable,
                           levels = c('Centre distribution*',
                                      'Age',
                                      'Sex',
                                      'Baseline GCS',
                                      'Marshall CT',
                                      'Refractory intracranial hypertension',
                                      'Six-month GOSE',
                                      'Baseline functional prognosis',
                                      'TILmax',
                                      'TILmedian',
                                      'TIL24')),
         Significant = p_val < 0.05,
         p_val = case_when(is.na(p_val)~'',
                           T~sprintf('%.3f',p_val)),
         DaysSinceICUAdmission = paste0('Day ',DaysSinceICUAdmission)) %>%
  arrange(Substudy,DaysSinceICUAdmission,MissingVariable,variable,value,Set) %>%
  pivot_wider(names_from=c(Set),
              values_from=c(n,FormattedLabel)) %>%
  relocate(Substudy,DaysSinceICUAdmission,MissingVariable,n_In,n_Out,variable,value,FormattedLabel_In,FormattedLabel_Out)

write.csv(long.miss.table,'../results/longitudinal_missingness_table.csv',row.names = F)

### X. Supplementary Figure S3: Correlation matrices between total scores of TIL and alternative scales.
## Load and prepare inter-scale Spearman's correlation results
# Load formatted Spearman's rho values for inter-scale comparisons
inter.scale.spearmans <- read.csv('../results/bootstrapping_results/CI_spearman_rhos_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((grepl('TIL',first))|(grepl('PILOT',first)))&((grepl('TIL',second))|(grepl('PILOT',second)))) %>%
  mutate(FirstMax = grepl('max',first),SecondMax = grepl('max',second)) %>%
  filter(FirstMax == SecondMax,
         metric == 'rho')

# Create dataframe of row-column reversals
rev.inter.scale.spearmans <- inter.scale.spearmans %>%
  rename(first=second,second=first)

# Combine formatted Spearman's rho values and complete formatting
inter.scale.spearmans <- rbind(inter.scale.spearmans,rev.inter.scale.spearmans) %>%
  mutate(BaseFirst = case_when(FirstMax ~ str_remove(first,'max'),
                               !FirstMax ~ str_remove(first,'mean')),
         BaseSecond = case_when(SecondMax ~ str_remove(second,'max'),
                                !SecondMax ~ str_remove(second,'mean'))) %>%
  mutate(BaseFirst = factor(BaseFirst,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')),
         BaseSecond = factor(BaseSecond,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi))

## Create and save plots of inter-scale maximum score Spearman correlations
# Create ggplot object
max.scale.correlations <- inter.scale.spearmans %>%
  filter(FirstMax) %>%
  ggplot(aes(x=BaseFirst,y=BaseSecond)) +
  geom_tile(aes(fill=median)) + 
  scale_fill_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-1,1),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(inter.scale.spearmans$BaseSecond))) +
  geom_text(aes(label=FormattedCI,color = as.factor(as.integer(abs(median)>.75))),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(fill = guide_colourbar(title="Spearman's correlation coefficient (p)",title.position = "top",title.hjust=.5,barwidth = 10, barheight = .5,ticks = FALSE))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_blank(),
        aspect.ratio = 1,
        legend.position = 'bottom',
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save plots of inter-scale maximum score Spearman correlations
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'max_scale_correlations.svg'),max.scale.correlations,device=svglite,units='in',dpi=600,width=2.5,height = 2.73)

## Create and save plots of inter-scale mean score Spearman correlations
# Create ggplot object
mean.scale.correlations <- inter.scale.spearmans %>%
  filter(!FirstMax) %>%
  ggplot(aes(x=BaseFirst,y=BaseSecond)) +
  geom_tile(aes(fill=median)) + 
  scale_fill_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-1,1),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(inter.scale.spearmans$BaseSecond))) +
  geom_text(aes(label=FormattedCI,color = as.factor(as.integer(abs(median)>.75))),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('white','black'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(fill = guide_colourbar(title="Spearman's correlation coefficient (p)",title.position = "top",title.hjust=.5,barwidth = 10, barheight = .5,ticks = FALSE))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_blank(),
        aspect.ratio = 1,
        legend.position = 'bottom',
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save plots of inter-scale mean score Spearman correlations
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'mean_scale_correlations.svg'),mean.scale.correlations,device=svglite,units='in',dpi=600,width=2.5,height = 2.73)

# Load formatted repeated measures correlation values for inter-scale comparisons
inter.scale.rmcorrs <- read.csv('../results/bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((grepl('Sum',first))|(grepl('TIL_Basic',first)))&((grepl('Sum',second))|(grepl('TIL_Basic',second)))) %>%
  filter(metric == 'rmcorr') %>%
  mutate(BaseFirst = plyr::mapvalues(first,
                                     from=c('TotalSum','TIL_Basic','PILOTSum','TIL_1987Sum','uwTILSum'),
                                     to=c('TIL24','TILBasic24','PILOT24','TIL198724','uwTIL24')),
         BaseSecond = plyr::mapvalues(second,
                                      from=c('TotalSum','TIL_Basic','PILOTSum','TIL_1987Sum','uwTILSum'),
                                      to=c('TIL24','TILBasic24','PILOT24','TIL198724','uwTIL24')))

# Create dataframe of row-column reversals
rev.inter.scale.rmcorrs <- inter.scale.rmcorrs %>%
  rename(BaseFirst=BaseSecond,BaseSecond=BaseFirst)

# Combine formatted repeated measures correlation values and complete formatting
inter.scale.rmcorrs <- rbind(inter.scale.rmcorrs,rev.inter.scale.rmcorrs) %>%
  mutate(BaseFirst = factor(BaseFirst,levels=c('TIL24','uwTIL24','TILBasic24','PILOT24','TIL198724')),
         BaseSecond = factor(BaseSecond,levels=c('TIL24','uwTIL24','TILBasic24','PILOT24','TIL198724')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi))

## Create and save plots of inter-scale repeated measures correlations
# Create ggplot object
daily.scale.correlations <- inter.scale.rmcorrs %>%
  ggplot(aes(x=BaseFirst,y=BaseSecond)) +
  geom_tile(aes(fill=median)) + 
  scale_fill_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-1,1),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(inter.scale.rmcorrs$BaseSecond))) +
  geom_text(aes(label=FormattedCI,color = as.factor(as.integer(abs(median)>.75))),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  # guides(fill = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = .5, barheight = 5,ticks = FALSE))+
  guides(fill = guide_colourbar(title="Repeated measures correlation coefficient (rrm)",title.position = "top",title.hjust=.5,barwidth = 10, barheight = .5,ticks = FALSE))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_blank(),
        aspect.ratio = 1,
        legend.position = 'bottom',
        legend.margin=margin(0,0,0,0))

# Create directory for current date and save plots of inter-scale repeated measures correlations
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'daily_scale_correlations.svg'),daily.scale.correlations,device=svglite,units='in',dpi=600,width=2.5,height = 2.73)

### XI. Supplementary Figure S4: Inter-item correlation matrices for daily scores of TIL and alternative scales.
## Load and prepare inter-item repeated-measures correlation coefficients
# Define item names
item.names <- c('Positioning','Sedation','Neuromuscular','Paralysis','CSFDrainage','Ventricular','FluidLoading','Vasopressor','Hyperventilation','Ventilation','Mannitol','Hypertonic','Temperature','ICPSurgery','DecomCraniectomy')

# Load formatted repeated measures correlation values for intra-scale comparisons
intra.scale.rmcorrs <- read.csv('../results/bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter((first %in% item.names)&(second %in% item.names),
         metric=='rmcorr') %>%
  mutate(first = factor(first,levels=item.names),
         second = factor(second,levels=item.names))

# Create dataframe of row-column reversals
rev.intra.scale.rmcorrs <- intra.scale.rmcorrs %>%
  rename(first=second,second=first)

# Combine formatted repeated measures correlation values and complete formatting
intra.scale.rmcorrs <- rbind(intra.scale.rmcorrs,rev.intra.scale.rmcorrs) %>%
  mutate(Scale = factor(Scale,levels=c('TIL','uwTIL','PILOT','TIL_1987')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi),
         BoxScore = case_when(lo<0&hi>0~NA,
                              T~median),
         LabelColor = case_when(is.na(BoxScore)|abs(median)<.27~'black',
                                T~'white'))

## Create and save plots of intra-scale repeated measures correlations
# Create ggplot object
component.correlations <- intra.scale.rmcorrs %>%
  ggplot(aes(x=first,y=factor(second,levels=rev(levels(second))))) +
  geom_tile(aes(fill=BoxScore)) + 
  scale_fill_gradient2(na.value='gray90',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-.36,.36),breaks=seq(-.36,.36,by=.12)) +
  geom_text(aes(label=FormattedCI,color = LabelColor),family = 'Roboto Condensed',size=3.8/.pt) +
  scale_color_manual(values = c('black','white'),breaks = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  # guides(fill = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = .5, barheight = 5,ticks = FALSE))+
  guides(fill = guide_colourbar(title="Repeated measures correlation coefficient (rrm)",title.position = "top",title.hjust=.5,barwidth = 10, barheight = .5,ticks = FALSE))+
  facet_wrap(~Scale,
             scales = 'free',
             nrow=2) +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7, color = 'black',margin = margin(r = 0),angle = 30,hjust=1),
        legend.title = element_text(size = 7, color = 'black',face = 'bold'),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0),angle = 30,hjust=1),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_blank(),
        aspect.ratio = 1,
        legend.position = 'bottom',
        legend.margin=margin(0,0,0,0),
        strip.text = element_text(size=7, color = "black",face = 'bold',margin = margin(b = .5)))

# Create directory for current date and save plots of intra-scale repeated measures correlations
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'component_intra_correlations.png'),component.correlations,units='in',dpi=600,width=7.5,height = 8)

### XII. Table 3: Optimised ranges for TIL categorisation
## Load and prepare information
# Load formatted TIL scores
formatted.TIL.scores <- read.csv('../formatted_data/formatted_TIL_scores.csv',
                                 na.strings = c("NA","NaN","", " "))

# Load formatted TIL(Basic) scores
formatted.TILBasic.scores <- read.csv('../formatted_data/formatted_TIL_Basic_scores.csv',
                                      na.strings = c("NA","NaN","", " "))

# Load refractory intracranial hypertension status
formatted.refractory.ICP.values <- read.csv('../formatted_data/formatted_outcome_and_demographics.csv',
                                            na.strings = c("NA","NaN","", " ")) %>%
  dplyr::select(GUPI,RefractoryICP)

# Load TIL validation resamples
TIL.validation.resamples <- read.csv('../results/bootstrapping_results/resamples/TIL_validation/TIL_validation_resamples.csv')

## Calculate 95% confidence intervals for table metrics
# Create empty lists to store results
categories <- c()
sensitivities <- c()
specificities <- c()
accuracies <- c()
resample.indices <- c()

# Iterate through 
for (curr.rs.idx in unique(TIL.validation.resamples$RESAMPLE_IDX)){
  
  # Get current imputation index
  curr.imp.idx <- unique(TIL.validation.resamples %>%
                           filter(RESAMPLE_IDX==curr.rs.idx) %>%
                           .$IMPUTATION_IDX)
  
  # Get current GUPIs
  curr.GUPIs <- unique(TIL.validation.resamples %>%
                         filter(RESAMPLE_IDX==curr.rs.idx) %>%
                         .$GUPI)
  
  # Define directory of current imputation
  curr.imp.dir <- file.path('../formatted_data','imputed_sets',sprintf("imp%03d",curr.imp.idx))
  
  # Load and filter formatted dynamic variable sets
  global.dynamic.var.set <- read.csv(file.path(curr.imp.dir,'dynamic_var_set.csv')) %>%
    dplyr::select(GUPI,TILTimepoint,TotalSum,TIL_Basic,ICPSurgery,DecomCraniectomy) %>%
    filter(GUPI %in% curr.GUPIs) %>%
    mutate(TIL_Basic = factor(TIL_Basic),
           SurgIndicator = factor(as.integer((ICPSurgery+DecomCraniectomy)>0)),
           SurgPred = factor(as.integer(TotalSum>=9)),
           TILBasicPred = factor(case_when(TotalSum>=9~4,
                                           TotalSum>=7~3,
                                           TotalSum>=3~2,
                                           TotalSum>=1~1,
                                           T ~ 0)))
  
  # Load and filter formatted static variable sets
  global.static.var.set <- read.csv(file.path(curr.imp.dir,'static_var_set.csv')) %>%
    dplyr::select(GUPI,TILmedian,TILmax,RefractoryICP) %>%
    filter(GUPI %in% curr.GUPIs) %>%
    mutate(RefractoryICP = factor(RefractoryICP),
           RefractoryMaxPred = factor(as.integer(TILmax>=14)),
           RefractoryMedianPred = factor(as.integer(TILmedian>=7.5)))
  
  # Add category labels
  categories <- c(categories,
                  c('Surgery','TILBasic','MaxRefractory','MedianRefractory'))
  
  # Add sensitivity scores
  sensitivities <- c(sensitivities,
                     c(sens_vec(global.dynamic.var.set$SurgIndicator,global.dynamic.var.set$SurgPred,event_level = 'second'),
                       sens_vec(global.dynamic.var.set$TIL_Basic,global.dynamic.var.set$TILBasicPred,estimator = 'macro'),
                       sens_vec(global.static.var.set$RefractoryICP,global.static.var.set$RefractoryMaxPred,event_level = 'second'),
                       sens_vec(global.static.var.set$RefractoryICP,global.static.var.set$RefractoryMedianPred,event_level = 'second')))
  
  # Add sensitivity scores
  specificities <- c(specificities,
                     c(spec_vec(global.dynamic.var.set$SurgIndicator,global.dynamic.var.set$SurgPred,event_level = 'second'),
                       spec_vec(global.dynamic.var.set$TIL_Basic,global.dynamic.var.set$TILBasicPred,estimator = 'macro'),
                       spec_vec(global.static.var.set$RefractoryICP,global.static.var.set$RefractoryMaxPred,event_level = 'second'),
                       spec_vec(global.static.var.set$RefractoryICP,global.static.var.set$RefractoryMedianPred,event_level = 'second')))
  
  # Add accuracy scores
  accuracies <- c(accuracies,
                  c(accuracy_vec(global.dynamic.var.set$SurgIndicator,global.dynamic.var.set$SurgPred),
                    accuracy_vec(global.dynamic.var.set$TIL_Basic,global.dynamic.var.set$TILBasicPred),
                    accuracy_vec(global.static.var.set$RefractoryICP,global.static.var.set$RefractoryMaxPred),
                    accuracy_vec(global.static.var.set$RefractoryICP,global.static.var.set$RefractoryMedianPred)))
  
  # Add resample index labels
  resample.indices <- c(resample.indices,
                        c(curr.rs.idx,curr.rs.idx,curr.rs.idx,curr.rs.idx))
}

# Compile all results into a dataframe
compiled.range.results <- data.frame(RESAMPLE_IDX=resample.indices,
                                     CATEGORY=categories,
                                     SENSITIVITY=sensitivities,
                                     SPECIFICITY=specificities,
                                     ACCURACY=accuracies)

# Save compiled results
write.csv(compiled.range.results,'../results/bootstrapping_results/compiled_TIL_range_results.csv',row.names = F)

# Calculate 95% confidence interval
CI.TIL.range.results <- read.csv('../results/bootstrapping_results/compiled_TIL_range_results.csv') %>%
  pivot_longer(cols=c('SENSITIVITY','SPECIFICITY','ACCURACY'),names_to = 'METRIC') %>%
  group_by(CATEGORY,METRIC) %>%
  summarise(lo=100*quantile(value,.025),
            median=100*quantile(value,.5),
            hi=100*quantile(value,.975),
            count = n()) %>%
  mutate(FormattedCI = sprintf('%.0f%% (%.0f–%.0f%%)',median,lo,hi)) %>%
  pivot_wider(id_cols = CATEGORY,names_from = METRIC,values_from = FormattedCI)

### XIII. Supplementary Table 4: TIL thresholds for detection of day of surgery
## Load and prepare information
# Load formatted TIL scores
formatted.TIL.scores <- read.csv('../formatted_data/formatted_TIL_scores.csv',
                                 na.strings = c("NA","NaN","", " "))

# Load TIL validation resamples
TIL.validation.resamples <- read.csv('../results/bootstrapping_results/resamples/TIL_validation/TIL_validation_resamples.csv')

## Calculate 95% confidence intervals for table metrics
# Create empty list to store results
listOfDataFrames <- vector(mode = "list", length = length(unique(TIL.validation.resamples$RESAMPLE_IDX)))

# Iterate through bootstrapping resamples
for (curr.rs.idx in unique(TIL.validation.resamples$RESAMPLE_IDX)){
  
  # Get current imputation index
  curr.imp.idx <- unique(TIL.validation.resamples %>%
                           filter(RESAMPLE_IDX==curr.rs.idx) %>%
                           .$IMPUTATION_IDX)
  
  # Get current GUPIs
  curr.GUPIs <- unique(TIL.validation.resamples %>%
                         filter(RESAMPLE_IDX==curr.rs.idx) %>%
                         .$GUPI)
  
  # Define directory of current imputation
  curr.imp.dir <- file.path('../formatted_data','imputed_sets',sprintf("imp%03d",curr.imp.idx))
  
  # Load and filter formatted dynamic variable sets
  curr.roc.curve <- read.csv(file.path(curr.imp.dir,'dynamic_var_set.csv')) %>%
    dplyr::select(GUPI,TILTimepoint,TotalSum,ICPSurgery,DecomCraniectomy) %>%
    filter(GUPI %in% curr.GUPIs) %>%
    mutate(SurgIndicator = factor(as.integer((ICPSurgery+DecomCraniectomy)>0))) %>%
    roc_curve(SurgIndicator, TotalSum, event_level='second') %>%
    mutate(youdensJ = specificity + sensitivity - 1)
  
  # Add current calculated dataframe to running list
  listOfDataFrames[[curr.rs.idx]] <- curr.roc.curve
}

# Compile all results into a dataframe
compiled.roc.curves <- do.call("rbind", listOfDataFrames)

# Save compiled results
write.csv(compiled.roc.curves,'../results/bootstrapping_results/compiled_surgery_ROCs_results.csv',row.names = F)

# Calculate 95% confidence interval
CI.surgery.ROC.results <- read.csv('../results/bootstrapping_results/compiled_surgery_ROCs_results.csv') %>%
  pivot_longer(cols=c('specificity','sensitivity','youdensJ'),names_to = 'METRIC') %>%
  group_by(.threshold,METRIC) %>%
  summarise(lo=100*quantile(value,.025),
            median=100*quantile(value,.5),
            hi=100*quantile(value,.975),
            count = n()) %>%
  mutate(FormattedCI = sprintf('%.0f%% (%.0f–%.0f%%)',median,lo,hi)) %>%
  pivot_wider(id_cols = .threshold,names_from = METRIC,values_from = FormattedCI)

### XIV. Supplementary Figure S5: Internal reliability of TIL and uwTIL
## Load and prepare information
# Load formatted TIL scores
formatted.TIL.scores <- read.csv('../formatted_data/formatted_TIL_scores.csv',
                                 na.strings = c("NA","NaN","", " "))

# Load TIL validation resamples
TIL.validation.resamples <- read.csv('../results/bootstrapping_results/resamples/TIL_validation/TIL_validation_resamples.csv')

# Define names of TIL component items
TIL.component.names <- c("CSFDrainage","DecomCraniectomy","FluidLoading","Hypertonic","ICPSurgery","Mannitol","Neuromuscular","Positioning","Sedation","Temperature","Vasopressor","Ventilation")

# Define names of uwTIL component items
uwTIL.component.names <- paste0('uw',TIL.component.names)

## Calculate 95% confidence intervals of Cronbach's alpha
# Create empty vectors to store results
scales <- c()
timepoints <- c()
alphas <- c()
resample.indices <- c()

# Iterate through bootstrapping resamples
for (curr.rs.idx in unique(TIL.validation.resamples$RESAMPLE_IDX)){
  
  # Get current imputation index
  curr.imp.idx <- unique(TIL.validation.resamples %>%
                           filter(RESAMPLE_IDX==curr.rs.idx) %>%
                           .$IMPUTATION_IDX)
  
  # Get current GUPIs
  curr.GUPIs <- unique(TIL.validation.resamples %>%
                         filter(RESAMPLE_IDX==curr.rs.idx) %>%
                         .$GUPI)
  
  # Define directory of current imputation
  curr.imp.dir <- file.path('../formatted_data','imputed_sets',sprintf("imp%03d",curr.imp.idx))
  
  # Load and filter formatted dynamic variable sets
  global.dynamic.var.set <- read.csv(file.path(curr.imp.dir,'dynamic_var_set.csv')) %>%
    filter(GUPI %in% curr.GUPIs) %>%
    dplyr::dplyr::select(GUPI,TILTimepoint,TIL.component.names,uwTIL.component.names)
  
  # Calculate TIL Cronbach's alpha
  curr.TIL.alphas <- global.dynamic.var.set %>%
    group_by(TILTimepoint) %>%
    group_map(~cronbach.alpha(.x[,TIL.component.names])$alpha) %>%
    unlist()
  
  # Calculate uwTIL Cronbach's alpha
  curr.uwTIL.alphas <- global.dynamic.var.set %>%
    group_by(TILTimepoint) %>%
    group_map(~cronbach.alpha(.x[,uwTIL.component.names])$alpha) %>%
    unlist()
  
  # Append vectors to running dataframes
  scales <- c(scales,rep('TIL',7),rep('uwTIL',7))
  timepoints <- c(timepoints,rep(seq(1,7),2))
  alphas <- c(alphas,curr.TIL.alphas,curr.uwTIL.alphas)
  resample.indices <- c(resample.indices,rep(curr.rs.idx,14))
}

# Compile all results into a dataframe
compiled.alpha.results <- data.frame(RESAMPLE_IDX=resample.indices,
                                     TILTimepoint = timepoints,
                                     SCALE=scales,
                                     ALPHA=alphas)

# Save compiled results
write.csv(compiled.alpha.results,'../results/bootstrapping_results/compiled_cronbach_alpha_results.csv',row.names = F)

# Calculate 95% confidence interval
CI.alpha.results <- read.csv('../results/bootstrapping_results/compiled_cronbach_alpha_results.csv') %>%
  group_by(SCALE,TILTimepoint) %>%
  summarise(lo=quantile(ALPHA,.025),
            median=quantile(ALPHA,.5),
            hi=quantile(ALPHA,.975),
            count = n()) %>%
  mutate(FormattedCI = sprintf('%.2f (%.2f–%.2f)',median,lo,hi),
         TILTimepoint = paste('Day',TILTimepoint))

# Create ggplot of Cronbach's alphas of TIL and uwTIL over time
cronbachs.aplha <- CI.alpha.results %>%
  ggplot() +
  geom_line(mapping=aes(x=TILTimepoint, y=median, color=SCALE, group = SCALE),
            lwd=1.3) +
  geom_point(mapping=aes(x=TILTimepoint, y=median, color=SCALE),
             size=2,
             position=position_dodge(0.1)) +
  geom_errorbar(mapping=aes(x=TILTimepoint, ymin=lo, ymax=hi, color=SCALE),
                width=.35,
                position=position_dodge(0.1)) +
  coord_cartesian(ylim = c(0.25,.75)) +
  xlab("Timepoint during ICU stay")+
  ylab("Cronbach's alpha")+
  scale_color_manual(values=c('#003f5c','#58508d'),guide=guide_legend(title = 'Scale',reverse = FALSE))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    axis.text.x = element_text(size = 6, color = 'black'),
    axis.text.y = element_text(size = 6, color = 'black'),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save plots of information coverage of TIL(Basic) over ICU stay
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'cronbachs_alpha.png'),cronbachs.aplha,units='in',dpi=600,width=3.75,height = 2.05)