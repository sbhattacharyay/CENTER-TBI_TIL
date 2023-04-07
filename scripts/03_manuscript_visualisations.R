#### Master Script 03: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. ICP24_lores vs. TIL24
# III. CPP24_lores vs. TIL24
# IV. Distributions of TILsummaries per study sub-group
# V. TILmean-TILmax correlations per study sub-group

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

# Import custom plotting functions
source('functions/plotting.R')

### II. 
## Load and prepare Spearman's Rho results
# Load Spearman's Rho confidence interval dataframe
CI_spearman_rhos <- read.csv('../bootstrapping_results/CI_spearman_rhos_results.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  mutate(FormattedCombos = factor(paste(second,first,sep = ' vs. '),
                                  levels = c('ICPmean vs. TILmean',
                                             'ICPmax vs. TILmax',
                                             'CPPmean vs. TILmean',
                                             'CPPmax vs. TILmax',
                                             'NAmean vs. TILmean',
                                             'NAmax vs. TILmax',
                                             'GCS vs. TILmean',
                                             'GCS vs. TILmax',
                                             'GOSE vs. TILmean',
                                             'GOSE vs. TILmax',
                                             'GOS vs. TILmean',
                                             'GOS vs. TILmax',
                                             'Pr(GOSE>1) vs. TILmean',
                                             'Pr(GOSE>3) vs. TILmean',
                                             'Pr(GOSE>4) vs. TILmean',
                                             'Pr(GOSE>5) vs. TILmean',
                                             'Pr(GOSE>6) vs. TILmean',
                                             'Pr(GOSE>7) vs. TILmean',
                                             'Pr(GOSE>1) vs. TILmax',
                                             'Pr(GOSE>3) vs. TILmax',
                                             'Pr(GOSE>4) vs. TILmax',
                                             'Pr(GOSE>5) vs. TILmax',
                                             'Pr(GOSE>6) vs. TILmax',
                                             'Pr(GOSE>7) vs. TILmax')),
         population = factor(population,
                             levels = c('PriorStudy','HighResolution','LowResolution')))

## Create `ggplot` object
spearmans_correlation_plot <- CI_spearman_rhos %>%
  ggplot() +
  coord_cartesian(xlim = c(-.74,.74)) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_errorbarh(aes(y = FormattedCombos, group = population, xmin = lo, xmax = hi, color = population, linetype = population),position=position_dodge(width=.5), height=.5)+
  geom_point(aes(y = FormattedCombos, group = population, x = median, color = population),position=position_dodge(width=.5),size=1)+
  scale_y_discrete(limits=rev) +
  xlab("Spearman's correlation coefficient")+
  scale_color_manual(values=c('darkgray','#bc5090','#003f5c'))+
  scale_linetype_manual(values=c('dashed','solid','solid'))+
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
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create directory for current date and save Spearman's correlation plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'spearmans_correlation.svg'),spearmans_correlation_plot,device= svglite,units='in',dpi=600,width=3.75,height = 5.5)

### III.
## Load and prepare mixed effects modelling results
# Load repeated measures correlation confidence interval dataframe
CI_rm_correlations <- read.csv('../bootstrapping_results/CI_rm_correlation_results.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  mutate(FormattedCombos = factor(paste(second,first,sep = ' vs. '),
                                  levels = c('ICP24 vs. TIL24',
                                             'CPP24 vs. TIL24',
                                             'NA24 vs. TIL24')),
         population = factor(population,
                             levels = c('PriorStudy','HighResolution','LowResolution')))

# Load mixed effects modelling confidence interval dataframe
CI_mixed_effects <- read.csv('../bootstrapping_results/CI_mixed_effects_results.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  mutate(FormattedCombos = factor(paste(second,first,sep = ' ~ '),
                                  levels = c('ICP24 ~ TIL24',
                                             'CPP24 ~ TIL24',
                                             'NA24 ~ TIL24')),
         population = factor(population,
                             levels = c('PriorStudy','HighResolution','LowResolution')))

## Create `ggplot` object
mixed_effect_coefficients <- CI_mixed_effects %>%
  ggplot() +
  coord_cartesian(xlim = c(-.80064385,0.6672615)) +
  scale_x_continuous(expand = expansion(mult = c(.01, .01)))+
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_errorbarh(aes(y = FormattedCombos, group = population, xmin = lo, xmax = hi, color = population, linetype = population),position=position_dodge(width=.5), height=.5)+
  geom_point(aes(y = FormattedCombos, group = population, x = median, color = population),position=position_dodge(width=.5),size=1)+
  scale_y_discrete(limits=rev) +
  xlab("Linear mixed effects model coefficient")+
  scale_color_manual(values=c('darkgray','#bc5090','#003f5c'))+
  scale_linetype_manual(values=c('dashed','solid','solid'))+
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
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create directory for current date and save Spearman's correlation plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'mixed_effect_coefficients.svg'),mixed_effect_coefficients,device= svglite,units='in',dpi=600,width=3.75,height = 2.05)

## Create `ggplot` object
rm_correlation_coefficients <- CI_rm_correlations %>%
  ggplot() +
  coord_cartesian(xlim = c(-0.2,0.45274355)) +
  scale_x_continuous(expand = expansion(mult = c(.01, .01)))+
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_errorbarh(aes(y = FormattedCombos, group = population, xmin = lo, xmax = hi, color = population, linetype = population),position=position_dodge(width=.5), height=.5)+
  geom_point(aes(y = FormattedCombos, group = population, x = median, color = population),position=position_dodge(width=.5),size=1)+
  scale_y_discrete(limits=rev) +
  xlab("Repeated measures correlation coefficient")+
  scale_color_manual(values=c('darkgray','#bc5090','#003f5c'))+
  scale_linetype_manual(values=c('dashed','solid','solid'))+
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
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create directory for current date and save Spearman's correlation plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'rm_correlation_coefficients.svg'),rm_correlation_coefficients,device= svglite,units='in',dpi=600,width=3.75,height = 2.05)


### IV. ICP24_lores vs. TIL24
## Load and prepare low-resolution ICP information
# Load and filter dataframe of low-resolution ICP information
ICP24_lores.df <- read.csv('../formatted_data/daily_correlations/lo_res_ICP24_TIL24.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalTIL),
         population = 'LowResolution')

ICP24_hires.df <- read.csv('../formatted_data/daily_correlations/hi_res_ICP24_TIL24.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalTIL),
         population = 'HighResolution')

# Designate outliers (based on 1.5*interquartile range)
ICP24_lores.df <- ICP24_lores.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(ICPmean))

ICP24_hires.df <- ICP24_hires.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(ICPmean))

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
  coord_cartesian(ylim = c(-4,40)) +
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

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'icp24_til24.svg'),ICP24.TIL24.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

### IV. Na24_lores vs. TIL24
## Load and prepare low-resolution Na information
# Load and filter dataframe of low-resolution Na information
Na24_lores.df <- read.csv('../formatted_data/daily_correlations/lo_res_Na24_TIL24.csv',
                          na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalTIL),
         population = 'LowResolution')

Na24_hires.df <- read.csv('../formatted_data/daily_correlations/hi_res_Na24_TIL24.csv',
                          na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalTIL),
         population = 'HighResolution')

# Designate outliers (based on 1.5*interquartile range)
Na24_lores.df <- Na24_lores.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(meanSodium))

Na24_hires.df <- Na24_hires.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(meanSodium))

combined.Na24.TIL24.df <- rbind(Na24_lores.df,Na24_hires.df) %>%
  mutate(population = factor(population,levels=c('LowResolution','HighResolution')))

## Create and save TIL24 vs. Na24 violin plots
# Create ggplot object for plot
Na24.TIL24.violin.plot <- ggplot() +
  geom_split_violin(data=combined.Na24.TIL24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = meanSodium, fill=population),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(data=combined.Na24.TIL24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = meanSodium, color=population),varwidth = TRUE,alpha = 0.35,stroke = 0,size=.5,dodge.width =.5) +
  geom_quasirandom(data=combined.Na24.TIL24.df %>% filter(!NotOutlier),mapping = aes(x = TotalTIL, y = meanSodium, color=population),varwidth = TRUE,alpha = 1,stroke = .2,size=.5,dodge.width =.5) +
  stat_summary(data=combined.Na24.TIL24.df,
               mapping = aes(x = TotalTIL, y = meanSodium, color=population),
               fun = median,
               fun.min = function(x) quantile(x,.25),
               fun.max = function(x) quantile(x,.75),
               geom = "crossbar",
               width = 0.5,
               show.legend = FALSE,
               position = position_dodge(width = .5),
               fill='white',
               lwd=1.3/.pt) +
  coord_cartesian(ylim = c(128.4,165.1250)) +
  scale_fill_manual(values=c('#003f5c','#bc5090')) +
  scale_color_manual(values=c('#003f5c','#bc5090')) +
  ylab('Na24') +
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

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'na24_til24.svg'),Na24.TIL24.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

### IV. CPP24_lores vs. TIL24
## Load and prepare low-resolution CPP information
# Load and filter dataframe of low-resolution CPP information
CPP24_lores.df <- read.csv('../formatted_data/daily_correlations/lo_res_CPP24_TIL24.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalTIL),
         population = 'LowResolution')

CPP24_hires.df <- read.csv('../formatted_data/daily_correlations/hi_res_CPP24_TIL24.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalTIL),
         population = 'HighResolution')

# Designate outliers (based on 1.5*interquartile range)
CPP24_lores.df <- CPP24_lores.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(CPPmean))

CPP24_hires.df <- CPP24_hires.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(CPPmean))

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
  coord_cartesian(ylim = c(22.37,105.5)) +
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

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'cpp24_til24.svg'),CPP24.TIL24.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

### IV. Distributions of TILsummaries per study sub-group
## Load and prepare formatted TILsummaries scores dataframe
# Load formatted TILsummaries scores dataframe and select relevant columns
formatted.TIL.means.maxes <- read.csv('../formatted_data/formatted_TIL_means_maxes.csv',
                                      na.strings = c("NA","NaN","", " ")) %>%
  mutate(TILmetric = factor(TILmetric,levels=c('TILmean','TILmax')))

prior.study.formatted.TIL.means.maxes <- read.csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv',
                                                  na.strings = c("NA","NaN","", " ")) %>%
  mutate(TILmetric = factor(TILmetric,levels=c('TILmean','TILmax')),
         Group = 'PriorStudySet')

compiled.TIL.means.maxes <- rbind(formatted.TIL.means.maxes,prior.study.formatted.TIL.means.maxes) %>%
  mutate(Group = plyr::mapvalues(Group,from=c("LowResolutionSet","HighResolutionSet","PriorStudySet"),to=c("ICPMR","ICPHR","PS(2016)"))) %>%
  mutate(Group = factor(Group,levels=c("ICPMR","ICPHR","PS(2016)")))

## Create and save TILmeans and TILmaxes violin plots
# Create ggplot object for plot
TIL.means.maxes.violin.plot <- compiled.TIL.means.maxes %>%
  ggplot(aes(x = factor(Group), y = value)) +
  geom_violin(aes(fill=Group),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  geom_boxplot(aes(color=Group),width=0.1,outlier.shape = NA,lwd=1.3/.pt) +
  geom_hline(yintercept = 38, color='#ffa600',alpha = 1, size=2/.pt)+
  coord_cartesian(ylim = c(0,38)) +
  scale_fill_manual(values=c('#003f5c','#bc5090','darkgray'))+
  scale_color_manual(values=c('#003f5c','#bc5090','darkgray'))+
  xlab('Population') +
  facet_wrap(~TILmetric,
             nrow = 2,
             scales = 'free_x',
             strip.position = "left") +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(0.05, "lines"),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_blank(),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    strip.placement = "outside",
    legend.position = 'none'
  )

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_mean_maxes.svg'),TIL.means.maxes.violin.plot,device= svglite,units='in',dpi=600,width=3.75,height = 3.75)

### V. TILmean-TILmax correlations per study sub-group
## Load and prepare formatted TILsummaries scores dataframe
# Load formatted TILsummaries scores dataframe and select relevant columns
long.formatted.TIL.means.maxes <- read.csv('../formatted_data/formatted_TIL_means_maxes.csv',
                                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TILmetric = factor(TILmetric,levels=c('TILmean','TILmax')),
         Group = factor(Group,levels=c('LowResolutionSet','HighResolutionSet'))) %>%
  pivot_wider(names_from = TILmetric,values_from = value)

long.prior.study.formatted.TIL.means.maxes <- read.csv('../formatted_data/prior_study_formatted_TIL_means_maxes.csv',
                                                  na.strings = c("NA","NaN","", " ")) %>%
  mutate(TILmetric = factor(TILmetric,levels=c('TILmean','TILmax')),
         Group = 'PriorStudySet') %>%
  pivot_wider(names_from = TILmetric,values_from = value)

compiled.TIL.means.maxes <- rbind(long.formatted.TIL.means.maxes,long.prior.study.formatted.TIL.means.maxes) %>%
  mutate(Group = plyr::mapvalues(Group,from=c("LowResolutionSet","HighResolutionSet","PriorStudySet"),to=c("ICPMR","ICPHR","PS(2016)"))) %>%
  mutate(Group = factor(Group,levels=c("ICPMR","ICPHR","PS(2016)")))

## Create and save TILmean-TILmax correlation plots
TILmean.TILmax.correlation.plots <- compiled.TIL.means.maxes %>%
  ggplot(aes(TILmax, TILmean)) +
  geom_quasirandom(varwidth = TRUE,alpha = 1,stroke = 0,size=.5) +
  geom_smooth(aes(color=Group),method = lm, se = TRUE) +
  scale_color_manual(values=c('#003f5c','#bc5090','darkgray'))+
  coord_cartesian(xlim = c(0,35),ylim = c(0,35))+
  facet_wrap(~Group,
             nrow = 1,
             scales = 'free') +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(aspect.ratio = 1,
        panel.background = element_blank(),
        panel.spacing = unit(0.05, "lines"),
        panel.border = element_rect(colour = 'black', fill=NA, size = 1/.pt),
        axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
        axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
        axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
        axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
        axis.line = element_blank(),
        strip.text = element_text(size=7, color = "black",face = 'bold',margin = margin(b = .5)),
        strip.background = element_blank(),
        strip.placement = "outside",
        legend.position = 'none')

# Create directory for current date and save TILmean-TILmax correlation plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_mean_til_max_correlations.svg'),TILmean.TILmax.correlation.plots,device= svglite,units='in',dpi=600,width=7.5,height = 2.9)

# Calculate correlations per group
TIL.means.maxes.rhos <- compiled.TIL.means.maxes %>%
  group_by(Group) %>%
  summarize(cor=cor(TILmax, TILmean,method='spearman'),
            beta = lm(formula = 'TILmean ~ TILmax')$coefficients[2])