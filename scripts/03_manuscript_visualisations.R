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

# Import custom plotting functions
source('functions/plotting.R')

### II. ICP24_lores vs. TIL24
## Load and prepare low-resolution ICP information
# Load and filter dataframe of low-resolution ICP information
ICP24_lores.df <- read.csv('../formatted_data/formatted_low_resolution_neuromonitoring.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  filter(variable == 'ICP24') %>%
  mutate(TotalTIL=as.factor(TotalTIL))

# Designate outliers (based on 1.5*interquartile range)
ICP24_lores.df <- ICP24_lores.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(value))

## Create and save TIL24 vs. ICP24 violin plots
# Create ggplot object for plot
ICP24_lores.TIL24.violin.plot <- ggplot() +
  geom_violin(data=ICP24_lores.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = value),scale = "width",trim=TRUE,fill='#9cc3dc',lwd=1.3/.pt) +
  geom_quasirandom(data=ICP24_lores.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = value),varwidth = TRUE,alpha = 0.15,stroke = 0,size=.5) +
  geom_quasirandom(data=ICP24_lores.df %>% filter(!NotOutlier),mapping = aes(x = TotalTIL, y = value),varwidth = TRUE,alpha = 1,color='red',stroke = .2,size=.5) +
  geom_boxplot(data=ICP24_lores.df,mapping = aes(x = TotalTIL, y = value),width=0.1,outlier.shape = NA,lwd=1.3/.pt) +
  coord_cartesian(ylim = c(-4,40)) +
  ylab('ICP24') +
  xlab('TIL24') +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold')
  )

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'icp24_lores_til24.svg'),ICP24_lores.TIL24.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

### III. CPP24_lores vs. TIL24
## Load and prepare low-resolution CPP information
# Load and filter dataframe of low-resolution CPP information
CPP24_lores.df <- read.csv('../formatted_data/formatted_low_resolution_neuromonitoring.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  filter(variable == 'CPP24') %>%
  mutate(TotalTIL=as.factor(TotalTIL))

# Designate outliers (based on 1.5*interquartile range)
CPP24_lores.df <- CPP24_lores.df %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(value))

## Create and save TIL24 vs. CPP24 violin plots
# Create ggplot object for plot
CPP24_lores.TIL24.violin.plot <- ggplot() +
  geom_violin(data=CPP24_lores.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = value),scale = "width",trim=TRUE,fill='#55d19a',lwd=1.3/.pt) +
  geom_quasirandom(data=CPP24_lores.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = value),varwidth = TRUE,alpha = 0.15,stroke = 0,size=.5) +
  geom_quasirandom(data=CPP24_lores.df %>% filter(!NotOutlier),mapping = aes(x = TotalTIL, y = value),varwidth = TRUE,alpha = 1,color='red',stroke = .2,size=.5) +
  geom_boxplot(data=CPP24_lores.df,mapping = aes(x = TotalTIL, y = value),width=0.1,outlier.shape = NA,lwd=1.3/.pt) +
  coord_cartesian(ylim = c(0,105)) +
  ylab('CPP24') +
  xlab('TIL24') +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold')
  )

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'cpp24_lores_til24.svg'),CPP24_lores.TIL24.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

### IV. Distributions of TILsummaries per study sub-group
## Load and prepare formatted TILsummaries scores dataframe
# Load formatted TILsummaries scores dataframe and select relevant columns
formatted.TIL.means.maxes <- read.csv('../formatted_data/formatted_TIL_means_maxes.csv',
                                      na.strings = c("NA","NaN","", " ")) %>%
  mutate(TILmetric = factor(TILmetric,levels=c('TILmean','TILmax')))

## Create and save TILmeans and TILmaxes violin plots
# Create ggplot object for plot
TIL.means.maxes.violin.plot <- formatted.TIL.means.maxes %>%
  ggplot(aes(x = factor(Group), y = value)) +
  geom_violin(aes(fill=TILmetric),scale = "width",trim=TRUE,lwd=1.3/.pt) +
  geom_quasirandom(varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  geom_boxplot(width=0.1,outlier.shape = NA,lwd=1.3/.pt) +
  geom_hline(yintercept = 38, color='#ffa600',alpha = 1, size=2/.pt)+
  coord_cartesian(ylim = c(0,38)) +
  scale_fill_manual(values=c('#9cc3dc','#55d19a'))+
  xlab('Study subgroup') +
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
         Group = factor(Group,levels=c('Total','ICP_lo_res','ICP_hi_res'))) %>%
  pivot_wider(names_from = TILmetric,values_from = value)

## Create and save TILmean-TILmax correlation plots
TILmean.TILmax.correlation.plots <- long.formatted.TIL.means.maxes %>%
  ggplot(aes(TILmax, TILmean)) +
  geom_quasirandom(varwidth = TRUE,alpha = 1,stroke = 0,size=.5) +
  geom_smooth(method = lm, se = TRUE,color='#bc5090') +
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
TIL.means.maxes.rhos <- long.formatted.TIL.means.maxes %>%
  group_by(Group) %>%
  summarize(cor=cor(TILmax, TILmean,method='spearman'))