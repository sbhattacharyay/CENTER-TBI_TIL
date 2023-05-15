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
library(lme4)
library(forcats)

# Import custom plotting functions
source('functions/plotting.R')

### II. 
## Load and prepare formatted TILsummaries scores dataframe
# Load formatted TILsummaries scores dataframe and select relevant columns
formatted.uwTIL.max.mean <- read.csv('../formatted_data/formatted_unweighted_TIL_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(uwTILSum = CSFDrainage+DecomCraniectomy+FluidLoading+Hypertonic+ICPSurgery+Mannitol+Neuromuscular+Positioning+Sedation+Temperature+Vasopressor+Ventilation) %>%
  filter(TILTimepoint<=7) %>%
  group_by(GUPI) %>%
  summarise(uwTILmax = max(uwTILSum,na.rm=T),
            uwTILmean = mean(uwTILSum,na.rm=T))

formatted.TIL.max <- read.csv('../formatted_data/formatted_TIL_max.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,TILmax) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_max.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,TIL_Basicmax)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_max.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,PILOTmax)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_max.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,TIL_1987max)) %>%
  left_join(formatted.uwTIL.max.mean%>%select(GUPI,uwTILmax)) %>%
  pivot_longer(cols=-GUPI,names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = str_remove(Scale,'max'),
         MeanMax='Max over first week in ICU')

formatted.TIL.mean <- read.csv('../formatted_data/formatted_TIL_mean.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,TILmean) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_mean.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,TIL_Basicmean)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_mean.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,PILOTmean)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_mean.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,TIL_1987mean)) %>%
  left_join(formatted.uwTIL.max.mean%>%select(GUPI,uwTILmean)) %>%
  pivot_longer(cols=-GUPI,names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = str_remove(Scale,'mean'),
         MeanMax='Mean over first week in ICU')

formatted.TIL.maxes.means <- rbind(formatted.TIL.max,formatted.TIL.mean) %>%
  mutate(Scale = factor(Scale,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')))

## Create and save TILmeans and TILmaxes violin plots
# Create ggplot object for plot
TIL.means.maxes.violin.plot <- formatted.TIL.maxes.means %>%
  ggplot(aes(x = Scale, y = Score)) +
  geom_violin(aes(fill=Scale),scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  geom_boxplot(aes(color=Scale),width=0.1,outlier.shape = NA,lwd=1.3/.pt) +
  coord_cartesian(ylim = c(0,31)) +
  scale_y_continuous(breaks = seq(0,31,5),minor_breaks = seq(0,38,1)) +
  scale_fill_manual(values=c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  scale_color_manual(values=c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  facet_wrap(~MeanMax,
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

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_mean_maxes.svg'),TIL.means.maxes.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2)

## Load formatted TIL scores over first week of ICU stay
formatted.TIL.scores <- read.csv('../formatted_data/formatted_TIL_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,TILTimepoint,TotalSum) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_Basic_scores.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,TILTimepoint,TIL_Basic)) %>%
  left_join(read.csv('../formatted_data/formatted_PILOT_scores.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,TILTimepoint,PILOTSum)) %>%
  left_join(read.csv('../formatted_data/formatted_TIL_1987_scores.csv',na.strings = c("NA","NaN","", " "))%>%select(GUPI,TILTimepoint,TIL_1987Sum)) %>%
  left_join(read.csv('../formatted_data/formatted_unweighted_TIL_scores.csv',na.strings = c("NA","NaN","", " "))%>%mutate(uwTILSum = CSFDrainage+DecomCraniectomy+FluidLoading+Hypertonic+ICPSurgery+Mannitol+Neuromuscular+Positioning+Sedation+Temperature+Vasopressor+Ventilation)%>%select(GUPI,TILTimepoint,uwTILSum)) %>%
  rename(TIL24=TotalSum,
         TILBasic24 = TIL_Basic,
         PILOT24 = PILOTSum,
         TIL198724 = TIL_1987Sum,
         uwTIL24 = uwTILSum) %>%
  filter(TILTimepoint<=7,TILTimepoint>0) %>%
  pivot_longer(cols=-c(GUPI,TILTimepoint),names_to = 'Scale',values_to = 'Score') %>%
  mutate(Scale = factor(Scale,levels=c('TIL24','uwTIL24','TILBasic24','PILOT24','TIL198724')),
         TILTimepoint = paste('Day',TILTimepoint))

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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_24s.png'),TIL.24s.violin.plot,units='in',dpi=600,width=7.5,height = 2.15)

## Load refractory status
refractory.ROC.cutpoints <- read.csv('../bootstrapping_results/compiled_ROC_refractory_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(YoudensJ = TPR-FPR,
         Scale = str_remove(Scale,'max')) %>%
  group_by(Scale) %>%
  slice(which.max(YoudensJ))

refractory.AUCs <- read.csv('../bootstrapping_results/CI_AUC_refractory_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(Scale = str_remove(Scale,'max'))

refractory.TIL.maxes <- formatted.TIL.maxes.means %>%
  left_join(read.csv('../formatted_data/formatted_outcome_and_demographics.csv',na.strings = c("NA","NaN","", " ")) %>% select(GUPI,RefractoryICP)) %>%
  filter(MeanMax=='Max over first week in ICU',
         !is.na(RefractoryICP)) %>%
  left_join(refractory.ROC.cutpoints %>% select(Scale,Threshold)) %>%
  mutate(Scale = factor(Scale,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')),
         RefractoryICP = plyr::mapvalues(RefractoryICP,c(0,1),c('No (n=707)','Yes (n=157)')))

refractory.TIL.maxes.plot <- ggplot() +
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
  geom_errorbar(data = refractory.ROC.cutpoints, aes(x = Scale, ymin = Threshold, ymax = Threshold),lwd=1.3/.pt) +
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
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.text = element_text(size = 7, color = "black",face = 'bold'),
    strip.placement = "outside",
    legend.position = 'bottom',
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    legend.margin=margin(0,0,0,0)
  )

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_maxes_refractory.svg'),refractory.TIL.maxes.plot,device= svglite,units='in',dpi=600,width=3.75,height = 2.3)


# Refractory ROC
refractory.ROC.curves <- read.csv('../bootstrapping_results/compiled_ROC_refractory_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(YoudensJ = TPR-FPR,
         Scale = str_remove(Scale,'max'))

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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_maxes_ROC.svg'),refractory.TIL.max.ROC,device= svglite,units='in',dpi=600,width=3.75,height = 2.3)





## Load correlation results
# Spearmans
chofi <- read.csv('../bootstrapping_results/CI_spearman_rhos_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((grepl('TIL',first))|(grepl('PILOT',first)))&((grepl('TIL',second))|(grepl('PILOT',second)))) %>%
  mutate(FirstMax = grepl('max',first),SecondMax = grepl('max',second)) %>%
  filter(FirstMax == SecondMax,
         metric == 'rho')

floofi <- chofi %>%
  rename(first=second,second=first)

chofi <- rbind(chofi,floofi) %>%
  mutate(BaseFirst = case_when(FirstMax ~ str_remove(first,'max'),
                               !FirstMax ~ str_remove(first,'mean')),
         BaseSecond = case_when(SecondMax ~ str_remove(second,'max'),
                                !SecondMax ~ str_remove(second,'mean'))) %>%
  mutate(BaseFirst = factor(BaseFirst,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')),
         BaseSecond = factor(BaseSecond,levels=c('TIL','uwTIL','TIL_Basic','PILOT','TIL_1987')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi))

max.scale.correlations <- chofi %>%
  filter(FirstMax) %>%
  ggplot(aes(x=BaseFirst,y=BaseSecond)) +
  geom_tile(aes(fill=median)) + 
  scale_fill_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-1,1),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(chofi$BaseSecond))) +
  geom_text(aes(label=FormattedCI,color = as.factor(as.integer(abs(median)>.75))),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  # guides(fill = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = .5, barheight = 5,ticks = FALSE))+
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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'max_scale_correlations.svg'),max.scale.correlations,device=svglite,units='in',dpi=600,width=2.5,height = 2.73)

mean.scale.correlations <- chofi %>%
  filter(!FirstMax) %>%
  ggplot(aes(x=BaseFirst,y=BaseSecond)) +
  geom_tile(aes(fill=median)) + 
  scale_fill_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-1,1),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(chofi$BaseSecond))) +
  geom_text(aes(label=FormattedCI,color = as.factor(as.integer(abs(median)>.75))),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('white','black'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  # guides(fill = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = .5, barheight = 5,ticks = FALSE))+
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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'mean_scale_correlations.svg'),mean.scale.correlations,device=svglite,units='in',dpi=600,width=2.5,height = 2.73)

# Repeated-measures correlation
chofi <- read.csv('../bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((grepl('Sum',first))|(grepl('TIL_Basic',first)))&((grepl('Sum',second))|(grepl('TIL_Basic',second)))) %>%
  filter(metric == 'rmcorr') %>%
  mutate(BaseFirst = plyr::mapvalues(first,
                                     from=c('TotalSum','TIL_Basic','PILOTSum','TIL_1987Sum','uwTILSum'),
                                     to=c('TIL24','TILBasic24','PILOT24','TIL198724','uwTIL24')),
         BaseSecond = plyr::mapvalues(second,
                                      from=c('TotalSum','TIL_Basic','PILOTSum','TIL_1987Sum','uwTILSum'),
                                      to=c('TIL24','TILBasic24','PILOT24','TIL198724','uwTIL24')))

floofi <- chofi %>%
  rename(BaseFirst=BaseSecond,BaseSecond=BaseFirst)

chofi <- rbind(chofi,floofi) %>%
  mutate(BaseFirst = factor(BaseFirst,levels=c('TIL24','uwTIL24','TILBasic24','PILOT24','TIL198724')),
         BaseSecond = factor(BaseSecond,levels=c('TIL24','uwTIL24','TILBasic24','PILOT24','TIL198724')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi))

daily.scale.correlations <- chofi %>%
  ggplot(aes(x=BaseFirst,y=BaseSecond)) +
  geom_tile(aes(fill=median)) + 
  scale_fill_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-1,1),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(chofi$BaseSecond))) +
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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'daily_scale_correlations.svg'),daily.scale.correlations,device=svglite,units='in',dpi=600,width=2.5,height = 2.73)

### Component analysis
TIL.24.components <- read.csv('../formatted_data/formatted_TIL_scores.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7) %>%
  select(-starts_with('TILPhysician')) %>%
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



TIL.24.components.plot <- TIL.24.components %>%
  ggplot(aes(x=TotalSum)) +
  geom_segment(x = 0, y = 0, xend = 31, yend = 31,alpha = 0.5,linetype = "dashed",lwd=.75, color = 'gray')+
  geom_col(aes(fill=Category,y=MedianScore)) +
  geom_text(data=TIL.24.components %>% filter(MedianScore>0),
            aes(label = MedianScore,y=MedianScore),
            position = position_stack(vjust = .5),
            color='white',
            size=5/.pt) +
  geom_col(data=TIL.24.components %>% select(TotalSum,SumInstanceCount) %>% unique() %>% mutate(ModCount = -SumInstanceCount*(31/5)/547),
           aes(y=ModCount),
           fill='lightgrey') +
  geom_text(data=TIL.24.components %>% select(TotalSum,SumInstanceCount) %>% unique() %>% mutate(ModCount = -SumInstanceCount*(31/5)/547),
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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TIL_24_components_plot.svg'),TIL.24.components.plot,device=svglite,units='in',dpi=600,width=7.5,height = 3.15)

## Component repeated-measures correlation
item.names <- c('Positioning','Sedation','Neuromuscular','CSFDrainage','FluidLoading','Vasopressor','Ventilation','Mannitol','Hypertonic','Temperature','ICPSurgery','DecomCraniectomy')
item.labels <- c('Positioning','Sedation','Paralysis','CSF drainage','Fluid loading','Vasopressors','Ventilation','Mannitol','Hypertonic saline','Temperature control','Intracranial surgery','Decompressive craniectomy')
other.score.names <- c('TotalSum','ICPmean','CPPmean','meanSodium','TILPhysicianConcernsICP','TILPhysicianConcernsCPP')
other.score.proto.labels <- c('TIL24','ICP24','CPP24','Na+24','Physician concern of ICP','Physician concern of CPP')
other.score.labels <- c('TIL24','ICP24EH','ICP24HR','CPP24EH','CPP24HR','Na+24','Physician concern of ICP','Physician concern of CPP')
CI.rmcorr.results <- read.csv('../bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Scale == 'TIL',
         metric == 'rmcorr') %>%
  filter(((first %in% item.names)&(second %in% other.score.names))|((second %in% item.names)&(first %in% other.score.names))) %>%
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
                                      to=other.score.proto.labels)) %>%
  mutate(OtherScore = factor(case_when((OtherScore=='ICP24')|(OtherScore=='CPP24')~paste0(OtherScore,sub(".*_","", Population)),
                                       TRUE ~ OtherScore),
                             levels=other.score.labels),
         BoxScore = case_when(lo<0&hi>0~NA,
                              T~median),
         LabelColor = case_when(is.na(BoxScore)|abs(median)<.46~'black',
                                T~'white'))

component.corrs <- CI.rmcorr.results %>%
  ggplot(aes(x=TILComponent,y=OtherScore)) +
  geom_tile(aes(fill=BoxScore)) + 
  scale_fill_gradient2(na.value='gray90',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-0.6141,0.6141),breaks=seq(-1,1,by=.25)) +
  scale_y_discrete(limits = rev(levels(CI.rmcorr.results$OtherScore))) +
  geom_text(aes(label=FormattedCI,color = LabelColor),family = 'Roboto Condensed',size=5/.pt) +
  scale_color_manual(values = c('black','white'),breaks = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  # guides(fill = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = .5, barheight = 5,ticks = FALSE))+
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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'component_corrs.svg'),component.corrs,device=svglite,units='in',dpi=600,width=7.5,height = 3.15)

## Component LMER coefficients
lmer.coeff.labels <- read_xlsx('../bootstrapping_results/coefficient_labels.xlsx')

CI.lmer.results <- read.csv('../bootstrapping_results/CI_mixed_effects_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Type == 'Component',
         !(Name %in% c('Group Var','Intercept')),
         metric == 'Coefficient') %>%
  left_join(lmer.coeff.labels) %>%
  mutate(DepVar = case_when(str_starts(Formula,'meanSodium')~'Na+24',
                            TRUE~paste0(sub(".*_","", Population),' ',sub("\\mean.*", "", Formula),'24'))) %>%
  mutate(Label = fct_reorder(Label, Order),
         DepVar = factor(DepVar,
                         levels=c('EH ICP24','HR ICP24','EH CPP24','HR CPP24','Na+24')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi),
         BoxScore = case_when(lo<0&hi>0~NA,
                              T~median),
         LabelColor = case_when(is.na(BoxScore)|abs(median)<3.375~'black',
                                T~'white'))

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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'lmer_coeffs.svg'),lmer.coeffs,device=svglite,units='in',dpi=600,width=7.5,height = 2.8)




### Extract confidence intervals of Spearmans
TIL_spearman_names <- read.csv('../bootstrapping_results/CI_spearman_rhos_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(first,second) %>%
  unique() %>%
  pivot_longer(cols=c(first,second)) %>%
  filter((grepl('TIL',value))|(grepl('PILOT',value))) %>%
  select(value) %>%
  unique() %>%
  .$value

CI.spearman.rhos <- read.csv('../bootstrapping_results/CI_spearman_rhos_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((first %in% TIL_spearman_names)&!(second %in% TIL_spearman_names))|(!(first %in% TIL_spearman_names)&(second %in% TIL_spearman_names))) %>%
  mutate(FirstMax = grepl('max',first),
         SecondMax = grepl('max',second),
         FirstMean = grepl('mean',first),
         SecondMean = grepl('mean',second)) %>%
  filter(!(FirstMax&SecondMean),
         !(FirstMean&SecondMax),
         metric == 'rho') %>%
  mutate(MaxOrMean = case_when(FirstMax|SecondMax~'Max',
                               FirstMean|SecondMean~'Mean')) %>%
  mutate(TILScore = case_when(first %in% TIL_spearman_names ~ first,
                              second %in% TIL_spearman_names ~ second),
         OtherScore = case_when(!(first %in% TIL_spearman_names) ~ first,
                                !(second %in% TIL_spearman_names) ~ second)) %>%
  filter(!(OtherScore %in% c('MarshallCT','RefractoryICP'))) %>%
  mutate(BaseTILScore = case_when(MaxOrMean=='Max'~str_remove(TILScore,'max'),
                                  MaxOrMean=='Mean'~str_remove(TILScore,'mean')),
         OtherScore = case_when(Population!='TIL'~paste(sub(".*_","", Population),OtherScore),
                                TRUE ~ OtherScore)) %>%
  mutate(OtherScore = case_when(OtherScore == "maxSodium" ~ "Na+max",
                                OtherScore == "meanSodium" ~ "Na+mean",
                                OtherScore == "GCSScoreBaselineDerived" ~ "GCS",
                                OtherScore == "GOSE6monthEndpointDerived" ~ "GOSE",
                                TRUE ~ OtherScore)) %>%
  mutate(BaseTILScore = factor(BaseTILScore,levels=rev(c("TIL",
                                                         "uwTIL",
                                                         "TIL_Basic",
                                                         "PILOT",
                                                         "TIL_1987"))),
         OtherScore = factor(OtherScore,levels=c("EH ICPmax",
                                                 "EH ICPmean",
                                                 "HR ICPmax",
                                                 "HR ICPmean",
                                                 "EH CPPmax",
                                                 "EH CPPmean",
                                                 "HR CPPmax",
                                                 "HR CPPmean",
                                                 "Na+max",
                                                 "Na+mean",
                                                 "GCS",
                                                 "GOSE",
                                                 "Pr(GOSE>1)",
                                                 "Pr(GOSE>3)",
                                                 "Pr(GOSE>4)",
                                                 "Pr(GOSE>5)",
                                                 "Pr(GOSE>6)",
                                                 "Pr(GOSE>7)")),
         MaxOrMean = plyr::mapvalues(MaxOrMean,
                                     from=c('Max','Mean'),
                                     to=c('Max score correlations','Mean score correlations')))

spearmans_correlation_plot <- CI.spearman.rhos %>%
  ggplot() +
  coord_cartesian(xlim = c(-.54,.54)) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_errorbarh(aes(y = OtherScore, xmin = lo, xmax = hi, color = BaseTILScore),position=position_dodge(width=.675), height=.5)+
  geom_point(aes(y = OtherScore, x = median, color = BaseTILScore),position=position_dodge(width=.675),size=1)+
  scale_y_discrete(limits=rev) +
  xlab("Spearman's correlation coefficient (p)")+
  scale_color_manual(values=rev(c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600')),guide=guide_legend(title = 'Scale',reverse = TRUE))+
  facet_wrap(~MaxOrMean,
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


### Extract confidence intervals of RMcorrs
TIL_rmcorrs_names <- read.csv('../bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(first,second) %>%
  unique() %>%
  pivot_longer(cols=c(first,second)) %>%
  filter((grepl('Sum',value))|(grepl('TIL_Basic',value))) %>%
  select(value) %>%
  unique() %>%
  .$value

CI.rmcorrs <- read.csv('../bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(((first %in% TIL_rmcorrs_names)&!(second %in% TIL_rmcorrs_names))|(!(first %in% TIL_rmcorrs_names)&(second %in% TIL_rmcorrs_names))) %>%
  filter(metric == 'rmcorr') %>%
  mutate(OtherScore = case_when(!(first %in% TIL_rmcorrs_names) ~ first,
                                !(second %in% TIL_rmcorrs_names) ~ second)) %>%
  filter(OtherScore %in% c('ICPmean','CPPmean','meanSodium','TILPhysicianConcernsICP','TILPhysicianConcernsCPP')) %>%
  mutate(OtherScore = case_when(OtherScore == "ICPmean" ~ "ICP24",
                                OtherScore == "CPPmean" ~ "CPP24",
                                OtherScore == "meanSodium" ~ "Na+24",
                                OtherScore == "TILPhysicianConcernsICP" ~ "Physician concern of ICP",
                                OtherScore == "TILPhysicianConcernsCPP" ~ "Physician concern of CPP",
                                TRUE ~ OtherScore)) %>%
  mutate(OtherScore = case_when(Population!='TIL'~paste(sub(".*_","", Population),OtherScore),
                                TRUE ~ OtherScore)) %>%
  mutate(Scale = factor(Scale,levels=rev(c("TIL",
                                           "uwTIL",
                                           "TIL_Basic",
                                           "PILOT",
                                           "TIL_1987"))),
         OtherScore = factor(OtherScore,levels=c("EH ICP24",
                                                 "HR ICP24",
                                                 "EH CPP24",
                                                 "HR CPP24",
                                                 "Na+24",
                                                 "Physician concern of ICP",
                                                 "Physician concern of CPP")))

rm_correlation_plot <- CI.rmcorrs %>%
  ggplot() +
  coord_cartesian(xlim = c(-.465,.465)) +
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


### Extract confidence intervals of mixed effect coefficients
TIL_mixed_effects_names <- read.csv('../bootstrapping_results/CI_mixed_effects_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Type=='TotalScore',
         metric=='Coefficient',
         !(Name%in%c('Intercept','Group Var'))) %>%
  select(Formula,Name,Scale) %>%
  unique() %>%
  mutate(Target=sub(" ~.*", "", Formula)) %>%
  filter(((grepl('Sum',Name))|(grepl('TIL_Basic',Name)))&(Target!='TotalSum')) %>%
  select(Formula) %>%
  unique() %>%
  .$Formula

CI.mixed.effects <- read.csv('../bootstrapping_results/CI_mixed_effects_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Formula %in% TIL_mixed_effects_names,
         Type=='TotalScore',
         metric=='Coefficient',
         !(Name%in%c('Intercept','Group Var')),
         metric == 'Coefficient')%>%
  mutate(Target=sub(" ~.*", "", Formula)) %>%
  mutate(Target = case_when(Target=='ICPmean'~'ICP24',
                            Target=='CPPmean'~'CPP24',
                            Target=='meanSodium'~'Na+24')) %>%
  mutate(Target = case_when(Population!='TIL'~paste(sub(".*_","", Population),Target),
                            TRUE ~ Target)) %>%
  mutate(FormattedFormula = paste0(Target,'~Scale')) %>%
  mutate(Scale = factor(Scale,levels=rev(c("TIL",
                                           "uwTIL",
                                           "TIL_Basic",
                                           "PILOT",
                                           "TIL_1987"))),
         FormattedFormula = factor(FormattedFormula,levels=c("EH ICP24~Scale",
                                                             "HR ICP24~Scale",
                                                             "EH CPP24~Scale",
                                                             "HR CPP24~Scale",
                                                             "Na+24~Scale")))

mixed_effect_correlation_plot <- CI.mixed.effects %>%
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


### IV. ICP24 vs. TIL24
## Load and prepare low-resolution ICP information
# Load and filter dataframe of low-resolution ICP information
ICP24_lores.df <- read.csv('../formatted_data/formatted_low_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'LowResolution') %>%
  filter(TILTimepoint<=7)

ICP24_hires.df <- read.csv('../formatted_data/formatted_high_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'HighResolution') %>%
  filter(TILTimepoint<=7)

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

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'icp24_til24.png'),ICP24.TIL24.violin.plot,units='in',dpi=600,width=7.5,height = 2.3)

### IV. Na24 vs. TIL24
## Load and prepare low-resolution Na information
# Load and filter dataframe of Na information
Na24.df <- read.csv('../formatted_data/formatted_daily_sodium_values.csv',
                    na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalSum)) %>%
  filter(TILTimepoint<=7) %>%
  group_by(TotalTIL) %>%
  mutate(NotOutlier = isnt_out_tukey(meanSodium))

## Create and save TIL24 vs. Na24 violin plots
# Create ggplot object for plot
Na24.TIL24.violin.plot <- ggplot() +
  geom_violin(data=Na24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = meanSodium),fill='#7a5195',scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(data=Na24.df %>% filter(NotOutlier),mapping = aes(x = TotalTIL, y = meanSodium),varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  geom_quasirandom(data=Na24.df %>% filter(!NotOutlier),mapping = aes(x = TotalTIL, y = meanSodium),color='darkred',varwidth = TRUE,alpha = 1,stroke = .2,size=.5,dodge.width =.5) +
  geom_boxplot(data=Na24.df,mapping=aes(x = TotalTIL, y = meanSodium),width=0.2,outlier.shape = NA,lwd=1.3/.pt) +
  coord_cartesian(ylim = c(128.4,165.1250)) +
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
ggsave(file.path('../plots',Sys.Date(),'na24_til24.png'),Na24.TIL24.violin.plot,units='in',dpi=600,width=7.5,height = 2.3)

### IV. CPP24 vs. TIL24
## Load and prepare low-resolution CPP information
# Load and filter dataframe of low-resolution CPP information
CPP24_lores.df <- read.csv('../formatted_data/formatted_low_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'LowResolution') %>%
  filter(TILTimepoint<=7)

CPP24_hires.df <- read.csv('../formatted_data/formatted_high_resolution_values.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  mutate(TotalTIL=as.factor(TotalSum),
         population = 'HighResolution') %>%
  filter(TILTimepoint<=7)

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

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'cpp24_til24.png'),CPP24.TIL24.violin.plot,units='in',dpi=600,width=7.5,height = 2.3)










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

#### component correlation results
# repeated-measures correlation
item.names <- c('Positioning','Sedation','Neuromuscular','Paralysis','CSFDrainage','Ventricular','FluidLoading','Vasopressor','Hyperventilation','Ventilation','Mannitol','Hypertonic','Temperature','ICPSurgery','DecomCraniectomy')
chofi <- read.csv('../bootstrapping_results/CI_rmcorr_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter((first %in% item.names)&(second %in% item.names),
         metric=='rmcorr') %>%
  mutate(first = factor(first,levels=item.names),
         second = factor(second,levels=item.names))

floofi <- chofi %>%
  rename(first=second,second=first)

chofi <- rbind(chofi,floofi) %>%
  mutate(Scale = factor(Scale,levels=c('TIL','uwTIL','PILOT','TIL_1987')),
         FormattedCI = sprintf('%.2f\n(%.2f–%.2f)',median,lo,hi),
         BoxScore = case_when(lo<0&hi>0~NA,
                              T~median),
         LabelColor = case_when(is.na(BoxScore)|abs(median)<.27~'black',
                                T~'white'))

component.correlations <- chofi %>%
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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'component_intra_correlations.png'),units='in',dpi=600,width=7.5,height = 8)

#### TIL cutoffs
## Calculate ideal TIL-Basic cutpoints for each scale
TIL.Basic.ROC.cutpoints <- read.csv('../bootstrapping_results/compiled_ROC_TILBasic_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(YoudensJ = TPR-FPR,
         Label = str_replace(Label,'TIL','TIL(Basic)24')) %>%
  group_by(Scale,Label) %>%
  slice(which.max(YoudensJ)) %>%
  filter(Scale == 'TIL')

TIL.Basic.AUCs <- read.csv('../bootstrapping_results/CI_AUC_TILBasic_results.csv',na.strings = c("NA","NaN","", " "))

# TIL_Basic ROC cutoffs
TIL.Basic.ROC.curves <- read.csv('../bootstrapping_results/compiled_ROC_TILBasic_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(YoudensJ = TPR-FPR,
         Label = str_replace(Label,'TIL','TIL(Basic)24'))

TIL.Basic.cutoffs.ROC <- TIL.Basic.ROC.curves %>%
  filter(Scale == 'TIL') %>%
  ggplot() +
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",lwd=.75, color = 'gray')+
  geom_line(aes(x=FPR,y=TPR,color=Threshold),lwd=1.3) +
  scale_color_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=8,limits = c(0,31),breaks=seq(0,31,by=5)) + 
  geom_point(data=TIL.Basic.ROC.cutpoints,mapping = aes(x=FPR,y=TPR),fill=NA, color="darkred", size=5, shape = 1)+
  xlab("False positive rate") +
  ylab("True positive rate") +
  facet_wrap(~Label,
             scales = 'free',
             nrow=1) +
  guides(color = guide_colourbar(title="TIL24 threshold for TIL(Basic)24 (>=)",barwidth = 10, barheight = .5,ticks = FALSE))+
  theme_classic(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    aspect.ratio = 1,
    panel.border = element_rect(colour = 'black', fill=NA, linewidth = .75),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line"),
    legend.margin=margin(0,0,0,0),
    plot.margin=margin(0,0,0,0),
    strip.text = element_text(size=7, color = "black",face = 'bold',margin = margin(b = .5)),
    strip.background = element_blank()
  )

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'til_basic_ROC.svg'),TIL.Basic.cutoffs.ROC,device= svglite,units='in',dpi=600,width=7.5,height = 2.45)

## Information content
#
TIL.Basic.TIL.information <- read.csv('../bootstrapping_results/compiled_MI_entropy_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint!=0) %>%
  group_by(TILTimepoint,METRIC) %>%
  mutate(group_idx = row_number()) %>%
  select(group_idx,TILTimepoint,METRIC,TotalSum,TIL_Basic) %>%
  pivot_longer(cols=c(TotalSum,TIL_Basic),
               names_to = 'Scale') %>%
  filter(!((METRIC=='MutualInfo')&(Scale=='TIL_Basic'))) %>%
  mutate(Scale = case_when(Scale=='TotalSum'~'TIL',
                           TRUE~'TIL_Basic'),
         METRIC = paste0(METRIC,'_',Scale)) %>%
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

information_coverage <- ggplot() +
  geom_line(data=TIL.Basic.TIL.information%>%filter(TILTimepoint!='Max'),
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

dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'info_coverage.svg'),information_coverage,device= svglite,units='in',dpi=600,width=3.75,height = 2.15)















## Load and prepare Spearman's Rho results
# Load Spearman's Rho confidence interval dataframe
CI_spearman_rhos <- rbind(read.csv('../bootstrapping_results/CI_spearman_rhos_results.csv',
                                   na.strings = c("NA","NaN","", " ")),
                          read.csv('../bootstrapping_results/1987_CI_spearman_rhos_results.csv',
                                   na.strings = c("NA","NaN","", " "))) %>%
  mutate(FormattedCombos = factor(paste(second,first,sep = ' vs. '),
                                  levels = c('TIL_1987mean vs. TILmean',
                                             'TIL_1987max vs. TILmax',
                                             'ICPmean vs. TILmean',
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
ggsave(file.path('../plots',Sys.Date(),'spearmans_correlation.svg'),spearmans_correlation_plot,device= svglite,units='in',dpi=600,width=3.75,height = 5.91)

### III.
## Load and prepare mixed effects modelling results
# Load repeated measures correlation confidence interval dataframe
CI_rm_correlations <- rbind(read.csv('../bootstrapping_results/CI_rm_correlation_results.csv',
                                     na.strings = c("NA","NaN","", " ")),
                            read.csv('../bootstrapping_results/1987_CI_rm_correlation_results.csv',
                                     na.strings = c("NA","NaN","", " "))) %>%
  mutate(FormattedCombos = factor(paste(second,first,sep = ' vs. '),
                                  levels = c('TIL_1987_24 vs. TIL24',
                                             'ICP24 vs. TIL24',
                                             'CPP24 vs. TIL24',
                                             'NA24 vs. TIL24')),
         population = factor(population,
                             levels = c('PriorStudy','HighResolution','LowResolution')))

# Load mixed effects modelling confidence interval dataframe
CI_mixed_effects <- rbind(read.csv('../bootstrapping_results/CI_mixed_effects_results.csv',
                                   na.strings = c("NA","NaN","", " ")),
                          read.csv('../bootstrapping_results/1987_CI_mixed_effects_results.csv',
                                   na.strings = c("NA","NaN","", " "))) %>%
  mutate(FormattedCombos = factor(paste(second,first,sep = ' ~ '),
                                  levels = c('TIL_1987_24 ~ TIL24',
                                             'ICP24 ~ TIL24',
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
ggsave(file.path('../plots',Sys.Date(),'mixed_effect_coefficients.svg'),mixed_effect_coefficients,device= svglite,units='in',dpi=600,width=3.75,height = 2.25)

## Create `ggplot` object
rm_correlation_coefficients <- CI_rm_correlations %>%
  ggplot() +
  coord_cartesian(xlim = c(-0.2,0.82)) +
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
ggsave(file.path('../plots',Sys.Date(),'rm_correlation_coefficients.svg'),rm_correlation_coefficients,device= svglite,units='in',dpi=600,width=3.75,height = 2.25)

## Distribution of TIL vs. TIL_Basic
formatted.TIL.Basic.scores <- read.csv('../formatted_data/formatted_TIL_Basic_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(TIL_Basic = factor(TIL_Basic)) %>%
  filter(!((TIL_Basic==0)&(TotalSum==5)))
TIL.Basic.violin.plot <- formatted.TIL.Basic.scores %>%
  ggplot(aes(x = TIL_Basic, y = TotalSum)) +
  geom_violin(scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5,fill='#7a5195') +
  geom_quasirandom(varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  geom_boxplot(width=0.1,outlier.shape = NA,lwd=1.3/.pt,color='#7a5195') +
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
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TIL_Basic_violin.svg'),TIL.Basic.violin.plot,device= svglite,units='in',dpi=600,width=3.75,height = 2.15)
ggsave(file.path('../plots',Sys.Date(),'TIL_Basic_violin.png'),TIL.Basic.violin.plot,units='in',dpi=600,width=3.75,height = 2.15)






### Calculate Component Counts
formatted.uwTIL.scores <- read.csv('../formatted_data/formatted_unweighted_TIL_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7) %>%
  select(-c(TILTimepoint,TILDate,ICUAdmTimeStamp,ICUDischTimeStamp,TotalSum,starts_with('TILPhysician'))) %>%
  pivot_longer(cols = -GUPI) %>%
  filter(value!=0) %>%
  group_by(name,value) %>%
  mutate(uniq_pt_count=n_distinct(GUPI),
         instance_count=n()) %>%
  group_by(name) %>%
  mutate(overall_uniq_pt_count = n_distinct(GUPI),
         overall_instance_count = n()) %>%
  select(name,value,uniq_pt_count,overall_uniq_pt_count,instance_count,overall_instance_count) %>%
  unique() %>%
  mutate(uniq_pt_perc = 100*uniq_pt_count/873,
         overall_uniq_pt_perc = 100*overall_uniq_pt_count/873,
         formatted_entry_1 = sprintf('%d (%.1f%%)',uniq_pt_count,uniq_pt_perc),
         formatted_entry_2 = sprintf('%d (%.1f%%)',overall_uniq_pt_count,overall_uniq_pt_perc))

















### IV. Distributions of TILsummaries per study sub-group

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

### VI. Na24-ICP24 correlations per study population
## Load and prepare formatted Na24-ICP24 dataframe
# Load formatted Na24-ICP24 dataframe for low resolution data
ICP24_Na24_lo_res.df <- inner_join(read.csv('../formatted_data/daily_correlations/lo_res_Na24_TIL24.csv',
                                            na.strings = c("NA","NaN","", " ")),
                                   read.csv('../formatted_data/daily_correlations/lo_res_ICP24_TIL24.csv',
                                            na.strings = c("NA","NaN","", " "))) %>%
  mutate(Group = 'ICPMR')

# Load formatted Na24-ICP24 dataframe for high resolution data
ICP24_Na24_hi_res.df <- inner_join(read.csv('../formatted_data/daily_correlations/hi_res_Na24_TIL24.csv',
                                            na.strings = c("NA","NaN","", " ")),
                                   read.csv('../formatted_data/daily_correlations/hi_res_ICP24_TIL24.csv',
                                            na.strings = c("NA","NaN","", " "))) %>%
  mutate(Group = 'ICPHR') %>%
  left_join(read.csv('../formatted_data/formatted_TIL_scores.csv',na.strings = c("NA","NaN","", " ")) %>%
              select(GUPI,DateComponent,TILTimepoint) %>%
              unique())

# Compile both low- and high-resolution data
ICP24_Na24.df <- rbind(ICP24_Na24_lo_res.df,ICP24_Na24_hi_res.df) %>%
  mutate(Group = factor(Group,levels=c("ICPMR","ICPHR")))

## Create and save TILmean-TILmax correlation plots
ICP24.Na24.correlation.plots <- ICP24_Na24.df %>%
  ggplot(aes(meanSodium, ICPmean)) +
  geom_quasirandom(varwidth = TRUE,alpha = 1,stroke = 0,size=.5) +
  geom_smooth(aes(color=Group),method = lm, se = TRUE) +
  scale_color_manual(values=c('#003f5c','#bc5090'))+
  coord_cartesian(xlim = c(128.4,165.1250),ylim = c(-4,40))+
  scale_x_continuous(limits = c(128.4,165.1250))+
  scale_y_continuous(limits = c(-4,40))+
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
ggsave(file.path('../plots',Sys.Date(),'icp_24_na_24_correlations.svg'),ICP24.Na24.correlation.plots,device= svglite,units='in',dpi=600,width=5.08,height = 2.9)

# Calculate correlations per group
ICP24_Na24_lores.rmcorr = rmcorr(GUPI,TotalTIL,meanSodium,ICP24_Na24.df %>% filter(Group=='ICPMR'))
ICP24_Na24_hires.rmcorr = rmcorr(GUPI,TotalTIL,meanSodium,ICP24_Na24.df %>% filter(Group=='ICPHR'))

ICP24_Na24_lores.mixed = lmer(ICPmean ~ meanSodium + (1 | GUPI), data = ICP24_Na24.df %>% filter(Group=='ICPMR'))
ICP24_Na24_hires.mixed = lmer(ICPmean ~ meanSodium + (1 | GUPI), data = ICP24_Na24.df %>% filter(Group=='ICPHR'))

TIL.means.maxes.rhos <- compiled.TIL.means.maxes %>%
  group_by(Group) %>%
  summarize(cor=cor(TILmax, TILmean,method='spearman'),
            beta = lm(formula = 'TILmean ~ TILmax')$coefficients[2])

### Component plots
formatted.TIL.max <- read.csv('../formatted_data/formatted_TIL_max.csv',na.strings = c("NA","NaN","", " "))

###### Calculate R2 for TIL explanation
kofi <- read.csv('../bootstrapping_results/compiled_mixed_effects_results.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(Formula %in% c("TotalSum ~ meanSodium + ICPmean","meanSodium ~ TotalSum","ICPmean ~ TotalSum")) %>%
  select(Formula,ResidVar,RandomEffectVar,PredictedValueVar,FittedValueVar,Population,resample_idx) %>%
  unique() %>%
  mutate(margR2 = FittedValueVar/(RandomEffectVar+FittedValueVar+ResidVar),
         condR2 = (RandomEffectVar+FittedValueVar)/(RandomEffectVar+FittedValueVar+ResidVar)) %>%
  pivot_longer(cols = c(margR2,condR2)) %>%
  group_by(Population,Formula,name) %>%
  summarise(median = median(value),
            lo = quantile(value,.025),
            hi = quantile(value,.975),
            count = n())

formatted.Na.scores <- read.csv('../formatted_data/formatted_daily_sodium_values.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(TILTimepoint<=7) %>%
  mutate(HypertonicSalineDose = factor(case_when(Hypertonic==3~'>0.3g/kg/24h',
                                                 Hypertonic==2~'≤0.3g/kg/24h',
                                                 Hypertonic==0~'0g/kg/24h'),
                                       levels=c('0g/kg/24h','≤0.3g/kg/24h','>0.3g/kg/24h'))) %>%
  group_by(HypertonicSalineDose) %>%
  mutate(NotOutlier = isnt_out_tukey(meanSodium))

Na24.HTS24.violin.plot <- ggplot() +
  geom_violin(data=formatted.Na.scores,mapping = aes(x = HypertonicSalineDose, y = ChangeInSodium),fill='#7a5195',scale = "width",trim=TRUE,lwd=1.3/.pt,alpha=.5) +
  geom_quasirandom(data=formatted.Na.scores,mapping = aes(x = HypertonicSalineDose, y = ChangeInSodium),varwidth = TRUE,alpha = 0.25,stroke = 0,size=.5) +
  #geom_quasirandom(data=formatted.Na.scores %>% filter(!NotOutlier),mapping = aes(x = HypertonicSalineDose, y = meanSodium),color='darkred',varwidth = TRUE,alpha = 1,stroke = .2,size=.5,dodge.width =.5) +
  geom_boxplot(data=formatted.Na.scores,mapping=aes(x = HypertonicSalineDose, y = ChangeInSodium),width=0.2,outlier.shape = NA,lwd=1.3/.pt) +
  coord_cartesian(ylim = c(-15,20)) +
  #scale_y_continuous(breaks=seq(120,170,by=10))+
  ylab('delta_Na+') +
  xlab('HTS') +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )


trial <- formatted.Na.scores %>%
  drop_na(ChangeInSodium) %>%
  group_by(TILTimepoint) %>%
  summarise(rho=cor(x=Hypertonic, y=ChangeInSodium, method = c("spearman")),
            pvals=cor.test(x=Hypertonic, y=ChangeInSodium, method = c("spearman"))$p.value)

Na.HTS.rmcorr = rmcorr(GUPI,Hypertonic,ChangeInSodium,formatted.Na.scores%>%drop_na(ChangeInSodium))


mixed.lmer <- lmer(ChangeInSodium ~ HTS + (1|GUPI), data = formatted.Na.scores%>%drop_na(ChangeInSodium)%>%mutate(HTS=factor(Hypertonic)))
summary(mixed.lmer)

mutate(Target = case_when(Population!='TIL'~paste(sub(".*_","", Population),Target),
                          TRUE ~ Target))


