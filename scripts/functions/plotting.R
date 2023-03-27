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

# Function to determine whether row is not outlier in group based on Tukey's fences
isnt_out_tukey <- function(x, k = 1.5, na.rm = TRUE) {
  quar <- quantile(x, probs = c(0.25, 0.75), na.rm = na.rm)
  iqr <- diff(quar)
  (quar[1] - k * iqr <= x) & (x <= quar[2] + k * iqr)
}