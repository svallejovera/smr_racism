### Indigena Data 2018 - 2021 ###
### Sebastián Vallejo ###

rm(list = ls(all=TRUE))
# .rs.restartR()

library(tidyverse)
library(tidylog)
library(igraph)
library(wesanderson)


#### Load da data ----------------------------------------------
setwd("HERE")
load("sub_paro_1.Rdata")
load("sub_paro_2.Rdata")
load("sub_paro_3.Rdata")
summary(sub_paro_1)
summary(sub_paro_2)

#### Make new data and see if we can join it to get one layout:
sub_paro_1_df <- as_long_data_frame(sub_paro_1)
sub_paro_1_df <- sub_paro_1_df %>%
  filter(from_membership != 8) %>%
  filter(from_membership != 33)

sub_paro_2_df <- as_long_data_frame(sub_paro_2)
sub_paro_2_df <- sub_paro_2_df %>%
  filter(from_membership != 4)

sub_paro_3_df <- as_long_data_frame(sub_paro_3)
sub_paro_3_df <- sub_paro_3_df %>%
  filter(from_membership != 17)

#### Bind
sub_paro <- rbind.data.frame(sub_paro_1_df,sub_paro_2_df,sub_paro_3_df)

#### Save
setwd("/Volumes/Extreme SSD/Twitter Data/Indigena_JOP/data_final")
save(sub_paro,file = "sub_paro_df.Rdata")





