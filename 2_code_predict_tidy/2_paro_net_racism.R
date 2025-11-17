### Indigena Data 2018 - 2021 ###
### Diana Dávila Gordillo, Joan C Timoneda, Sebastián Vallejo Vera ###

rm(list = ls(all=TRUE))
# .rs.restartR()

library(tidyverse)
library(tidylog)
library(igraph)
library(wesanderson)
library(writexl)

# Load da data ----------------------------------------------

## Network data: ----
setwd("here")
load("sub_paro_final.Rdata")

summary(sub_paro_final) 

## Predicted data: ----
text_predict <- read_csv("here/text_paro_predicted_clean.csv")
text_predict$racism <- ifelse(str_detect(text_predict$prediction,"No Racism"), "No Racism", NA)
text_predict$racism <- ifelse(str_detect(text_predict$prediction,"Covert Racism"), "Covert Racism", text_predict$racism)
text_predict$racism <- ifelse(str_detect(text_predict$prediction,"Overt Racism"), "Overt Racism", text_predict$racism)

table(text_predict$racism)

## Example paper:
covert_indi <- text_predict %>%
  filter(from_member_label=="1. Indígena Community") %>%
  filter(racism=="Overt Racism")

# Merge both datasets ----
text_id <- E(sub_paro_final)$text_id
text_id <- as.data.frame(text_id)

text_id <- left_join(text_id,text_predict[,c("text_id","racism")])

text_id$text_id[1] == E(sub_paro_final)$text_id[1]
text_id$text_id[2020487] == E(sub_paro_final)$text_id[2020487]

# Add to network ----
E(sub_paro_final)$racismo <- text_id$racism

# Save -----
setwd("here")
save(sub_paro_final, file = "sub_paro_final_pred.Rdata")



