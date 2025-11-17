### Paro Indigena 2018-2021 ###
### Diana Dávila Gordillo, Joan C Timoneda, Sebastián Vallejo Vera ###
### H1 ###

rm(list = ls(all=TRUE))
# .rs.restartR()

library(tidyverse)
library(igraph)
library(wesanderson)
library(MASS)
library(nnet) # for the multinom()-function
library(MNLpred)

#### Load da data ----------------------------------------------
setwd("here")
load("sub_indigena_pred.Rdata")

summary(sub_indigena)

V(sub_indigena)$membership_name[V(sub_indigena)$membership==2] <-  "Indigena"
V(sub_indigena)$membership_name[V(sub_indigena)$membership==3] <-  "Government"
V(sub_indigena)$membership_name[V(sub_indigena)$membership==23] <-  "APais"
V(sub_indigena)$membership_name[V(sub_indigena)$membership==11] <-  "International Right"

sub_indigena_df <- as_long_data_frame(sub_indigena)
sub_indigena_df$racismo <- ifelse(is.na(sub_indigena_df$racismo),"No Racism",sub_indigena_df$racismo)

# H1 b: In Degree and Prob of Racism (Produce) -----

sub_indigena_df$racismo <- factor(sub_indigena_df$racismo, levels = c("No Racism", "Covert Racism", "Overt Racism"))
sub_indigena_df$from_ind_ln <- log(sub_indigena_df$from_ind+1)
sub_indigena_df$to_ind_ln <- log(sub_indigena_df$to_ind)

sub_indigena_df <- sub_indigena_df %>%
  group_by(from) %>%
  mutate(total_tweets_from = n(),
         ln_total_tweets_from = log(total_tweets_from)) %>%
  ungroup() 

to_indigena_df <- sub_indigena_df %>%
  filter(to_membership_name=="Government" | to_membership_name=="Indigena") %>%
  group_by(to) %>%
  mutate(total_tweets_to = n(),
         ln_total_tweets_to = log(total_tweets_to),
         to_Indigena = ifelse(to_membership_name=="Indigena",1,0)) %>%
  ungroup() %>%
  distinct(text_id,.keep_all=T) 

# Comparison Overt Racism in Indigena community
comparison_ind <- to_indigena_df[to_indigena_df$to_Indigena==1 & to_indigena_df$racismo=="Overt Racism",]
comparison_gov <- to_indigena_df[to_indigena_df$to_Indigena==0 & to_indigena_df$racismo=="Overt Racism",]
comparison_gov <- comparison_gov %>%
  distinct(to_name)

writexl::write_xlsx(comparison_ind,path = "/Users/sebastianvallejo/Library/CloudStorage/OneDrive-UniversityOfHouston/Papers and Chapters/Racism NPL/data_final/data/comparison_ind.xlsx")
writexl::write_xlsx(comparison_gov,path = "/Users/sebastianvallejo/Library/CloudStorage/OneDrive-UniversityOfHouston/Papers and Chapters/Racism NPL/data_final/data/comparison_gov.xlsx")

bots_ind <- readxl::read_xlsx('/Users/sebastianvallejo/Library/CloudStorage/OneDrive-UniversityOfHouston/Papers and Chapters/Racism NPL/data_final/data/bots_ind.xlsx')
bots_ind$community <- "Pro-Indigena"
bots_gov <- readxl::read_xlsx('/Users/sebastianvallejo/Library/CloudStorage/OneDrive-UniversityOfHouston/Papers and Chapters/Racism NPL/data_final/data/bots_gov.xlsx')
bots_gov$community <- "Pro-Government"

bots_ind %>%
  distinct(user_id,.keep_all = T) %>%
  summarise(mean(bot_score),
            sd(bot_score))

bots_gov %>%
  distinct(user_id,.keep_all = T) %>%
  summarise(mean(bot_score),
            sd(bot_score))

bots_ind %>%
  bind_rows(bots_gov) %>%
  group_by(community) %>%
  mutate(mean_bot = mean(bot_score)) %>%
  ungroup() %>%
  ggplot(aes(x=bot_score, fill = community, color= community)) +
  geom_density(alpha=.6)
 

to_indigena_df %>%
  group_by(to_Indigena,racismo) %>%
  summarise(mean(to_ind_ln)) 

to_indigena_df %>%
  ungroup() %>%
  group_by(to_Indigena,racismo) %>%
  summarise(mean(as.numeric(rt_user_following_count),na.rm=T))

## Multinomial model ----

h1_b <- multinom(racismo ~ to_ind_ln + to_Indigena, 
                 data = to_indigena_df,
                 Hess = TRUE)
summary(h1_b)

### Predicted probabilities ----

#### In Degree ----
pred_h1_b_ind <- mnl_pred_ova(model = h1_b,
                              data = to_indigena_df,
                              x = "to_ind_ln",
                              by = 1,
                              seed = "random", # default
                              nsim = 35, # faster
                              probs = c(0.025, 0.975)) 

ggplot(data = pred_h1_b_ind$plotdata, aes(x = to_ind_ln, 
                                          y = mean,
                                          ymin = lower, ymax = upper,
                                          fill=racismo,color=racismo)) +
  geom_ribbon() + # Confidence intervals
  geom_line() + # Mean
  facet_wrap(~racismo, scales = "free_y", ncol = 3) +
  theme_minimal() +
  scale_color_viridis_d(option="E",alpha = 0.7) +
  scale_fill_viridis_d(option="E",alpha = 0.4) +
  theme(legend.position = "bottom") +
  labs(x="In-Degree",y="Predicted Prob.", color="", fill = "") 

#### Community ----
pred_h1_b_comm <- mnl_pred_ova(model = h1_b,
                               data = to_indigena_df,
                               x = "to_Indigena",
                               by = 1,
                               seed = "random", # default
                               nsim = 35, # faster
                               probs = c(0.025, 0.975)) 

pred_h1_b_comm$plotdata$community <- ifelse(pred_h1_b_comm$plotdata$to_Indigena==1,"Pro-Indigena","Pro-Government")

ggplot(data = pred_h1_b_comm$plotdata, aes(x = (community), 
                                           y = mean,
                                           ymin = lower, ymax = upper,
                                           fill=racismo,color=racismo)) +
  geom_pointrange() + # Confidence intervals
  geom_point() + # Mean
  facet_wrap(~racismo, scales = "free_y", ncol = 3)  +
  theme_minimal() +
  scale_color_viridis_d(option="E",alpha = 0.7) +
  scale_fill_viridis_d(option="E",alpha = 0.4) +
  theme(legend.position = "bottom") +
  labs(x="",y="Predicted Prob.", color="", fill = "") 

#### In Degree + Community ----
pred_ind_comm <- mnl_fd_ova(model = h1_b,
                            data = to_indigena_df,
                            x = "to_ind_ln",
                            by = 1,
                            z = "to_Indigena",
                            z_values = c(0,1),
                            nsim = 35)

pred_ind_comm$plotdata$community <- ifelse(pred_ind_comm$plotdata$to_Indigena==1,"Pro-Indigena","Pro-Government")
ggplot(data = pred_ind_comm$plotdata, aes(x = (to_ind_ln), 
                                          y = mean,
                                          ymin = lower, ymax = upper,
                                          fill=community,color=community,
                                          linetype = community)) +
  geom_ribbon() + # Confidence intervals
  geom_line() + # Mean
  facet_wrap(~racismo, scales = "free_y", ncol = 3)  +
  theme_minimal() +
  scale_color_viridis_d(option="H",alpha = 0.5) +
  scale_fill_viridis_d(option="H",alpha = 0.4) +
  theme(legend.position = "bottom") +
  labs(x="In-Degree",y="Predicted Prob.", color="", fill = "", linetype = "") 

setwd("here")
ggsave("pred_prob_h1b_indigena.jpg",w=8,h=3)

