### Paro Data 2019 ###
### Diana Dávila Gordillo, Joan C Timoneda, Sebastián Vallejo Vera ###
### Appendix G Bots ###

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
load("sub_paro_final_pred.Rdata")

summary(sub_paro_final)

V(sub_paro_final)$membership_name[V(sub_paro_final)$membership==3] <-  "Indigena"
V(sub_paro_final)$membership_name[V(sub_paro_final)$membership==2] <-  "Government"
V(sub_paro_final)$membership_name[V(sub_paro_final)$membership==1] <-  "International Left"
V(sub_paro_final)$membership_name[V(sub_paro_final)$membership==4] <-  "International Right"

sub_paro_df <- as_long_data_frame(sub_paro_final)
sub_paro_df$racismo <- ifelse(is.na(sub_paro_df$racismo),"No Racism",sub_paro_df$racismo)

# H1 b: In Degree and Prob of Racism (Produce) -----

sub_paro_df$racismo <- factor(sub_paro_df$racismo, levels = c("No Racism", "Covert Racism", "Overt Racism"))
sub_paro_df$from_ind_ln <- log(sub_paro_df$from_ind+1)
sub_paro_df$to_ind_ln <- log(sub_paro_df$to_ind)

sub_paro_df <- sub_paro_df %>%
  group_by(from) %>%
  mutate(total_tweets_from = n(),
         ln_total_tweets_from = log(total_tweets_from)) %>%
  ungroup() 

to_paro_df <- sub_paro_df %>%
  filter(to_membership_name=="Government" | to_membership_name=="Indigena") %>%
  group_by(to) %>%
  mutate(total_tweets_to = n(),
         ln_total_tweets_to = log(total_tweets_to),
         to_Indigena = ifelse(to_membership_name=="Indigena",1,0)) %>%
  ungroup() %>%
  distinct(text_id,.keep_all=T) 

# # Sample 
# set.seed(1986)
# 
# sample_ind <- to_paro_df %>%
#   slice_sample(n=20000)
#   
# proportions(table(sample_ind$to_Indigena,sample_ind$racismo))
# proportions(table(to_paro_df$to_Indigena,to_paro_df$racismo))

# save(sample_ind,file = "sample_ind.Rdata")
# 
# sample_names <- sample_ind %>%
#   distinct(to_name)
# 
# writexl::write_xlsx(sample_names,path = "sample_names.xlsx")

load("sample_ind.Rdata")
bots_sample <- readxl::read_xlsx('here/bots_sample.xlsx')
bots_sample$to_name <- bots_sample$username
sample_w_bots <- left_join(sample_ind,bots_sample)
sample_w_bots <- sample_w_bots %>%
  filter(!is.na(bot_score))

## Multinomial model ----

h1_b <- multinom(racismo ~ to_ind_ln + to_Indigena + bot_score, 
                 data = sample_w_bots,
                 Hess = TRUE)
summary(h1_b)

### Predicted probabilities ----

#### In Degree + Community ----
pred_ind_comm <- mnl_fd_ova(model = h1_b,
                            data = sample_w_bots,
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

ggsave("pred_prob_sample.jpg",w=8,h=3)

#### In Degree + Community ----
pred_h1_b_comm <- mnl_pred_ova(model = h1_b,
                               data = sample_w_bots,
                               x = "bot_score",
                               by = 0.05,
                               seed = "random", # default
                               nsim = 35, # faster
                               probs = c(0.025, 0.975)) 

# pred_h1_b_comm$plotdata$community <- ifelse(pred_h1_b_comm$plotdata$to_Indigena==1,"Pro-Indigena","Pro-Government")
ggplot(data = pred_h1_b_comm$plotdata, aes(x = bot_score, 
                                          y = mean,
                                          ymin = lower, ymax = upper,
                                          fill=racismo,color=racismo)) +
  geom_ribbon() + # Confidence intervals
  geom_line() + # Mean
  facet_wrap(~racismo, scales = "free_y", ncol = 3) +
  theme_minimal() +
  coord_cartesian(xlim = c(0,0.6)) +
  scale_color_viridis_d(option="E",alpha = 0.7) +
  scale_fill_viridis_d(option="E",alpha = 0.4) +
  theme(legend.position = "bottom") +
  labs(x="Prob. Bot",y="Predicted Prob.", color="", fill = "") 

ggsave("pred_prob_sample_bot.jpg",w=8,h=3)
