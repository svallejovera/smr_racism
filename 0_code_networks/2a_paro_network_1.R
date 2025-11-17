### Paro Data 2018 - 2021 ###
### Sebastián Vallejo ###

rm(list = ls(all=TRUE))
# .rs.restartR()

library(tidyverse)
library(igraph)
library(RColorBrewer)


#### Load da data ----------------------------------------------
setwd("HERE")
load("paro_final.Rdata")
paro_final <- paro_final[c(1:2000000),] # To replicate, you will need to run it in parts...
paro_final <- paro_final[c(2000001:4000000),] # To replicate, you will need to run it in parts...
paro_final <- paro_final[c(4000001:6000000),] # To replicate, you will need to run it in parts...

#### Take only the user and the RTed user -------------------------------------
paro_final <- paro_final %>%
  ungroup() %>%
  filter(paro_final$rt_user_name!="")

## I use the screen_name but could also use the name ##
t_batch <- paro_final %>%
  select(user_username,rt_user_username)
t_batch <- as.matrix(t_batch) # has to be matrix

## Begin the graph
net_paro_1 <- graph.empty() # empty (not necessary but following EC)
net_paro_1 <- add.vertices(net_paro_1, length(unique(c(t_batch))),
                             name=as.character(unique(c(t_batch)))) # Add vertices by looking at our base matrix
net_paro_1 <- add.edges(net_paro_1, t(t_batch)) # Traspose my matrix for the edges

E(net_paro_1)$text_id <- paro_final$text_id # ID of Text (unique to this network)
E(net_paro_1)$rt_tweet_id <- paro_final$rt_tweet_id # ID of original T
E(net_paro_1)$user_followers_count <- paro_final$user_followers_count # Friends of userT
E(net_paro_1)$rt_user_follower_count <- paro_final$rt_user_follower_count # Followers of userT
E(net_paro_1)$user_following_count <- paro_final$user_following_count # Friends of rtT
E(net_paro_1)$rt_user_following_count <- paro_final$rt_user_following_count # Followers of rtT
E(net_paro_1)$created_at <- paro_final$created_at # time of T
E(net_paro_1)$rt_created_at <- paro_final$rt_created_at # time of userT
E(net_paro_1)$user_verified <- paro_final$user_verified # verified of rtT
E(net_paro_1)$rt_user_verified <- paro_final$rt_user_verified # verified of userT
E(net_paro_1)$user_username <- paro_final$user_username # name of rtT
E(net_paro_1)$rt_user_username <- paro_final$rt_user_username # name of userT
E(net_paro_1)$rt_like_count <- paro_final$rt_like_count # fav of rt
E(net_paro_1)$rt_reply_count <- paro_final$rt_reply_count # RTs of rt
E(net_paro_1)$rt_retweet_count <- paro_final$rt_retweet_count # Reply to

## Summary and save
summary(net_paro_1) # Vertices = XXX ; Edges = YYY

#### Trimming the network #### ----------------------------------------------

### Out- and in-degree to selected nodes
out_d <- degree(net_paro_1, mode="out")
in_d <- degree(net_paro_1, mode="in")

# Let's keep the nodes with degrees larger than our cutpoint values
select.nodes<-which(out_d>=3 | in_d>=2)
net_paro_1 <- induced.subgraph(graph=net_paro_1,vids=select.nodes)

# Let's eliminate unconnected users, since we got rid of a bunch 
# of nodes
membership_net <- cluster_walktrap(net_paro_1)
V(net_paro_1)$membership<-membership_net$membership # the communities detected

summary(net_paro_1) # Vertices = XXX ; Edges = YYY

save(net_paro_1, membership_net, file = "net_paro_1.Rdata")

# load("net_paro_1.Rdata")

#### Identify Communities:

# Highest in degree by community
top_rt_member <-data.frame(degree(net_paro_1, mode="in"), V(net_paro_1)$membership)
colnames(top_rt_member) <- c("in_degree","membership")
top_rt_member <- top_rt_member[order(top_rt_member$in_degree, decreasing=T), ]
head(top_rt_member,45)

#                   in_degree    membership
# CONAIE_Ecuador       1687          1
# MashiRafael           776          2
# confeniae1            449          1
# MarcoAnibal           324          2
# ApawkiCastro          227          1
# jaimevargasnae        223          1
# RCSomosPueblo         185          2
# InclusionEc           181          3
# sachacristo1          179          1
# ecuachaskiecua        169          1
# garay                 167         20
# Chevron_Toxico        164         30
# Micc_Ec               143          1
# ArellaProano          141         20
# MyrianeLibre          140         20
# PattyAviBoom          120         20

# There seems to be four distinct communities which makes sense:
# 1: paro
# 2: Opposition (ex-PAIS)
# 7: Pro Gov

#### Now I only keep the four main communities:
main <- rownames(sort(table(V(net_paro_1)$membership),decreasing=T)[c(1:4)])
select.nodes<-which(V(net_paro_1)$membership==as.numeric(main[1]) | 
                      V(net_paro_1)$membership==as.numeric(main[2]) | 
                      V(net_paro_1)$membership==as.numeric(main[3]) |
                      V(net_paro_1)$membership==as.numeric(main[4]))

sub_paro_1 <- induced.subgraph(graph=net_paro_1,vids=select.nodes)

### New layout

## Layout ##
layout_sub <- layout_with_fr(sub_paro_1, grid = c("nogrid")) # 82 secs

#### Load membership and layout to network ---------------------------

V(sub_paro_1)$l1<-layout_sub[,1] # the layout across x
V(sub_paro_1)$l2<-layout_sub[,2] # the layout across y
V(sub_paro_1)$outd<-degree(sub_paro_1, mode="out")
V(sub_paro_1)$ind<-degree(sub_paro_1, mode="in")
E(sub_paro_1)$color <- "lightgray"


save(sub_paro_1, file = "sub_paro_1.Rdata")
# save(sub_paro_2, file = "sub_paro_2.Rdata")
# save(sub_paro_3, file = "sub_paro_3.Rdata")


