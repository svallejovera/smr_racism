### Paro Data 2018 - 2021 ###
### Sebastián Vallejo ###

rm(list = ls(all=TRUE))
# .rs.restartR()

library(tidyverse)
library(igraph)
library(RColorBrewer)


#### Load da data ----------------------------------------------
setwd("HERE")
load("sub_paro_df.Rdata")
paro_final <- sub_paro

## I use the screen_name but could also use the name ##
t_batch <- paro_final %>%
  select(user_username,rt_user_username)
t_batch <- as.matrix(t_batch) # has to be matrix

## Begin the graph
net_paro <- graph.empty() # empty (not necessary but following EC)
net_paro <- add.vertices(net_paro, length(unique(c(t_batch))),
                           name=as.character(unique(c(t_batch)))) # Add vertices by looking at our base matrix
net_paro <- add.edges(net_paro, t(t_batch)) # Traspose my matrix for the edges

E(net_paro)$text_id <- paro_final$text_id # ID of Text (unique to this network)
E(net_paro)$rt_tweet_id <- paro_final$rt_tweet_id # ID of original T
E(net_paro)$user_followers_count <- paro_final$user_followers_count # Friends of userT
E(net_paro)$rt_user_follower_count <- paro_final$rt_user_follower_count # Followers of userT
E(net_paro)$user_following_count <- paro_final$user_following_count # Friends of rtT
E(net_paro)$rt_user_following_count <- paro_final$rt_user_following_count # Followers of rtT
E(net_paro)$created_at <- paro_final$created_at # time of T
E(net_paro)$rt_created_at <- paro_final$rt_created_at # time of userT
E(net_paro)$user_verified <- paro_final$user_verified # verified of rtT
E(net_paro)$rt_user_verified <- paro_final$rt_user_verified # verified of userT
E(net_paro)$user_username <- paro_final$user_username # name of rtT
E(net_paro)$rt_user_username <- paro_final$rt_user_username # name of userT
E(net_paro)$rt_like_count <- paro_final$rt_like_count # fav of rt
E(net_paro)$rt_reply_count <- paro_final$rt_reply_count # RTs of rt
E(net_paro)$rt_retweet_count <- paro_final$rt_retweet_count # Reply to

## Summary and save
summary(net_paro) # Vertices = XXX ; Edges = YYY

#### Trimming the network #### ----------------------------------------------

### Out- and in-degree to selected nodes
out_d <- degree(net_paro, mode="out")
in_d <- degree(net_paro, mode="in")

# Let's keep the nodes with degrees larger than our cutpoint values

# Let's eliminate unconnected users, since we got rid of a bunch 
# of nodes
membership_net <- cluster_walktrap(net_paro)
V(net_paro)$membership<-membership_net$membership # the communities detected

summary(net_paro) # Vertices = XXX ; Edges = YYY

save(net_paro, membership_net, file = "net_paro.Rdata")

# load("net_paro.Rdata")

#### Identify Communities:

# Highest in degree by community
top_rt_member <-data.frame(degree(net_paro, mode="in"), V(net_paro)$membership)
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
main <- rownames(sort(table(V(net_paro)$membership),decreasing=T)[c(1:4)])
select.nodes<-which(V(net_paro)$membership==as.numeric(main[1]) | 
                      V(net_paro)$membership==as.numeric(main[2]) | 
                      V(net_paro)$membership==as.numeric(main[3]) |
                      V(net_paro)$membership==as.numeric(main[4]))

sub_paro_final <- induced.subgraph(graph=net_paro,vids=select.nodes)

### New layout

## Layout ##
layout_sub <- layout_with_fr(sub_paro_final, grid = c("nogrid")) # 82 secs

#### Load membership and layout to network ---------------------------

V(sub_paro_final)$l1<-layout_sub[,1] # the layout across x
V(sub_paro_final)$l2<-layout_sub[,2] # the layout across y
V(sub_paro_final)$outd<-degree(sub_paro_final, mode="out")
V(sub_paro_final)$ind<-degree(sub_paro_final, mode="in")
E(sub_paro_final)$color <- "lightgray"


save(sub_paro_final, file = "sub_paro_final.Rdata")


