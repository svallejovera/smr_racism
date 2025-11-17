### Indigena Data 2018 - 2021 ###
### Sebastián Vallejo ###

rm(list = ls(all=TRUE))
# .rs.restartR()

library(tidyverse)
library(igraph)
library(RColorBrewer)


#### Load da data ----------------------------------------------
setwd("HERE")
load("indigena_final.Rdata")

#### Take only the user and the RTed user -------------------------------------
indigena_final <- indigena_final %>%
  ungroup() %>%
  filter(indigena_final$rt_user_name!="")

## I use the screen_name but could also use the name ##
t_batch <- indigena_final %>%
  select(user_username,rt_user_username)
t_batch <- as.matrix(t_batch) # has to be matrix

## Begin the graph
net_indigena <- graph.empty() # empty (not necessary but following EC)
net_indigena <- add.vertices(net_indigena, length(unique(c(t_batch))),
                               name=as.character(unique(c(t_batch)))) # Add vertices by looking at our base matrix
net_indigena <- add.edges(net_indigena, t(t_batch)) # Traspose my matrix for the edges

E(net_indigena)$text_id <- indigena_final$text_id # ID of Text (unique to this network)
E(net_indigena)$rt_tweet_id <- indigena_final$rt_tweet_id # ID of original T
E(net_indigena)$user_followers_count <- indigena_final$user_followers_count # Friends of userT
E(net_indigena)$rt_user_follower_count <- indigena_final$rt_user_follower_count # Followers of userT
E(net_indigena)$user_following_count <- indigena_final$user_following_count # Friends of rtT
E(net_indigena)$rt_user_following_count <- indigena_final$rt_user_following_count # Followers of rtT
E(net_indigena)$created_at <- indigena_final$created_at # time of T
E(net_indigena)$rt_created_at <- indigena_final$rt_created_at # time of userT
E(net_indigena)$user_verified <- indigena_final$user_verified # verified of rtT
E(net_indigena)$rt_user_verified <- indigena_final$rt_user_verified # verified of userT
E(net_indigena)$user_username <- indigena_final$user_username # name of rtT
E(net_indigena)$rt_user_username <- indigena_final$rt_user_username # name of userT
E(net_indigena)$rt_like_count <- indigena_final$rt_like_count # fav of rt
E(net_indigena)$rt_reply_count <- indigena_final$rt_reply_count # RTs of rt
E(net_indigena)$rt_retweet_count <- indigena_final$rt_retweet_count # Reply to

## Summary and save
summary(net_indigena) # Vertices = XXX ; Edges = YYY

#### Trimming the network #### ----------------------------------------------

### Out- and in-degree to selected nodes
out_d <- degree(net_indigena, mode="out")
in_d <- degree(net_indigena, mode="in")

# Let's keep the nodes with degrees larger than our cutpoint values
select.nodes<-which(out_d>=3 | in_d>=2)
net_indigena <- induced.subgraph(graph=net_indigena,vids=select.nodes)

# Let's eliminate unconnected users, since we got rid of a bunch 
# of nodes
membership_net <- cluster_walktrap(net_indigena)
V(net_indigena)$membership<-membership_net$membership # the communities detected

summary(net_indigena) # Vertices = XXX ; Edges = YYY

save(net_indigena, membership_net, file = "net_indigena.Rdata")

# load("net_indigena.Rdata")

#### Identify Communities:

# Highest in degree by community
top_rt_member <-data.frame(degree(net_indigena, mode="in"), V(net_indigena)$membership)
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
# 1: Indigena
# 2: Opposition (ex-PAIS)
# 7: Pro Gov

#### Now I only keep the four main communities:
main <- rownames(sort(table(V(net_indigena)$membership),decreasing=T)[c(1:4)])
select.nodes<-which(V(net_indigena)$membership==as.numeric(main[1]) | 
                      V(net_indigena)$membership==as.numeric(main[2]) | 
                      V(net_indigena)$membership==as.numeric(main[3]) |
                      V(net_indigena)$membership==as.numeric(main[4]))

sub_indigena <- induced.subgraph(graph=net_indigena,vids=select.nodes)

### New layout

## Layout ##
layout_sub <- layout_with_fr(sub_indigena, grid = c("nogrid")) # 82 secs

# Prettify
my.color <- brewer.pal(n=4, "RdBu")

temp <- rep(1,length(V(sub_indigena)$membership))
new.color <- data.frame(t(col2rgb(temp)/255))
new.color <- rgb(new.color, alpha=.05)

new.color[V(sub_indigena)$membership==as.numeric(main[1])] <- my.color[1] ####
new.color[V(sub_indigena)$membership==as.numeric(main[2])] <- my.color[2] ####
new.color[V(sub_indigena)$membership==as.numeric(main[3])] <- my.color[3] ####
new.color[V(sub_indigena)$membership==as.numeric(main[4])] <- my.color[4] ####


#### Load membership and layout to network ---------------------------

V(sub_indigena)$l1<-layout_sub[,1] # the layout across x
V(sub_indigena)$l2<-layout_sub[,2] # the layout across y
V(sub_indigena)$outd<-degree(sub_indigena, mode="out")
V(sub_indigena)$ind<-degree(sub_indigena, mode="in")
E(sub_indigena)$color <- "lightgray"


save(sub_indigena,new.color, file = "sub_indigena.Rdata")


