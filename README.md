# Replication materials
Replication materials for "Machines Do See Color: Using LLMs to Classify Overt and Covert Racism in Text" (Forthcoming), by Diana Dávila Gordillo, Joan C Timoneda, and Sebastián Vallejo Vera, at __Sociological Methods & Research__.

> __Abstract:__
> Extant work has identified two discursive forms of racism: overt and covert. While both forms have received attention in scholarly work, research on covert racism has been limited. Its subtle and context-specific nature has made it difficult to systematically identify covert racism in text, especially in large corpora. In this article, we first propose a theoretically driven and generalizable process to identify and classify covert and overt racism in text. This process allows researchers to construct coding schemes and build labeled datasets. We use the resulting dataset to train XLM-RoBERTa, a cross-lingual model for supervised classification with a cutting-edge contextual understanding of text. We show that XLM-R and XLM-R-Racismo, our pretrained model, outperform other state-of-the-art approaches in classifying racism in large corpora. We illustrate our approach using a corpus of tweets relating to the Ecuadorian \textit{ind\'igena} community between 2018 and 2021. 

A link to the latest pre-print is available [here](CHANGE). The published version is [here](CHANGE).

This README file provides an overview of the replications materials for the article. 
The [Code](https://github.com/svallejovera/smr_racism#code) section describes the code used to pre-train and fine-tune the models, predict labels in the data, and a replication of the main analysis.
Given the size of the models, we have not uploaded them to any repository. However, the best performing model described in the paper, XLM-R-Racismo, whose full technical model name is ‘xlm-r-racismo-es-v2’, is available for public use at huggingface.co: https://huggingface.co/timoneda/xlm-r-racismo-es-v2.
The [Data](https://github.com/svallejovera/smr_racism#data) section describes the main dataset required to reproduce the tables and figures in the paper. 
The [Analysis](https://github.com/svallejovera/gender_inst_speeches#Analysis) section summarizes the purpose of each R or python script. 

## Code

### Pre-Training and Fine-Tuning
- `code_fine_tune/1_pre_train_model.py`: python code to add tokens to and further pre-train an xlm-roberta-large model. Further pre-trained model is called: xlm-r-racismo-PT. Execution time:
- `code_fine_tune/2_CV_model_PT.py`: python code to fine-tune our xlm-r-racismo-PT model using the `data/training_set.xlsx` training set with 10-fold cross-valiation to evaluate performance. 
- `code_fine_tune/2_CV_model_noPT.py`: python code to fine-tune a xlm-roberta-large model using the `data/training_set.xlsx` training set with 10-fold cross-valiation to evaluate performance. 
- `code_fine_tune/3_train_final_model_PT.py`: python code to fine-tune our xlm-r-racismo-PT model using the `data/training_set.xlsx` training set (fully fine-tuned model available at at huggingface.co: https://huggingface.co/timoneda/xlm-r-racismo-es-v2). 
- `code_fine_tune/3_train_final_model_noPT.py`: python code to fine-tune our xlm-r-racismo-PT model using the `data/training_set.xlsx` training set. 

### Prediction and Tidying
- `code_predict_tidy/1_pre_train_model.py`: python code to add tokens to and further pre-train an xlm-roberta-large model. Further pre-trained model is called: xlm-r-racismo-PT. Execution time:
- `code_predict_tidy/2_CV_model_PT.py`: python code to fine-tune our xlm-r-racismo-PT model using the `data/training_set.xlsx` training set with 10-fold cross-valiation to evaluate performance. 

### Empirical Analysis

## Data
  - `data/data_rep.Rdata`: dataset with information on the number of speeches delivered in every topic by a legisator, across differnt legislative sessions. The unit of analysis is the session-legislator-topic. We include all control variables. It contains the following variables:
      - **perodo**: legislative session *k*
      - **candidatoa**: candidate *i* name 
      - **female**: indicator variable that takes value of 1 if legislator is a woman and 0 otherwise
      - **topics_sp**: topic *j*     
      - **total_sp_leg_type_topic**: total number of speeches delivered by candidate *i* on topic *j* during session *k*
      - **total_sp_leg_type_topic2**: total number of speeches **not mentioning women** delivered by candidate *i* on topic *j* during session *k*
      - **total_sp_leg_period**: total number of speeches delivered by candidate *i* during session *k*
      - **total_mentions_muj**: total number of speeches delivered mentioning women by candidate *i* during session *k*  
      - **mesa**: indicator variable that takes value of 1 if the legislator is part of the executive board of the Chamber of Deputies (*Mesa*) and 0 otherwise
      - **membc1**:**membc31**: indicator variable that takes value of 1 if the legislator is a member of committee *p* and 0 otherwise (see `data/committees.xlsx` for the names of the committees matching the numbers)
      - **chairc1**:**chairc31**: indicator variable that takes value of 1 if the legislator is president of committee *p* and 0 otherwise (see `data/committees.xlsx` for the names of the committees matching the numbers)
      - **alianza**: indicator variable that takes value of 1 if the legislator is part of the *Alianza* coalition and 0 otherwise
      - **other**: indicator variable that takes value of 1 if the legislator is part of the *Other* coalition and 0 otherwise
      - **tenure**: the number of terms the legislator has served in the chamber
      - **lndist**: logged distance between main town of the legislator’s district and La Moneda, the presiden- tial building located in Santiago, computed using Google Maps
      - **porcRural**: share of rural inhabitants in the district, obtained from Chilean census information
      - **margenlist**: difference between the votes received by the legislator and her list’s partner
  - `data/keyword mujer.xlsx`: dictionary with words associated with women-related topics.
  - `data/committees.xlsx`: names of the committees matching the numbers (see variables from `data/data_rep.Rdata`.
  - `data/topics_committees.xlsx`: names of topics that are matched to committees.

## Analysis
  - [figure 2.R](https://github.com/svallejovera/gender_inst_speeches/blob/main/code/figure%202.R) to replicate Figure 2 of the paper, where we show the proportion of speeches delivered by women by topic.

<img src = "https://github.com/svallejovera/gender_inst_speeches/blob/main/figures/figure%202.jpeg">

  - [figure 3.R](https://github.com/svallejovera/gender_inst_speeches/blob/main/code/figure%203.R) to replicate Figure 3 of the paper, where we show gender difference in speechmaking by topic.

<img src = "https://github.com/svallejovera/gender_inst_speeches/blob/main/figures/figure%203.jpg">

  - [figure 4.R](https://github.com/svallejovera/gender_inst_speeches/blob/main/code/figure%204.R) to replicate Figures 4a and 4b of the paper, where we show gender differences in Law and Crime speeches when we include and exclude women issues.

<img src = "https://github.com/svallejovera/gender_inst_speeches/blob/main/figures/figure%204a.jpg">
<img src = "https://github.com/svallejovera/gender_inst_speeches/blob/main/figures/figure%204b.jpg">

  - [figure 5.R](https://github.com/svallejovera/gender_inst_speeches/blob/main/code/figure%205.R) to replicate Figure 5 of the paper, where we plot the relationship between being a women a delivering non women-related speeches and women-related speeches.

<img src = "https://github.com/svallejovera/gender_inst_speeches/blob/main/figures/figure%205.jpg">

  - [table 1b appendix.R](https://github.com/svallejovera/gender_inst_speeches/blob/main/code/table%201b%20appendix.R) to replicate Table 1.B of the Appendix, where we estimate negative binomial models for the determinants of speech topic. We use the odds ratio of the variable *female* to plot Figure 3.

## Supervised Machine Learning Model (XLM-RoBERTa) for Topics:

`data/data_rep.Rdata` contains the training set used to train an XLM-RoBERTa model to identify topics in speeches. A full tutorial on how to train Transformer-based models can be found [here](https://colab.research.google.com/drive/1rWh6JVhJ4aZmdTYZUYVYo3AOGb2TOi6b?usp=sharing#scrollTo=Classification_Models). A complete course on Computational Text Analysis can be found [here](https://svallejovera.github.io/cpa_uwo/index.html)
