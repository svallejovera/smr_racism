# Replication materials
Replication materials for "Machines Do See Color: Using LLMs to Classify Overt and Covert Racism in Text" (Forthcoming), by Diana Dávila Gordillo, Joan C Timoneda, and Sebastián Vallejo Vera, at __Sociological Methods & Research__.

> __Abstract:__
> Extant work has identified two discursive forms of racism: overt and covert. While both forms have received attention in scholarly work, research on covert racism has been limited. Its subtle and context-specific nature has made it difficult to systematically identify covert racism in text, especially in large corpora. In this article, we first propose a theoretically driven and generalizable process to identify and classify covert and overt racism in text. This process allows researchers to construct coding schemes and build labeled datasets. We use the resulting dataset to train XLM-RoBERTa, a cross-lingual model for supervised classification with a cutting-edge contextual understanding of text. We show that XLM-R and XLM-R-Racismo, our pretrained model, outperform other state-of-the-art approaches in classifying racism in large corpora. We illustrate our approach using a corpus of tweets relating to the Ecuadorian indígena community between 2018 and 2021. 

A link to the latest pre-print is available [here](https://arxiv.org/abs/2401.09333). The published version is [here](CHANGE).

This README file provides an overview of the replications materials for the article. 
The [Code](https://github.com/svallejovera/smr_racism#code) section describes the code used to create networks (see Appendix), pre-train and fine-tune the models, predict labels in the data, and a replication of the main analysis.
Given the size of the models, we have not uploaded them to any repository. However, the best performing model described in the paper, XLM-R-Racismo, whose full technical model name is ‘xlm-r-racismo-es-v2’, is available for public use at huggingface.co: https://huggingface.co/timoneda/xlm-r-racismo-es-v2.
The [Data](https://github.com/svallejovera/smr_racism#data) section describes the main dataset required to reproduce the tables and figures in the paper. All the data is available here: https://doi.org/10.7910/DVN/3G6YXY 

## Code

### Networks
- `0_code_networks/1_indigena_network.R`: R code to create networks and communities from Twitter data from the indigena query using community detecting algorithms. 
- `0_code_networks/2a_paro_network_1.R`: R code to create networks and communities from Twitter data from the paro query using community detecting algorithms. Requires multiple files because data is too large. 
- `0_code_networks/2b_paro_data_cropped.R`: R code to create networks and communities from Twitter data from the paro query using community detecting algorithms. 
- `0_code_networks/2c_paro_network_final.R`: R code to create networks and communities from Twitter data from the paro query using community detecting algorithms. 

### Pre-Training and Fine-Tuning
- `1_code_fine_tune/1_pre_train_model.py`: python code to add tokens to and further pre-train an xlm-roberta-large model. Further pre-trained model is called: xlm-r-racismo-PT. Execution time:
- `1_code_fine_tune/2_CV_model_PT.py`: python code to fine-tune our xlm-r-racismo-PT model using the `data/training_set.xlsx` training set with 10-fold cross-valiation to evaluate performance. 
- `1_code_fine_tune/2_CV_model_noPT.py`: python code to fine-tune a xlm-roberta-large model using the `data/training_set.xlsx` training set with 10-fold cross-valiation to evaluate performance. 
- `1_code_fine_tune/3_train_final_model_PT.py`: python code to fine-tune our xlm-r-racismo-PT model using the `data/training_set.xlsx` training set (fully fine-tuned model available at at huggingface.co: https://huggingface.co/timoneda/xlm-r-racismo-es-v2). 
- `1_code_fine_tune/3_train_final_model_noPT.py`: python code to fine-tune our xlm-r-racismo-PT model using the `data/training_set.xlsx` training set. 

### Prediction and Tidying
- `2_code_predict_tidy/1_apply_model_indigena.py`: python code to use our fine-tuned xlm-r-racismo-es-v2 model to predict covert and overt racism in the `data/to_predict/text_indigena_to_predict.xlsx` dataset.
- `2_code_predict_tidy/1_apply_model_paro.py`: python code to use our fine-tuned xlm-r-racismo-es-v2 model to predict covert and overt racism in the `data/to_predict/text_paro_to_predict.xlsx` dataset.
- `2_code_predict_tidy/2_indigena_net_racism.R`: R code to merge and tidy predicted data (`data/to_predict/text_indigena_predicted_clean.csv`) and Twitter-network data (`tw_networks/sub_indigena.Rdata`).
- `2_code_predict_tidy/2_paro_net_racism.R`: R code to merge and tidy predicted data (`data/to_predict/text_paro_predicted_clean.csv`) and Twitter-network data (`tw_networks/sub_paro.Rdata`).  

### Empirical Analysis
- `3_code_empirics/0_detect_bots.py`: python code to identify bots from sample data (`data/bots/sample_names.xlsx`).
- `3_code_empirics/1_hypothesis_1b_paro.R`: R code to estimate the multinomial model from Section 5 Covert and Overt Racism in Ecuadorian Twitter and Figure 1 from the main paper.
- `3_code_empirics/2_racism_by_bots.R`: R code to estimate the multinomial model from Appendix G and Figure G1 from the Appendix.

## Data
All data files are available at this public repository: https://doi.org/10.7910/DVN/3G6YXY. Here are the names and the content of each data file:

- `data/final/sub_indigena_pred.Rdata`: Final data file with network data and predicted labels for each tweet from the indigena Twitter query.
- `data/final/sub_paro_pred.Rdata`: Final data file with network data and predicted labels for each tweet from the paro Twitter query.
- `data/to_train/pre_train_ecuador.txt`: unstructured corpus of twitter data used to further pretrain the xlm-roberta-large model.
- `data/to_train/training_set.xlsx`: training set used to fine-tune all supervised machine learning models.
- `data/to_predict/text_indigena_to_predict.xlsx`: corpus of tweets collected between 2018 and 2021 related to the indígena community in Ecuador.
- `data/to_predict/text_indigena_to_predict.xlsx`: corpus of tweets collected during 2019 indigena uprising in Ecuador.
- `data/tw_networks/sub_indigena.Rdata` and `data/tw_networks/sub_paro.Rdata`: Twitter data (igraph object) collected between 2018 and 2021 related to the indígena community in Ecuador and the 2019 indigena uprising in Ecuador. The unit of analysis is the tweet (original and retweeted). The variables of interest are:
  
      - **membership**: cummunity the user belongs to (estimated using community detection alrogithm from `0_code_networks')
      - **ind**: in-degree of user.
      - **to**: user tweeting or retweeting a tweet.
      - **text_id**: tweet or retweet individual id.

- `data/bots/sample_names.xlsx`: sample from full `data/tw_networks/sub_paro.Rdata` used to identify bots.

A full tutorial on how to train Transformer-based models can be found [here](https://colab.research.google.com/drive/1rWh6JVhJ4aZmdTYZUYVYo3AOGb2TOi6b?usp=sharing#scrollTo=Classification_Models). A complete course on Computational Text Analysis can be found [here](https://svallejovera.github.io/cpa_uwo/index.html)
