
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import os
import re

tokenizer = XLMRobertaTokenizer.from_pretrained("timoneda/xlm-r-racismo-es-v2")

classifier = XLMRobertaForSequenceClassification.from_pretrained('timoneda/xlm-r-racismo-es-v2', num_labels=3)
classifier.eval()

label_dict = {0: "No Racism", 1: "Covert Racism", 2: "Overt Racism"}

def predict(sentence):
    sentence = tokenizer.encode_plus(sentence, return_tensors='pt')
    outputs = classifier(sentence["input_ids"],
                    attention_mask=sentence["attention_mask"])
    outputs = outputs[0].detach().numpy()
    predicted_label = np.argmax(outputs)
    label = label_dict[predicted_label]
    return predicted_label, label


# Now we take all the texts and predict the topic:
text_paro = pd.read_excel(r'text_paro_to_predict.xlsx')
text_paro.head(5)

prediction = []

for i in tqdm(text_paro['text_clean']):
    pred_temp = predict(i)
    prediction.append(pred_temp)


text_paro['prediction'] = prediction
text_paro

text_paro.to_csv(r'text_paro_predicted.csv')
