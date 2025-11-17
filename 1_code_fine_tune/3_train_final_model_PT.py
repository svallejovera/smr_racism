#####
## Classifier for Overt and Covert Racism 
## Diana Dávila Gordillo, Joan C Timoneda, and Sebastián Vallejo V
## Date: Nov. 10, 2025
#####

# First, our XLM-Roberta Tokenizer and our XLM-Roberta Classifier:
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# Some other libraries to adjust hyper-parameters:
from transformers import get_linear_schedule_with_warmup, AdamW

# Torch: A Tensor library like NumPy, with strong GPU support
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

# Performance statistics:
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score

# And the rest to help:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import random

def good_update_interval(total_iters, num_desired_updates):
    exact_interval = total_iters / num_desired_updates
    order_of_mag = len(str(total_iters)) - 1
    round_mag = order_of_mag - 1
    update_interval = int(round(exact_interval, -round_mag))
    if update_interval == 0:
        update_interval = 1
    return update_interval

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Load your labeled data
data = pd.read_excel("training_set.xlsx")
print("training_set.xlsx")

# Check your data
len(data)

# Set the main variable to be used for classification:
var = 'racismo_final' ## 
print(data[var].value_counts())

# Re index and change type/name:
data = data.sample(frac=1).reset_index(drop=True)
data["label"] = data[var].astype('category')
data["label"] = data["label"].cat.codes

# Check that it worked
print(data["label"].value_counts())

# Tokenize all of the sentences and map the tokens to their word IDs.
# Record the length of each sequence (in terms of RoBERTa tokens).

# Choose tokenizer from PT model: 
tokenizer = AutoTokenizer.from_pretrained("xlm-r-racismo-PT", do_lower_case=True)

input_ids = []
lengths = []
for x, row in data.iterrows():
    encoded_sent = tokenizer.encode(
                        row['text_clean'], 
                        add_special_tokens = True,
                   )
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

print('{:>10,} comments'.format(len(input_ids)))
print('   Min length: {:,} tokens'.format(min(lengths)))
print('   Max length: {:,} tokens'.format(max(lengths)))
print('Median length: {:,} tokens'.format(np.median(lengths)))

# We will trunctate the text input since RoBERTa can only handle 512 tokens at a time
# Also, the more tokens you have, the mode memory your computer requires

max_len = 120 

num_truncated = np.sum(np.greater(lengths, max_len))
num_sentences = len(lengths)
prcnt = float(num_truncated) / float(num_sentences)
print('{:,} of {:,} sentences ({:.1%}) in the training set are longer than {:} tokens.'.format(num_truncated, num_sentences, prcnt, max_len))

# Create tokenized data
labels = []
input_ids = []
attn_masks = []

for x, row in data.iterrows():
    encoded_dict = tokenizer.encode_plus(row['text_clean'], 
                                              max_length=max_len,
                                              padding='max_length',
                                              truncation=True,
                                              return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attn_masks.append(encoded_dict['attention_mask'])
    labels.append(row['label'])

# Convert into tensor matrix.
input_ids = torch.cat(input_ids, dim=0)
attn_masks = torch.cat(attn_masks, dim=0)

# Labels list to tensor.
labels = torch.tensor(labels)

# Create TensorDataset.
dataset = TensorDataset(input_ids, attn_masks, labels)

#########
# Specify key model parameters here:
model_name = "xlm-r-racismo-PT_tok" # <- The model you will choose. It has to match the tokenizer
lr = 2e-5 # <- Learning rate... usually between 2e-5 and 2e-6
epochs = 4 # <- No more than 5 or you will start overfitting
batch_size = 32 # <- Best if multiple of 2^x... The more the better but also the more GPU
#########

seed_val = 6
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.cuda.empty_cache() #Clear GPU cache if necessary

training_stats = [] # Store training and validation loss, validation accuracy, and timings.
fold_stats = []

total_t0 = time.time() # Measure the total training time

# ======================================== #
#               Training                   #
# ======================================== #

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H%M')

# Define data loaders for training and testing data in this fold
train_dataloader = torch.utils.data.DataLoader(
                       dataset,
                       batch_size=batch_size)

# Initiate model parameters for each fold
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16)
device = torch.device('cuda:0')
desc = model.to(device)
optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
total_steps = (int(len(dataset)/batch_size)+1) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 20, num_training_steps = total_steps)

# Run the training loop for defined number of epochs
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0 # Reset the total loss for this epoch.
    model.train() # Put the model into training mode.
    update_interval = good_update_interval( # Pick an interval on which to print progress updates.
                    total_iters = len(train_dataloader),
                    num_desired_updates = 10)

    predictions_t, true_labels_t = [], []
    for step, batch in enumerate(train_dataloader):
        if (step % update_interval) == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed), end='\r')
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # Perform a forward pass --returns the loss and the "logits"
        loss = model(b_input_ids,attention_mask=b_input_mask,labels=b_labels)[0]
        logits = model(b_input_ids,attention_mask=b_input_mask,labels=b_labels)[1]

        # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end.
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions_t.append(logits)
        true_labels_t.append(label_ids)

    # Combine the results across all batches.
    flat_predictions_t = np.concatenate(predictions_t, axis=0)
    flat_true_labels_t = np.concatenate(true_labels_t, axis=0)
    # For each sample, pick the label (0, 1) with the highest score.
    predicted_labels_t = np.argmax(flat_predictions_t, axis=1).flatten()
    acc_t = accuracy_score(predicted_labels_t, flat_true_labels_t)

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.3f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    print("  Training accuracy: {:.3f}".format(acc_t))

model.save_pretrained('/xlm-r-racismo-es-v2/')
