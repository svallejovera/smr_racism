#####
## Classifier for Overt and Covert Racism: Further Pre-Training 
## Diana Dávila Gordillo, Joan C Timoneda, and Sebastián Vallejo V
## Date: Nov. 10, 2025
#####

from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch
import matplotlib.pyplot as plt
import pandas as pd
import random

# Load Model and Tokenizer
maskedlm_model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

# Add Tokens
tokenizer.add_tokens(["longo"])
tokenizer.add_tokens(["rocoto"])
tokenizer.add_tokens(["huangudo"])
tokenizer.add_tokens(["longanizo"])
tokenizer.add_tokens(["emplumado"])
tokenizer.add_tokens(["indiada"])
tokenizer.add_tokens(["cholo"])
tokenizer.add_tokens(["guangudo"])
tokenizer.add_tokens(["bobolongo"])
tokenizer.add_tokens(["jíbaro"])
tokenizer.add_tokens(["plumífero"])
tokenizer.add_tokens(["pndj"])
tokenizer.add_tokens(["hp"])
tokenizer.add_tokens(["hdlgp"])
tokenizer.add_tokens(["mamaverga"])
tokenizer.add_tokens(["mmvs"])
tokenizer.add_tokens(["hijodeputa"])
tokenizer.add_tokens(["pendejo"])
tokenizer.add_tokens(["mmv"])
tokenizer.add_tokens(["mmvgs"])
tokenizer.add_tokens(["hdp"])

# Custom Initialized Vectors for insults:
maskedlm_model.resize_token_embeddings(len(tokenizer)) 

stupid_id = tokenizer.convert_tokens_to_ids("imbécil")
idiot_id = tokenizer.convert_tokens_to_ids("idiot")
asshole_id = tokenizer.convert_tokens_to_ids("asshole")
fucker_id = tokenizer.convert_tokens_to_ids("fucker")

stupid_embedding = maskedlm_model.get_input_embeddings().weight[stupid_id]
idiot_embedding = maskedlm_model.get_input_embeddings().weight[idiot_id]
asshole_embedding = maskedlm_model.get_input_embeddings().weight[asshole_id]
fucker_embedding = maskedlm_model.get_input_embeddings().weight[fucker_id]

insults_embedding = torch.mean(torch.stack([stupid_embedding, idiot_embedding, asshole_embedding, fucker_embedding]), dim=0)

maskedlm_model.get_input_embeddings().weight[-1].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-2].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-3].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-4].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-5].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-6].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-7].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-8].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-9].data[:] = insults_embedding
maskedlm_model.get_input_embeddings().weight[-10].data[:] = insults_embedding

# Custom Initialized Vectors for racist slur:
maskedlm_model.resize_token_embeddings(len(tokenizer)) 

nixxer_id = tokenizer.convert_tokens_to_ids("nigger")
negro_id = tokenizer.convert_tokens_to_ids("negro")
inferior_id = tokenizer.convert_tokens_to_ids("inferior")
indigena_id = tokenizer.convert_tokens_to_ids("indígena")

nixxer_embedding = maskedlm_model.get_input_embeddings().weight[nixxer_id]
negro_embedding = maskedlm_model.get_input_embeddings().weight[negro_id]
inferior_embedding = maskedlm_model.get_input_embeddings().weight[inferior_id]
indigena_embedding = maskedlm_model.get_input_embeddings().weight[indigena_id]

slur_embedding = torch.mean(torch.stack([nixxer_embedding, negro_embedding, inferior_embedding, indigena_embedding]), dim=0)

maskedlm_model.get_input_embeddings().weight[-11].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-12].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-13].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-14].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-15].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-16].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-17].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-18].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-19].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-20].data[:] = slur_embedding
maskedlm_model.get_input_embeddings().weight[-21].data[:] = slur_embedding

## Data:
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="pre_train_ecuador.txt",
    block_size=32,
)

# Custom DataCollator for Language Modeling
class CustomDataCollatorForLM(DataCollatorForLanguageModeling):

    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.custom_masked_tokens = []
        self.custom_masked_probability = mlm_probability

    def add_custom_masked_tokens(self, tokens, probability):
        self.custom_masked_tokens.extend(tokens)
        self.custom_masked_probability = probability
    
    def collate_batch(self, features):
        inputs = self.tokenizer.pad({"input_ids": [f.input_ids for f in features]}, return_tensors="pt")
        labels = inputs["input_ids"].clone() 

        for i, feature in enumerate(features):
            masked_indices = random.sample(range(len(feature.input_ids)), int(self.custom_masked_probability * len(feature.input_ids)))

            for idx in masked_indices:
                if feature.input_ids[idx] in self.custom_masked_tokens:
                    labels[i, idx] = self.tokenizer.mask_token_id

        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}
     
# Set up training arguments
training_args = TrainingArguments(   
    output_dir="./",
    overwrite_output_dir=True,
    learning_rate=3e-05, 
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=80000,
    #save_total_limit=2,
)

# Set up the Trainer with custom data collator
trainer = Trainer(
    model=maskedlm_model,
    args=training_args,
    data_collator=CustomDataCollatorForLM(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    train_dataset=dataset,
)

# Add custom masked tokens with probability
custom_masked_tokens = ["longo", "rocoto", "huangudo", "longanizo","emplumado",
                        "indiada","cholo","guangudo","bobolongo","jíbaro",
                        "plumífero","pndj","hp","hdlgp","mamaverga","mmvs",
                        "hijodeputa","pendejo","mmv","mmvgs","hdp"] ## all data is in small caps 
custom_masked_probability = 0.15  

# Train:
trainer.data_collator.add_custom_masked_tokens(custom_masked_tokens, custom_masked_probability)

# Start pretraining
trainer.train()

# Save models and tokenizer
trainer.save_model("xlm-r-racismo-PT")
tokenizer.save_pretrained("xlm-r-racismo-PT_tok")
