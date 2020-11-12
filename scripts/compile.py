import torch
# import torch_neuron
# import numpy as np
# import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# model_name = "deepset/roberta-base-squad2" # Model for QA
# model_name = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english" # Tiny, untrained model for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english" # Sentiment Model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")

pt_batch = tokenizer(
     ["i am happy", "i am sad"],
     padding=True,
     truncation=True,
     return_tensors="pt"
    )

# pt_outputs = model(**pt_batch)
# pt_predictions = F.softmax(pt_outputs[0], dim=1)
# print(pt_predictions)

model_neuron = torch.neuron.trace(model, example_inputs=[pt_batch])
## Export to saved model
model_neuron.save("model.pt")

tokenizer_neuron = torch.neuron.trace(tokenizer)
tokenizer_neuron.save('tokenizer.pt')

# Now try with pipeline