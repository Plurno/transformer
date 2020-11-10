# from haystack import Finder
# from haystack.reader.farm import FARMReader
# # from haystack.reader.transformers import TransformersReader
# from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
# from haystack.retriever.sparse import ElasticsearchRetriever
# from haystack.utils import print_answers


# document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="ahrq")

# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
# # reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=False)

# retriever = ElasticsearchRetriever(document_store=document_store)
# finder = Finder(reader, retriever)

# prediction = finder.get_answers(question="Does ahrq offer minority supplements for grants?", top_k_retriever=10, top_k_reader=5)
# print_answers(prediction, details="medium")

import torch
import numpy as np
import os
import torch_neuron

from haystack.reader.farm import FARMReader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
model = reader.inferencer.model

model.eval()
test_dict = {}
model_neuron = torch.neuron.trace(model, example_inputs=test_dict)
