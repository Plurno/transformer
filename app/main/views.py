import os
import pickle
import torch
from time import time

from flask import request, current_app, jsonify, abort, url_for
from . import main

@main.route('/')
def index():
    return jsonify({"hello":"Ryan"})

@main.route('/model_info')
def model_info():
    model_head = current_app.finder.reader.inferencer.model.prediction_heads
    return jsonify({"prediction_heads": dir(model_head[0])})

@main.route('/load')
def load():
    direct = os.listdir('app/static/data/language_model')
    if len(direct) == 0:
        inputs = pickle.load(open('app/static/batch.p', 'rb'))
        entry = (inputs['input_ids'], inputs['padding_mask'], inputs['segment_ids'])
        current_app.finder.reader.inferencer.model.language_model.model.eval()
        traced_model = torch.jit.trace(current_app.finder.reader.inferencer.model.language_model.model, entry) # Trace with AWS Neuron instead of pytorch
        torch.jit.save(traced_model, 'app/static/data/language_model/traced_model.pt')
    # direct = os.listdir('app/static/data/question_head') # Language Head Model infers very quickly.  Unnecessary to compile.
    # if len(direct) == 0:
    #     pass
    current_app.finder.reader.inferencer.model.language_model.model = torch.jit.load('app/static/data/language_model/traced_model.pt')
    current_app.finder.reader.inferencer.model.language_model.model.eval()
    return jsonify({'done':'loading'})

@main.route('/get_answers/<query>')
def get_answers(query=None):
    start = time()
    if query=='default': query = 'what does ahrq stand for'
    response = current_app.finder.get_answers(query, top_k_retriever=1, top_k_reader=1)
    return jsonify({'doc 1 text': response,
                    'Elapsed time': time()-start})


