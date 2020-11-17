from flask import request, current_app, jsonify, abort, url_for
from . import main

@main.route('/')
def index():
    return jsonify({"hello":"world"})

@main.route('/search/<query>')
def search(query):
    answer = current_app.reader.infer(query)
    if len(answer) > 0 and answer[0]['answer'] != '<s>':
        return jsonify({'Query': str(query),
                        'Answer': answer[0]['answer'],
                        'Score': answer[0]['score']})
    else:
        return jsonify({'Query': str(query),
                        'Answer': None,
                        'Score': 0})
