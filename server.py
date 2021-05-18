import logging
import os

import flask
import torch
from flask import jsonify, Flask
from gevent import monkey

monkey.patch_all()
from gevent.pywsgi import WSGIServer
from transformers import AutoTokenizer

from args import model_name_or_path
from dispatcher import Dispatcher
from gpu_runner import serve_gpu
from model import BertForSentimentClassification, ManagedBertModel

app = Flask(__name__)
model, dispatcher = None, None

BERT = BertForSentimentClassification.from_pretrained(model_name_or_path)


def classify_sentiment(model, text):
    with torch.no_grad():
        tokens = tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        seq = torch.tensor(tokens_ids)
        seq = seq.unsqueeze(0)
        attn_mask = (seq != 0).long()
        logit = model(seq, attn_mask)
        prob = torch.sigmoid(logit.unsqueeze(-1))
        prob = prob.item()
        soft_prob = prob > 0.5
        if soft_prob == 1:
            return 'positive', int(prob * 100)
        else:
            return 'negative', int(100 - prob * 100)


@app.route('/', methods=['GET'])
def health_check():
    logging.info("health check ping.")
    return jsonify({'status': 'healthy'}), 200


@serve_gpu(model=BERT, gpu_id=0)
@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    text = flask.request.form['text']
    sent, prob = classify_sentiment(BERT, text)
    return jsonify({'sentiment': sent, 'prob': prob}), 200


@app.route('/api/distributed', methods=['POST'])
def distributed():
    inputs = flask.request.form.getlist('text')
    return jsonify(dispatcher.predict(inputs)), 200


if __name__ == '__main__':
    # init our tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dispatcher = Dispatcher(ManagedBertModel, batch_size=64, worker_num=2, cuda_devices=(0,))
    # app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
    WSGIServer(("0.0.0.0", 5000), app).serve_forever()
