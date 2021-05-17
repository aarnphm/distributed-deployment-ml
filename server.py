from typing import Dict

import flask
import torch
from flask import jsonify, Flask, request
import logging
from transformers import pipeline, AutoTokenizer, AutoConfig
from model import BERT
from gpu_runner import serve_gpu
from args import model_name_or_path
import os

app = Flask(__name__)
app.config.from_object(__name__)
classifier = pipeline("zero-shot-classification")


def news_categorization(text) -> Dict:
    categories = [
        "War",
        "Sports",
        "Business",
        "Science",
        "Biology",
        "News"
    ]
    res = classifier(text, categories, multi_label=True)
    return dict((key, value) for key, value in zip(res['labels'], res['scores']))


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
    text = request.args['text']
    sent, prob = classify_sentiment(BERT, text)
    return jsonify({'sentiment': sent, 'prob': prob}), 200


# we use transformers.pipeline for this task
@app.route('/api/categorize', methods=['POST'])
def categorize():
    data = flask.request.form
    text = data['text']
    print(f"got categorization request of length {len(text)}")
    if len(text) < 10:
        return "too short", 400

    return jsonify(news_categorization(text)), 200


if __name__ == '__main__':
    config = AutoConfig.from_pretrained(model_name_or_path)
    # init our tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    app.run(host="0.0.0.0",debug=True, port=int(os.environ.get("PORT", 5000)))
