import os
import torch
import flask
from flask import jsonify, Flask
import logging
from transformers import AutoTokenizer, AutoConfig
from model import BertForSentimentClassification
from gpu_runner import serve_gpu
from args import model_name_or_path, rank, world_size

app = Flask(__name__)

# BERT = wrap_ddp(BSC.from_pretrained(model_name_or_path), rank, world_size)
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


if __name__ == '__main__':
    # init our tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
