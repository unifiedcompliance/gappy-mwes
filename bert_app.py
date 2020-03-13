import argparse
import glob
import os
import random
import numpy as np
import time
import requests
from collections import Counter
import json
from flask import Flask, render_template, request
from stanfordnlp.server import CoreNLPClient
from bert_utils import predict

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)


app = Flask(__name__)

BERT_MODEL='bert-base-multilingual-cased'
OUTPUT_DIR='../transformer/transformers/examples/unified_compliance_512_50_18022020/'


os.environ['CORENLP_HOME'] = '/home/sritanu/stanford-corenlp-full-2018-10-05'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

client = CoreNLPClient(annotators=['tokenize','ssplit'], timeout=100000, memory='32G')

def load_model():
    config_class, model_class, tokenizer_class = BertConfig, BertForTokenClassification, BertTokenizer
    global tokenizer
    tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR, do_lower_case=False)
    global model
    model = model_class.from_pretrained(OUTPUT_DIR)
    #global device
    #device = torch.device("cuda")
    #model.to(device)

@app.route('/', methods=['GET','POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        start1 = time.time()
        try:
            sent = request.get_json()
            params = request.args
            #print(sent)
            sent = sent['text']
            drop_adjectives = (params['drop_adjectives'] == 'True')
            drop_determiners = (params['drop_determiners'] == 'True')
        except:
            errors.append("Unable to get text. Please make sure it's valid and try again.")
            return render_template('index.html', errors=errors)


        if sent:
            sent = sent.strip()
            ann = client.annotate(sent)
            sentence = ann.sentence
            tokens = []

            for i in sentence:
                token = []
                for j in i.token:
                    token.append(j.word)
                tokens.append(token)

            preds = predict(tokens, model, tokenizer)

            sentences = json.dumps({'sentences': preds},
                                    sort_keys = False, indent = 4, separators = (',', ': '), ensure_ascii=False)
            #print(sentences)
            print("--- %s SECONDS ---" % (time.time() - start1))
    return sentences

if __name__ == '__main__':
    print("Loading the neural net model")
    #global model
    load_model()
    print("Model loaded")
    #app.debug = True
    app.run(threaded=False, host='0.0.0.0')
