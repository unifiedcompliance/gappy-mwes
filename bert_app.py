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
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from bert_utils import predict
from opformat_utils import  analysis

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

def get_nlp():
    snlp = stanfordnlp.Pipeline(lang="en", treebank='en_lines', use_gpu=False, tokenize_pretokenized=True)
    global nlp
    nlp = StanfordNLPLanguage(snlp)

def get_doc(sent):
    doc = nlp(sent)
    return doc


def load_model():
    config_class, model_class, tokenizer_class = BertConfig, BertForTokenClassification, BertTokenizer
    global tokenizer
    tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR, do_lower_case=False)
    global model
    model = model_class.from_pretrained(OUTPUT_DIR)

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
            orig_sent = []
            beginCharOffset = []
            endCharOffset = []

            #Recreates the original sentences back from the tokens and stores
            #it in a list
            for i in sentence:
                sent = ""
                endchar = 0
                for j in i.token:
                    if j.tokenBeginIndex != 0:
                        if endchar != j.beginChar:
                            sent += " "
                    sent += bytes(j.originalText, 'utf-8').decode('utf-8')
                    endchar = j.endChar
                orig_sent.append(sent.strip())

            print(orig_sent)

            #Gets the tokens
            for i in sentence:
                token = []
                for j in i.token:
                    token.append(j.word)
                tokens.append(token)

            #Gets the beginCharOffset and endCharOffset of each token wrt sentence.
            for i in sentence:
                beChar = []
                enChar = []
                offset = i.characterOffsetBegin
                for j in i.token:
                    beChar.append(j.beginChar - offset)
                    enChar.append(j.endChar - offset)
                beginCharOffset.append(beChar)
                endCharOffset.append((enChar))

            print(beginCharOffset)
            print(endCharOffset)

            predictions = predict(tokens, model, tokenizer)

            sents = []

            for token in tokens:
                sents.append(' '.join([i for i in token]))

            mwe_list = []
            sentences_list = []

            for sent_idx,sent in enumerate(sents):
                print("Sentence Index: {}".format(sent_idx))
                print(orig_sent[sent_idx])
                start_time = time.time()
                doc = get_doc(sent)
                preds = [predictions[sent_idx]]
                annotated_sentences, mwe = analysis(sent_idx, preds, doc, orig_sent,
                                                    beginCharOffset, endCharOffset,
                                                    drop_adjectives=drop_adjectives,
                                                    drop_determiners=drop_determiners)


                mwe_list.append(mwe)
                sentences_list.append(annotated_sentences[0])




            sentences = json.dumps({'sentences': sentences_list},
                                    sort_keys = False, indent = 4, separators = (',', ': '), ensure_ascii=False)
            print("--- %s SECONDS ---" % (time.time() - start1))
    return sentences

if __name__ == '__main__':
    print("Getting the stanford nlp set up")
    get_nlp()
    print("Stanford Nlp object set up")
    print("Loading the neural net model")
    #global model
    load_model()
    print("Model loaded")
    #app.debug = True
    app.run(threaded=False, host='0.0.0.0')
