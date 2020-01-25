import os
import time
import requests
import operator
import re
import nltk
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from collections import Counter
import json
from analysis_json import get_inputs_X_test_enc, analysis, utils
from Input_Output_task import get_num_classes, get_idx, model_ELMo_H_combined, get_tests
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from stanfordnlp.server import CoreNLPClient

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CORENLP_HOME'] = "stanford-corenlp-full-2018-10-05/"

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


app = Flask(__name__)
#app.config.from_object(os.environ['APP_SETTINGS'])

client = CoreNLPClient(annotators=['tokenize','ssplit'], timeout=60000, memory='16G')

def embed_elmo2():
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        print("POI2 - B - F - A")
        start_time = time.time() 
        embed = hub.Module('module/module_elmo3')
        print("--- %s seconds ---" % (time.time() - start_time))
        print("POI2 - B - F - B")
        start_time = time.time()
        embeddings = embed(sentences, signature="default", as_dict=True)["elmo"]
        print("--- %s seconds ---" % (time.time() - start_time))
        session = tf.train.MonitoredSession()

    return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_elmo2()

def get_nlp():
    start_time = time.time()
    snlp = stanfordnlp.Pipeline(lang="en", treebank='en_lines', use_gpu=False, tokenize_pretokenized=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    global nlp
    start_time = time.time()
    nlp = StanfordNLPLanguage(snlp)
    print("--- %s seconds ---" % (time.time() - start_time))
    #start_time = time.time()
    #doc = nlp(sent)
    #print("--- %s seconds ---" % (time.time() - start_time))

def get_doc(sent):
    start_time = time.time()
    doc = nlp(sent)
    print("--- %s seconds ---" % (time.time() - start_time))
    return doc

def load_model():
    
    n_classes, w2idx, idx2l, max_length, input_dim = utils()
    global model
    model = model_ELMo_H_combined(max_length, input_dim, n_classes)
    #global graph
    #graph = tf.get_default_graph()
    #global session 
    #session = tf.Session()
    #global models_stored_for_later 
    #models_stored_for_later = [model, graph, session]
    #model._make_predict_function()
    #global session
    #session = tf.Session()
    #K.set_session(session)
    #global graph
    #graph = tf.get_default_graph()
    #return model

@app.route('/', methods=['GET','POST'])
def index():
    #return "welcome" 
    #K.clear_session()
    errors = []
    results = {}
    if request.method == "POST":
        start1 = time.time()
        try:
            print("Reading in sentence to analyse MWEs")
            start_time = time.time()
            #sent = request.form['text']
            sent = request.get_json()
            sent = sent['text']
            print("--- %s seconds ---" % (time.time() - start_time))
            #r = requests.get(url)
            #print(sent)
        except:
            errors.append("Unable to get text. Please make sure it's valid and try again.")
            return render_template('index.html', errors=errors)

        if sent:
            #model = load_model()
            #model, graph, session = models_stored_for_later[0], models_stored_for_later[1], models_stored_for_later[2]
            sent = sent.strip()
            ann = client.annotate(sent)
            sentence = ann.sentence
            tokens = []
            orig_sent = []

            for i in sentence:
                sent = ""
                endchar = 0
                for j in i.token:
                    if j.tokenBeginIndex != 0:
                        if endchar != j.beginChar:
                            sent += " "
                    sent += j.originalText
                    endchar = j.endChar
                orig_sent.append(sent.strip())

            for i in sentence:
                token = []
                for j in i.token:
                    token.append(j.word)
                tokens.append(token)
            
            #sent = ' '.join([i for i in tokens])
            sents = []
            
            for token in tokens:
                sents.append(' '.join([i for i in token]))
            
            start_time = time.time()
            print("Getting in the doc tokens and dictionary")
            print("POI1")
            mwe_list = []
            sentences_list = []
            for sent_idx,sent in enumerate(sents):
                doc = get_doc(sent) #POI1
                print("--- %s seconds ---" % (time.time() - start_time))
                #print(model.summary())
                X_test, dep_test = get_tests(doc)
                start_time = time.time()
                print("Getting predictions")
                #model._make_predict_function()
                print("POI2")
                start_time = time.time()
                for sent_toks in X_test:
                    sent = " ".join(sent_toks)
                weight = embed_fn([sent])
                print("--- %s seconds ---" % (time.time() - start_time))
                print("POI2-P")
                start_time = time.time()
                inputs, X_test_enc = get_inputs_X_test_enc(X_test, dep_test, weight, model) #POI2
                print("--- %s seconds ---" % (time.time() - start_time))
                #with tf.Session as sess:
                #    init = tf.global_variables_initializer()
                #    sess.run(init)
                #    with graph.as_default():
                #global graph
                #with graph.as_default():
                if len(X_test_enc) > 0:
                    start_time = time.time()
                    print("POI3")
                    preds = model.predict(inputs) #POI3
                    print(preds.shape)
                    print(preds)
                    print("--- %s seconds ---" % (time.time() - start_time))
                else:
                    preds = []
                    #with session.graph.as_default():
                start_time = time.time()
                print("POI4")
                annotated_sentences, mwe = analysis(sent_idx, preds, X_test_enc, doc, orig_sent) #POI4
                print("Done")
                print("--- %s seconds ---" % (time.time() - start_time))
                #print(results
                mwe_list.append(mwe)
                sentences_list.append(annotated_sentences[0])
            
            sentences = json.dumps({'sentences': sentences_list},
                                    sort_keys = False, indent = 4, separators = (',', ': '))
            results = [mwe_list, sentences]
            print("--- %s SECONDS ---" % (time.time() - start1))
    #return render_template('index.html', errors=errors, results=results)
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
