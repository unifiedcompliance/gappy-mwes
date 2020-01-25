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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


app = Flask(__name__)
#app.config.from_object(os.environ['APP_SETTINGS'])

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
    snlp = stanfordnlp.Pipeline(lang="en", treebank='en_lines', use_gpu=False)
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
            start_time = time.time()
            print("Getting in the doc tokens and dictionary")
            print("POI1")
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
                print("--- %s seconds ---" % (time.time() - start_time))
                    #with session.graph.as_default():
            else:
                preds = []
                    
            start_time = time.time()
            print("POI4")
            annotated_sentences, mwe_list = analysis(preds, X_test_enc, doc) #POI4
            print("Done")
            print("--- %s seconds ---" % (time.time() - start_time))
            #print(results)
            sentences = json.dumps({'sentences': annotated_sentences},
                                sort_keys = False, indent = 4, separators = (',', ': '))
            results = [mwe_list, sentences]
            print("--- %s SECONDS/SENTENCE ---" % (time.time() - start1))
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
