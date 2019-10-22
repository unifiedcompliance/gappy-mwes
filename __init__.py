import os
import requests
import operator
import re
import nltk
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from collections import Counter
import json
from analysis_json import get_inputs_X_test_enc, analysis, utils
from Input_Output_task import get_num_classes, get_idx, model_ELMo_H_combined
from keras import backend as K
import tensorflow as tf
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage

app = Flask(__name__)
#app.config.from_object(os.environ['APP_SETTINGS'])

def get_doc(sent):
    snlp = stanfordnlp.Pipeline(lang="en", treebank='en_lines')
    nlp = StanfordNLPLanguage(snlp)
    doc = nlp(sent)

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
        try:
            sent = request.form['text']
            #r = requests.get(url)
            #print(sent)
        except:
            errors.append("Unable to get text. Please make sure it's valid and try again.")
            return render_template('index.html', errors=errors)

        if sent:
            #model = load_model()
            #model, graph, session = models_stored_for_later[0], models_stored_for_later[1], models_stored_for_later[2]
            sent = sent.strip()
            doc = get_doc(sent)
            #print(model.summary())
            print("Getting predictions")
            #model._make_predict_function()
            inputs, X_test_enc = get_inputs_X_test_enc(doc, model)
            #with tf.Session as sess:
            #    init = tf.global_variables_initializer()
            #    sess.run(init)
            #    with graph.as_default():
            #global graph
            #with graph.as_default():
            preds = model.predict(inputs)
                #with session.graph.as_default():
                    

            results = analysis(preds, X_test_enc, doc)
            print("Done")
            #print(results)
            results = json.dumps({'sentences': results},
                                sort_keys = False, indent = 4, separators = (',', ': '))
        
    return render_template('index.html', errors=errors, results=results)
    



if __name__ == '__main__':
    print("Loading the neural net model")
    #global model
    load_model()
    print("Model loaded")
    #app.debug = True
    app.run(threaded=False, host='0.0.0.0')