import os
import requests
import operator
import re
import nltk
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from collections import Counter
import json
from analysis_json import get_inputs_X_test_enc, analysis
from Input_Output_task import get_num_classes, get_idx, model_ELMo_H_combined
from keras import backend as K
import tensorflow as tf

app = Flask(__name__)
#app.config.from_object(os.environ['APP_SETTINGS'])

global model
global graph
global session

def utils():
    DOC_PATH = '../docs.pkl'
    IDX_PATH = '../idxs.pkl'
    n_classes = get_num_classes(DOC_PATH)
    w2idx, idx2l = get_idx(IDX_PATH)
    max_length = 265
    input_dim = 1024

    return n_classes, w2idx, idx2l, max_length, input_dim

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
            print(sent)
        except:
            errors.append("Unable to get text. Please make sure it's valid and try again.")
            return render_template('index.html', errors=errors)

        if sent:
            #model = load_model()
            #model, graph, session = models_stored_for_later[0], models_stored_for_later[1], models_stored_for_later[2]
            print(model.summary())
            print("Starting Analysis")
            #model._make_predict_function()
            inputs, X_test_enc = get_inputs_X_test_enc(sent, model)
            #with tf.Session as sess:
            #    init = tf.global_variables_initializer()
            #    sess.run(init)
            #    with graph.as_default():
            #global graph
            #with graph.as_default():
            preds = model.predict(inputs)
                #with session.graph.as_default():
                    

            results = analysis(preds, X_test_enc, sent)
            print("Analysis Complete")
            print(results)
            results = json.dumps({'sentences': results},
                                sort_keys = False, indent = 4, separators = (',', ': '))
        
    return render_template('index.html', errors=errors, results=results)
    



if __name__ == '__main__':
    print("Loading the neural net model")
    #global model
    load_model()
    print("Model loaded")
    #app.debug = True
    app.run(threaded=False)