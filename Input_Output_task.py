import re, h5py, pickle, os
import numpy as np
from collections import Counter 
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from corpus_reader import * 
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout,concatenate, Conv1D, BatchNormalization, CuDNNLSTM, Lambda, \
                        Multiply, Add, Activation, Flatten, MaxPooling1D 
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from keras.regularizers import l2
import keras.initializers
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from models.layers import *
from BIO_to_dataset import read
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import tensorflow_hub as hub
import time
import copy

#gpu_options = tf.GPUOptions(visible_device_list="0")
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
#sess.global_variables_initializer().run()
#sess.tables_initializer().run()

#elmo.get_input_info_dict


def get_sents(doc):
    sents = []
    for token in doc:
        idx = token.i + 1 
        x = str(idx) + "\t" + token.text+ "\t" + token.lemma_ + "\t" + token.pos_ + "\t" + token.tag_ + "\t" + str(token.head.i+1) + "\t" + token.dep_
        sents.append(x)
    return sents


def read_sents(sents):
    seqs = []
    seqs_i = []
    for i in sents:
        j_list = i.split('\t')
        seqs_i.append((j_list[1], j_list[2], j_list[3], int(j_list[5]), j_list[6]))
    seqs.append(seqs_i)
    return seqs

def load_elmo(X, max_length):	# the aim is to create a numpy array of shape (sent_num, max_sent_size, 1024)
			filename =  'embeddings/ELMO_EN.hdf5'#.format(self.lang)
			elmo_dict = h5py.File(filename, 'r')
			lst = []
			not_in_dict = 0
			for sent_toks in X:
				sent = "\t".join(sent_toks)
				if sent in elmo_dict:
				    item = list(elmo_dict[sent])	# ELMo representations for all words in the sentence
				else:
				    print("NO", sent, "is not in ELMO")
				    not_in_dict +=1		
				    item = list(np.zeros((len(sent_toks), 1024)))
				min_lim = len(item)	#len(sent_toks)
				for i in range(min_lim, max_length):	# Here, we do padding, to make all sentences the same size
				    item.append([0]*1024)

				lst.append(item)
			if len(X):
				print('not found sentences:', not_in_dict)

			print('ELMO Loaded ...')
			return np.array(lst, dtype = np.float32)

def load_adjacency(dep, direction, max_length):

		if direction == 1:
			dep_adjacency = [adjacencyHead2Dep(d, max_length) for d in dep]
		elif direction == 0:
			dep_adjacency = [adjacencyDep2Head(d, max_length) for d in dep]
		elif direction == 3:
			dep_adjacency = [adjacencySelf(d, max_length) for d in dep]
		
		return np.array(dep_adjacency)

def adjacencyDep2Head(sentDep, max_length):
    adjacencyMatrix = np.zeros((max_length,max_length), dtype=np.int)
    for i in range(len(sentDep)):
        if sentDep[i] != 0:
            adjacencyMatrix[i][sentDep[i]-1] = 1
            # adjacencyMatrix[sentDep[i]-1][i] = -1
    return adjacencyMatrix

def adjacencyHead2Dep(sentDep, max_length):
    adjacencyMatrix = np.zeros((max_length,max_length), dtype=np.int)
    for i in range(len(sentDep)):
        if sentDep[i] != 0:
            #adjacencyMatrix[i][sentDep[i]-1] = 1
            adjacencyMatrix[sentDep[i]-1][i] = 1
    return adjacencyMatrix

def adjacencySelf(sentDep, max_length):
    adjacencyMatrix = np.zeros((max_length,max_length), dtype=np.int)
    for i in range(len(sentDep)):
            adjacencyMatrix[i][i] = 1
    return adjacencyMatrix

def model_ELMo_H_combined(max_length, input_dim, n_classes):
		"""concat gcn and self_att and then apply highway on the output, finally concatenate with gcn. num of highway layers = 3, bias = -2"""

		# some variables to set 
		hidden = 300
		dim = 200 #hidden * 2 
		k = 1
		n_attn_heads = 4
		l2_reg = 5e-4/2
		dropout_rate = 0.5
		n_layers = 4 

		X_in = Input(shape=(max_length, input_dim), name='x-data')
		A_in = [Input(shape=(max_length, max_length), name='A-edge_{}'.format(i)) for i in range(True)]

		# stacked CNNs
		cn_stacked = Conv1D(100, k, activation="relu", padding="same")(X_in)
		cn_stacked = Conv1D(100, k, activation="relu", padding="same")(cn_stacked) 
		# parallel CNN
		cn = Conv1D(100, k, activation="relu", padding="same")(X_in) 
		cn_concat = concatenate([cn, cn_stacked])         
		cn_concat = Dropout(dropout_rate)(cn_concat)

		att = MultiHeadAttention(n_head=n_attn_heads, d_model=dim, d_k=dim, d_v=dim, dropout=0.5, mode=1)
		self_att = att(q=cn_concat, k=cn_concat, v=cn_concat)[0]

		gcn = SpectralGraphConvolution(200, activation='relu')([X_in] + A_in)
		gcn = Dropout(dropout_rate)(gcn)


		conc = concatenate([gcn, self_att])
		gate = BatchNormalization()(Highway(n_layers = n_layers, value=conc, gate_bias=-2)) 
		lstm = Bidirectional(LSTM(hidden,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(gate)
        
        #if max_length < 300:
		#	lstm = Bidirectional(LSTM(hidden,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(gate)
		#else:
		#	print('CuDNNLSTM')
		#	lstm = Bidirectional(CuDNNLSTM(hidden,return_sequences=True, name='lstm'))(gate)
		#	lstm = Dropout(0.5)(lstm)
        
		output = Dense(n_classes, activation='softmax', name='dense')(lstm)

		# Build the model
		model = Model(inputs=[X_in] + A_in, outputs=output)
		model.load_weights('results/EN_GCN_model_ELMo_H_combined_results_50/weights-improvement-50-1.00.hdf5')
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae','acc'])
		#print(model.summary())
		return model


def get_final_preds(preds, idx2l):
    final_preds = []
    pred = np.argmax(preds[0],-1)
    pred = [idx2l[p] for p in pred]
    final_preds.append(pred)
    
    return final_preds


def labels2MWE(labels, mainTest_sents):
    mainTest_split  = read(mainTest_sents)
    predTest_split = copy.deepcopy(mainTest_split)
    for sent_id, sent in enumerate(labels):
        for word_id, word in enumerate(sent):
            if word=='<PADLABEL>':
                continue
            else:
                predTest_split[sent_id][word_id].append(word)
    
    return predTest_split


def get_words_tags(predTest):
    words = []
    tags = []
    for i in predTest:
        #print(i[0],i[1],i[7])
        words.append(i[1])
        tags.append(i[7])
    
    return words, tags


def extract_mwe(words, tags):
    mwe_list = []
    mwe_indices = {}
    for idx,val in enumerate(tags):
        if val == 'B':
            name = words[idx]
            mwe = name
            mwe_index = [idx+1]
            #print(name, val)
            for j in range(idx+1, len(tags)):
                if tags[j] == 'I':
                    mwe += " " + words[j]
                    #print(words[j], tags[j])
                    mwe_index.append(j+1)
                else:
                    break
            if len(mwe.split(" ")) > 1:
                mwe_list.append(mwe)
                mwe_indices[mwe] = mwe_index
    return mwe_list, mwe_indices

def embed_elmo2(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        print("POI2 - B - F - A")
        start_time = time.time() 
        embed = hub.Module(module)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("POI2 - B - F - B")
        start_time = time.time()
        embeddings = embed(sentences, signature="default", as_dict=True)["elmo"]
        print("--- %s seconds ---" % (time.time() - start_time))
        session = tf.train.MonitoredSession()

    return lambda x: session.run(embeddings, {sentences: x})

def elmo_vectors(x):
    #print("POI2 - B - F - A")
    start_time = time.time() 
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
    #print("--- %s seconds ---" % (time.time() - start_time))

    #print("POI2 - B - F - B")
    start_time = time.time()
    #x = sess.run(embeddings)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #return x
    
    with tf.Session() as sess:
        #print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #print("--- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
        #sess.run(tf.tables_initializer())
        #print("--- %s seconds ---" % (time.time() - start_time))
        # return average of ELMo features
        start_time = time.time()
        x = sess.run(embeddings)
        #print("--- %s seconds ---" % (time.time() - start_time))
        return x

def get_tests(doc):
    sents = get_sents(doc)
    test = read_sents(sents)
    X_test = [[x[0].replace('.',"$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem] for elem in test]
    dep_test = [[x[3] for x in elem] for elem in test]

    return X_test, dep_test

def inputoutput(dep_test, idx2l, weight, max_length = 511, input_dim = 1024):

    """
    #input_str = "{risk factor} Determine whether the institution has appropriate standards and processes for risk-based auditing and internal risk assessments that : Describe the process for assessing and documenting risk and control factors and its application in the formulation of audit plans , resource allocations , audit scopes , and audit cycle frequency"
    #snlp = stanfordnlp.Pipeline(lang="en", treebank='en_lines')
    #nlp = StanfordNLPLanguage(snlp)
    #doc = nlp(sentence)
    #print("POI2 - B - A")
    #start_time = time.time()
    #sents = get_sents(doc)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #print("POI2 - B - B")
    #start_time = time.time()
    #test = read_sents(sents)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #print("POI2 - B - C")
    #start_time = time.time()
    #X_test = [[x[0].replace('.',"$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem] for elem in test]
    #print("--- %s seconds ---" % (time.time() - start_time))
    #print("POI2 - B - D")
    #start_time = time.time()
    #dep_test = [[x[3] for x in elem] for elem in test]
    #print("--- %s seconds ---" % (time.time() - start_time))
    #max_length = 265

    #with open('../docs.pkl', 'rb') as f:
    #    docs = pickle.load(f)

    #words = docs['words']
    #vocab_size = docs['vocab_size']
    #n_classes = docs['n_classes']
    #n_poses = docs['n_poses']

    #with open('../idxs.pkl', 'rb') as f:
    #    idxs = pickle.load(f)

    #w2idx = idxs['w2idx']
    #l2idx = idxs['l2idx']
    #pos2idx = idxs['pos2idx']
    #idx2w = idxs['idx2w']
    #idx2l = idxs['idx2l']
    #idx2pos = idxs['idx2pos']

    print("POI2 - B - E")
    start_time = time.time()
    try:
        X_test_enc = [w2idx[w] for w in X_test[0]]
        X_test_enc = pad_sequences([X_test_enc], maxlen=max_length, padding='post')
    except Exception as e:
        X_test_enc = []
    print("--- %s seconds ---" % (time.time() - start_time))
    #e = Embedder('../../pytorch/144/')
    #for i in range(10):
    #    weight = e.sents2elmo(X_test)
    
    for sent_toks in X_test:
        sent = " ".join(sent_toks)

    print("POI2 - B - F")
        
    #weight = elmo_vectors([sent])
    #embed_fn = embed_elmo2('module/module_elmo2')
    #weight = embed_fn([sent])
    """
    start_time = time.time()
    #print(weight.shape)
    #print(weight[0].shape)
    lim, elmo_n = weight[0].shape
    #print(lim, elmo_n)
    weight = weight[0].reshape(1, lim, input_dim)
    #print(weight.shape)
    test_weights = np.zeros((1, max_length, input_dim))
    test_weights[:, :lim, :] = weight
    #print(test_weights.shape)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #test_weights = load_elmo(X_test, max_length)

    #print("POI2 - B - G")
    start_time = time.time()   
    test_adjacency = load_adjacency([dep_test[0]], 1, max_length)
    test_adjacency_matrices = [test_adjacency]
    inputs = [test_weights]
    inputs += test_adjacency_matrices
    #print("--- %s seconds ---" % (time.time() - start_time))
    #input_dim = len(test_weights[0][0])
    #model = model_ELMo_H_combined(max_length, input_dim, n_classes)
    return inputs

def _predTest(preds, doc, idx2l):
    final_preds = get_final_preds(preds, idx2l)
    sents = get_sents(doc)
    predTest = labels2MWE(final_preds, [sents])

    return predTest

def get_num_classes(DOC_PATH):
    
    with open(DOC_PATH, 'rb') as f:
        docs = pickle.load(f)
    
    n_classes = docs['n_classes']

    return n_classes

def get_idx(IDX_PATH):
    
    with open(IDX_PATH, 'rb') as f:
        idxs = pickle.load(f)

    w2idx = idxs['w2idx']
    idx2l = idxs['idx2l']

    return w2idx, idx2l

if __name__ == "__main__":
    print("Please enter the sentence")
    input_str = input()
    
    DOC_PATH = 'docs.pkl'
    IDX_PATH = 'idxs.pkl'
    n_classes = get_num_classes(DOC_PATH)
    w2idx, idx2l = get_idx(IDX_PATH) 
    max_length = 503
    input_dim = 1024
    
    model = model_ELMo_H_combined(max_length, input_dim, n_classes)

    snlp = stanfordnlp.Pipeline(lang="en", treebank='en_lines')
    nlp = StanfordNLPLanguage(snlp)
    doc = nlp(input_str)
    inputs, X_test_enc = inputoutput(doc, n_classes, w2idx, idx2l, max_length, input_dim)
    preds = model.predict(inputs, batch_size=16, verbose=1)
    predTest = _predTest(preds, X_test_enc, doc, idx2l)

    words, tags = get_words_tags(predTest[0])
    mwe_list = extract_mwe(words, tags)

    print("The multi word expressions are:")
    for mwe in mwe_list:
        print(mwe)




