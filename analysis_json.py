from Input_Output_task import inputoutput, get_num_classes, get_idx, _predTest, get_words_tags, extract_mwe
import time


def utils():
    DOC_PATH = 'docs_new.pkl'
    IDX_PATH = 'idxs_new.pkl'
    n_classes = get_num_classes(DOC_PATH)
    w2idx, idx2l = get_idx(IDX_PATH)
    max_length = 511
    input_dim = 1024

    return n_classes, w2idx, idx2l, max_length, input_dim


def get_inputs(dep_test, weight, model):

    #doc = get_doc(sent)
    print("POI2 - A")
    start_time = time.time()
    n_classes, w2idx, idx2l, max_length, input_dim = utils()
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("[INFO] Getting inputs and X_test enc")
    print("POI2 - B")
    start_time = time.time()
    inputs = inputoutput(dep_test, idx2l, weight, max_length, input_dim)
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("[INFO] Done")
    return inputs

def get_words_pos_upos_tags(predTest, doc):
    words = []
    tags = []
    pos = []
    upos = []
    #characterOffsetBegin = []
    #characterOffsetEnd = []
    for i in predTest:
        #print(i[0],i[1],i[7])
        words.append(i[1])
        pos.append(i[3])
        upos.append(i[4])
        tags.append(i[7])
    
    #for token in doc:
    #    characterOffsetBegin.append(token.idx)
    #    characterOffsetEnd.append(token.idx + len(token.text))
    
    
    return words, pos, upos, tags


def extract_mwe_json(words, pos, upos, tags, sent_idx, characterOffsetBegin, characterOffsetEnd):
    vis = [True]*len(words)
    mwe_list = []
    for idx,val in enumerate(tags):
        name = words[idx]
        if val == 'B' and vis[idx]:
            mwe = [{"index":idx+1, "word":name, "pos":pos[idx], "upos":upos[idx], "mwe": val,
                    "characterOffsetBegin": characterOffsetBegin[sent_idx][idx],
                    "characterOffsetEnd": characterOffsetEnd[sent_idx][idx]}]
            #print(name, val)
            vis[idx] = False
            for j in range(idx+1, len(tags)):
                if tags[j] == 'I':
                    mwe.append({"index":j+1, "word":words[j], "pos":pos[j], "upos":upos[j],
                                "mwe": tags[j], "characterOffsetBegin": characterOffsetBegin[sent_idx][j],
                                "characterOffsetEnd": characterOffsetEnd[sent_idx][j]})
                    #print(words[j], tags[j])
                    vis[j] = False
                else:
                    break
            #print(mwe)
            mwe_list.append(mwe)
        elif val=='I' and vis[idx]:
            mwe_list.append([{"index":idx+1, "word":name, "pos":pos[idx], "upos":upos[idx], 
                              "mwe": val,"characterOffsetBegin": characterOffsetBegin[sent_idx][idx],
                              "characterOffsetEnd": characterOffsetEnd[sent_idx][idx]}])
            vis[idx] = False
    return mwe_list

def analysis(sent_idx, preds, doc, orig_sent, characterOffsetBegin, characterOffsetEnd):

    #print("[INFO] Getting predTest")
    #doc = get_doc(sent)
    n_classes, w2idx, idx2l, max_length, input_dim = utils()
    

    #print(preds.shape)
    #print(X_test_enc.shape)
    
    predTest = _predTest(preds, doc, idx2l)

    words, tags = get_words_tags(predTest[0])
    mwe_list = extract_mwe(words, tags)

    #print("[INFO} Done")
    words, pos, upos, tags = get_words_pos_upos_tags(predTest[0], doc)
    mwe_list_json = extract_mwe_json(words, pos, upos, tags, sent_idx, characterOffsetBegin, characterOffsetEnd)  
        
    
    annotated_sentences = []
    tokens = []
    deps = []
    #mwe = []
    #characterOffsetBegin = 0
    #characterOffsetEnd = 0
    for token in doc:
        #print(token)
        index = token.i + 1 
        word = token.text
        lemma = token.lemma_
        pos = token.pos_
        upos = token.tag_
        #feats = nlp.vocab.morphology.tag_map[upos]
        mwe_tag = predTest[0][index-1][7]
        dep = token.dep_
        governor = token.head.i + 1
        governorGloss = str(token.head)
        #characterOffsetBegin = token.idx
        #characterOffsetEnd = characterOffsetBegin + len(word)
        #print("{} - {} -> {}".format(word, characterOffsetBegin, characterOffsetEnd))
        #print("{} - {} -> {}".format(word, token.i, token.))
        #print(characterOffsetBegin[sent_idx])
        #print(characterOffsetEnd[sent_idx])
        tokens.append({'index': index, 'word': word, 'lemma': lemma, 'characterOffsetBegin':characterOffsetBegin[sent_idx][index-1], 'characterOffsetEnd':characterOffsetEnd[sent_idx][index-1], 'pos': pos, 'upos': upos,'mwe':mwe_tag})
        deps.append({'dep': dep, 'governor': governor, 'governorGloss': governorGloss, 'dependent': index, 'dependentGloss': word})
        #if mwe_tag == 'B' or mwe_tag == 'I':
        #    mwe.append({'index': index, 'word': word,'mwe':mwe_tag})
        
        #if in[characterOffsetEnd + 1] != ' ':
        #    characterOffsetBegin = characterOffsetEnd + 1
        #else:
        #    characterOffsetBegin = characterOffsetEnd

        #characterOffsetBegin = characterOffsetEnd + 1   
        #characterOffsetEnd = characterOffsetBegin
        
    annotated_sentences.append({'index': sent_idx,'sentence':orig_sent[sent_idx], 'basicDependencies': deps, 'tokens': tokens, 'mwe': mwe_list_json})

    return annotated_sentences, mwe_list

