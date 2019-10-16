from Input_Output_task import inputoutput, get_num_classes, get_idx, _predTest, get_words_tags
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage


def utils():
    DOC_PATH = '../docs.pkl'
    IDX_PATH = '../idxs.pkl'
    n_classes = get_num_classes(DOC_PATH)
    w2idx, idx2l = get_idx(IDX_PATH)
    max_length = 265
    input_dim = 1024

    return n_classes, w2idx, idx2l, max_length, input_dim

def get_doc(sent):
    snlp = stanfordnlp.Pipeline(lang="en", treebank='en_lines')
    nlp = StanfordNLPLanguage(snlp)
    doc = nlp(sent)

    return doc


def get_inputs_X_test_enc(sent, model):

    doc = get_doc(sent)
    n_classes, w2idx, idx2l, max_length, input_dim = utils()
    print("[INFO] Getting inputs and X_test enc")
    inputs, X_test_enc = inputoutput(doc, n_classes, w2idx, idx2l, max_length, input_dim)
    print("[INFO] Done")
    return inputs, X_test_enc

def extract_mwe(words, tags):
    vis = [True]*len(words)
    mwe_list = []
    for idx,val in enumerate(tags):
        name = words[idx]
        if val == 'B' and vis[idx]:
            mwe = [{"index":idx+1, "word":name, "mwe": val}]
            #print(name, val)
            vis[idx] = False
            for j in range(idx+1, len(tags)):
                if tags[j] == 'I':
                    mwe.append({"index":j+1, "word":words[j], "mwe": tags[j]})
                    #print(words[j], tags[j])
                    vis[j] = False
                else:
                    break
            #print(mwe)
            mwe_list.append(mwe)
        elif val=='I' and vis[idx]:
            mwe_list.append([{"index":idx+1, "word":name, "mwe": val}])
            vis[idx] = False
    return mwe_list


def analysis(preds, X_test_enc, sent):

    print("[INFO] Getting predTest")
    doc = get_doc(sent)
    n_classes, w2idx, idx2l, max_length, input_dim = utils()
    predTest = _predTest(preds, X_test_enc, doc, idx2l)
    print("[INFO} Done")
    words, tags = get_words_tags(predTest[0])
    mwe_list = extract_mwe(words, tags)
    
    annotated_sentences = []
    tokens = []
    deps = []
    #mwe = []
    characterOffsetBegin = 0
    characterOffsetEnd = 0
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
        characterOffsetEnd += len(word)
        #print("{} - {} -> {}".format(word, characterOffsetBegin, characterOffsetEnd))
        #print("{} - {} -> {}".format(word, token.i, token.))
        tokens.append({'index': index, 'word': word, 'lemma': lemma, 'characterOffsetBegin':characterOffsetBegin, 'characterOffsetEnd':characterOffsetEnd, 'pos': pos, 'upos': upos,'mwe':mwe_tag})
        deps.append({'dep': dep, 'governor': governor, 'governorGloss': governorGloss, 'dependent': index, 'dependentGloss': word})
        #if mwe_tag == 'B' or mwe_tag == 'I':
        #    mwe.append({'index': index, 'word': word,'mwe':mwe_tag})
        
        #if in[characterOffsetEnd + 1] != ' ':
        #    characterOffsetBegin = characterOffsetEnd + 1
        #else:
        #    characterOffsetBegin = characterOffsetEnd

        characterOffsetBegin = characterOffsetEnd + 1   
        characterOffsetEnd = characterOffsetBegin
        
    annotated_sentences.append({'index': 0,'sentence':str(doc), 'basicDependencies': deps, 'tokens': tokens, 'mwe': mwe_list})

    return annotated_sentences

