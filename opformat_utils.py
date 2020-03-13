from spacy_nounchunks import nounChunks
from BIO_to_dataset import read
import copy

def get_sents(doc):
    sents = []
    for token in doc:
        idx = token.i + 1
        x = str(idx) + "\t" + token.text+ "\t" + token.lemma_ + "\t" + token.pos_ + "\t" + token.tag_ + "\t" + str(token.head.i+1) + "\t" + token.dep_
        sents.append(x)
    return sents

def labels2MWE(labels, mainTest_sents):
    mainTest_split  = read(mainTest_sents)
    predTest_split = copy.deepcopy(mainTest_split)
    for sent_id, sent in enumerate(labels):
        for word_id, word in enumerate(sent):
            predTest_split[sent_id][word_id].append(word)

    return predTest_split


def _predTest(preds, doc):
    sents = get_sents(doc)
    predTest = labels2MWE(preds, [sents])
    print(predTest)

    return predTest

def get_words_tags(predTest):
    words = []
    tags = []
    for i in predTest:
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
            for j in range(idx+1, len(tags)):
                if tags[j] == 'I':
                    mwe += " " + words[j]
                    mwe_index.append(j+1)
                else:
                    break
            if len(mwe.split(" ")) > 1:
                mwe_list.append(mwe)
                mwe_indices[mwe] = mwe_index
    return mwe_list, mwe_indices


def get_words_pos_upos_tags(predTest, doc):
    words = []
    tags = []
    pos = []
    upos = []
    for i in predTest:
        words.append(i[1])
        pos.append(i[3])
        upos.append(i[4])
        tags.append(i[7])

    return words, pos, upos, tags

def extract_mwe_json(words, pos, upos, tags, sent_idx, characterOffsetBegin, characterOffsetEnd):
    vis = [True]*len(words)
    mwe_list = {}
    for idx,val in enumerate(tags):
        name = words[idx]
        if val == 'B' and vis[idx]:
            mwe = [{"index":idx+1, "word":name, "pos":pos[idx], "upos":upos[idx], "mwe": val,
                    "characterOffsetBegin": characterOffsetBegin[sent_idx][idx],
                    "characterOffsetEnd": characterOffsetEnd[sent_idx][idx]}]
            vis[idx] = False
            for j in range(idx+1, len(tags)):
                if tags[j] == 'I':
                    name += (" " + words[j])
                    mwe.append({"index":j+1, "word":words[j], "pos":pos[j], "upos":upos[j],
                                "mwe": tags[j], "characterOffsetBegin": characterOffsetBegin[sent_idx][j],
                                "characterOffsetEnd": characterOffsetEnd[sent_idx][j]})
                    vis[j] = False
                else:
                    break
            mwe_list[name] = mwe

    return mwe_list

def mwe_is_a_substring_of_nounchunk(string1, string2):
    if (string1.find(string2) == -1):
        return False
    else:
        return True

def nounchunk_is_a_substring_of_mwe(string1, string2):
    if (string1.find(string2) == -1):
        return False
    else:
        return True


def analysis(sent_idx, preds, doc, orig_sent, characterOffsetBegin,
             characterOffsetEnd, drop_adjectives=True, drop_determiners=True):

    #n_classes, w2idx, idx2l, max_length, input_dim = utils()

    predTest = _predTest(preds, doc)

    words, tags = get_words_tags(predTest[0])
    mwe_list, mwe_indices = extract_mwe(words, tags)

    words, pos, upos, tags = get_words_pos_upos_tags(predTest[0], doc)
    mwe_dict = extract_mwe_json(words, pos, upos, tags, sent_idx, characterOffsetBegin, characterOffsetEnd)
    noun_chunks_list, noun_chunk_mwe, noun_chunk_indices = nounChunks(orig_sent[sent_idx], drop_adjectives, drop_determiners)

    print("MWE detected by neural model")
    print(mwe_list)

    print("Noun chunks detected")
    print(noun_chunks_list)


    same_mwe = []
    mwe_same = []
    for mwe in mwe_list:
        for nc in noun_chunks_list:
            if mwe == nc:
                print("Found same mwe and noun chunk  {}".format(nc))
                same_mwe.append(nc)
            elif nounchunk_is_a_substring_of_mwe(mwe, nc):
                print("Found nounchunk is a subset of mwe {} {}".format(mwe, nc))
                same_mwe.append(nc)
            elif mwe_is_a_substring_of_nounchunk(nc,mwe):
                print("Found mwe is a subset of nounchunk {} {}".format(mwe, nc))
                mwe_same.append(mwe)

    mwe_list_ref = []
    for mwe in mwe_list:
        if mwe not in mwe_same:
            mwe_list_ref.append(mwe)
        else:
            del mwe_dict[mwe]

    for mwe in same_mwe:
        del noun_chunk_mwe[mwe]

    mwe_list_json = []
    for k,v in mwe_dict.items():
        mwe_list_json.append(v)

    for k,v in noun_chunk_mwe.items():
        mwe_list_json.append(v)

    for nc in noun_chunks_list:
        if nc not in same_mwe:
            mwe_list_ref.append(nc)

    annotated_sentences = []
    tokens = []
    deps = []
    for token in doc:
        index = token.i + 1
        word = token.text
        lemma = token.lemma_
        pos = token.pos_
        upos = token.tag_
        mwe_tag = predTest[0][index-1][7]
        dep = token.dep_
        governor = token.head.i + 1
        governorGloss = str(token.head)

        tokens.append({'index': index, 'word': word, 'lemma': lemma, 'characterOffsetBegin':characterOffsetBegin[sent_idx][index-1], 'characterOffsetEnd':characterOffsetEnd[sent_idx][index-1], 'pos': pos, 'upos': upos,'mwe':mwe_tag})
        deps.append({'dep': dep, 'governor': governor, 'governorGloss': governorGloss, 'dependent': index, 'dependentGloss': word})

    annotated_sentences.append({'index': sent_idx,'sentence':orig_sent[sent_idx], 'basicDependencies': deps, 'tokens': tokens, 'mwe': mwe_list_json})

    return annotated_sentences, mwe_list_ref
