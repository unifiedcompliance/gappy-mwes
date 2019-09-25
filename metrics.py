def sent_extractor(collection):
        sents = []
        tok_det = []
        for i in collection:
                if i=="\n":
                    sents.append(tok_det)
                    tok_det = []
                else:
                    tok_det.append(i)

        if len(tok_det) > 0:
            sents.append(tok_det)
        
        return sents

def read(sents):
		seqs = []
		for i in sents:
			seqs_i = []
			for j in i:
				j_list = j.split('\t')
				seqs_i.append([j_list[0], j_list[1], j_list[7]])
			seqs.append(seqs_i)
		
		return seqs


def mwe_tag_index_extractor(tags):
    mwe_tags_start = {}
    for idx, sentence in enumerate(tags):
        print("Working on index {}".format(idx))
        mwe_tags_start[idx] = []
        for i,j in enumerate(sentence):
            mwe_indices = []
            if j[2] == 'B':
                print("------- Found B-tag at {}".format(i))
                k = i
                mwe_indices.append(k)
                if k+1 < len(sentence):
                    k += 1
                else:
                    break
                while sentence[k][2] == 'I':
                    print("---------------- Appended k: {}".format(k))
                    mwe_indices.append(k)
                    k += 1
                    if k >= len(sentence):
                        break
                mwe_tags_start[idx].append(mwe_indices)
    return mwe_tags_start


def metrics(test_actual_mwe_tags, test_pred_mwe_tags):
    exact_matches_tp = 0
    exact_matches_fn = 0
    exact_matches_fp = 0
    #tp_list = []
    #fp_list = []
    #fn_list = []
    for k,v in test_actual_mwe_tags.items():
        test_actual_tag_indices = v
        test_pred_tag_indices = test_pred_mwe_tags[k]
        print(k)
        print(test_actual_tag_indices)
        print(test_pred_tag_indices)
        exact_match_tp = 0
        exact_match_fn = 0
        exact_match_fp = 0
        for i in test_actual_tag_indices:
            print("Searching for a match for {}".format(i))
            if i in test_pred_tag_indices:
                print("True positive")
                exact_match_tp += 1
            else:
                print("True Negative")
                exact_match_fn += 1

        for i in test_pred_tag_indices:
            print("Searching for a match for {} in the actual test".format(i))
            if i not in test_actual_tag_indices:
                print("False positive")
                exact_match_fp += 1

        #tp_list.append(exact_match_tp)
        #fp_list.append(exact_match_fp)
        #fn_list.append(exact_match_fn)

        exact_matches_tp += exact_match_tp
        exact_matches_fn += exact_match_fn
        exact_matches_fp += exact_match_fp
        
    return exact_matches_tp, exact_matches_fn, exact_matches_fp

def calc_precision(tp, fp):
    return tp / (tp+fp)

def calc_recall(tp, fn):
    return tp / (tp+fn)

def metrics_f1_score(actual_test_file, pred_test_file):

    with open(actual_test_file,'r') as f:
        test_actual = f.readlines()
    
    with open(pred_test_file,'r') as f:
        test_pred = f.readlines()
    
    test_actual_sent = sent_extractor(test_actual) 
    test_pred_sent = sent_extractor(test_pred)

    test_actual_tags = read(test_actual_sent)
    test_pred_tags = read(test_pred_sent)

    test_actual_mwe_tags = mwe_tag_index_extractor(test_actual_tags)
    test_pred_mwe_tags = mwe_tag_index_extractor(test_pred_tags)

    tp, fn, fp = metrics(test_actual_mwe_tags, test_pred_mwe_tags)

    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, fn)

    f1_score = 2 / ((1.0/recall) + (1.0/precision))

    return f1_score