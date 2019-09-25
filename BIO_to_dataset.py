import os
import copy

def read(sents):
		seqs = []
		for i in sents:
			seqs_i = []
			for j in i:
				j_list = j.split('\t')
				seqs_i.append(j_list)
			seqs.append(seqs_i)
		
		return seqs

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



def labels2MWE(labels, mainTest, predOut):
    """
    labels: predicted labels
    mainTest: the file that contains predicted labels should be matched with gold file
    predOut: output

    This function is used to convert BIO labeling back to the dataset format for evaluation.
    """
    with open(mainTest) as bt:
        mainTest = bt.readlines()

    mainTest_sents = sent_extractor(mainTest)
    mainTest_split  = read(mainTest_sents)
    predTest_split = copy.deepcopy(mainTest_split)

    for sent_id, sent in enumerate(labels):
        for word_id, word in enumerate(sent):
            if word=='<PADLABEL>':
                continue
            else:
                predTest_split[sent_id][word_id][7] = word
    
    with open(predOut, 'w') as f:
        for i in predTest_split:
            for j in i:
                f.write('\t'.join(j)) 
            f.write('\n')




    

