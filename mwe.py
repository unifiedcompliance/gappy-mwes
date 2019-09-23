import os

class Mwe:
    def __init__(self, path):
        
        trainFile = os.path.join(path, 'train.txt')
        devFile = os.path.join(path, 'dev.txt')
        testFile = os.path.join(path, 'test.txt')
        

        with open(trainFile) as f:
            self.train_collection = f.readlines() 
        
        with open(testFile) as f:
            self.test_collection = f.readlines()
        
        with open(devFile) as f:
            self.dev_collection = f.readlines()


    def sent_extractor(self, collection):
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