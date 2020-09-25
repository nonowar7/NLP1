from sklearn.feature_extraction.text import CountVectorizer
import sys
#import json
#import numpy as np

VALID_PARAMETERS_NUMBER = 4

class MLETrain:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        prog, file_name, q_file, e_file = sys.argv
        return file_name, q_file, e_file

    def readTaggedCorpus(self, tagged_corpus):
        with open(tagged_corpus, 'r', encoding="utf8") as f:
            return f.read().splitlines()

    def writeDictToFile(self, file_name, _dict):
        with open(file_name, 'w') as f:
            for item in _dict:
                entry = item + '\t' + str(_dict[item])
                f.write("%s\n" % entry)

    def keepOnlyTags(self, txt):
        return ' '.join(w.split('/')[-1] for w in txt.split())

    def getTransitions(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(1, 3), preprocessor=self.keepOnlyTags)
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def getEmissions(self, train_data):
        vec = CountVectorizer(lowercase=False, token_pattern=r"\S+")
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        names = [s.replace('/', ' ') for s in names]
        return dict(zip(names, values))

    def getQ(self, t1, t2, t3):
        lambdas = [0.9, 0.09, 0.01]
        try:
            return lambdas[0] * self.transitions[" ".join([t2, t1, t3])] / self.transitions[" ".join([t2, t1])] +\
            lambdas[1] * self.transitions[" ".join([t1, t3])] / self.transitions[t1] +\
                lambdas[2] * self.transitions[t3] / sum(self.transitions.values())
        except:
            return 0

    def getE(self, w, t):
        try:
            return self.emissions[" ".join([w, t])] / self.transitions[t]
        except:
            return 0

    def estimateMLE(self):
        data_file, q_file, e_file = self.readParameters(VALID_PARAMETERS_NUMBER)
        train_data = self.readTaggedCorpus(data_file)
        self.transitions = self.getTransitions(train_data)
        self.emissions = self.getEmissions(train_data)
        # add word signatures to emissions
        self.writeDictToFile(q_file, self.transitions)
        self.writeDictToFile(e_file, self.emissions)


mle = MLETrain()
mle.estimateMLE()
