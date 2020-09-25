from sklearn.feature_extraction.text import CountVectorizer
import sys
#import json
#import numpy as np

VALID_PARAMETERS_NUMBER = 4


def readParameters():
    if len(sys.argv) != VALID_PARAMETERS_NUMBER:
        print("invalid input")
        return None
    prog, file_name, q_file, e_file = sys.argv
    return file_name, q_file, e_file


def readTaggedCorpus(tagged_corpus):
    with open(tagged_corpus, 'r', encoding="utf8") as f:
        return f.read().splitlines()


def writeDictToFile(file_name, _dict):
    with open(file_name, 'w') as f:
        for item in _dict:
            entry = item + '\t' + str(_dict[item])
            f.write("%s\n" % entry)


def keepOnlyTags(txt):
    return ' '.join(w.split('/')[-1] for w in txt.split())


def getTransitions(train_data):
    vec = CountVectorizer(lowercase=False, ngram_range=(1, 3), preprocessor=keepOnlyTags)
    values = vec.fit_transform(train_data).sum(axis=0).A1
    names = vec.get_feature_names()
    return dict(zip(names, values))


def getEmissions(train_data):
    vec = CountVectorizer(lowercase=False, token_pattern=r"\S+")
    values = vec.fit_transform(train_data).sum(axis=0).A1
    names = vec.get_feature_names()
    names = [s.replace('/', ' ') for s in names]
    return dict(zip(names, values))


def estimateMLE():
    data_file, q_file, e_file = readParameters()
    train_data = readTaggedCorpus(data_file)
    transitions = getTransitions(train_data)
    emissions = getEmissions(train_data)
    # add word signatures to emissions
    # add getQ function
    # add getE function
    writeDictToFile(q_file, transitions)
    writeDictToFile(e_file, emissions)


estimateMLE()
