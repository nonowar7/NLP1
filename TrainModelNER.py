from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import sys
import numpy as np
import pickle
VALID_PARAMETERS_NUMBER = 3

class TrainModel:
    def __init__(self):
        pass

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, features_file, model_file = sys.argv
        return features_file, model_file

    def readInputFile(self, file_name):
        with open(file_name, 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        return content

    def getFeaturesAndTags(self, features_input):
        X, Y = [], []
        for line in features_input:
            split_arr = line.split()
            features_dict = {}
            for f_v in split_arr[1:]:
                f_v = f_v.split('=')
                features_dict[f_v[0]] = f_v[1]
            X.append(features_dict)
            Y.append(split_arr[0])
        return X, Y

    def getFeaturesAsDict(self, X):
        dv = DictVectorizer(sparse=True)
        features_vec = dv.fit_transform(X)
        return dv, features_vec

    def trainWithSGD(self, X, Y):
        def batches(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        clf = SGDClassifier(loss='log', alpha=0.0000005)
        rows = X.get_shape()[0]
        for i in range(100):
            print(i)
            X, Y = shuffle(X, Y)
            for batch in batches(range(rows), 100000):
                clf.partial_fit(X[batch[0]:batch[-1]+1], Y[batch[0]:batch[-1]+1], np.unique(Y))
        return clf

    def saveModelToFile(self, model, file_name):
        pickle.dump(model, open(file_name, 'wb'))

    def saveFeaturesMapToFile(self, known_words, dv, features_map_file):
        pickle.dump([known_words,dv], open(features_map_file ,'wb'))

    def createKnownWordsDictionary(self,X,Y):
        known_words = {}
        for x,y in zip(X,Y):
            if 'w0' not in x:
                continue
            word = x['w0']
            if word not in known_words:
                known_words[word] = {}
            known_words[word][y] = 1
        return known_words

    def runTrainModel(self):
        features_file, model_file = self.readParameters(VALID_PARAMETERS_NUMBER)
        features_input = self.readInputFile(features_file)
        X, Y = self.getFeaturesAndTags(features_input)
        known_words = self.createKnownWordsDictionary(X,Y)
        dv, features_vec = self.getFeaturesAsDict(X)
        clf = self.trainWithSGD(features_vec, Y)
        self.saveFeaturesMapToFile(known_words, dv, 'ner_feature_map_file_improve')
        self.saveModelToFile(clf, model_file)


model = TrainModel()
model.runTrainModel()