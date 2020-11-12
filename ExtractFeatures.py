from sklearn.feature_extraction.text import CountVectorizer
import sys
import time
import math
NGRAM = 5

class ExtractFeatures:
    def __init__(self):
        self.num_input_param = 3
        self.num_rare_word = 5

    def readParameters(self):
        if len(sys.argv) != self.num_input_param:
            print("invalid input")
            return None
        syst, corpus_file, features_file = sys.argv
        return corpus_file, features_file

    def readInputFile(self, file_name):
        with open(file_name, 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        lines = []
        for line in content:
            new_line = "STARTword/STARTtag STARTword/STARTtag " + line + " ENDword/ENDtag ENDword/ENDtag"
            lines.append(new_line)
        return lines

    def writeFeaturesToFile(self, features_list, features_file):
        with open(features_file, 'w') as f:
            for item in features_list:
                f.write("%s\n" % item)

    def keepOnlyWords(self, txt):
        return ' '.join(w.rsplit('/', 1)[0] for w in txt.split())

    def getWordsCount(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(1, 1),
                              preprocessor=self.keepOnlyWords, token_pattern=r"\S+", min_df=self.num_rare_word)
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def getFeaturesFromInput(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(NGRAM, NGRAM), token_pattern=r"\S+")
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def extract(self, sent, i, prev_prev_tag, prev_tag, rare):
        features = ""
        for j, token in enumerate(sent[i - 2:i + 3]):
            len_tok = len(token)
            if rare and j == 2:
                for k in range(min(4, math.floor(len_tok/2))):
                    features = " ".join([features, '='.join([''.join(['PC_', str(k)]), token[0:k + 1]])])
                    features = " ".join([features, '='.join([''.join(['SC_', str(k)]), token[len_tok-1- k:len_tok]])])
                if any(str.isdigit(c) for c in token):
                    features = " ".join([features, '='.join(['number', str(1)])])
                if any(str.upper(c) for c in token):
                    features = " ".join([features, '='.join(['uppercase', str(1)])])
                if '-' in token:
                    features = " ".join([features, '='.join(['hyphen', str(1)])])
                continue
            features = " ".join([features, "=".join(["".join(['w', str(j - 2)]), token])])
        features = " ".join([features, "=".join(['pt', prev_tag])])
        features = " ".join([features, "=".join(['ppt', ",".join([prev_prev_tag, prev_tag])])])
        return features

    def createFeaturesDict(self, features_count, known_words):
        features_list = []
        for entry in features_count:
            words_and_tags = [x.rsplit('/', 1) for x in entry.split()]
            if len(words_and_tags) != NGRAM:
                print('feature string is not the right size')
                continue
            words, tags = [x[0] for x in words_and_tags], [x[1] for x in words_and_tags]
            rare_word = True if words[2] not in known_words else False
            feature_str = " ".join([tags[2], self.extract(words, 2, tags[0], tags[1], rare_word)])
            features_list.append(feature_str)
        return features_list

    def runExtractFeatures(self):
        a = time.time()
        corpus_file, features_file = self.readParameters()
        train_data = self.readInputFile(corpus_file)
        features_count = self.getFeaturesFromInput(train_data)
        known_words = self.getWordsCount(train_data)
        features_list = self.createFeaturesDict(features_count, known_words)
        self.writeFeaturesToFile(features_list, features_file)
        print(time.time()-a)


def main():
    ef = ExtractFeatures()
    ef.runExtractFeatures()

if __name__== "__main__":
    main()