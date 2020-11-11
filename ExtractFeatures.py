from sklearn.feature_extraction.text import CountVectorizer
import sys
import time
VALID_PARAMETERS_NUMBER = 3

class ExtractFeatures:
    def __init__(self):
        self.words_count = {}

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
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

    def keepOnlyTags(self, txt):
        return ' '.join(w.rsplit('/', 1)[0] for w in txt.split())

    def getWordsCount(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(1, 1), preprocessor=self.keepOnlyTags, token_pattern=r"\S+")
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def getFeaturesFromInput(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(5, 5), token_pattern=r"\S+")
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def createFeaturesDict(self, features_count):
        features_list = []
        for entry in features_count:
            words_tags = entry.split()
            if len(words_tags) != 5:
                print(len(words_tags))
                continue
            list_of_word_tag = []
            for word_tag in words_tags:
                list_of_word_tag.append(word_tag.rsplit('/', 2))
            label_str = list_of_word_tag[2][1]
            word = list_of_word_tag[2][0]
            len_word = len(word)
            rare_word = True if self.words_count.get(word, 0 ) <= 5 else False
            '''
            if rare_word:
                for i in range(5):
                    if i == 2:
                        continue
                    label_str = " ".join([label_str, "=".join(["".join(['word', str(i - 2)]), list_of_word_tag[i][0]])])
                for i, c in enumerate(word):
                    if i > 3:
                        break
                    label_str = " ".join([label_str, '='.join([''.join(['PrefixC', str(i)]), word[0:i+1]])])
                    label_str = " ".join([label_str, '='.join([''.join(['SuffixC', str(i)]), word[len_word-1-i:len_word]])])
                if any(str.isdigit(c) for c in word):
                    label_str = " ".join([label_str, '='.join(['number', str(1)])])
                else:
                    label_str = " ".join([label_str, '='.join(['number', str(0)])])
                if any(str.upper(c) for c in word):
                    label_str = " ".join([label_str, '='.join(['uppercase', str(1)])])
                else:
                    label_str = " ".join([label_str, '='.join(['uppercase', str(0)])])
                if '-' in word:
                    label_str = " ".join([label_str, '='.join(['hyphen', str(1)])])
                else:
                    label_str = " ".join([label_str, '='.join(['hyphen', str(0)])])
            '''
            for i in range(5):
                label_str = " ".join([label_str, "=".join(["".join(['word', str(i-2)]), list_of_word_tag[i][0]])])
            label_str = " ".join([label_str, "=".join(['tag-1' ,list_of_word_tag[1][1]])])
            label_str = " ".join([label_str, "=".join(['tag-2tag-1', ";".join([list_of_word_tag[0][1], list_of_word_tag[1][1]])])])
            features_list.append(label_str)
        return features_list

    def runExtractFeatures(self):
        a = time.time()
        corpus_file, features_file = self.readParameters(VALID_PARAMETERS_NUMBER)
        train_data = self.readInputFile(corpus_file)
        features_count = self.getFeaturesFromInput(train_data)
        words_count = self.getWordsCount(train_data)
        features_list = self.createFeaturesDict(features_count)
        self.writeFeaturesToFile(features_list, features_file)
        print(time.time()-a)


def extract(sent, i, prev_tag, prev_prev_tag):
    features = ""
    for j, token in enumerate(sent[i-2:i+3]):
        features = " ".join([features, "=".join(["".join(['word', str(j-2)]), token])])
    features = " ".join([features, "=".join(['tag-1', prev_tag])])
    features = " ".join([features, "=".join(['tag-2tag-1', ";".join([prev_prev_tag, prev_tag])])])
    return features

'''
ef = ExtractFeatures()
ef.runExtractFeatures()
'''
