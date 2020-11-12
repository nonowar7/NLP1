from sklearn.feature_extraction.text import CountVectorizer
from ExtractFeatures import ExtractFeatures
import sys
import time
import pickle
VALID_PARAMETERS_NUMBER = 5

class FeaturesTagger:
    def __init__(self):
        self.num_rare_word = 2

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, input_file, model_file, feature_map_file, features_output_file = sys.argv
        return input_file, model_file, feature_map_file, features_output_file

    def readInputFile(self, file_name):
        with open(file_name, 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        lines = []
        for line in content:
            new_line = "STARTword STARTword " + line + " ENDword"
            lines.append(new_line)
        return lines

    def loadFeatureMap(self, file_name):
        return pickle.load(open(file_name, "rb"))

    def loadModel(self, file_name):
        return pickle.load(open(file_name, "rb"))

    def writeOutputsToFile(self, file_name, outputs):
        with open(file_name, 'w') as f:
            for output in outputs:
                f.write("%s\n" % output)

    def getOutputSequence(self, input, tags_sequence, outputs):
        output = ""
        for word, tag in zip(input.split()[2:len(input.split())], tags_sequence):
            output = " ".join([output, "/".join([word, tag])])
        outputs.append(output[1:])
        return outputs

    def getWordsCount(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(1, 1), token_pattern=r"\S+", min_df=self.num_rare_word)
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def predictTagsSequence(self, tokens, model, dv, known_words):
        prev_prev_tag, prev_tag = 'STARTtag', 'STARTtag'
        tags = []
        for i, token in enumerate(tokens[2:len(tokens)-1]):
            rare_word = True if token not in known_words else False
            features = ExtractFeatures().extract(tokens, i+2, prev_prev_tag, prev_tag, rare_word).split()
            features_dict = {}
            for f_v in features:
                f_v = f_v.split('=')
                features_dict[f_v[0]] = f_v[1]
            x = dv.transform([features_dict])
            y = model.predict(x)[0]
            tags.append(y)
            prev_tag = y
            prev_prev_tag = prev_tag
        return tags

    def check(self, outputs):
        with open('ass1-tagger-dev', 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        good, count = 0, 0
        for correct_line, predicted_line in zip(content[:1499], outputs):
            correct_tokens, predicted_tokens = correct_line.split(), predicted_line.split()
            for correct_token, predicted_token in zip(correct_tokens, predicted_tokens):
                if correct_token == predicted_token:
                    good += 1
                count += 1
        print(good/count)

    def runFeaturesTagger(self):
        a = time.time()
        input_file, model_file, feature_map_file, features_output_file = self.readParameters(VALID_PARAMETERS_NUMBER)
        inputs = self.readInputFile(input_file)
        known_words = self.getWordsCount(inputs)
        model = self.loadModel(model_file)
        dv = self.loadFeatureMap(feature_map_file)
        outputs = []
        for i, input in enumerate(inputs):
            print(i)
            if i >= 1499:
                break
            tags = self.predictTagsSequence(input.split(), model, dv, known_words)
            outputs = self.getOutputSequence(input, tags, outputs)
        self.writeOutputsToFile(features_output_file, outputs)
        self.check(outputs)
        print(time.time()-a)


ft = FeaturesTagger()
ft.runFeaturesTagger()