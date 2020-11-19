from sklearn.feature_extraction.text import CountVectorizer
from ExtractFeatures import ExtractFeatures
import sys
import utils
import pickle
VALID_PARAMETERS_NUMBER = 5

class FeaturesTagger:
    def __init__(self):
        self.num_rare_word = 2
        self.my_known = {}
        self.dv = None

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, input_file, model_file, feature_map_file, features_output_file = sys.argv
        return input_file, model_file, feature_map_file, features_output_file

    def readInputFile(self, file_name):
        tokens, content = utils.getOnlyTokens(file_name)
        lines = []
        for line in tokens:
            new_line = "STARTword STARTword STARTword " + line + " ENDword ENDword ENDword"
            lines.append(new_line)
        return lines, content

    def loadFeatureMap(self, file_name):
        data = pickle.load(open(file_name, "rb"))
        self.dv = data[1]
        self.my_known = data[0]

    def loadModel(self, file_name):
        return pickle.load(open(file_name, "rb"))

    def writeOutputsToFile(self, file_name, outputs):
        with open(file_name, 'w') as f:
            for output in outputs:
                f.write("%s\n" % output)

    def getOutputSequence(self, input, tags_sequence, outputs):
        output = ""
        for word, tag in zip(input.split()[3:len(input.split())], tags_sequence):
            output = " ".join([output, "/".join([word, tag])])
        outputs.append(output[1:])
        return outputs

    def getWordsCount(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(1, 1), token_pattern=r"\S+", min_df=self.num_rare_word)
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def predictTagsSequence(self, tokens, model):
        prev_prev_prev_tag, prev_prev_tag, prev_tag = 'STARTtag', 'STARTtag', 'STARTtag'
        tags = []
        for i, token in enumerate(tokens[3:len(tokens)-3]):
            rare_word = True if token not in self.my_known else False
            features = ExtractFeatures().extract(tokens, i+3, prev_prev_prev_tag, prev_prev_tag, prev_tag, rare_word).split()
            features_dict = {}
            for f_v in features:
                f_v = f_v.split('=')
                features_dict[f_v[0]] = f_v[1]
            x = self.dv.transform([features_dict])
            y = model.predict(x)[0]
            tags.append(y)
            prev_prev_prev_tag = prev_prev_tag
            prev_prev_tag = prev_tag
            prev_tag = y
        return tags

    def accuracyEvaluation(self, outputs, golden):
        good, count = 0, 0
        for correct_line, predicted_line in zip(golden, outputs):
            correct_tokens, predicted_tokens = correct_line.split(), predicted_line.split()
            for correct_token, predicted_token in zip(correct_tokens, predicted_tokens):
                # for ner accuracy evaluation
                if correct_token.endswith('/O'):
                    continue
                if correct_token == predicted_token:
                    good += 1
                count += 1
        print(good/count)

    def runFeaturesTagger(self):
        input_file, model_file, feature_map_file, features_output_file = self.readParameters(VALID_PARAMETERS_NUMBER)
        inputs, golden = self.readInputFile(input_file)
        model = self.loadModel(model_file)
        self.loadFeatureMap(feature_map_file)
        outputs = []
        for i, input in enumerate(inputs):
            tags = self.predictTagsSequence(input.split(), model)
            outputs = self.getOutputSequence(input, tags, outputs)
        self.writeOutputsToFile(features_output_file, outputs)
        #self.accuracyEvaluation(outputs, golden)


ft = FeaturesTagger()
ft.runFeaturesTagger()