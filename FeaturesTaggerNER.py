from ExtractFeaturesNER import ExtractFeaturesNER
import sys
import time
import utils
import pickle
VALID_PARAMETERS_NUMBER = 5

class FeaturesTaggerNER:
    def __init__(self):
        self.num_rare_word = 2
        self.all_tags = {}
        self.dv = None
        self.model = None
        self.my_known = {}

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
            new_line = "STARTword STARTword " + line + " ENDword ENDword"
            lines.append(new_line)
        return lines, content

    def loadFeatureMap(self, file_name):
        data = pickle.load(open(file_name, "rb"))
        self.dv = data[1]
        self.my_known = data[0]

    def loadModel(self, file_name):
        self.model = pickle.load(open(file_name, "rb"))

    def getAllClasses(self):
        classes = self.model.classes_
        for c in classes:
            self.all_tags[c] = 1
        self.all_tags['STARTtag'] = 1

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


    def initializeViterbiDicts(self):
        P = {(0, 'STARTtag', 'STARTtag'): 0}
        T = {}
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                if t1 == 'STARTtag' or t2 == 'STARTtag':
                    continue
                P[(0, t1, t2)] = float('-inf')
        return P, T

    def getScores(self, needed_param):
        (tokens, token, i, prev_prev_tag, prev_tag) = needed_param
        rare_word = True if token not in self.my_known else False
        features = ExtractFeaturesNER().extract(tokens, i, prev_prev_tag, prev_tag, rare_word).split()
        features_dict = {}
        for f_v in features:
            f_v = f_v.split('=')
            features_dict[f_v[0]] = f_v[1]
        x = self.dv.transform([features_dict])
        probs = self.model.predict_proba(x)[0]
        classes = self.model.classes_
        return dict(zip(classes, probs))

    def possibleTags(self, i, words):
        if i <= 0:
            return ['STARTtag']
        if len(words) >= 3 and words[i-1] in self.my_known:
            return self.my_known[words[i-1]]
        d = self.all_tags.copy()
        d.pop("STARTtag")
        return d

    def predictTagsSequence(self, tokens):
        words = tokens[2:len(tokens)-2]
        N = len(tokens[2:len(tokens)-2])
        P, T = self.initializeViterbiDicts()
        for i in range(1, N+1):
            tags = self.possibleTags(i, words)
            for tag in tags:
                prev_tags = self.possibleTags(i-1, words)
                for prev_tag in prev_tags:
                    best_prob = float('-inf')
                    best_tag = None
                    prev_prev_tags = self.possibleTags(i-2, words)
                    for prev_prev_tag in prev_prev_tags:
                        needed_param = (tokens, words[i-1], i+1, prev_prev_tag, prev_tag)
                        this_prob = P.get((i-1, prev_prev_tag,prev_tag), -10000) + \
                                    self.getScores(needed_param)[tag]
                        if this_prob > best_prob:
                            best_tag = prev_prev_tag
                            best_prob = this_prob
                    P[(i, prev_tag, tag)] = best_prob
                    T[(i, prev_tag, tag)] = best_tag

        output = []
        last_tag, one_before_last = None, None
        best_prob = float('-inf')
        prev_tags = self.possibleTags(N-1, words)
        prev_prev_tags = self.possibleTags(N-2, words)
        for prev_tag in prev_tags:
            for prev_prev_tag in prev_prev_tags:
                needed_param = (tokens, words[N-1], N + 2, prev_prev_tag, prev_tag)
                last_tag_dict = self.getScores(needed_param)
                for tag in last_tag_dict:
                    this_prob = P.get((N, prev_tag, tag), -10000) + last_tag_dict[tag]
                    if this_prob > best_prob:
                        best_prob = this_prob
                        last_tag = tag
                        one_before_last = prev_tag

        output.append(last_tag)
        if N > 1:
            output.append(one_before_last)
        index = 0
        for i in range(N-2, 0, -1):
            output.append(T[(i+2, output[index+1], output[index])])
            index += 1
        output.reverse()
        return output


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
        a = time.time()
        input_file, model_file, feature_map_file, features_output_file = self.readParameters(VALID_PARAMETERS_NUMBER)
        inputs, golden = self.readInputFile(input_file)
        self.loadModel(model_file)
        self.getAllClasses()
        self.loadFeatureMap(feature_map_file)
        outputs = []
        for i, input in enumerate(inputs):
            print(i)
            tags = self.predictTagsSequence(input.split())
            outputs = self.getOutputSequence(input, tags, outputs)
        self.writeOutputsToFile(features_output_file, outputs)
        #self.accuracyEvaluation(outputs, golden)
        print(time.time()-a)


ft = FeaturesTaggerNER()
ft.runFeaturesTagger()

