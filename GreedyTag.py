import sys
import numpy as np
from collections import defaultdict
import Language
VALID_PARAMETERS_NUMBER = 6

class GreedyTag:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.start_signatures = {}
        self.end_signatures = {}
        self.knownWords = {}
        self.all_tags = {}
        self.inputs = []
        self.outputs = []

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, file_name, q_file, e_file, greedy_output_file, extra_file = sys.argv
        return file_name, q_file, e_file, greedy_output_file, extra_file

    def readInputFile(self, file_name):
        with open(file_name, 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        lines = []
        for line in content:
            new_line = "START START " + line + " END END"
            lines.append(new_line)
        return lines

    def readEstimates(self, file_name):
        _dict = {}
        with open(file_name, 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        for line in content:
            k, v = line.split('\t')
            _dict[k] = int(v)
        return _dict

    def getSignaturesAndUnknown(self):
        for entry in self.emissions:
            if not entry.startswith('^'):
                word = entry.split()[0]
                if word not in self.knownWords:
                    self.knownWords[word] = 0
            key = entry[1:]
            sign, signature_tag = key[0], key[1:]
            if sign == '^':
                self.start_signatures[signature_tag] = self.emissions[entry]
            if sign == '~':
                self.end_signatures[signature_tag] = self.emissions[entry]

    def writeOutputsToFile(self, file_name):
        with open(file_name, 'w') as f:
            for output in self.outputs:
                f.write("%s\n" % output)

    def getOutputSequence(self, input, tags_sequence):
        output = ""
        for word, tag in zip(input.split()[2:len(input.split())-1], tags_sequence):
            output = " ".join([output, "/".join([word, tag])])
        self.outputs.append(output[1:])

    def getAllTags(self):
        for entry in self.transitions:
            tags = entry.split()
            if len(tags) != 1:
                continue
            if tags[-1] not in self.all_tags:
                self.all_tags[tags[-1]] = 0
            self.all_tags[tags[-1]] += self.transitions[entry]

    def getQ(self, prev_prev_t, prev_t, t):
        epsilon = sys.float_info.epsilon
        lambdas = [0.6, 0.3, 0.1]
        first_prob = self.transitions.get(' '.join([prev_prev_t, prev_t, t]), 0)
        second_prob = self.transitions.get(' '.join([prev_t, t]), 0)
        third_prob = self.transitions.get(t, 0)
        return (lambdas[0]*first_prob / (second_prob + epsilon)) +\
               (lambdas[1]*second_prob / (third_prob + epsilon)) + \
               (lambdas[2]*third_prob / sum(self.all_tags.values()))

    def getE(self, w, t):
        # since tag_list is taken from training, transitions[t] is never 0
        if w.startswith('_UNK_'):
            return self.getUnknownScore(w,t)
        w_t = " ".join([w, t])
        prob_w_t = self.emissions.get(w_t, 0) / self.transitions[t]
        if prob_w_t > 0:
            return prob_w_t
        return sys.float_info.epsilon

    def getUnknownScore(self, w, t):
        w = w.split('_UNK_')[-1]
        w_t = " ".join([w, t])
        for pattern in self.end_signatures:
            if w_t.endswith(pattern):
                return self.end_signatures.get(pattern, 0) / self.transitions[t]
        for pattern in self.start_signatures:
            only_pattern, only_tag = pattern.split()[0], pattern.split()[-1]
            if w_t.startswith(only_pattern) and w_t.endswith(only_tag):
                return self.start_signatures.get(pattern, 0) / self.transitions[t]
        if Language.startWithCapitalLower(w):
            entry = " ".join(['Aa', t])
            return self.start_signatures.get(entry, 0) / self.transitions[t]
        return sys.float_info.epsilon

    def greedyAlgorithm(self, sequence):
        tags_sequence = []
        words = sequence.split()
        prev_tag, prev_prev_tag = 'START', 'START'
        for i, word in enumerate(words):
            prob_i = 0
            tag_i = None
            if i <= 1 or i >= len(words)-2:
                continue
            if word not in self.knownWords:
                word = '_UNK_' + word
            for tag in self.all_tags:
                current_prob = self.getE(word, tag)*self.getQ(prev_prev_tag, prev_tag, tag)
                if current_prob > prob_i:
                    prob_i = current_prob
                    tag_i = tag
                # assign some small prob, than handle word signatures
            prev_prev_tag = prev_tag
            prev_tag = tag_i
            tags_sequence.append(tag_i)
        return tags_sequence

    def check(self):
        with open('ass1-tagger-dev', 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        good, count = 0, 0
        for correct_line, predicted_line in zip(content, self.outputs):
            correct_tokens, predicted_tokens = correct_line.split(), predicted_line.split()
            for correct_token, predicted_token in zip(correct_tokens, predicted_tokens):
                if correct_token == predicted_token:
                    good += 1
                count += 1
        print(good/count)

    def runTagger(self):
        file_name, q_file, e_file, greedy_output_file, extra_file = greedyTagger.readParameters(VALID_PARAMETERS_NUMBER)
        self.emissions = greedyTagger.readEstimates(e_file)
        self.transitions = greedyTagger.readEstimates(q_file)
        self.getSignaturesAndUnknown()
        self.getAllTags()
        self.inputs = self.readInputFile(file_name)
        for input in self.inputs:
            tags_sequence = self.greedyAlgorithm(input)
            self.getOutputSequence(input, tags_sequence)
        self.writeOutputsToFile(greedy_output_file)
        self.check()


greedyTagger = GreedyTag()
greedyTagger.runTagger()

