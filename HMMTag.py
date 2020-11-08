import sys
import numpy as np
import math
import itertools
import Language
import time
VALID_PARAMETERS_NUMBER = 6
class HMMTag:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.signatures = {}
        self.emissions_prob = {}
        self.transitions_prob = {}
        self.patterns_prob = {}
        self.unallowedTagsSequences = {}
        self.knownWords = {}
        self.all_tags = {}
        self.inputs = []
        self.outputs = []
        self.close_group = {}

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
            #new_line = "START START " + line + " END END"
            new_line = line
            lines.append(new_line)
        return lines

    def writeOutputsToFile(self, file_name):
        with open(file_name, 'w') as f:
            for output in self.outputs:
                f.write("%s\n" % output)

    def getOutputSequence(self, input, tags_sequence):
        output = ""
        for word, tag in zip(input.split(), tags_sequence):
            output = " ".join([output, "/".join([word, tag])])
        self.outputs.append(output[1:])

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
                word, tag = entry.split()[0], entry.split()[-1]
                if word not in self.knownWords:
                    self.knownWords[word] = []
                self.knownWords[word].append(tag)
            else:
                self.signatures[entry[1:]] = self.emissions[entry]

    def getAllTags(self):
        for entry in self.transitions:
            tags = entry.split()
            if len(tags) != 1:
                continue
            if tags[-1] not in self.all_tags:
                self.all_tags[tags[-1]] = 0
            self.all_tags[tags[-1]] += self.transitions[entry]

    def calculateKnownProbabilities(self):
        for t in self.all_tags:
            for w in self.knownWords:
                w_t = (w, t)
                self.emissions_prob[w_t] = self.emissions.get(" ".join([w, t]), 0) / (self.transitions[t])
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                for t3 in self.all_tags:
                    epsilon = sys.float_info.epsilon
                    lambdas = [0.6, 0.3, 0.1]
                    first_prob = self.transitions.get(' '.join([t1, t2, t3]), 0)
                    second_prob = self.transitions.get(' '.join([t2, t3]), 0)
                    third_prob = self.transitions.get(t3, 0)
                    self.transitions_prob[(t1, t2, t3)] = (lambdas[0] * first_prob / (second_prob + epsilon)) + \
                           (lambdas[1] * second_prob / (third_prob + epsilon)) + \
                           (lambdas[2] * third_prob / sum(self.all_tags.values()))
        for pattern in self.signatures:
            pat = pattern.split()
            self.emissions_prob[(pat[0], pat[1])] = self.signatures.get(pattern, 0) / self.transitions[pat[1]]
            #self.patterns_prob[pattern] = self.signatures.get(pattern, 0) / self.transitions[tag]

    def getQ(self, prev_prev_t, prev_t, t):
        return self.transitions_prob[(prev_prev_t, prev_t, t)]

    def getE(self, w, t):
        # since tag_list is taken from training, transitions[t] is never 0
        return max(self.emissions_prob.get((w, t), 0), sys.float_info.epsilon)

    # start start 'all tags'
    # start 'all tags' 'all tags'
    def possibleTags(self, i, word='_@@_'):
        if i <= 0:
            return ['START']
        if word in self.knownWords:
            return self.knownWords[word]
        d = self.all_tags.copy()
        d.pop("START")
        return d

    def ungrammaticalTagsSequences(self):
        self.close_group = Language.getCloseClassPOS(self.emissions)
        tags = self.all_tags.copy()
        tags.pop('START')
        for tag in tags:
            for prev_tag in tags:
                if prev_tag ==tag and tag in self.close_group:
                    self.unallowedTagsSequences[(prev_tag, tag)] = 1

    def initializeViterbiDicts(self):
        P = {(0, 'START', 'START'): 0}
        T = {}
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                if t1 == 'START' or t2 == 'START':
                    continue
                P[(0, t1, t2)] = float('-inf')
        return P, T

    def viterbiAlgorithm(self, words):
        N = len(words)
        P, T = self.initializeViterbiDicts()
        for i in range(1, N+1):
            word = words[i-1]
            tags = self.possibleTags(i, word)
            for tag in tags:
                prev_tags = self.possibleTags(i-1)
                for prev_tag in prev_tags:
                    best_prob = float('-inf')
                    best_tag = None
                    prev_prev_tags = self.possibleTags(i-2)
                    for prev_prev_tag in prev_prev_tags:
                        this_prob = P.get((i-1,prev_prev_tag,prev_tag), float('-inf'))\
                                    + math.log(self.getE(word, tag))\
                                    + math.log(self.getQ(prev_prev_tag, prev_tag, tag))
                        if this_prob > best_prob:
                            best_prob = this_prob
                            best_tag = prev_prev_tag
                    P[(i, prev_tag, tag)] = best_prob
                    T[(i, prev_tag, tag)] = best_tag
        output = (N) * [None]
        last_tag, one_before_last = None, None
        best_prob = float('-inf')
        tags = self.possibleTags(N)
        prev_tags = self.possibleTags(N-1)
        for tag in tags:
            for prev_tag in prev_tags:
                this_prob = P.get((N, prev_tag, tag), float('-inf')) + math.log(self.getQ(prev_tag, tag, 'END'))
                if this_prob > best_prob:
                    best_prob = this_prob
                    last_tag = tag
                    one_before_last = prev_tag
        output[N-1] = last_tag
        output[N-2] = one_before_last
        for i in range(N-2, 0, -1):
            output[i-1] = T[(i+2, output[i], output[i+1])]
        return output

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

    def replaceWithSignatures(self, words):
        sequence = []
        for word in words:
            if word not in self.knownWords:
                sequence.append(Language.replaceRareWords(word))
            else:
                sequence.append(word)
        return sequence

    def runTagger(self):
        a = time.time()
        file_name, q_file, e_file, greedy_output_file, extra_file = hmmTagger.readParameters(VALID_PARAMETERS_NUMBER)
        self.emissions = hmmTagger.readEstimates(e_file)
        self.transitions = hmmTagger.readEstimates(q_file)
        self.getSignaturesAndUnknown()
        self.getAllTags()
        self.ungrammaticalTagsSequences()
        self.calculateKnownProbabilities()
        self.inputs = self.readInputFile(file_name)
        for i , input in enumerate(self.inputs):
            if i%100 == 0:
                print(time.time() - a)
            words = input.split()
            sequence = self.replaceWithSignatures(words)
            tags_sequence = self.viterbiAlgorithm(sequence)
            self.getOutputSequence(input, tags_sequence)
        self.writeOutputsToFile(greedy_output_file)
        print(time.time()-a)
        self.check()

hmmTagger = HMMTag()
hmmTagger.runTagger()