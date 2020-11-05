import sys
import numpy as np
import math
import itertools
import Language
VALID_PARAMETERS_NUMBER = 6
class HMMTag:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.start_signatures = {}
        self.end_signatures = {}
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

    def getAllTags(self):
        for entry in self.transitions:
            tags = entry.split()
            if len(tags) != 1:
                continue
            if tags[-1] not in self.all_tags:
                self.all_tags[tags[-1]] = 0
            self.all_tags[tags[-1]] += self.transitions[entry]

    def calculateKnownProbabilites(self):
        for t in self.all_tags:
            for w in self.knownWords:
                w_t = " ".join([w, t])
                self.emissions_prob[w_t] = self.emissions.get(w_t, 0) / (self.transitions[t] + sys.float_info.epsilon)
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                for t3 in self.all_tags:
                    epsilon = sys.float_info.epsilon
                    lambdas = [0.6, 0.3, 0.1]
                    first_prob = self.transitions.get(' '.join([t1, t2, t3]), 0)
                    second_prob = self.transitions.get(' '.join([t2, t3]), 0)
                    third_prob = self.transitions.get(t3, 0)
                    self.emissions_prob[" ".join([t1, t2, t3])] = (lambdas[0] * first_prob / (second_prob + epsilon)) + \
                           (lambdas[1] * second_prob / (third_prob + epsilon)) + \
                           (lambdas[2] * third_prob / sum(self.all_tags.values()))
        for pattern in self.end_signatures:
            tag = pattern.split()[-1]
            self.patterns_prob[pattern] = self.end_signatures.get(pattern, 0) / self.transitions[tag]
        for pattern in self.start_signatures:
            tag = pattern.split()[-1]
            self.patterns_prob[pattern] = self.start_signatures.get(pattern, 0) / self.transitions[tag]

    def getQ(self, prev_prev_t, prev_t, t):
        return self.emissions_prob[" ".join([prev_prev_t, prev_t, t])]
        '''
        epsilon = sys.float_info.epsilon
        lambdas = [0.6, 0.3, 0.1]
        first_prob = self.transitions.get(' '.join([prev_prev_t, prev_t, t]), 0)
        second_prob = self.transitions.get(' '.join([prev_t, t]), 0)
        third_prob = self.transitions.get(t, 0)
        return (lambdas[0]*first_prob / (second_prob + epsilon)) +\
               (lambdas[1]*second_prob / (third_prob + epsilon)) + \
               (lambdas[2]*third_prob / sum(self.all_tags.values()))
        '''


    def getE(self, w, t):
        # since tag_list is taken from training, transitions[t] is never 0
        if w.startswith('_UNK_'):
            w = w.split('_UNK_')[-1]
            w_t = " ".join([w, t])
            if w_t in self.patterns_prob:
                return self.patterns_prob[w_t]
            return sys.float_info.epsilon
            #return max(self.getUnknownScore(w,t), sys.float_info.epsilon)
        w_t = " ".join([w, t])
        return max(self.emissions_prob[w_t], sys.float_info.epsilon)
        #prob_w_t = self.emissions.get(w_t, 0) / self.transitions[t]
        #return max(prob_w_t, sys.float_info.epsilon)

    def getUnknownScore(self, w, t):
        w = w.split('_UNK_')[-1]
        w_t = " ".join([w, t])
        if w_t in self.patterns_prob:
            return self.patterns_prob[w_t]
        '''
        if Language.startWithCapitalLower(w):
            entry = " ".join(['Aa', t])
            return self.start_signatures.get(entry, 0) / self.transitions[t]
        '''
        return sys.float_info.epsilon


    # start start 'all tags'
    # start 'all tags' 'all tags'
    def possibleTags(self, i):
        if i > 1:
            return self.all_tags
        return ['START']

    def ungrammaticalTagsSequences(self):
        self.close_group = Language.getCloseClassPOS(self.emissions)
        for tag in self.all_tags:
            for prev_tag in self.all_tags:
                if not tag.isalpha() and not prev_tag.isalpha():
                    self.unallowedTagsSequences[(prev_tag, tag)] = 0
                if prev_tag == tag and tag in self.close_group:
                    self.unallowedTagsSequences[(prev_tag, tag)] = 0

    def viterbiAlgorithm(self, sequence):
        words = sequence.split()
        N = len(words)-1
        print(N)
        P = {(0, 'START', 'START'): 0}
        T = {}

        for i in range(1, N+1):
            word = words[i]
            best_prob = float('-inf')
            best_tag = None
            if word not in self.knownWords:
                word = '_UNK_' + word
            for tag in self.all_tags:
                for prev_tag in self.possibleTags(i):
                    if (prev_tag, tag) in self.unallowedTagsSequences:
                        P[(i, prev_tag, tag)] = float('-inf')
                        T[(i, prev_tag, tag)] = None
                        continue
                    for prev_prev_tag in self.possibleTags(i - 1):
                        this_prob = P[(i-1,prev_prev_tag,prev_tag)]\
                                    + math.log(self.getE(word, tag))\
                                    + math.log(self.getQ(prev_prev_tag, prev_tag, tag))
                        if this_prob > best_prob:
                            best_prob = this_prob
                            best_tag = tag
                    P[(i, prev_tag, tag)] = best_prob
                    T[(i, prev_tag, tag)] = best_tag

        output = (N + 1) * [None]
        last_positions = [N]
        for pos in last_positions:
            chosen_prob = float('-inf')
            chosen_tag = None
            for tup in (x for x in P if x[0] == pos):
                prob = P[tup]
                if prob > chosen_prob:
                    chosen_prob = prob
                    chosen_tag = T[tup]
            output[pos] = chosen_tag

        for i in range(N-1, 0, -1):
            chosen_prob = float('-inf')
            chosen_tag = None
            for tup in (x for x in P if x[0] == i and x[2] == output[i+1]):
                prob = P[tup]
                if prob > chosen_prob:
                    chosen_prob = prob
                    chosen_tag = T[tup]
            output[i] = chosen_tag

    def runTagger(self):
        file_name, q_file, e_file, greedy_output_file, extra_file = hmmTagger.readParameters(VALID_PARAMETERS_NUMBER)
        self.emissions = hmmTagger.readEstimates(e_file)
        self.transitions = hmmTagger.readEstimates(q_file)
        self.getSignaturesAndUnknown()
        self.ungrammaticalTagsSequences()
        self.getAllTags()
        self.calculateKnownProbabilites()
        self.inputs = self.readInputFile(file_name)
        for input in self.inputs:
            tags_sequence = self.viterbiAlgorithm(input)
            #self.getOutputSequence(input, tags_sequence)
        #self.writeOutputsToFile(greedy_output_file)
        #self.check()

hmmTagger = HMMTag()
hmmTagger.runTagger()