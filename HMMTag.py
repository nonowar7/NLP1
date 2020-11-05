import sys
import numpy as np
VALID_PARAMETERS_NUMBER = 6
class HMMTag:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.prob_q = {}
        self.prob_e = {}
        self.start_signatures = {}
        self.end_signatures = {}
        self.all_tags = []
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
            new_line = "START " + line
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

    def getAllTags(self):
        for entry in self.emissions:
            tag = entry.split()[-1]
            if tag in self.all_tags:
                continue
            self.all_tags.append(tag)

    def getQ(self, t, prev_prev_t, prev_t):
        lambdas = [0.9, 0.09, 0.01]
        seq = [' '.join([prev_prev_t, prev_t, t]), ' '.join([prev_t, t]), t]
        vals = [sys.float_info.epsilon, sys.float_info.epsilon, sys.float_info.epsilon]
        try:
            vals[0] = lambdas[0] * self.transitions[seq[0]] / self.transitions[seq[1]]
        except:
            pass
        try:
            vals[1] = lambdas[1]*self.transitions[seq[1]] / self.transitions[seq[2]]
        except:
            pass
        try:
            vals[2] = lambdas[2]*self.transitions[seq[2]] / sum(self.transitions.values())
        except:
            pass
        return sum(vals)

    def getE(self, w, t):
        try:
            return self.emissions[" ".join([w, t])] / self.transitions[t]
        except:
            return sys.float_info.epsilon

    # start start 'all tags'
    # start 'all tags' 'all tags'
    def possibleTags(self, i):
        if i <= 1:
            return ['START']
        return self.all_tags

    def viterbiAlgorithm(self, sequence):
        N = len(sequence)-1
        print(N)
        P = {(0, 'START', 'START'): 1}
        T = {}

        for i in range(1, N+1):
            print(i)
            best_prob = float('-inf')
            best_tag = 'NN'
            for tag in self.all_tags:
                for prev_tag in self.possibleTags(i):
                    for prev_prev_tag in self.possibleTags(i - 1):
                            this_prob = P[(i-1,prev_prev_tag,prev_tag)]\
                                        + np.log(self.getE(sequence[i], tag))\
                                        + np.log(self.getQ(tag, prev_prev_tag, prev_tag))
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
        self.getAllTags()
        self.inputs = self.readInputFile(file_name)
        for input in self.inputs:
            tags_sequence = self.viterbiAlgorithm(input.split())
            #self.getOutputSequence(input, tags_sequence)
        #self.writeOutputsToFile(greedy_output_file)
        #self.check()

hmmTagger = HMMTag()
hmmTagger.runTagger()