import sys
import math
import Language
import utils
VALID_PARAMETERS_NUMBER = 6

class HMMTag:
    def __init__(self, lambdas):
        self.emissions = {}
        self.transitions = {}
        self.signatures = {}
        self.emissions_prob = {}
        self.transitions_prob = {}
        self.known_words = {}
        self.all_tags = {}
        self.inputs = []
        self.outputs = []
        self.lambdas = lambdas

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, file_name, q_file, e_file, greedy_output_file, extra_file = sys.argv
        return file_name, q_file, e_file, greedy_output_file, extra_file

    def readInputFile(self, file_name):
        tokens, content = utils.getOnlyTokens(file_name)
        return tokens, content

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
                ent = entry.split()
                word, tag = ent[0], ent[-1]
                if word not in self.known_words:
                    self.known_words[word] = []
                self.known_words[word].append(tag)
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
            for w in self.known_words:
                w_t = (w, t)
                self.emissions_prob[w_t] = self.emissions.get(" ".join([w, t]), 0) / (self.transitions[t])
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                for t3 in self.all_tags:
                    epsilon = sys.float_info.epsilon
                    first_prob = self.transitions.get(' '.join([t1, t2, t3]), 0)
                    second_prob = self.transitions.get(' '.join([t2, t3]), 0)
                    third_prob = self.transitions.get(t3, 0)
                    self.transitions_prob[(t1, t2, t3)] = (self.lambdas[0] * first_prob / (second_prob + epsilon)) + \
                           (self.lambdas[1] * second_prob / (third_prob + epsilon)) + \
                           (self.lambdas[2] * third_prob / sum(self.all_tags.values()))
        for pattern in self.signatures:
            pat = pattern.split()
            self.emissions_prob[(pat[0], pat[1])] = self.signatures.get(pattern, 0) / self.transitions[pat[1]]

    def getQ(self, prev_prev_t, prev_t, t):
        return max(self.transitions_prob[(prev_prev_t, prev_t, t)], sys.float_info.epsilon)

    def getE(self, w, t):
        # since tag_list is taken from training, transitions[t] is never 0
        return max(self.emissions_prob.get((w, t), 0), sys.float_info.epsilon)

    # start start 'all tags'
    # start 'all tags' 'all tags'
    def possibleTags(self, i, words):
        if i <= 0:
            return ['STARTtag']
        if len(words) >= 3 and words[i-1] in self.known_words:
            return self.known_words[words[i-1]]
        d = self.all_tags.copy()
        d.pop("STARTtag")
        return d


    def initializeViterbiDicts(self):
        P = {(0, 'STARTtag', 'STARTtag'): 0}
        T = {}
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                if t1 == 'STARTtag' or t2 == 'STARTtag':
                    continue
                P[(0, t1, t2)] = float('-inf')
        return P, T

    def viterbiAlgorithm(self, words):
        N = len(words)
        P, T = self.initializeViterbiDicts()
        for i in range(1, N+1):
            word = words[i-1]
            tags = self.possibleTags(i, words)
            for tag in tags:
                prev_tags = self.possibleTags(i-1, words)
                for prev_tag in prev_tags:
                    best_prob = float('-inf')
                    best_tag = None
                    prev_prev_tags = self.possibleTags(i-2, words)
                    for prev_prev_tag in prev_prev_tags:
                        this_prob = P.get((i-1,prev_prev_tag,prev_tag), -10000)\
                                    + math.log(self.getE(word, tag))\
                                    + math.log(self.getQ(prev_prev_tag, prev_tag, tag))
                        if this_prob > best_prob:
                            best_prob = this_prob
                            best_tag = prev_prev_tag
                    P[(i, prev_tag, tag)] = best_prob
                    T[(i, prev_tag, tag)] = best_tag
        output = []
        last_tag, one_before_last = None, None
        best_prob = float('-inf')
        tags = self.possibleTags(N, words)
        prev_tags = self.possibleTags(N-1, words)
        for tag in tags:
            for prev_tag in prev_tags:
                this_prob = P.get((N, prev_tag, tag), -10000) + math.log(self.getQ(prev_tag, tag, 'ENDtag'))
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

    def accuracyEvaluation(self, golden):
        good, count = 0, 0
        for correct_line, predicted_line in zip(golden, self.outputs):
            correct_tokens, predicted_tokens = correct_line.split(), predicted_line.split()
            for correct_token, predicted_token in zip(correct_tokens, predicted_tokens):
                # for ner accuracy evaluation
                if correct_token.endswith('/O'):
                    continue
                if correct_token == predicted_token:
                    good += 1
                count += 1
        print(good/count)

    def replaceWithSignatures(self, words):
        sequence = []
        for i, word in enumerate(words):
            if word not in self.known_words:
                sym = Language.replaceRareWords(word, i)
                sequence.append(sym)
            else:
                sequence.append(word)
        return sequence

    def runTagger(self):
        file_name, q_file, e_file, greedy_output_file, extra_file = hmmTagger.readParameters(VALID_PARAMETERS_NUMBER)
        self.emissions = hmmTagger.readEstimates(e_file)
        self.transitions = hmmTagger.readEstimates(q_file)
        self.getSignaturesAndUnknown()
        self.getAllTags()
        self.calculateKnownProbabilities()
        self.inputs, golden = self.readInputFile(file_name)
        for i , input in enumerate(self.inputs):
            if len(input) == 0:
                continue
            words = self.replaceWithSignatures(input.split())
            tags_sequence = self.viterbiAlgorithm(words)
            self.getOutputSequence(input, tags_sequence)
        self.writeOutputsToFile(greedy_output_file)
        #self.accuracyEvaluation(golden)


lambdas = [[0.9, 0.09, 0.01]]
for lam_values in lambdas:
    hmmTagger = HMMTag(lam_values)
    hmmTagger.runTagger()