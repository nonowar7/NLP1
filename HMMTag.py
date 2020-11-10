import sys
import math
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
        self.rare_words = {}
        self.knownWords = {}
        self.all_tags = {}
        self.inputs = []
        self.outputs = []
        self.close_group = {}
        self.count_replace = {}
        self.training_close_group_words = {"in":0, "to":0, "for":0, "and":0, "or":0,
                                           "but":0, "the":0, "a":0, "an":0, "of":0, ".":0, ",":0,
                                           "is":0, "was":0, "are":0, "were":0}

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, file_name, q_file, e_file, greedy_output_file, extra_file = sys.argv
        return file_name, q_file, e_file, greedy_output_file, extra_file

    def readInputFile(self, file_name):
        with open(file_name, 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        return content

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

    def calculateKnownProbabilities(self, lambdas):
        for t in self.all_tags:
            for w in self.knownWords:
                w_t = (w, t)
                self.emissions_prob[w_t] = self.emissions.get(" ".join([w, t]), 0) / (self.transitions[t])
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                for t3 in self.all_tags:
                    epsilon = sys.float_info.epsilon
                    first_prob = self.transitions.get(' '.join([t1, t2, t3]), 0)
                    second_prob = self.transitions.get(' '.join([t2, t3]), 0)
                    third_prob = self.transitions.get(t3, 0)
                    self.transitions_prob[(t1, t2, t3)] = (lambdas[0] * first_prob / (second_prob + epsilon)) + \
                           (lambdas[1] * second_prob / (third_prob + epsilon)) + \
                           (lambdas[2] * third_prob / sum(self.all_tags.values()))
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
        if len(words) >= 3 and words[i-1] in self.knownWords:
            return self.knownWords[words[i-1]]
        if 'RARE_rare' == words[i-1]:
            return ['NNP', 'NN', 'JJ', 'VB']
        d = self.all_tags.copy()
        d.pop("STARTtag")
        return d

    def ungrammaticalTagsSequences(self):
        self.close_group = Language.getCloseClassPOS(self.emissions)
        tags = self.all_tags.copy()
        tags.pop('STARTtag')
        for tag in tags:
            for prev_tag in tags:
                if prev_tag == tag and tag in self.close_group:
                    self.unallowedTagsSequences[(prev_tag, tag)] = 1

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
        output = (N) * [None]
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
        output[N-1] = last_tag
        output[N-2] = one_before_last
        for i in range(N-2, 0, -1):
            output[i-1] = T[(i+2, output[i], output[i+1])]
        return output

    def check(self):
        mistakes = {}
        corrects = {}
        with open('ass1-tagger-dev', 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        good, count = 0, 0
        check = {}
        for correct_line, predicted_line in zip(content, self.outputs):
            correct_tokens, predicted_tokens = correct_line.split(), predicted_line.split()
            for correct_token, predicted_token in zip(correct_tokens, predicted_tokens):
                for a in self.rare_words:
                    if correct_token.split('/')[0] == a:
                        if correct_token.split('/')[-1] not in check:
                            check[correct_token.split('/')[-1]] = []
                        check[correct_token.split('/')[-1]].append(a)
                if correct_token == predicted_token:
                    TOK = correct_token
                    if TOK not in corrects:
                        corrects[TOK] = 0
                    corrects[TOK] += 1
                    good += 1
                else:
                    TOK = (correct_token, predicted_token)
                    if TOK not in mistakes:
                        mistakes[TOK] = 0
                    mistakes[TOK] += 1
                count += 1
        '''
        for mistake in mistakes:
            correct, mis = mistake
            if mistakes[mistake] > 5:
                print(mistake)
                print(mistakes[mistake])
                print(corrects.get(correct,0))
        '''
        print(good/count)
        '''
        for x in check:
            print(x)
            print(len(check[x]))
        '''

    def replaceWithSignatures(self, words):
        sequence = []
        for i, word in enumerate(words):
            if word not in self.knownWords:
                sym = Language.replaceRareWords(word)
                if sym not in self.count_replace:
                    self.count_replace[sym] = 0
                self.count_replace[sym] += 1
                sequence.append(sym)
                if sym == 'RARE_rare':
                    if word not in self.rare_words:
                        self.rare_words[word] = 0
                    self.rare_words[word] += 1
            else:
                sequence.append(word)
        return sequence

    def runTagger(self):
        lambdas = [[0.8, 0.19, 0.01]]
        lambdas = [[0.95, 0.04, 0.01]]
        lambdas = [[0.97, 0.02, 0.01]]
        lambdas = [[0.9, 0.09, 0.01]]
        for _lambdas in lambdas:
            a = time.time()
            file_name, q_file, e_file, greedy_output_file, extra_file = hmmTagger.readParameters(VALID_PARAMETERS_NUMBER)
            self.emissions = hmmTagger.readEstimates(e_file)
            self.transitions = hmmTagger.readEstimates(q_file)
            self.getSignaturesAndUnknown()
            self.getAllTags()
            self.ungrammaticalTagsSequences()
            self.calculateKnownProbabilities(_lambdas)
            self.inputs = self.readInputFile(file_name)
            for i , input in enumerate(self.inputs):
                if len(input) == 0:
                    continue
                words = self.replaceWithSignatures(input.split())
                tags_sequence = self.viterbiAlgorithm(words)
                self.getOutputSequence(input, tags_sequence)
            self.writeOutputsToFile(greedy_output_file)
            print(time.time()-a)
            print(_lambdas)
            self.check()
            print(self.count_replace)

hmmTagger = HMMTag()
hmmTagger.runTagger()