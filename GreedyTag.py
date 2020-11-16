import sys
import Language
import time
import utils
import math
VALID_PARAMETERS_NUMBER = 6

class GreedyTag:
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
        self.tag_for_rare = []

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, file_name, q_file, e_file, greedy_output_file, extra_file = sys.argv
        return file_name, q_file, e_file, greedy_output_file, extra_file

    def readInputFile(self, file_name):
        tokens, content = utils.getOnlyTokens(file_name)
        lines = []
        for line in tokens:
            new_line = "STARTword STARTword " + line + " ENDword ENDword"
            lines.append(new_line)
        return lines, content

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
        return max(self.emissions_prob.get((w, t), 0), sys.float_info.epsilon)
        # since tag_list is taken from training, transitions[t] is never 0


    def possibleTags(self, word):
        if word in self.known_words:
            return self.known_words[word]
        if word == 'RARE_rare':
            return self.tag_for_rare
        d = self.all_tags.copy()
        d.pop("STARTtag")
        d.pop("ENDtag")
        return d

    def greedyAlgorithm(self, words):
        tags_sequence = []
        prev_tag, prev_prev_tag = 'STARTtag', 'STARTtag'
        for i, word in enumerate(words):
            prob_i = float('-inf')
            tag_i = None
            if i <= 1 or i >= len(words)-2:
                continue
            tags = self.possibleTags(word)
            for tag in tags:
                current_prob = math.log(self.getE(word, tag)) + \
                               math.log(self.getQ(prev_prev_tag, prev_tag, tag))
                if current_prob > prob_i:
                    prob_i = current_prob
                    tag_i = tag
            prev_prev_tag = prev_tag
            prev_tag = tag_i
            tags_sequence.append(tag_i)
        return tags_sequence

    def accuracyEvaluation(self, golden):
        good, count = 0, 0
        for correct_line, predicted_line in zip(golden, self.outputs):
            correct_tokens, predicted_tokens = correct_line.split(), predicted_line.split()
            for correct_token, predicted_token in zip(correct_tokens, predicted_tokens):
                if correct_token == predicted_token:
                    good += 1
                count += 1
        print(good/count)

    def replaceWithSignatures(self, words, rare_word_counter):
        sequence = []
        for i, word in enumerate(words):
            if word not in self.known_words:
                sym = Language.replaceRareWords(word, i-2)
                sequence.append(sym)
                if sym not in rare_word_counter:
                    rare_word_counter[sym] = 0
                rare_word_counter[sym] += 1
            else:
                sequence.append(word)
        return sequence


    def getMostCommonTagForRare(self):
        for signature in self.signatures:
            entry = signature.split()
            if entry[0] != 'RARE_rare':
                continue
            self.tag_for_rare.append(entry[1])


    def runTagger(self):
        rare_word_counter = {}
        a = time.time()
        file_name, q_file, e_file, greedy_output_file, extra_file = greedyTagger.readParameters(VALID_PARAMETERS_NUMBER)
        self.emissions = greedyTagger.readEstimates(e_file)
        self.transitions = greedyTagger.readEstimates(q_file)
        self.getSignaturesAndUnknown()
        self.getMostCommonTagForRare()
        self.getAllTags()
        self.calculateKnownProbabilities()
        self.inputs, golden = self.readInputFile(file_name)
        for i, input in enumerate(self.inputs):
            if len(input) == 0:
                continue
            words = self.replaceWithSignatures(input.split(), rare_word_counter)
            tags_sequence = self.greedyAlgorithm(words)
            self.getOutputSequence(input, tags_sequence)
        self.writeOutputsToFile(greedy_output_file)
        print(time.time()-a)
        self.accuracyEvaluation(golden)


lambdas = [[0.01, 0.09, 0.9]]
for lam_values in lambdas:
    print(lam_values)
    greedyTagger = GreedyTag(lam_values)
    greedyTagger.runTagger()


