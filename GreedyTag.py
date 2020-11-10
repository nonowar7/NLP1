import sys
import Language
import time
import math
VALID_PARAMETERS_NUMBER = 6

class GreedyTag:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.signatures = {}
        self.emissions_prob = {}
        self.transitions_prob = {}
        self.known_words = {}
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
            new_line = "STARTword STARTword " + line + " ENDword ENDword"
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

    def calculateKnownProbabilities(self, lambdas):
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
                    self.transitions_prob[(t1, t2, t3)] = (lambdas[0] * first_prob / (second_prob + epsilon)) + \
                           (lambdas[1] * second_prob / (third_prob + epsilon)) + \
                           (lambdas[2] * third_prob / sum(self.all_tags.values()))
        for pattern in self.signatures:
            pat = pattern.split()
            self.emissions_prob[(pat[0], pat[1])] = self.signatures.get(pattern, 0) / self.transitions[pat[1]]

    def getQ(self, prev_prev_t, prev_t, t):
        return max(self.transitions_prob[(prev_prev_t, prev_t, t)], sys.float_info.epsilon)

    def getE(self, w, t):
        return max(self.emissions_prob.get((w, t), 0), sys.float_info.epsilon)
        # since tag_list is taken from training, transitions[t] is never 0


    def greedyAlgorithm(self, words):
        tags_sequence = []
        prev_tag, prev_prev_tag = 'STARTtag', 'STARTtag'
        for i, word in enumerate(words):
            prob_i = float('-inf')
            tag_i = None
            if i <= 1 or i >= len(words)-2:
                continue
            for tag in self.all_tags:
                current_prob = math.log(self.getE(word, tag)) + \
                               math.log(self.getQ(prev_prev_tag, prev_tag, tag))
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

    def replaceWithSignatures(self, words):
        sequence = []
        for i, word in enumerate(words):
            if word not in self.known_words:
                sym = Language.replaceRareWords(word)
                sequence.append(sym)
            else:
                sequence.append(word)
        return sequence

    def runTagger(self):
        lambdas = [[0.6, 0.3, 0.1]]
        for _lambdas in lambdas:
            a = time.time()
            file_name, q_file, e_file, greedy_output_file, extra_file = greedyTagger.readParameters(VALID_PARAMETERS_NUMBER)
            self.emissions = greedyTagger.readEstimates(e_file)
            self.transitions = greedyTagger.readEstimates(q_file)
            self.getSignaturesAndUnknown()
            self.getAllTags()
            self.calculateKnownProbabilities(_lambdas)
            self.inputs = self.readInputFile(file_name)
            for i, input in enumerate(self.inputs):
                if len(input) == 0:
                    continue
                words = self.replaceWithSignatures(input.split())
                tags_sequence = self.greedyAlgorithm(words)
                self.getOutputSequence(input, tags_sequence)
            self.writeOutputsToFile(greedy_output_file)
            print(time.time()-a)
            print(_lambdas)
            self.check()


greedyTagger = GreedyTag()
greedyTagger.runTagger()

