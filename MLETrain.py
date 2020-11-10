from sklearn.feature_extraction.text import CountVectorizer
import sys
import Language

VALID_PARAMETERS_NUMBER = 4

class MLETrain:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}

    def readParameters(self, num_param):
        if len(sys.argv) != num_param:
            print("invalid input")
            return None
        syst, file_name, q_file, e_file = sys.argv
        return file_name, q_file, e_file

    def readTaggedCorpus(self, tagged_corpus):
        with open(tagged_corpus, 'r', encoding="utf8") as f:
            content = f.read().splitlines()
        lines = []
        for line in content:
            new_line = "STARTword/STARTtag STARTword/STARTtag " + line + " ENDword/ENDtag"
            lines.append(new_line)
        return lines

    def writeDictToFile(self, file_name, _dict):
        with open(file_name, 'w') as f:
            for item in _dict:
                entry = item + '\t' + str(_dict[item])
                f.write("%s\n" % entry)

    def keepOnlyTags(self, txt):
        return ' '.join(w.rsplit('/', 1)[1] for w in txt.split())

    def getTransitions(self, train_data):
        vec = CountVectorizer(lowercase=False, ngram_range=(1, 3), preprocessor=self.keepOnlyTags, token_pattern=r"\S+")
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        return dict(zip(names, values))

    def getEmissions(self, train_data, token_pattern=r"\S+"):
        vec = CountVectorizer(lowercase=False, token_pattern=token_pattern, min_df=1)
        values = vec.fit_transform(train_data).sum(axis=0).A1
        names = vec.get_feature_names()
        names = [s.replace('/', ' ') for s in names]
        return dict(zip(names, values))

    def replaceWithSignatures(self, train_data):
        signatures = Language.getSignatures()
        sig_dict = {}
        for signature in signatures:
            print(self.getEmissions(train_data, signatures[signature]))

    def estimateMLE(self):
        data_file, q_file, e_file = self.readParameters(VALID_PARAMETERS_NUMBER)
        train_data = self.readTaggedCorpus(data_file)
        self.transitions = self.getTransitions(train_data)
        self.emissions = self.getEmissions(train_data)
        self.emissions.update(Language.addSignatureWords(self.emissions))
        self.emissions.update(Language.addRareWords(self.emissions))
        self.writeDictToFile(q_file, self.transitions)
        self.writeDictToFile(e_file, self.emissions)

mle = MLETrain()
mle.estimateMLE()

