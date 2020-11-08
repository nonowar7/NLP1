###
# 1. when doing t/sum in getQ, sum of all transitions or only single transitions?
# 2. upper lower case matters? start sentence
# 3. every word has to be labeled in its turn
import string

def getCloseClassPOS(emissions):
    POS = {}
    prepositions = ["in", "to", "for", "In"]
    conjunctions = ["and", "or", "but"]
    determiners = ["the", "a", "an"]
    close_groups = [prepositions, conjunctions, determiners]
    for entry in emissions:
        word, tag = entry.split()[0], entry.split()[-1]
        for group in close_groups:
            if word in group and emissions[entry] > 50:
                if tag not in POS:
                    POS[tag] = 0

    return POS

def getOpenClassPOS():
    # verbs, noun, adjectives, adverbs
    return {'NOUNS': ['NNS', 'NNP', 'NNPS', 'NN'],
            'VERBS': ['VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB'],
            'ADJECTIVES': ['JJR', 'JJS', 'JJ'],
            'ADVERBS': ['RBS', 'RBR', 'RB']}


def getSignatures():
    signatures_regex = {'^VERBS': r'[^ ]*(re|dis|over|mis|out|ed|es|s|ing|en|ise)[^ ]*',
                        '^NOUNS': r'[^ ]*(co|sub|ment|ion|ship)[^ ]*',
                        '^ADJ': r'[^ ]*(ful|ble|al)[^ ]*',
                        '^ADV': r'[^ ]*(ly|wise|wards)[^ ]*'}
    return signatures_regex

def replaceRareWords(word):
    if word.startswith('in'):
        return 'PREFIX_in'
    if word.startswith('im'):
        return 'PREFIX_im'
    if word.startswith('un'):
        return 'PREFIX_un'
    if word.startswith('il'):
        return 'PREFIX_il'
    if word.startswith('ir'):
        return 'PREFIX_ir'
    if word.startswith('re'):
        return 'PREFIX_re'
    if word.startswith('dis'):
        return 'PREFIX_dis'
    if word.startswith('mis'):
        return 'PREFIX_mis'
    if word.endswith('ing'):
        return 'SUFFIX_ing'
    if word.endswith('ed'):
        return 'SUFFIX_ed'
    if word.endswith('s'):
        return 'SUFFIX_s'
    if word.endswith('es'):
        return 'SUFFIX_es'
    if word.endswith('en'):
        return 'SUFFIX_en'
    if word.endswith('ise'):
        return 'SUFFIX_ise'
    if word.endswith('ment'):
        return 'SUFFIX_ment'
    if word.endswith('ship'):
        return 'SUFFIX_ship'
    if word.endswith('ion'):
        return 'SUFFIX_ion'
    if word.endswith('al'):
        return 'SUFFIX_al'
    if word.endswith('ful'):
        return 'SUFFIX_ful'
    if word.endswith('ly'):
        return 'SUFFIX_ly'
    if word.endswith('ble'):
        return 'SUFFIX_ble'
    if word.endswith('wards'):
        return 'SUFFIX_wards'
    if any(str.isdigit(c) for c in word):
        return 'NUMBERS_numbers'
    '''
    if startCapitalLower(word):
        return 'CAPITAL_low'
    '''
    return 'RARE_rare'


def startCapitalLower(w):
    if not w[0].isupper():
        return False
    for c in w[1:]:
        if not c.islower():
            return False
    return True


def commonSignatures():
    signatures = {'SUFFIX': ['ing', 's', 'es', 'en', 'ise', 'ed', 'ment', 'ion', 'ship',
                                       'ful', 'ble', 'al', 'ly', 'wards'],
                          'PREFIX': ['re', 'dis', 'mis', 'out', 'co', 'sub', 'ir', 'il', 'im', 'in', 'un']}
    return signatures


def addSignatureWords(emissions):
    signatures = commonSignatures()
    signature_words = {}
    for group in signatures:
        for k in emissions:
            for sig in signatures[group]:
                if sig in k.split()[0]:
                    entry = "".join(['^', " ".join([("_".join([group, sig])), k.split()[-1]])])
                    if entry not in signature_words:
                        signature_words[entry] = 0
                    signature_words[entry] += emissions[k]
    for k in emissions:
        '''
        if startCapitalLower(k.split()[0]):
            entry = "".join(['^', " ".join(['CAPITAL_low', k.split()[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
            continue
        '''

        if any(str.isdigit(c) for c in k.split()[0]):
            entry = "".join(['^', " ".join(['NUMBERS_numbers', k.split()[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
            continue
    return signature_words


def addRareWords(emissions):
    rare_words = {}
    for k in emissions:
        if emissions[k] <= 5:
            entry = "".join(['^', " ".join(['RARE_rare', k.split()[-1]])])
            if entry not in rare_words:
                rare_words[entry] = 0
            rare_words[entry] += emissions[k]
    return rare_words


