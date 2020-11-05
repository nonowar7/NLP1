###
# 1. when doing t/sum in getQ, sum of all transitions or only single transitions?
# 2. upper lower case matters? start sentence
# 3. every word has to be labeled in its turn

import numpy as np
#import string


def getOpenClassPOS():
    # verbs, noun, adjectives, adverbs
    return {'NOUNS': ['NNS', 'NNP', 'NNPS', 'NN'],
            'VERBS': ['VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB'],
            'ADJECTIVES': ['JJR', 'JJS', 'JJ'],
            'ADVERBS': ['RBS', 'RBR', 'RB']}


def addSignatureWords(emissions, sigs, sign, strFunc):
    signature_words = {}
    for sig in sigs:
        for k in emissions:
            if strFunc(k.split()[0], sig):
                entry = ''.join(['^', ' '.join([''.join([sign, sig]), k.split()[-1]])])
                if entry not in signature_words:
                    signature_words[entry] = 0
                signature_words[entry] += emissions[k]
    for sig_word in signature_words:
        emissions[sig_word] = signature_words[sig_word]
    return emissions


def captialLower(emissions):
    signature_words = {}
    for k in emissions:
        if startWithCapitalLower(k.split()[0]):
            entry = ''.join(['^', ' '.join([''.join(['^', 'Aa']), k.split()[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
    for sig_word in signature_words:
        emissions[sig_word] = signature_words[sig_word]
    return emissions


def handleSignatureWords(emissions):
    numbers = [str(i) for i in np.arange(10)]
    prefixes = ['dis', 're', 'un', 'ir', 'in', 'im', 'il']
    suffixes = ['ed', 'ing', 's', 'es', 'ly']
    sig_kinds = [prefixes, suffixes, numbers]
    sig_signs = ['^', '~', '^']
    funcs = [str.startswith, str.endswith, str.startswith]
    for sigs, sign, func in zip(sig_kinds, sig_signs, funcs):
        emissions = addSignatureWords(emissions, sigs, sign, func)
    emissions = captialLower(emissions)
    return emissions


def startWithCapitalLower(word):
    if len(word) > 1 and str(word[0]).isupper() and str(word[1]).islower():
        return True
    return False
