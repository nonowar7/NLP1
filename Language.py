###
# 1. when doing t/sum in getQ, sum of all transitions or only single transitions?
# 2. upper lower case matters? start sentence
# 3. every word has to be labeled in its turn

import numpy as np
#import string

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
    if word.endswith('es'):
        return 'SUFFIX_es'
    if word.endswith('ise'):
        return 'SUFFIX_ise'
    if word.endswith('ment'):
        return 'SUFFIX_ment'
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
    return 'RARE_rare'


def commonSignatures():
    signatures = {'SUFFIX': ['ing', 's', 'es', 'en', 'ise', 'ed', 'ment', 'ion', 'ship',
                                       'ful', 'ble', 'al', 'ly',  'wise', 'wards'],
                          'PREFIX': ['re', 'dis', 'over', 'mis', 'out', 'co', 'sub', 'ir', 'il']}
    return signatures


def aaddSignatureWords(emissions):
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



