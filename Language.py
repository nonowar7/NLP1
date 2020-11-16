
def replaceRareWords(word, i):
    if word.endswith('ing'):
        return 'SUFFIX_ing'
    if word.endswith('ed'):
        return 'SUFFIX_ed'
    if word.endswith('es'):
        return 'SUFFIX_es'
    if word.endswith("'s"):
        return "SUFFIX_'s"
    if word.endswith('s'):
        return 'SUFFIX_s'
    if word.endswith('en'):
        return 'SUFFIX_en'
    if word.endswith('ise'):
        return 'SUFFIX_ise'
    if word.endswith('ness'):
        return 'SUFFIX_ness'
    if word.endswith('ship') and len(word) != 4:
        return 'SUFFIX_ship'
    if word.endswith('nce'):
        return 'SUFFIX_nce'
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
    if word.endswith('ous'):
        return 'SUFFIX_ous'
    if word.endswith('er'):
        return 'SUFFIX_er'
    if word.startswith('in') and len(word) != 2:
        return 'PREFIX_in'
    if word.startswith('un'):
        return 'PREFIX_un'
    if word.startswith('il'):
        return 'PREFIX_il'
    if word.startswith('ir'):
        return 'PREFIX_ir'
    if word.startswith('dis'):
        return 'PREFIX_dis'
    if word.startswith('mis'):
        return 'PREFIX_mis'
    if word.startswith('co'):
        return 'PREFIX_co'
    if word.startswith('sub') and len(word) != 3:
        return 'PREFIX_sub'
    if i == 0:
        return 'First_first'
    if isFirstUpper(word):
        return 'First_upper'
    if word.isupper():
        return 'CAPITAL_captial'
    if hyphenAdj(word):
        return 'ADJ_-'
    if any(str.isdigit(c) for c in word):
        return 'NUMBERS_numbers'
    if isLong(word):
        return 'LONG_long'
    return 'RARE_rare'


def isFirstUpper(w):
    if w[0].isupper() and len(w) > 5 and not w[1:].isupper():
        return True
    return False

def isLong(w):
    if len(w)>12:
        return True
    return False


def hyphenAdj(w):
    if '-' in w and '--' not in w and len(w) > 5 and (any(str.isdigit(c) for c in w) or any(str.isalpha(c) for c in w)):
        return True
    return False


def commonSignatures():
    signatures = {'SUFFIX': ['ing',"'s", 'es', 's', 'en', 'ise', 'ed', 'ship', 'ness', 'nce', 'ful', 'ble', 'al', 'ly', 'wards', 'ous', 'er'],
                  'PREFIX': ['co', 'sub', 'dis', 'mis', 'ir', 'il', 'in', 'un']}
    return signatures


def addSignatureWords(emissions, first_words):
    signature_words = {}
    for token in first_words:
        word_tag = token.rsplit('/', 1)
        entry = "".join(['^', " ".join(['FIRST_first', word_tag[-1]])])
        if entry not in signature_words:
            signature_words[entry] = 0
        signature_words[entry] += 1

    signatures = commonSignatures()
    funcs = {'SUFFIX': str.endswith, 'PREFIX': str.startswith}

    for group in signatures:
        for k in emissions:
            for sig in signatures[group]:
                if funcs[group](k.split()[0], sig) and k.split()[0] != sig:
                    entry = "".join(['^', " ".join([("_".join([group, sig])), k.split()[-1]])])
                    if entry not in signature_words:
                        signature_words[entry] = 0
                    signature_words[entry] += emissions[k]
    for k in emissions:
        split_k = k.split()
        if any(str.isdigit(c) for c in split_k[0]) and not any(str.isalpha(c) for c in split_k[0]):
            entry = "".join(['^', " ".join(['NUMBERS_numbers', split_k[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
        if hyphenAdj(split_k[0]):
            entry = "".join(['^', " ".join(['ADJ_-', split_k[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
        if isFirstUpper(split_k[0]):
            entry = "".join(['^', " ".join(['First_upper', split_k[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
        if split_k[0].isupper():
            entry = "".join(['^', " ".join(['CAPITAL_capital', split_k[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
        if isLong(split_k[0]):
            entry = "".join(['^', " ".join(['LONG_long', split_k[-1]])])
            if entry not in signature_words:
                signature_words[entry] = 0
            signature_words[entry] += emissions[k]
    return signature_words


def addRareWords(emissions):
    rare_words = {}
    for k in emissions:
        if emissions[k] <= 3:
            entry = "".join(['^', " ".join(['RARE_rare', k.split()[-1]])])
            if entry not in rare_words:
                rare_words[entry] = 0
            rare_words[entry] += emissions[k]
    limited_rare = {k:v for k,v in rare_words.items() if v >= 100}
    return limited_rare



