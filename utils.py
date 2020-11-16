
def getOnlyTokens(file_name):
    with open(file_name, 'r', encoding="utf8") as f:
        content = f.read().splitlines()
    lines = []
    for line in content:
        tokens = line.split()
        new_line = ""
        for token in tokens:
            new_token = token.rsplit("/", 1)[0]
            new_line = " ".join([new_line, new_token])
        lines.append(new_line[1:])
    return lines, content

