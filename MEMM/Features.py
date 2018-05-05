
def form(words:[], tags:[], i:int):
    return words[i]

def suf3(words:[], tags:[], i:int):
    return words[i][-3:]

def suf2(words:[], tags:[], i:int):
    return words[i][-2:]

def pt(words, tags, i):
    if i == 0:
        return None
    return tags[i-1]

