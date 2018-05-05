import re

def form(words:[], tags:[], i:int):
    return words[i]

# suffixes

def suf4(words:[], tags:[], i:int):
    return words[i][-4:]

def suf3(words:[], tags:[], i:int):
    return words[i][-3:]

def suf2(words:[], tags:[], i:int):
    return words[i][-2:]

def suf1(words:[], tags:[], i:int):
    return words[i][-1:]

# prefixes

def pref4(words, tags, i):
    return words[i][:4]

def pref3(words, tags, i):
    return words[i][:3]

def pref2(words, tags, i):
    return words[i][:2]

def pref1(words, tags, i):
    return words[i][:1]

# prev tags

def pt1(words, tags, i):
    if i < 1:
        return ''
    return tags[i-1]

def pt21(words, tags, i):
    if i < 1:
        return ''
    if i < 2:
        return tags[i-1]
    return ''.join((tags[i-2], tags[i-1]))

# prev word

def pw1(words, tags, i):
    if i < 1:
        return ''
    return words[i-1]

def pw2(words, tags, i):
    if i < 2:
        return ''
    return words[i-2]

# forward word

def fw1(words, tags, i):
    if i >=  len(words) - 1:
        return ''
    return words[i+1]

def fw2(words, tags, i):
    if i >=  len(words) - 2:
        return ''
    return words[i+2]

# kuku

num_re = re.compile("[0-9]")
def num(words, tags, i):
    word = words[i]
    return '1' if num_re.search(word) else '0'

upper_re = re.compile("[A-Z]")
def upper(words, tags, i):
    word = words[i]
    return '1' if upper_re.search(word) else '0'

def hyph(words, tags, i):
    word = words[i]
    return '1' if '-' in word else '0'
