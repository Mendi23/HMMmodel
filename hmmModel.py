import re
from collections import Counter
from itertools import product


class TransitionTable:

    def __init__(self, k = 3, items = None):
        self.k = k
        self.counter = Counter()
        self.total = 0
        if items:
            self.addFromList(items)

    def addFromList(self, items):
        "items shouldn't contain lists"
        items = tuple(items)
        itemsLen = len(items)
        self.total += itemsLen
        if self.k == 1:
            self.counter += Counter(items)
        else:
            x = product(range(itemsLen), range(1, self.k + 1))
            y = filter(lambda p: p[0]+p[1] <= itemsLen, x)
            self.counter += Counter([items[i:i + j] for i, j in y])

    def addKeyValue(self, key, value):
        self.counter[tuple(key)] += value



class TagsParser:

    def __init__(self, endLineTag = ("."), wordDelim = (" "), tagDelim = '/'):
        self.endLine = endLineTag
        self.wordDelim = ''.join(wordDelim)
        self.tagDelim = tagDelim

    def parseFile(self, filePath):
        tags = []
        with open(filePath) as f:
            for line in f:
                t = re.split(f"[{self.wordDelim}]", line.strip())
                for word in t:
                    tagPair = tuple(word.split(self.tagDelim))
                    tags.append(tagPair)
                    if tagPair[-1] in self.endLine:
                        yield tags
                        tags = []

class StorageParser:

    def __init__(self, wordDelim = " ", valueDelim = "\t"):
        self.wordDelim = wordDelim
        self.valueDelim = valueDelim

    def Load(self, filePath):
        with open(filePath) as f:
            for line in map(lambda x: x.split(self.valueDelim), f):
                yield line[0].split(self.wordDelim), int(line[-1])

    def Save(self, filePath, counter:dict):
        with open(filePath, 'w') as f:
            for tags, count in counter.items():
                f.write(f"{self.wordDelim.join(tags)}{self.valueDelim}{count}\n")

class HmmModel:

    def __init__(self, nOrder = 2):
        self.tagsTransitions = TransitionTable(k = nOrder+1)
        self.wordTags = TransitionTable(1)
        self.nOrder = nOrder

    def computeFromFile (self, filePath):
        START = "start"
        p = TagsParser()
        for tags in p.parseFile(filePath):
            self.tagsTransitions.addFromList(
                [START] * self.nOrder + list(map(lambda t: t[-1], tags))
            )
            self.wordTags.addFromList(tags)

    def loadTransitions(self, QfilePath = None, EfilePath = None):
        if QfilePath:
            for key, value in StorageParser().Load(QfilePath):
                self.tagsTransitions.addKeyValue(key, value)

        if EfilePath:
            for key, value in StorageParser().Load(EfilePath):
                self.wordTags.addKeyValue(key, value)

    def writeQ(self, filepath):
        StorageParser().Save(filepath, self.tagsTransitions.counter)

    def writeE(self, filepath):
        StorageParser().Save(filepath, self.wordTags.counter)

    #compute q(t3|t1,t2)
    def getQ(self, t1, t2, t3):
        pass

    def getE(self, s, t):
        pass

