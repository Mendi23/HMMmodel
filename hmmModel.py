import re
from collections import Counter
from itertools import product
import numpy as np

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

    def computeUnknown(self, threshold, token):
        unkCounter = Counter()
        for pair, count in filter(lambda x: x[1] < threshold, self.counter.items()):
            unkCounter[(token, pair[1])] += count
        self.counter += unkCounter

    def getCount(self, key = None):
        if key:
            return self.counter[key]
        return self.total


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
                    tagPair = tuple(word.rsplit(self.tagDelim, 1))
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

    class INVALID_INTERPOLATION(ValueError):
        pass


    def __init__(self, nOrder = 2, unkThreshold = 5):
        self.tagsTransitions = TransitionTable(k = nOrder+1)
        self.wordTags = TransitionTable(1)
        self.nOrder = nOrder
        self.signatures = {
            '^Aa':re.compile("^[A-Z][a-z]"),
            '^ing':re.compile("ing$"),
            '^ought':re.compile("ought$"),
        }
        self.unkThreshold = unkThreshold
        self.unknownToken = "*UNK*"

    def computeFromFile (self, filePath):
        START = "start"
        p = TagsParser()
        for tags in p.parseFile(filePath):
            self.tagsTransitions.addFromList(
                [START] * self.nOrder + list(map(lambda t: t[-1], tags))
            )

            self.wordTags.addFromList(list(self._getSignatures(tags)))
            self.wordTags.addFromList(list(map(lambda t: (t[0].lower(),t[1]), tags)))

        self.wordTags.computeUnknown(self.unkThreshold, self.unknownToken)

    def _getSignatures(self, tags):
        for word, tag in tags:
            for match in filter(lambda r: r[1].search(word), self.signatures.items()):
                yield (match[0], tag)

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
    def getQ(self, t1, t2, t3, hyperParam = (0.4, 0.4, 0.2)):

        if sum(hyperParam) != 1:
            raise HmmModel.INVALID_INTERPOLATION()

        c = self.tagsTransitions.getCount((t3))
        bc = self.tagsTransitions.getCount((t2, t3))
        abc = self.tagsTransitions.getCount((t1, t2, t3))
        ab = self.tagsTransitions.getCount((t1, t2)) or 1
        b = self.tagsTransitions.getCount((t2)) or 1
        tot = self.tagsTransitions.getCount() or 1

        return sum(np.array((abc/ab, bc/b, c/tot))*hyperParam)

    #compute e(s|t)
    def getE(self, s, t):
        pass

