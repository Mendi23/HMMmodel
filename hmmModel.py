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
        x = product(range(itemsLen), range(1, self.k + 1))
        y = filter(lambda p: p[0]+p[1] <= itemsLen, x)
        self.counter += Counter([items[i:i + j] for i, j in y])



class TagsParser:

    def __init__(self, endLineTag = ["."], wordDelim = [" "], tagDelim = '/'):
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

class HmmModel:

    def __init__(self, filePath, nOrder = 2):
        self.tagsTransitions = TransitionTable(nOrder+1)
        self.wordTags = TransitionTable(1)
        START = "start"
        p = TagsParser()
        for tags in p.parseFile(filePath):
            self.tagsTransitions.addFromList(
                [START]*nOrder + list(map(lambda t: t[-1], tags))
            )
            self.wordTags.addFromList(tags)

    def writeq(self, filepath):
        with open(filepath, 'w') as f:
            for tags, count in self.tagsTransitions.counter.items():
                f.write("{}\t{}\n".format(' '.join(tags), count))


    def getq(self):
        pass

    def gete(self, str):
        pass



x = HmmModel("./DataSets/ass1-tagger-train")
#x = HmmModel("Testing.txt")
x.writeq("Uriel.out")

