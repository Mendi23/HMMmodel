from functools import reduce
from collections import defaultdict, Iterable, Counter


# BUG: it setValue when loading transitions, root.value is not updates
# FIX: need to add field "sum of words".

class Tree(defaultdict):
    def __init__(self):
        super(Tree, self).__init__(Tree)
        self._value = 0

    def updateValue(self, indexes, val=1):
        it = self
        it._value += val
        for i in indexes:
            it = it[i]
            it._value += val

    def setValue(self, indexes, val):
        self._getDirectIndex(indexes)._value = val

    def getValue(self, indexes=None):
        return self._getDirectIndex(indexes)._value

    def _getDirectIndex(self, indexes=None):
        if not indexes:
            return self
        assert isinstance(indexes, Iterable)
        return reduce(lambda acc, i: acc[i], indexes, self)

    # def getItems(self, indexes=None):
    #     return ((tag, val._value) for tag, val in self._getDirectIndex(indexes).items())

    def getAllItems(self, cur=()):
        if cur:
            yield cur + (self._value,)
        for k in self.keys():
            yield from self[k].getAllItems(cur + (k,))


class NgramTransitions(Tree):
    def __init__(self, k=3):
        super(NgramTransitions, self).__init__()
        self._k = k

    def addFromList(self, items):
        itemsLen = len(items)
        for j in range(0, itemsLen - self._k + 1):
            self.updateValue(items[j: j + self._k])


class EmissionTable:
    def __init__(self):
        self._countersByTag = defaultdict(Counter)
        self._words = set()

    def getCount(self, word, tag):
        return self._countersByTag[tag][word]

    def addFromIterable(self, items, value=1):
        """
        items: iterable items must be a tuple of (word, tag)
        """
        for word, tag in items:
            self._countersByTag[tag][word] += value
            self._words.add(word)

    def computeUnknown(self, threshold):
        return filter(lambda x: x[1] > 0,
                      ((tag, sum(filter(lambda x: x < threshold, counter.values())))
                       for tag, counter in self._countersByTag.items()))

    def getAllItems(self):
        for tag in self._countersByTag.keys():
            for word, count in self._countersByTag[tag].items():
                yield (word, tag, count)

    def wordExists(self, word):
        return word in self._words
