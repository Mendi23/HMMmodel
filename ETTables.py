from functools import reduce
from collections import defaultdict, Iterable, Counter


# BUG: it setValue when loading transitions, root.value is not updates
# FIX: need to add field "sum of words".

class Tree(defaultdict):
    def __init__(self):
        super(Tree, self).__init__(Tree)
        self._value = 0

    def updateValue(self, index, val=1):
        it = self
        it._value += val
        for i in index:
            it = it[i]
            it._value += val

    def setValue(self, index, val):
        self._getDirectIndex(index)._value = val

    def getValue(self, index=None):
        return self._getDirectIndex(index)._value

    def _getDirectIndex(self, index=None):
        if not isinstance(index, Iterable) or not index:
            return self
        return reduce(lambda acc, i: acc[i], index, self)

    def getItems(self, index=None):
        return ((tag, val._value) for tag, val in self._getDirectIndex(index).items())

    def _getAllItemsRec(self, cur):
        m = [cur + (self._value,)]
        for x in self.keys():
            m += self[x]._getAllItemsRec(cur + (x,))
        return m

    def getAllItems(self):
        d = []
        for k in self.keys():
            d += self[k]._getAllItemsRec((k,))
        return d


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
        self._counter = defaultdict(Counter)

    def getCount(self, word, tag):
        return self._counter[tag][word]

    def addFromIterable(self, items, value=1):
        """
        items: iterable items must be a tuple of (word, tag)
        """
        for pair in items:
            word, tag = pair
            self._counter[tag][word] += value

    def computeUnknown(self, threshold):
        return filter(lambda x: x[1] > 0,
                      ((tag, sum(filter(lambda x: x < threshold, counter.values())))
                       for tag, counter in self._counter.items()))

    def getAllItems(self):
        for tag in self._counter.keys():
            for wordCount in self._counter[tag].items():
                yield (wordCount[0], tag, wordCount[1])
