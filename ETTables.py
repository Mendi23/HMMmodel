from functools import reduce
from collections import defaultdict, Iterable, Counter


class Tree(defaultdict):
    def __init__(self):
        super(Tree, self).__init__(Tree)
        self._value = 0

    def updateValue(self, index, val = 1):
        iter = self
        iter._value += val
        for i in index:
            iter = iter[i]
            iter._value += val

    def getValue(self, index = None):
        return self._getDirectIndex(index)._value

    def _getDirectIndex(self, index = None):
        if not isinstance(index, Iterable):
            return self
        return reduce(lambda acc, i: acc[i], index, self)

    def getItems(self, index = None):
        return map(lambda x: (x[0], x[1]._value), self._getDirectIndex(index).items())

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
    def __init__(self, k = 3):
        super(NgramTransitions, self).__init__()
        self._k = k

    def addFromList(self, items):
        itemsLen = len(items)
        for j in range(0, itemsLen - self._k + 1):
            self.updateValue(items[j: j + self._k])


class EmissionTable:
    def __init__(self):
        self._counter = defaultdict(Counter)

    """
    items: iterable items must be a tuple of (word, tag)
    """
    def addFromIterable(self, items, value = 1):
        for word, tag in items:
            self._counter[tag][word] += value

    def computeUnknown(self, threshold):
        return filter(lambda x: x[1] > 0,
                      map(lambda tag: (tag, sum(filter(lambda x: x < threshold,
                                                       self._counter[tag].values()))),
                          self._counter.keys()))

    def getAllItems(self):
        for i in self._counter.keys():
            for j in self._counter[i].items():
                yield (j[0], i, j[1])
