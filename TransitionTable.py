from collections import Sized
from itertools import chain
from typing import Hashable

from collections.__init__ import Counter


class TransitionTable:

    def __init__(self):
        self._counter = Counter()
        self._total = 0

    def addFromList(self, items):
        """items shouldn't contain lists"""
        items = tuple(items)
        itemsLen = len(items)
        self._total += itemsLen
        self._addFromList(items, itemsLen)

    def _addFromList(self, items, itemsLen):
        self._counter += Counter(items)

    def computeUnknown(self, threshold, token):
        unkCounter = Counter()
        for pair, count in filter(lambda x: x[1] < threshold, self._counter.items()):
            unkCounter[(token, pair[1])] += count
        self._counter += unkCounter

    def getCount(self, key=None):
        if key:
            return self._getCount(key)
        return self._total

    def addKeyValue(self, key, value):
        assert isinstance(key, Hashable)
        self._counter[key] += value

    def _getCount(self, key):
        return self._counter[key]

    def getAllItems(self):
        return self._counter.items()


class KTransitionTable(TransitionTable):

    def __init__(self, k=3):
        super().__init__()
        self._counters = [Counter()] * k
        self._k = k

    def _addFromList(self, items, itemsLen):
        for j in range(1, self._k + 1):
            self._counters[j - 1] += Counter(
                [items[i:i + j]
                 for i in filter(lambda i: i + j <= itemsLen, range(itemsLen))])
        # indexPairs = product(range(itemsLen), range(2, self._k + 1))
        # indexPairs = filter(lambda p: p[0] + p[1] <= itemsLen, indexPairs)
        # self._counters += Counter([items[i:i + j] for i, j in indexPairs])

    def addKeyValue(self, key, value):
        assert isinstance(key, Hashable)
        assert isinstance(key, Sized)
        self._counters[len(key)][key] += value

    def _getCount(self, key):
        return self._counters[len(key) - 1][key]

    def getAllItems(self):
        return chain.from_iterable(map(lambda x: x.items(), self._counters))
