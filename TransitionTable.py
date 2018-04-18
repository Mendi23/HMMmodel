from collections import Sized
from itertools import chain, product
from typing import Hashable
from abc import ABC, abstractmethod

from collections.__init__ import Counter

class _TransitionTable(ABC):

    def __init__(self):
        self._counter = Counter()
        self._total = 0

    def addFromList(self, items):
        """items shouldn't contain lists"""
        items = tuple(items)
        itemsLen = len(items)
        self._total += itemsLen
        self._addFromList(items, itemsLen)

    def getCount(self, key=None):
        if key:
            return self._counter[key]
        return self._total

    def addKeyValue(self, key, value):
        assert isinstance(key, Hashable)
        self._total += value
        self._counter[key] += value

    @abstractmethod
    def _addFromList(self, items, itemsLen):
        pass

    def getAllItems(self):
        return self._counter.items()


class TransitionTable(_TransitionTable):

    def _addFromList(self, items, itemsLen):
        self._counter += Counter(items)

    def computeUnknown(self, threshold, token):
        unkCounter = Counter()
        for pair, count in filter(lambda x: x[1] < threshold, self._counter.items()):
            unkCounter[(token, pair[1])] += count
        self._counter += unkCounter

class KTransitionTable(_TransitionTable):

    def __init__(self, k=3):
        super().__init__()
        self._k = k

    def _addFromList(self, items, itemsLen):
        indexPairs = product(range(itemsLen), range(1, self._k + 1))
        indexPairs = filter(lambda p: p[0] + p[1] <= itemsLen, indexPairs)
        self._counter += Counter([items[i:i + j] for i, j in indexPairs])

    # def getAllItems(self):
    #     return chain.from_iterable(map(lambda x: x.items(), self._counters))
