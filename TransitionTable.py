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

    def getCount(self, key = None):
        if key:
            return self._getCount(key)
        return self._total

    def addKeyValue(self, key, value):
        assert isinstance(key, Hashable)
        self._total += value
        self._addKeyValue(key, value)

    @abstractmethod
    def _addFromList(self, items, itemsLen):
        pass

    def _getCount(self, key):
        return self._counter[key]

    def _addKeyValue(self, key, value):
        self._counter[key] += value

    def getAllItems(self):
        return self._counter.items()


class TransitionTable(_TransitionTable):
    def _addFromList(self, items, itemsLen):
        for item in items:
            self._counter[item] += 1

    def computeUnknown(self, threshold, token, resetThreshold = 0):
        unkCounter = Counter()
        upperLimit = max(threshold, resetThreshold)
        lowerLimit = min(threshold, resetThreshold)

        for pair, count in filter(lambda x: x[1] < upperLimit and x[1] >= lowerLimit,
                                  self._counter.items()):
            unkCounter[(token, pair[1])] += count

        if (resetThreshold < threshold):
            self._counter += unkCounter
        else:
            self._counter -= unkCounter


class KTransitionTable(_TransitionTable):
    def __init__(self, k = 3):
        super().__init__()
        self._counters = [Counter() for _ in range(k)]
        self._k = k

    def _addFromList(self, items, itemsLen):
        for j in range(0, self._k):
            for i in range(itemsLen - j):
                self._counter[items[i:i + j + 1]] += 1
                # self._counters[j] += Counter(
                #     [items[i:i + j + 1] for i in range(itemsLen - j)])

                # indexPairs = product(range(itemsLen), range(1, self._k + 1))
                # indexPairs = filter(lambda p: p[0] + p[1] <= itemsLen, indexPairs)
                # self._counter += Counter([items[i:i + j] for i, j in indexPairs])

                # def _addKeyValue(self, key, value):
                #     assert isinstance(key, Sized)
                #     self._counters[len(key) - 1][key] += value
                #
                # def _getCount(self, key):
                #     return self._counters[len(key) - 1][key]
                #
                # def getAllItems(self):
                #     return chain.from_iterable(map(lambda x: x.items(), self._counters))
