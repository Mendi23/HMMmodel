from typing import Hashable
from abc import ABC, abstractmethod

from collections.__init__ import Counter


class _CountTable(ABC):
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
        # print(key, self._getCount(key))
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


class EmissionTable(_CountTable):
    def _addFromList(self, items, itemsLen):
        for item in items:
            self._counter[item] += 1

    def computeUnknown(self, threshold):
        unkCounter = Counter()
        for pair, count in filter(lambda x: x[1] < threshold, self._counter.items()):
            unkCounter[pair[1]] += count
        return unkCounter


class KTransitionTable(_CountTable):
    def __init__(self, k = 3):
        super().__init__()
        self._k = k

    def _addFromList(self, items, itemsLen):
        for j in range(0, self._k):
            for i in range(itemsLen - j):
                self._counter[items[i:i + j + 1]] += 1
