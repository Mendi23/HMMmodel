from functools import reduce
from collections import defaultdict, Iterable, Counter


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

    def setValue(self, val, indexes=None):
        self._getDirectIndex(indexes)._value = val

    def getValue(self, indexes=None):
        return self._getDirectIndex(indexes)._value

    def _getDirectIndex(self, indexes=None):
        if not indexes:
            return self
        assert isinstance(indexes, Iterable)
        return reduce(lambda acc, i: acc[i], indexes, self)

    def getItems(self, indexes=None):
        return ((tag, val._value) for tag, val in self._getDirectIndex(indexes).items())

    def getAllItems(self, cur=()):
        if cur:
            yield cur + (self._value,)
        for k in self.keys():
            yield from self[k].getAllItems(cur + (k,))

    def getKeys(self, indexes=None):
        return self._getDirectIndex(indexes).keys()


class NgramTransitions(Tree):
    def __init__(self, k=3):
        super(NgramTransitions, self).__init__()
        self._k = k

    def addFromList(self, items):
        itemsLen = len(items)
        for j in range(0, itemsLen - self._k + 1):
            self.updateValue(items[j: j + self._k])


class EmissionTable:
    def __init__(self, dict_t = None):
        self._countersByWord = defaultdict(Counter)
        if dict_t:
            assert isinstance(dict_t, dict)
            for key, val in dict_t.items():
                counter = Counter(val)
                self._countersByWord[key] = counter

    def getCount(self, word, tag):
        return self._countersByWord[word][tag]

    def addFromIterable(self, items, value=1):
        for word, tag in items:
            self._countersByWord[word][tag] += value

    def computeUnknown(self, threshold):
        return reduce(lambda x, y: x + y,
            filter(lambda counter: sum(counter.values()) < threshold,
                self._countersByWord.values()), Counter())

    def getAllItems(self):
        for word, tagsCounter in self._countersByWord.items():
            for tag, count in tagsCounter.items():
                yield (word, tag, count)

    def wordExists(self, word, threshold=1):
        return word in self._countersByWord and (len(self._countersByWord[word]) > threshold or
               sum(self._countersByWord[word].values()) >= threshold)

    def wordCount(self, word):
        return sum(self._countersByWord[word].values()) if word in self._countersByWord else 0

    def wordTags(self, word):
        if self.wordExists(word):
            return self._countersByWord[word].keys()
        return None

    def items(self):
        return self._countersByWord.items()
