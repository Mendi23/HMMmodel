from math import log
from collections import deque

from hmmModel import HmmModel
from parsers import OutputParser


class GreedyTagger:
    def __init__(self, hmmmodel: HmmModel, k=3, endLineTag=".", hyperParams=(0.4, 0.4, 0.2)):
        self.hyperParams = hyperParams
        self._k = k
        self._model = hmmmodel
        self._allTags = hmmmodel.getAllTags()
        self._endLineTag = endLineTag
        self._startQ = [self._model.startTag] * self._k
        self._queue = deque(self._startQ)

    def tagLine(self, wordsLine, outParser: OutputParser):
        q = self._queue
        first = True
        for word in wordsLine:
            q.popleft()
            argmax = max(self._allTags,
                         key=lambda tag: self._calcQ(tag) * self._calcE(word, tag))
            q.append(argmax)
            outParser.append(word, argmax, first)
            first = False


    def _calcE(self, word, tag):
        return self._model.getE(word, tag)

    def _calcQ(self, tag):
        return self._model.getQ(tuple(self._queue) + (tag,), self.hyperParams)
