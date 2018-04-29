from collections import deque
import numpy as np
from hmmModel import HmmModel
from parsers import OutputParser
from functools import lru_cache

#BUG: "." is a WordEvent! not have to be end line. ("Dr.", "Mr." etc)


def scaleArray (arr, start = 0, end = 1):
    """ scale numbers inside an np.arr to be between "start" to "end" """
    return np.interp(arr, (0, sum(arr)), (start, end))


class GreedyTagger:
    def __init__ (self, hmmmodel: HmmModel, k = 3, endLineTag = ".",
            QHyperParam = (0.4, 0.4, 0.2), unkSigHyperParam = None):
        self.QHyperParam = QHyperParam
        self.unkSigHP = unkSigHyperParam
        self._k = k
        self._model = hmmmodel
        self._allTags = hmmmodel.getAllTags()
        self._endLineTag = endLineTag
        self._startQ = [self._model.startTag] * self._k
        self._queue = deque(self._startQ)

        if not unkSigHyperParam:
            self.unkSigHP = (2,) * hmmmodel.getNumOfEvents() + (1,)
        self.unkSigHP = scaleArray(self.unkSigHP)

    def tagLine (self, wordsLine, outParser: OutputParser):
        for word in wordsLine:
            self._queue.popleft()

            argmax = max(self._allTags,
                key = lambda tag: self._calcQ(tag) * self._calcE(word, tag))

            self._queue.append(argmax)
            outParser.append(word, argmax)

            if argmax == self._endLineTag:
                self._queue = deque(self._startQ)
        outParser.breakLine()

    def _calcE (self, word, tag):
        if self._model.wordExists(word):
            return self._model.getE(word, tag)

        hyperParam = self._calcHPunkWord(word)
        tagProb = self._model.getEventRatioTuple(tag) + (self._model.getUnknownTagRatio(tag),)
        return sum(hyperParam * tagProb)

    def _calcQ (self, tag):
        return self._model.getQ(tuple(self._queue) + (tag,), self.QHyperParam)

    @lru_cache(maxsize = 64)
    def _calcHPunkWord (self, word):
        return scaleArray((self._model.getWordEventMask(word) +
                           (not self._model.wordExists(word),)) * self.unkSigHP)
