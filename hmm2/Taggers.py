from collections import deque

import numpy as np

from utils.Viterbi import ViterbiTrigramTaggerAbstract
from utils.hmmModel import HmmModel
from functools import lru_cache


def scaleArray(arr, start=0, end=1):
    """ scale numbers inside an np.arr to be between "start" to "end" """
    return np.interp(arr, (0, sum(arr)), (start, end))


class GreedyTagger:
    def __init__(self, hmmmodel: HmmModel, k=3,
                 QHyperParam=None, unkSigHyperParam=None):
        self._k = k
        self._model = hmmmodel
        self._allTags = list(hmmmodel.getAllTags())
        self._startQ = [self._model.startTag] * self._k
        self._queue = deque(self._startQ)

        if not QHyperParam:
            QHyperParam = [0.58, 0.28, 0.13]
        self.QHyperParam = tuple(scaleArray(QHyperParam))
        if not unkSigHyperParam:
            eventsLen = hmmmodel.getNumOfEvents()
            unkSigHyperParam = (2,) * eventsLen + (1,)
        self.unkSigHP = np.array(unkSigHyperParam)

    def tagLine(self, wordsLine):
        output = []
        self._queue = deque(self._startQ)
        for word in wordsLine:
            self._queue.popleft()

            argmax = max(self._allTags,
                         key=lambda tag: self._calcQ(tuple(self._queue) + (tag,)) * self._calcE(word, tag))

            self._queue.append(argmax)
            output.append((word, argmax))

        return output

    def _calcE(self, word, tag):
        if self._model.wordExists(word):
            return self._model.getE(word, tag)

        hyperParam = self._calcHPunkWord(word)
        tagProb = self._model.getEventRatioTuple(tag) + (self._model.getUnknownTagRatio(tag),)
        return sum(hyperParam * tagProb)

    def _calcQ(self, tags):
        return self._model.getQ(tags, self.QHyperParam)

    @lru_cache(maxsize=64)
    def _calcHPunkWord(self, word):
        return scaleArray((self._model.getWordEventMask(word) +
                           (not self._model.wordExists(word),)) * self.unkSigHP)


class ViterbiTrigramTagger(GreedyTagger):
    def __init__(self, hmmmodel: HmmModel, QHyperParam=None, unkSigHyperParam=None):
        super().__init__(hmmmodel, 3, QHyperParam, unkSigHyperParam)
        self._viterbi = ViterbiTrigramTaggerAbstract(self._model.startTag,
                                                     self._getPossibleTs,
                                                     self._getPossibleRs,
                                                     self._getCellVal)

    def tagLine(self, wordsLine):
        return self._viterbi.tagLine(wordsLine)

    def _getPossibleRs(self, line, i):
        return self._model.getWordTags(line[i])

    def _getPossibleTs(self, line, i):
        return self._model.getWordTags(line[i - 1])

    def _getCellVal(self, line, i, tagsTriplet):
        q = self._calcQ(tagsTriplet)
        e = self._calcE(line[i], tagsTriplet[-1])
        val = e * q
        if val == 0:
            return -np.inf
        return np.log(val)
