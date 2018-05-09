from collections import deque

import numpy as np

from utils.Viterbi import ViterbiTrigramTaggerAbstract
from utils.hmmModel import HmmModel


def scaleArray(arr, start=0, end=1):
    """ scale numbers inside an np.arr to be between "start" to "end" """
    return np.interp(arr, (0, sum(arr)), (start, end))

class GreedyTagger:
    def __init__(self, hmmmodel: HmmModel, k=3,
                 QHyperParam=None, unkSigHyperParam=None):
        if not QHyperParam:
            QHyperParam = (0.59, 0.28, 0.13)

        assert k == len(QHyperParam)

        self.QHyperParam = tuple(scaleArray(QHyperParam))
        self._k = k
        self._model = hmmmodel
        self._allTags = list(hmmmodel.getAllTags())

        if not unkSigHyperParam:
            eventsLen = hmmmodel.getNumOfEvents()
            unkSigHyperParam = (2,) * eventsLen + (1,)
        self.unkSigHP = np.array(unkSigHyperParam)

    def tagLine(self, wordsLine):
        """
        tags a sequence of words using the hmm Greedy algorithm
        :param wordsLine: iterator on words to tag, by order.
        :return: list of (word, tag) tupples
        """
        output = []
        tagsQueue = deque([self._model.startTag] * self._k)
        for word in wordsLine:
            tagsQueue.popleft()
            argmax = max(self._allTags,
                key=lambda tag: self._calcQ(tuple(tagsQueue) + (tag,)) * self._calcE(word, tag))
            tagsQueue.append(argmax)
            output.append((word, argmax))
        return output

    def _calcE(self, word, tag):
        if self._model.wordExists(word):
            return self._model.getE(word, tag)

        hyperParam = self._calcHPunkWord(word)
        tagProb = self._model.getAllEventsQ(tag) + (self._model.getUnknownQ(tag),)
        return sum(hyperParam * tagProb)

    def _calcQ(self, tags):
        return self._model.getQ(tags, self.QHyperParam)

    def _calcHPunkWord(self, word):
        return scaleArray((self._model.getWordEventMask(word) +
                           (not self._model.wordExists(word),)) * self.unkSigHP)

class ViterbiTrigramTagger(GreedyTagger):
    def __init__(self, hmmmodel: HmmModel, QHyperParam=None, unkSigHyperParam=None):
        super().__init__(hmmmodel, 3, QHyperParam, unkSigHyperParam)
        self._viterbi = ViterbiTrigramTaggerAbstract(self._model.startTag,
            self._getPossibleTags, self._getCellVal)

    def tagLine(self, wordsLine):
        return self._viterbi.tagLine(wordsLine)

    def _getPossibleTags(self, line, i):
        return self._model.getWordTags(line[i])

    def _getCellVal(self, line, i, tagsTriplet):
        q = self._calcQ(tagsTriplet)
        e = self._calcE(line[i], tagsTriplet[-1])
        val = e * q
        if val == 0:
            return -np.inf
        return np.log(val)
