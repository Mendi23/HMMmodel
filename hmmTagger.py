from math import log
from collections import deque
import numpy as np
from hmmModel import HmmModel
from parsers import OutputParser


class GreedyTagger:
    def __init__ (self, hmmmodel: HmmModel, k = 3, endLineTag = ".",
                  QHyperParam = (0.4, 0.4, 0.2),
                  sigHyperParam = None,
                  unkHyperParam = (0.9, 0.1)):
        self.QHyperParam = QHyperParam
        self.sigHyperParam = sigHyperParam
        self.unkHyperParam = np.array(unkHyperParam)
        self._k = k
        self._model = hmmmodel
        self._allTags = hmmmodel.getAllTags()
        self._endLineTag = endLineTag
        self._startQ = [self._model.startTag] * self._k
        self._queue = deque(self._startQ)

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
        word = word.lower()
        if self._model.wordExists(word):
            return self._model.getE(word, tag)
        return sum(np.array([self._model.getBySignature(word, tag, self.sigHyperParam),
                             self._model.getUnknownTag(tag)]
                            ) * self.unkHyperParam)

    def _calcQ (self, tag):
        return self._model.getQ(tuple(self._queue) + (tag,), self.QHyperParam)
