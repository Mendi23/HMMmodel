from collections import deque, namedtuple, defaultdict
from itertools import product

import numpy as np
from hmmModel import HmmModel
from parsers import OutputParser
from functools import lru_cache


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
                key = lambda tag: self._calcQ(tuple(self._queue) + (tag,)) * self._calcE(word, tag))

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

    def _calcQ (self, tags):
        return self._model.getQ(tags, self.QHyperParam)

    @lru_cache(maxsize = 64)
    def _calcHPunkWord (self, word):
        return scaleArray((self._model.getWordEventMask(word) +
                           (not self._model.wordExists(word),)) * self.unkSigHP)

class ViterbiTagger(GreedyTagger):
    TagVal = namedtuple("TagVal", "prev tag val")
    zeroTagVal = TagVal(None, "empty", -np.inf)

    @staticmethod
    def TagValVal(tagVal):
        return tagVal.val
    @staticmethod
    def TagValPrev(tagVal):
        return tagVal.prev

    def __init__ (self, hmmmodel: HmmModel, k = 3, endLineTag = ".",
            QHyperParam = (0.4, 0.4, 0.2), unkSigHyperParam = None):
        super().__init__(hmmmodel, k, endLineTag, QHyperParam, unkSigHyperParam)

        self.startTag = self._model.startTag

    def tagLine (self, line, outParser: OutputParser):
        if not line:
            outParser.breakLine()

        lineLength = len(line)
        vTable = [
            defaultdict(lambda: defaultdict(lambda: self.zeroTagVal))
            for _ in range(lineLength + 1)
        ]

        vTable[0][self.startTag][self.startTag] = self.TagVal(None, "start", np.log(1.0))

        maxTagVal = self.zeroTagVal
        for i, word in enumerate(line, 1):
            prevWord = line[i-2]
            possibleTs = [self.startTag] if i == 1 else self._model.getWordTags(prevWord)
            possibleRs = [self.startTag] if i <= 2 else self._model.getWordTags(word)
            assert possibleTs and possibleRs
            for t, r in product(possibleTs, possibleRs):
                cell = vTable[i][t][r] = max(
                    (self._calcVTableCell(vTable[i - 1][it][t], it, t, r, word)
                     for it in vTable[i - 1].keys()),
                    key=self.TagValVal)

                if i == lineLength:
                    maxTagVal = max(maxTagVal, cell, key=self.TagValVal)

        m = vTable

        # reversed((tagval for tagval in iter()))
        # outParser.append()

    def _calcVTableCell (self, VCell, it, t, r, word):
        q = self._calcQ((it, t, r))
        e = self._calcE(word, r)
        if 0 in (q, e):
            return self.zeroTagVal

        val = np.sum(np.log((q, e))) + VCell.val
        return self.TagVal(VCell, r, val)
