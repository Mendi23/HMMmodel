from collections import deque, namedtuple, defaultdict
import numpy as np
from hmmModel import HmmModel
from parsers import OutputParser
from functools import lru_cache


# BUG: "." is a WordEvent! not have to be end line. ("Dr.", "Mr." etc)


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
                key = lambda tag: self._calcQ(tag, tuple(self._queue)) * self._calcE(word, tag))

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

    def _calcQ (self, tag, prevTags):
        return self._model.getQ(prevTags + (tag,), self.QHyperParam)

    @lru_cache(maxsize = 64)
    def _calcHPunkWord (self, word):
        return scaleArray((self._model.getWordEventMask(word) +
                           (not self._model.wordExists(word),)) * self.unkSigHP)


class ViterbiTagger(GreedyTagger):
    TagVal = namedtuple("TagVal", "t r val")
    zeroTagVal = TagVal(0,0,0)

    def __init__ (self, hmmmodel: HmmModel, k = 3, endLineTag = ".",
            QHyperParam = (0.4, 0.4, 0.2), unkSigHyperParam = None):
        super().__init__(hmmmodel, k, endLineTag, QHyperParam, unkSigHyperParam)

        self._allTags = [self._model.startTag, ] + self._allTags


    def tagLine (self, line, outParser: OutputParser):
        vTable = [
            defaultdict(lambda: defaultdict(lambda: self.zeroTagVal))
            for _ in range(len(line) + 1)]


        for i, word in enumerate([self._model.startTag, ] + line):
            for t in self._allTags:
                for r in self._allTags:
                    vTable[i][t][r] = self._calcTagVal(vTable, i, t, r, word)

        m = vTable

    def _calcTagVal (self, vTable, i, t, r, word):
        start_tag = self._model.startTag
        if i == 0 and t == r == start_tag:
            return self.TagVal(start_tag, start_tag, val = 1)
        elif \
                i == 0 \
                or (t == r == start_tag) \
                or (t != start_tag and
                    (i == 1 or r == start_tag)):
            return self.TagVal(t, r, val = 0)

        return max(
            (self.TagVal(it, t, self._calcVTableVal(vTable, i, it, t, r, word))
                for it in self._allTags),
            key = lambda tv: tv.val)

    def _calcVTableVal (self, vTable, i, it, t, r, word):
        return vTable[i - 1][it][t].val * self._calcQ(r, (it, t)) * self._calcE(word, r)
