import re
from itertools import chain

import numpy as np

from TransitionTable import TransitionTable, KTransitionTable
from parsers import TagsParser, StorageParser


class HmmModel:
    class INVALID_INTERPOLATION(ValueError):
        pass

    def __init__(self, nOrder = 2, unkThreshold = 5):
        self.tagsTransitions = KTransitionTable(k = nOrder + 1)
        self.wordTags = TransitionTable()
        self.nOrder = nOrder
        self.signatures = {
            '^Aa': re.compile("^[A-Z][a-z]"),
            '^ing': re.compile("ing$"),
            '^ought': re.compile("ought$"),
        }
        self.unkThreshold = unkThreshold
        self.unknownToken = "*UNK*"

    def computeFromFile(self, filePath):
        START = "start"
        startTags = [START] * self.nOrder
        for tags in TagsParser().parseFile(filePath):
            self.tagsTransitions.addFromList(
                chain(startTags, map(lambda t: t[-1], tags)))

            self.wordTags.addFromList(self._getTagsWithSignatures(tags))

        self.wordTags.computeUnknown(self.unkThreshold, self.unknownToken)

    def reComputeUnknown(self, newThreshold = 5):
        if (newThreshold != self.unkThreshold):
            self.wordTags.computeUnknown(self.unkThreshold, self.unknownToken, newThreshold)
            self.unkThreshold = newThreshold

    def _getTagsWithSignatures(self, tags):
        for word, tag in tags:
            yield (word.lower(), tag)
            for match in filter(lambda r: r[1].search(word), self.signatures.items()):
                yield (match[0], tag)

    def loadTransitions(self, QfilePath = None, EfilePath = None):
        parser = StorageParser()
        for key, value in parser.Load(QfilePath):
            self.tagsTransitions.addKeyValue(key, value)
        for key, value in parser.Load(EfilePath):
            self.wordTags.addKeyValue(key, value)

    def writeQ(self, filePath):
        StorageParser().Save(filePath, self.tagsTransitions.getAllItems())

    def writeE(self, filePath):
        StorageParser().Save(filePath, self.wordTags.getAllItems())

    def getQ(self, t1, t2, t3, hyperParam = (0.4, 0.4, 0.2)):
        """ compute q(t3|t1,t2) """

        if sum(hyperParam) != 1:
            raise HmmModel.INVALID_INTERPOLATION()

        c = self.tagsTransitions.getCount((t3))
        bc = self.tagsTransitions.getCount((t2, t3))
        abc = self.tagsTransitions.getCount((t1, t2, t3))
        ab = self.tagsTransitions.getCount((t1, t2)) or 1
        b = self.tagsTransitions.getCount((t2)) or 1
        tot = self.tagsTransitions.getCount() or 1

        return sum(np.array((abc / ab, bc / b, c / tot)) * hyperParam)

    def getE(self, s, t):
        """ compute e(s|t) """
        count = self.wordTags.getCount((s,t))
        tot = self.tagsTransitions.getCount(t) or 1
        return count / tot

