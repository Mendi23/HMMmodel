import re
from itertools import chain
from collections import Counter

import numpy as np

from TransitionTable import EmissionTable, KTransitionTable
from parsers import TagsParser, StorageParser

class HmmModel:
    class INVALID_INTERPOLATION(ValueError):
        pass

    def __init__(self, nOrder = 2, unkThreshold = 5):
        self.tagsTransitions = KTransitionTable(k = nOrder + 1)
        self.wordTags = EmissionTable()
        self.eventsTags = Counter()
        self.unknownCounter = Counter()
        self.nOrder = nOrder
        self.unkThreshold = unkThreshold

        self.unknownToken = "*UNK*"
        self.eventChar = '^'
        self.signatures = {
            '^Aa': re.compile("^[A-Z][a-z]"),
            '^ing': re.compile("ing$"),
            '^ought': re.compile("ought$"),
        }

    def computeFromFile(self, filePath):
        START = "start"
        startTags = [START] * self.nOrder
        for tags in TagsParser().parseFile(filePath):
            self.tagsTransitions.addFromList(
                chain(startTags, map(lambda t: t[-1], tags)))

            self.wordTags.addFromList(self._getWordsCheckSignatures(tags))

        self.unknownCounter = self.wordTags.computeUnknown(self.unkThreshold)

    def reComputeUnknown(self, newThreshold = 5):
        if (newThreshold != self.unkThreshold):
            self.unknownCounter = self.wordTags.computeUnknown(newThreshold)
            self.unkThreshold = newThreshold

    def _getWordsCheckSignatures(self, tags):
        for word, tag in tags:
            yield (word.lower(), tag)
            for match in filter(lambda r: r[1].search(word), self.signatures.items()):
                self.eventsTags[(match[0], tag)] += 1

    def loadTransitions(self, QfilePath = None, EfilePath = None):
        parser = StorageParser()
        for key, value in parser.Load(QfilePath):
            self.tagsTransitions.addKeyValue(key, value)
        for key, value in parser.Load(EfilePath):
            if (key[0] == self.unknownToken):
                self.unknownCounter[key[1]] += value
            elif (key[0].startswith(self.eventChar)):
                self.eventsTags[key] += value
            else:
                self.wordTags.addKeyValue(key, value)

    def writeQ(self, filePath):
        StorageParser().Save(filePath, self.tagsTransitions.getAllItems())

    def writeE(self, filePath):
        StorageParser().Save(filePath,
                             list(self.wordTags.getAllItems()) + list(self.eventsTags.items())
                             + list(map(lambda x: ((self.unknownToken, x[0]), x[1]) ,
                                   self.unknownCounter.items())))

    def getQ(self, paramList, hyperParam = None):
        """ compute q(t_n|t_1,t_2,...t_n-1) """

        if (not hyperParam):
            hyperParam = 1/(self.nOrder+1)*self.nOrder,

        if sum(hyperParam) != 1:
            raise HmmModel.INVALID_INTERPOLATION()

        c = self.tagsTransitions.getCount((t3,))
        bc = self.tagsTransitions.getCount((t2, t3))
        abc = self.tagsTransitions.getCount((t1, t2, t3))
        ab = self.tagsTransitions.getCount((t1, t2)) or 1
        b = self.tagsTransitions.getCount((t2,)) or 1
        tot = self.tagsTransitions.getCount() or 1
        # print("c:", c, "bc:", bc, "abc:", abc, "ab:", ab, "b:", b, "tot:", tot)

        return sum(np.array((abc / ab, bc / b, c / tot)) * hyperParam)

    def getE(self, s, t):
        """ compute e(s|t) """
        count = self.wordTags.getCount((s, t))
        tot = self.tagsTransitions.getCount((t,)) or 1
        return count / tot

