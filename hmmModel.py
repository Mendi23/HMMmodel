import re
from collections import Counter
from itertools import chain
from ETTables import EmissionTable, NgramTransitions
from parsers import TagsParser, StorageParser
import numpy as np


class HmmModel:
    class INVALID_INTERPOLATION(ValueError):
        pass

    def __init__(self, nOrder=2, unkThreshold=5):
        self._tagsTransitions = NgramTransitions(k=nOrder + 1)
        self._wordTags = EmissionTable()
        self._eventsTags = EmissionTable()
        self._unknownCounter = Counter()
        self.nOrder = nOrder
        self.unkThreshold = unkThreshold

        self.endTag = "END"
        self.startTag = "start"
        self.unknownToken = "*UNK*"
        self.eventChar = eventChart = '^'
        self.signatures = {eventChart + 'Aa': re.compile("^[A-Z][a-z]"),
                           eventChart + 'ing': re.compile("ing$"),
                           eventChart + 'ought': re.compile("ought$"), }

    def computeFromFile(self, filePath):
        endTags = [self.endTag] * self.nOrder
        startTags = [self.startTag] * self.nOrder
        total = 0
        for tags in TagsParser().parseFile(filePath):
            total += len(tags)
            self._tagsTransitions.addFromList(startTags + [t[-1] for t in tags] + endTags)
            self._wordTags.addFromIterable(self._getWordsCheckSignatures(tags))
        self._unknownCounter = Counter(dict(self._wordTags.computeUnknown(self.unkThreshold)))
        self._tagsTransitions.setValue((), total)

    def reComputeUnknown(self, newThreshold=5):
        if newThreshold != self.unkThreshold:
            self._unknownCounter = Counter(dict(self._wordTags.computeUnknown(newThreshold)))
            self.unkThreshold = newThreshold

    def _getWordsCheckSignatures(self, tags):
        for word, tag in tags:
            yield (word.lower(), tag)
            self._eventsTags.addFromIterable((
                (signature[0], tag) for signature in
                filter(lambda r: r[1].search(word), self.signatures.items())
            ))

    def loadTransitions(self, QfilePath=None, EfilePath=None):
        parser = StorageParser()
        for key, value in parser.Load(QfilePath):
            self._tagsTransitions.setValue(key, value)
            if len(key) == 1 and key[0] not in (self.startTag, self.endTag):
                self._tagsTransitions.updateValue((), value)

        for key, value in parser.Load(EfilePath):
            if key[0] == self.unknownToken:
                self._unknownCounter[key[1]] = value
            elif key[0].startswith(self.eventChar):
                self._eventsTags.addFromIterable((key,), value)
            else:
                self._wordTags.addFromIterable((key,), value)

    def writeQ(self, filePath):
        StorageParser().Save(filePath, self._tagsTransitions.getAllItems())

    def writeE(self, filePath):
        StorageParser().Save(filePath,
                             chain(self._wordTags.getAllItems(),
                                   self._eventsTags.getAllItems(),
                                   ((self.unknownToken,) + keyVal for keyVal in self._unknownCounter.items())))

    def getQ(self, params, hyperParam=None):
        """ compute q(t_n|t_1,t_2,...t_n-1)
        based on the folowing equation:
        """

        if not hyperParam:
            hyperParam = (1,) + (0,) * self.nOrder

        if sum(hyperParam) != 1:
            raise self.INVALID_INTERPOLATION()

        getTagValue = self._tagsTransitions.getValue
        countValues = (getTagValue(params[i:]) / (getTagValue(params[i:-1]) or 1)
                       for i in range(min(self.nOrder + 1, len(params))))
        return sum(np.array(hyperParam) * np.fromiter(countValues, float))

    # result = 0
    # for i in range(min(self.nOrder, len(paramList)-1)):
    #   result += hyperParam[i]*self.tagsTransitions.getValue(paramList[
    #   i:])/self.tagsTransitions.getValue(paramList[i:-1])

    def getE(self, w, t):
        """ compute e(w|t) """
        return self._wordTags.getCount(w, t) / (self._tagsTransitions.getValue((t,) or 1))

    def getAllTags(self):
        return self._tagsTransitions.keys()

    def getUnknownTag(self):
        return self._unknownCounter.most_common(1)[0]
