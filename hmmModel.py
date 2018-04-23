import re
from itertools import chain
from ETTables import EmissionTable, NgramTransitions
from parsers import TagsParser, StorageParser
import numpy as np


class HmmModel:
    class INVALID_INTERPOLATION(ValueError):
        pass

    def __init__(self, nOrder=2, unkThreshold=5):
        self.tagsTransitions = NgramTransitions(k=nOrder + 1)
        self.wordTags = EmissionTable()
        self.eventsTags = EmissionTable()
        self.unknownCounter = {}
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
            self.tagsTransitions.addFromList(startTags + [t[-1] for t in tags] + endTags)
            self.wordTags.addFromIterable(self._getWordsCheckSignatures(tags))
        self.unknownCounter = dict(self.wordTags.computeUnknown(self.unkThreshold))
        self.tagsTransitions.setValue((), total)

    def reComputeUnknown(self, newThreshold=5):
        if newThreshold != self.unkThreshold:
            self.unknownCounter = dict(self.wordTags.computeUnknown(newThreshold))
            self.unkThreshold = newThreshold

    def _getWordsCheckSignatures(self, tags):
        for word, tag in tags:
            yield (word.lower(), tag)
            self.eventsTags.addFromIterable((
                (signature[0], tag) for signature in
                filter(lambda r: r[1].search(word), self.signatures.items())
            ))

    def loadTransitions(self, QfilePath=None, EfilePath=None):
        parser = StorageParser()
        for key, value in parser.Load(QfilePath):
            self.tagsTransitions.setValue(key, value)
            if len(key) == 1 and key[0] not in (self.startTag, self.endTag):
                self.tagsTransitions.updateValue((), value)

        for key, value in parser.Load(EfilePath):
            if key[0] == self.unknownToken:
                self.unknownCounter[key[1]] = value
            elif key[0].startswith(self.eventChar):
                self.eventsTags.addFromIterable((key,), value)
            else:
                self.wordTags.addFromIterable((key,), value)

    def writeQ(self, filePath):
        StorageParser().Save(filePath, self.tagsTransitions.getAllItems())

    def writeE(self, filePath):
        StorageParser().Save(filePath,
                             chain(self.wordTags.getAllItems(),
                                   self.eventsTags.getAllItems(),
                                   ((self.unknownToken,) + keyVal for keyVal in self.unknownCounter.items())))

    def getQ(self, params, hyperParam=None):
        """ compute q(t_n|t_1,t_2,...t_n-1)
        based on the folowing equation:
        """

        if not hyperParam:
            hyperParam = (1,) + (0,) * self.nOrder

        if sum(hyperParam) != 1:
            raise self.INVALID_INTERPOLATION()

        getTagValue = self.tagsTransitions.getValue
        countValues = (getTagValue(params[i:]) / (getTagValue(params[i:-1]) or 1)
                       for i in range(min(self.nOrder + 1, len(params))))
        return sum(np.array(hyperParam) * np.fromiter(countValues, float))

    # result = 0
    # for i in range(min(self.nOrder, len(paramList)-1)):
    #   result += hyperParam[i]*self.tagsTransitions.getValue(paramList[
    #   i:])/self.tagsTransitions.getValue(paramList[i:-1])

    def getE(self, w, t):
        """ compute e(w|t) """
        return self.wordTags.getCount(w, t) / (self.tagsTransitions.getValue((t,) or 1))
