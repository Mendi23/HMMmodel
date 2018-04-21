import re
from itertools import chain
from collections import Counter
from ETTables import EmissionTable, NgramTransitions
from parsers import TagsParser, StorageParser


class HmmModel:
    class INVALID_INTERPOLATION(ValueError):
        pass

    def __init__(self, nOrder = 2, unkThreshold = 5):
        self.tagsTransitions = NgramTransitions(k = nOrder + 1)
        self.wordTags = EmissionTable()
        self.eventsTags = EmissionTable()
        self.unknownCounter = {}
        self.nOrder = nOrder
        self.unkThreshold = unkThreshold

        self.endTag = "END"
        self.unknownToken = "*UNK*"
        self.eventChar = '^'
        self.signatures = {
            '^Aa': re.compile("^[A-Z][a-z]"),
            '^ing': re.compile("ing$"),
            '^ought': re.compile("ought$"),
        }

    def computeFromFile(self, filePath):
        endTags = [self.endTag] * self.nOrder
        for tags in TagsParser().parseFile(filePath):

            self.tagsTransitions.addFromList(list((map(lambda t: t[-1], tags))) + endTags)

            self.wordTags.addFromIterable(self._getWordsCheckSignatures(tags))

        self.unknownCounter = dict(self.wordTags.computeUnknown(self.unkThreshold))

    def reComputeUnknown(self, newThreshold = 5):
        if (newThreshold != self.unkThreshold):
            self.unknownCounter = dict(self.wordTags.computeUnknown(newThreshold))
            self.unkThreshold = newThreshold

    def _getWordsCheckSignatures(self, tags):
        for word, tag in tags:
            yield (word.lower(), tag)
            self.eventsTags.addFromIterable(map(lambda x: (x[0], tag),
                                                filter(lambda r: r[1].search(word),
                                                       self.signatures.items())))

    def loadTransitions(self, QfilePath = None, EfilePath = None):
        parser = StorageParser()
        for key, value in parser.Load(QfilePath):
            self.tagsTransitions.updateValue(key, value)
        for key, value in parser.Load(EfilePath):
            if (key[0] == self.unknownToken):
                self.unknownCounter[key[1]] += value
            elif (key[0].startswith(self.eventChar)):
                self.eventsTags.addFromIterable(key, value)
            else:
                self.wordTags.addFromIterable(key, value)

    def writeQ(self, filePath):
        StorageParser().Save(filePath, self.tagsTransitions.getAllItems())

    def writeE(self, filePath):
        StorageParser().Save(filePath,
                             chain(self.wordTags.getAllItems(), self.eventsTags.getAllItems(),
                                   map(lambda x: (self.unknownToken,) + x,
                                       self.unknownCounter.items())))

    def getQ(self, word, paramList):
        """ compute q(t_n|t_1,t_2,...t_n-1) """
        pass

    def getE(self, s, t):
        """ compute e(s|t) """
        pass
