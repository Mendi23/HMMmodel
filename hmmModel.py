import re
from collections import Counter
from functools import lru_cache
from itertools import chain

import numpy as np

from ETTables import EmissionTable, NgramTransitions
from parsers import TagsParser, StorageParser

def scaleArray(arr, start = 0, end = 1):
    """ scale numbers inside an np.arr to be between "start" to "end" """
    return np.interp(arr, (0, sum(arr)), (start, end))


class HmmModel:

    def __init__ (self, nOrder = 2, unkThreshold = 5):
        self._tagsTransitions = NgramTransitions(k = nOrder + 1)
        self._wordTags = EmissionTable()
        self._eventsTags = EmissionTable()
        self._unknownCounter = Counter()
        self.nOrder = nOrder
        self.unkThreshold = unkThreshold

        self.endTag = "END"
        self.startTag = "start"
        self.unknownToken = "*UNK*"
        self.eventChar = eventChart = '^'
        self.signatures = {
            eventChart + 'ought': re.compile("ought$", re.I),
            eventChart + 'ing': re.compile("ing$", re.I),
            eventChart + 'Aa': re.compile("^[A-Z][a-z]"),
            eventChart + 'AA': re.compile("^[A-Z]+$"),
            eventChart + '$$': re.compile("[^a-z]", re.I),
        }

    def computeFromFile (self, filePath):
        endTags = [self.endTag] * self.nOrder
        startTags = [self.startTag] * self.nOrder
        for tags in TagsParser().parseFile(filePath):
            self._tagsTransitions.addFromList(startTags + [t[-1] for t in tags] + endTags)
            self._wordTags.addFromIterable(self._getWordsCheckSignatures(tags))
        self._unknownCounter = self._wordTags.computeUnknown(self.unkThreshold)

    def reComputeUnknown (self, newThreshold = 5):
        if newThreshold != self.unkThreshold:
            self._unknownCounter = self._wordTags.computeUnknown(newThreshold)
            self.unkThreshold = newThreshold

    @lru_cache()
    def _signaturesFilterOnWord (self, word):
        return [(sig, regex.search(word)) for sig, regex in self.signatures.items()]

    def _getWordsCheckSignatures (self, tags):
        for word, tag in tags:
            yield (word.lower(), tag)
            self._eventsTags.addFromIterable((
                (signature, tag) for signature, match in self._signaturesFilterOnWord(word) if match))

    def loadTransitions (self, QfilePath = None, EfilePath = None):
        parser = StorageParser()
        for key, value in parser.Load(QfilePath):
            self._tagsTransitions.setValue(value, key)

        total = sum(map(lambda keyVal: keyVal[1], self._tagsTransitions.getItems()))
        self._tagsTransitions.setValue(total)

        for key, value in parser.Load(EfilePath):
            if key[0] == self.unknownToken:
                self._unknownCounter[key[1]] = value
            elif key[0].startswith(self.eventChar):
                self._eventsTags.addFromIterable((key,), value)
            else:
                self._wordTags.addFromIterable((key,), value)

    def writeQ (self, filePath):
        StorageParser().Save(filePath, self._tagsTransitions.getAllItems())

    def writeE (self, filePath):
        StorageParser().Save(filePath,
            chain(self._wordTags.getAllItems(),
                self._eventsTags.getAllItems(),
                ((self.unknownToken,) + keyVal for keyVal in self._unknownCounter.items())))

    def getAllTags (self):
        return list(filter(lambda tag: tag != self.startTag, self._tagsTransitions.keys()))

    @lru_cache(maxsize=8)
    def _getHyperParam(self, hyperParam, size):
        """ if not declared - declare hyper params in Descending order """
        if not hyperParam:
            hyperParam = np.arange(size, 0, -1)

        return scaleArray(hyperParam)[:size]

    # @lru_cache(maxsize=2048)
    def getQ (self, params, hyperParam = None):
        """
        compute q(t_n|t_1,t_2,...t_n-1)
        based on the folowing equation:
        w_i * Score(params[i:]) / Score(params[i:-1])
        :parameter hyperParam: should be with same len as params and sums up to 1.
        """
        paramsSize = self.nOrder + 1
        hyperParam = self._getHyperParam(hyperParam, paramsSize)

        getTagValue = self._tagsTransitions.getValue
        countValues = (getTagValue(params[i:]) / (getTagValue(params[i:-1]) or 1)
                       for i in range(min(paramsSize, len(params))))
        return sum(hyperParam * np.fromiter(countValues, float))

    # @lru_cache(maxsize=1024)
    def getE (self, w, t):
        """ compute e(w|t) """
        return self._wordTags.getCount(w, t) / (self._tagsTransitions.getValue((t,)) or 1)

    def wordExists (self, word):
        return self._wordTags.wordExists(word)

    def getBySignature (self, word, tag, hyperParam = None):
        hyperParam = self._getHyperParam(hyperParam, len(self.signatures))

        possibleScores = ((self._eventsTags.getCount(sig, tag) if match else 0)
                          for sig, match in self._signaturesFilterOnWord(word))
        return sum(np.fromiter(possibleScores, float) * hyperParam)

    def getUnknownTag (self, tag):
        return self._unknownCounter[tag] / sum(self._unknownCounter.values())
