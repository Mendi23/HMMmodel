import re
from collections import Counter
from functools import lru_cache
from itertools import chain
from ETTables import EmissionTable, NgramTransitions
from parsers import TagsParser, StorageParser
import numpy as np


class HmmModel:
    class INVALID_INTERPOLATION(ValueError):
        pass

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
            eventChart + 'Aa': re.compile("^[A-Z][a-z]"),
            eventChart + 'ing': re.compile("ing$", re.I),
            eventChart + 'ought': re.compile("ought$", re.I),
            eventChart + 'AA': re.compile("^[A-Z]+$"),
            eventChart + '$$': re.compile("[^a-z]", re.I),
        }
        self.signatures_scores = np.full(len(self.signatures), 1./len(self.signatures))

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
        return [sig for sig, regex in self.signatures.items() if regex.search(word)]

    def _getWordsCheckSignatures (self, tags):
        for word, tag in tags:
            yield (word.lower(), tag)
            self._eventsTags.addFromIterable((
                (signature, tag) for signature in self._signaturesFilterOnWord(word)))

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

    @lru_cache()
    def getQ (self, params, hyperParam = None):
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

    @lru_cache()
    def getE (self, w, t):
        """ compute e(w|t) """
        return self._wordTags.getCount(w, t) / (self._tagsTransitions.getValue((t,) or 1))

    def getAllTags (self):
        return list(filter(lambda tag: tag != self.startTag, self._tagsTransitions.keys()))

    @lru_cache()
    def wordExists (self, word):
        return self._wordTags.wordExists(word)

    @lru_cache()
    def getBySignature (self, word, tag):
        possibleScores = [self._eventsTags.getCount(sig[0], tag)
            for sig in self._signaturesFilterOnWord(word)]
        return max(possibleScores + [0, ])

    @lru_cache()
    def getUnknownTag (self, tag):
        return self._unknownCounter[tag]
