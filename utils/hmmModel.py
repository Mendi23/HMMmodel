import re
from collections import Counter
from functools import lru_cache
from itertools import chain

import numpy as np

from utils.ETTables import EmissionTable, NgramTransitions
from utils.parsers import TagsParser, StorageParser


class HmmModel:
    class _WordEvent:
        def __init__ (self, regex, count = 0):
            self.regex = regex
            self.count = count

    def __init__ (self, nOrder = 2, unkThreshold = 5):
        self._tagsTransitions = NgramTransitions(k=nOrder + 1)
        self._wordTags = EmissionTable()
        self._eventsTags = EmissionTable()
        self._unknownCounter = Counter()
        self._totalUnknown = 0
        self.nOrder = nOrder
        self.unkThreshold = unkThreshold

        self.endTag = "*END*"
        self.startTag = "*start*"
        self.unknownToken = "*UNK*"
        self.eventChar = eventChart = '^'
        self._eventsActions = {
            eventChart + 'num': self._WordEvent(re.compile("^[0-9\.]+$", re.I)),
            eventChart + '[0-9]': self._WordEvent(
                re.compile(r"([0-9][^0-9\.]|[^0-9\.][0-9])", re.I)),
            eventChart + '_ought': self._WordEvent(re.compile("ought$", re.I)),
            eventChart + '_ing': self._WordEvent(re.compile("ing$", re.I)),
            eventChart + '_ate': self._WordEvent(re.compile("ate$", re.I)),
            eventChart + '_es': self._WordEvent(re.compile("es$", re.I)),
            eventChart + '_ed': self._WordEvent(re.compile("ed$", re.I)),
            eventChart + 'en_': self._WordEvent(re.compile("^en", re.I)),
            eventChart + 'em_': self._WordEvent(re.compile("^em", re.I)),
            eventChart + '_ous': self._WordEvent(re.compile("ous$", re.I)),
            eventChart + '_cal': self._WordEvent(re.compile("cal$", re.I)),
            eventChart + '_ish': self._WordEvent(re.compile("ish$", re.I)),
            eventChart + 'adj_': self._WordEvent(re.compile("^(un|in|non)", re.I)),
            eventChart + 'AA': self._WordEvent(re.compile("^[A-Z]+$")),
            eventChart + 'Mr.': self._WordEvent(re.compile("[a-z]\.$", re.I)),
            eventChart + 'Aa': self._WordEvent(re.compile("^[A-Z][a-z]")),
            eventChart + '$$': self._WordEvent(re.compile("[^a-z0-9]", re.I)),
        }

    def computeFromFile (self, filePath):
        endTags = [self.endTag] * self.nOrder
        startTags = [self.startTag] * self.nOrder
        for tags in TagsParser().parseFile(filePath):
            self._tagsTransitions.addFromList(startTags + [t[-1] for t in tags] + endTags)
            self._eventsTags.addFromIterable(self._getWordsAppliedEvents(tags))
            self._wordTags.addFromIterable((word.lower(), tag) for word, tag in tags)
        self._unknownCounter = self._wordTags.computeUnknown(self.unkThreshold)
        self._totalUnknown = sum(self._unknownCounter.values())

    def reComputeUnknown (self, newThreshold = 5):
        if newThreshold != self.unkThreshold:
            self._unknownCounter = self._wordTags.computeUnknown(newThreshold)
            self.unkThreshold = newThreshold
            self._totalUnknown = sum(self._unknownCounter.values())

    def _getWordsAppliedEvents (self, tags):
        for word, tag in tags:
            for signature in self._eventsFilterOnWord(word):
                self._eventsActions[signature].count += 1
                yield signature, tag

    def loadTransitions (self, QfilePath, EfilePath):
        parser = StorageParser()
        for key, value in parser.Load(QfilePath):
            self._tagsTransitions.setValue(value, key)

        total = sum(map(lambda keyVal: keyVal[1], self._tagsTransitions.getItems()))
        self._tagsTransitions.setValue(total)

        for key, value in parser.Load(EfilePath):
            if key[0] == self.unknownToken:
                self._unknownCounter[key[1]] = value
                self._totalUnknown += value
            elif key[0].startswith(self.eventChar):
                self._eventsTags.addFromIterable((key,), value)
                self._eventsActions[key[0]].count += value
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
        return filter(lambda tag: tag != self.startTag, self._tagsTransitions.keys())

    def getWordTags (self, word):
        tags = self._wordTags.wordTags(word.lower())
        if not tags:
            eventsTags = (self._eventsTags.wordTags(key) for key in self._eventsFilterOnWord(word))
            tags = chain(chain.from_iterable(eventsTags), self._unknownCounter.keys())
        return set(tags)

    @lru_cache(maxsize=2 ** 17)
    def getQ (self, params, hyperParam):
        """
        compute q(t_n|t_1,t_2,...t_n-1)
        based on the folowing equation:
        w_i * Score(params[i:]) / Score(params[i:-1])
        :parameter hyperParam: should be with same len as params and sums up to 1.
        """
        getTagValue = self._tagsTransitions.getValue
        countValues = (getTagValue(params[i:]) / (getTagValue(params[i:-1]) or 1)
                       for i in range(min(self.nOrder + 1, len(params))))
        return sum(hyperParam * np.fromiter(countValues, float))

    @lru_cache(maxsize=2 ** 12)
    def getE (self, w, t):
        """ compute e(w|t) """
        return self._wordTags.getCount(w.lower(), t) / self._tagsTransitions.getValue((t,))

    def wordExists (self, word, threshold=1):
        return self._wordTags.wordExists(word.lower(), threshold)

    @lru_cache()
    def getEventTagRatio (self, eventKey, tag):
        return self._eventsTags.getCount(eventKey, tag) / self._tagsTransitions.getValue((tag,))
    @lru_cache()
    def getUnknownTagRatio (self, tag):
        return self._unknownCounter[tag] / self._tagsTransitions.getValue((tag,))

    @lru_cache()
    def getEventRatioTuple (self, tag):
        return tuple(
            (self.getEventTagRatio(eventKey, tag) for eventKey in self._eventsActions.keys()))

    def getNumOfEvents (self):
        return len(self._eventsActions)

    def _eventsFilterOnWord (self, word):
        return (key for key, action in self._eventsActions.items() if action.regex.search(word))

    def getWordEventMask (self, word):
        return tuple((bool(val.regex.search(word)) for val in self._eventsActions.values()))
