import re
from collections import Counter
from functools import lru_cache
from itertools import chain
import numpy as np

from utils.ETTables import EmissionTable, NgramTransitions
from utils.parsers import TagsParser, StorageParser


class HmmModel:
    def __init__(self, nOrder=2, unkThreshold=5):
        self._tagsTransitions = NgramTransitions(k=nOrder + 1)
        self._wordTags = EmissionTable()
        self._eventsTags = EmissionTable()
        self._unknownCounter = Counter()
        self.nOrder = nOrder
        self._unkThreshold = unkThreshold

        self.endTag = "*END*"
        self.startTag = "*start*"
        self.unknownToken = "*UNK*"
        self.eventChar = eventChart = '^'
        self._wordEvents = {
            eventChart + 'num': re.compile("^[0-9\.,\-]+$", re.I),
            eventChart + '_ought': re.compile("ought$", re.I),
            eventChart + '_ing': re.compile("ing$", re.I),
            eventChart + '_ate': re.compile("ate$", re.I),
            eventChart + '_es': re.compile("es$", re.I),
            eventChart + '_ed': re.compile("ed$", re.I),
            eventChart + 'en_': re.compile("^en", re.I),
            eventChart + 'em_': re.compile("^em", re.I),
            eventChart + '_ous': re.compile("ous$", re.I),
            eventChart + '_cal': re.compile("cal$", re.I),
            eventChart + '_ish': re.compile("ish$", re.I),
            eventChart + 'adj_': re.compile("^(un|in|non)", re.I),
            eventChart + 'AA': re.compile("^[A-Z]+$"),
            eventChart + 'Mr.': re.compile("[a-z]\.$", re.I),
            eventChart + 'Aa': re.compile("^[A-Z][a-z]"),
            eventChart + '$$': re.compile("[^a-z0-9]", re.I),
        }

    def computeFromFile(self, filePath):
        """
        fit the model by parameters computed from input file (using TagsParser for decoding)
        :param filePath:
        """
        endTags = [self.endTag] * self.nOrder
        startTags = [self.startTag] * self.nOrder
        for tags in TagsParser().parseFile(filePath):
            self._tagsTransitions.addFromList(startTags + [t[-1] for t in tags] + endTags)
            self._eventsTags.addFromIterable(self._getWordsAppliedEvents(tags))
            self._wordTags.addFromIterable((word.lower(), tag) for word, tag in tags)
        self._unknownCounter = self._wordTags.computeUnknown(self._unkThreshold)

    def reComputeUnknown(self, newThreshold=5):
        """
        re-compute the unknown Tags count according to a new newThreshold for max appearances
        :param newThreshold: int
        """
        if newThreshold != self._unkThreshold:
            self._unknownCounter = self._wordTags.computeUnknown(newThreshold)
            self._unkThreshold = newThreshold

    def _getWordsAppliedEvents(self, tags):
        """
        :param tags: iterator of (word, tag) tuples
        :return: Generator object. yield (signature, tag) corresponding to the matched word-events
        """
        for word, tag in tags:
            for signature in self._eventsFilterOnWord(word):
                yield signature, tag

    def loadTransitions(self, QfilePath, EfilePath):
        """
        fit the model using parameters stores in files (using StorageParser for decoding)
        :param QfilePath: file containing counts for all n-grams in range (1, nOrder)
        :param EfilePath: file containing words count
        """
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

    def saveTransitions(self, QfilePath, EfilePath):
        """
        save the model's transition counts to files (using StorageParser)
        :param QfilePath: file to store n-gram counts
        :param EfilePath: file to store word, unknown and word-events counts
        """
        StorageParser().Save(QfilePath, self._tagsTransitions.getAllItems())
        StorageParser().Save(EfilePath,
            chain(self._wordTags.getAllItems(), self._eventsTags.getAllItems(),
                ((self.unknownToken,) + keyVal for keyVal in self._unknownCounter.items())))

    def getAllTags(self):
        """
        :return: generator object. yield all tags that exist in the model
        """
        return filter(lambda tag: tag != self.startTag, self._tagsTransitions.keys())

    def getWordTags(self, word):
        """
        if word wasn't in training data return tags that appeared with word-events or unknown words
        :param word:
        :return: set object. all tags that appear in training data with word
        """
        tags = self._wordTags.wordTags(word.lower())
        if not tags:
            eventsTags = (self._eventsTags.wordTags(key) for key in self._eventsFilterOnWord(word))
            tags = chain(chain.from_iterable(eventsTags), self._unknownCounter.keys())
        return set(tags)

    @lru_cache(maxsize=None)
    def getQ(self, params, hyperParam):
        """
        compute q(t_n|t_1,t_2,...t_n-1)
        :param params: (t_1, ..., t_n)
        :param hyperParam: array-like. weights for q interpolation
        :return: sum of: (w_i * Score(params[i:]) / Score(params[i:-1]))
        """
        getTagValue = self._tagsTransitions.getValue
        countValues = (getTagValue(params[i:]) / (getTagValue(params[i:-1]) or 1)
                       for i in range(min(self.nOrder + 1, len(params))))
        return sum(hyperParam * np.fromiter(countValues, float))

    @lru_cache(maxsize=None)
    def getE(self, word, tag):
        """
        compute e(w|t)
        :param word:
        :param tag:
        :return: count(word, tag) / count(tag)
        """
        return self._wordTags.getCount(word.lower(), tag) / self._tagsTransitions.getValue((tag,))

    def wordExists(self, word, threshold=1):
        """
        :param word:
        :param threshold: optinal. minimum appearances to be considered exist
        :return: bool.
        """
        return self._wordTags.wordExists(word.lower(), threshold)

    def getEventQ(self, eventKey, tag):
        """
        :param eventKey:
        :param tag:
        :return: count(event, tag) / count(tag)
        """
        return self._eventsTags.getCount(eventKey, tag) / self._tagsTransitions.getValue((tag,))

    @lru_cache(maxsize=None)
    def getUnknownQ(self, tag):
        """
        :param tag:
        :return: count(unknown-words, tag) / count(tag)
        """
        return self._unknownCounter[tag] / self._tagsTransitions.getValue((tag,))

    @lru_cache(maxsize=None)
    def getAllEventsQ(self, tag):
        """
        :param tag:
        :return: tuple. count(event, tag) / count(tag) for every word-event
        """
        return tuple(
            (self.getEventQ(eventKey, tag) for eventKey in self._wordEvents.keys()))

    def getNumOfEvents(self):
        """
        :return: num of word-events
        """
        return len(self._wordEvents)

    @lru_cache(maxsize=None)
    def _eventsFilterOnWord(self, word):
        """
        :param word:
        :return: generator object. all word-events that apply to word
        """
        return (key for key, regex in self._wordEvents.items() if regex.search(word))

    @lru_cache(maxsize=None)
    def getWordEventMask(self, word):
        """
        :param word:
        :return: tuple. mask with mask[i] = 1 iff wordEvents[i] applies to word
        """
        return tuple((bool(regex.search(word)) for regex in self._wordEvents.values()))
