"""
features is zero based
"""
import random
from collections import namedtuple, defaultdict, Counter
from functools import lru_cache
from itertools import product

import scipy.sparse as sp
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression

from utils.Viterbi import ViterbiTrigramTaggerAbstract
from utils.parsers import MappingParser


class MemmTagger:

    def __init__(self, featuresFuncs=None, model=None):
        self._featuresFuncs = featuresFuncs
        self.tags_dict = {}
        self.features_dict = {}
        self.t_i, self.f_i = 1, 1
        self.model: LogisticRegression = model
        self.tags = [None]

    """
    input: list: word, list: tags, index: i
    output: list: strings of "feat=val"
    """
    def extractFeatures(self, words, tags, i):
        assert self._featuresFuncs
        return (MappingParser.featureValue(feat, val) for feat, val in
                filter(lambda x: x[1], ((feature[0], feature[1](words, tags, i))
                                        for feature in self._featuresFuncs)))

    """
    input: list: strings of "feat=val"
    output: list: sparse matrix libLinear
    """
    def transformToLiblinear(self, features):
        sorted_features = list(filter(lambda x: x,
            (self.features_dict.get(feature, None)
             for feature in features)))

        # sparse = sp.dok_matrix((1, len(self.features_dict) + 1), dtype=np.int64)
        # sparse[0, sorted_features] = 1
        # return sparse

        featCount = len(sorted_features)
        data = np.ones(featCount)
        rows = np.zeros(featCount)
        return sp.coo_matrix((data, (rows, sorted_features)),
            shape=(1, len(self.features_dict) + 1), dtype=np.int64)

    """
    input: string: tag, list: string of "feat=val"
    output: int: tag value, sparse matrix libLinear
    """
    def transformTaggedToLiblinear(self, tag, features):
        return self.tags_dict[tag], self.transformToLiblinear(features)

    """
    input: string: tagged line
    output: generator of "tag feat1=val1 feat2=val2"
    """
    def extractStringFromTaggedLine(self, line):
        return (MappingParser.TagFeatToString(tag, features)
                for (tag, features) in self.extractTagFeaturesFromTaggedLine(line))
        # return MappingParser.TagFeatToString(self.extractTagFeaturesFromTaggedLine(line))

    """
    input: string: tagged line
    output: generator of (tag, list: "feat=val")
    """
    def extractTagFeaturesFromTaggedLine(self, line):
        words, tags = tuple(zip(*line))
        for i in range(len(line)):
            yield tags[i], self.extractFeatures(words, tags, i)

    """
    input: list of (tag, list: "feat=val")
    output: list of string: "6 5:1 7:1 9:1"
    """
    def fitFeatures(self, inputiter, transform=True):
        result = []
        for (tag, features) in inputiter:
            if self.tags_dict.setdefault(tag, self.t_i) == self.t_i:
                self.tags.append(tag)
                self.t_i += 1

            for feature in features:
                if self.features_dict.setdefault(feature, self.f_i) == self.f_i:
                    self.f_i += 1

            if transform:
                sorted_features = sorted((self.features_dict[feature] for feature in features))
                result.append(MappingParser.TagVecToString(self.tags_dict[tag], sorted_features))
        return result

    """
    input: file where each line is "tag feat1=val1 feat2=val2"
    output: list of string: "6 5:1 7:1 9:1"
    """
    def fitFeaturesFromFile(self, inputFile, transform=True):
        with open(inputFile) as fIn:
            mapping = ((tag, MappingParser.splitFeatures(features))
                       for tag, features in
                       (MappingParser.splitTagFeatures(line.strip()) for line in fIn))
            return self.fitFeatures(mapping, transform)

    def getTagsMapping(self):
        return self.tags_dict

    def getFeaturesMapping(self):
        return self.features_dict

    def getModelParams(self):
        if self.model:
            return self.model.get_params()

    def saveModelTofile(self, filePath):
        pickle.dump(self.model, open(filePath, 'wb'))

    def loadModelFromFile(self, filePath):
        self.model = pickle.load(open(filePath, 'rb'))

    def fitModel(self, x_train, y_train):
        assert self.model
        self.model.fit(x_train, y_train)

    def saveParams(self, filePath, modelFile=None):
        MappingParser.saveDictsToFile([self.features_dict, self.tags_dict], filePath)
        if modelFile:
            assert self.model
            self.saveModelTofile(modelFile)

    def loadParams(self, filePath, modelFile=None):
        self.features_dict, self.tags_dict = tuple(MappingParser.getDictsFromFile(filePath))
        self.t_i, self.f_i = len(self.tags_dict), len(self.features_dict)
        self.tags = [None] * (len(self.tags_dict) + 1)
        for key, value in self.tags_dict.items():
            self.tags[value] = key
        if modelFile:
            self.loadModelFromFile(modelFile)

    def getNumOfFeatures(self):
        return len(self.features_dict)


class GreedyTagger(MemmTagger):

    def tagLine(self, line):
        assert self.model
        lineLength = len(line)
        tags = []
        for i in range(lineLength):
            featVec = self.transformToLiblinear(self.extractFeatures(line, tags, i))
            tags.append(self.tags[int(self.model.predict(featVec)[0])])
        return zip(line, tags)


class ViterbiTrigramTagger(GreedyTagger):
    def __init__(self, featuresFuncs=None, model=None):
        super().__init__(featuresFuncs, model)
        self._viterbi = ViterbiTrigramTaggerAbstract('*start*',
            self._getPossibleTsOrRs,
            self._getPossibleTsOrRs,
            self._getCellVal)

    def tagLine(self, line):
        self.tagsZeroBased = list(filter(lambda t: t, self.tags))
        return self._viterbi.tagLine(line)

    def _getPossibleTsOrRs(self, line, i):
        return self.tagsZeroBased

    def _getCellVal(self, line, i, tagsTriplet):
        tags_window = {i: t for i, t in zip(range(i - 2, i + 1), tagsTriplet)}

        features_vec = self.transformToLiblinear(self.extractFeatures(line, tags_window, i))
        all_props = np.array(self.model.predict_log_proba(features_vec)[0])

        tag_i = self.tags_dict[tagsTriplet[-1]] - 1
        # desired_indexes = all_props.argsort()[-5:]
        # desired_indexes = np.argwhere(all_props > np.median(all_props))
        # if tag_i not in desired_indexes:
        # return -np.inf
        return all_props[tag_i]


class ViterbiTrigramTagger_other(GreedyTagger):
    TagVal = namedtuple("TagVal", "prev tag val")
    zeroTagVal = TagVal(None, "empty", -np.inf)
    startTagVal = TagVal(None, '', np.log(1.0))

    @staticmethod
    def TagValVal(tagVal):
        return tagVal.val

    def tagLine(self, line):
        self.tagsZeroBased = list(filter(lambda t: t, self.tags))

        if not line:
            return None
        lineLength = len(line)
        vTable = [
            defaultdict(lambda: defaultdict(lambda: self.zeroTagVal))
            for _ in range(lineLength + 1)
        ]

        maxTagVal = self.zeroTagVal
        for i in range(len(line)):
            for t, it in product(self.tagsZeroBased, self.tagsZeroBased):
                tags_window = {i - 2: it, i - 1: t, i: None}
                features_vec = self.transformToLiblinear(self.extractFeatures(line, tags_window, i))
                all_props = self.model.predict_log_proba(features_vec)[0]

                # max_i: int = np.argmax(all_props)
                for tag_i, val in enumerate(all_props):
                    tag = self.tags[tag_i + 1]
                    # val = all_props[tag_i]
                    cell = self._setVTableCell(vTable, i, (it, t, tag,), val)

                    if i == len(line) - 1:
                        maxTagVal = max(maxTagVal, cell, key=self.TagValVal)

        output = []
        self._appendSelectedTags(maxTagVal, line, len(line) - 1, output)
        return output

    @staticmethod
    def _setVTableCell(vTable, i, tagsTriplet, val):
        _tagger = ViterbiTrigramTagger_other
        it, t, r = tagsTriplet
        if i == 0:
            prevCell = _tagger.startTagVal
            t = prevCell.tag
        elif i == 1:
            prevCell = vTable[0][_tagger.startTagVal.tag][t]
        else:
            prevCell = vTable[i - 1][it][t]

        oldCell = vTable[i][t][r]
        newCell = _tagger.TagVal(prevCell, r, val + prevCell.val)
        cell = vTable[i][t][r] = max(oldCell, newCell, key=_tagger.TagValVal)
        return cell

    def _appendSelectedTags(self, tagVal, line, i, output):
        if i > 0 and tagVal.prev:
            self._appendSelectedTags(tagVal.prev, line, i - 1, output)
        output.append((line[i], tagVal.tag))
