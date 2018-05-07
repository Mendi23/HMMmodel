"""
features is zero based
"""

from collections import OrderedDict, namedtuple, defaultdict
from itertools import product

import scipy.sparse as sp
import numpy as np
import pickle


class MemmTagger:

    def __init__(self, featuresFuncs=None, model=None):
        self._featuresFuncs = featuresFuncs
        self.tags_dict = {}
        self.features_dict = {}
        self.t_i, self.f_i = 1, 1
        self.model = model
        self.tags = [None]

    def extractFeatures(self, words, tags, i):
        assert self._featuresFuncs
        return ' '.join(
            '='.join(pair) for pair in
            filter(lambda x: x[1], (
                (feature[0], feature[1](words, tags, i))
                for feature in self._featuresFuncs)))

    def transform(self, features):
        sorted_features = sorted(filter(lambda x: x, (self.features_dict.get(feature, None) for \
                                                      feature in features.split())))
        featCount = len(sorted_features)
        data = np.ones(featCount)
        rows = np.zeros(featCount)
        return sp.csr_matrix((data, (rows, sorted_features)), shape=(1, len(self.features_dict)+1), dtype=np.int64)

    def extractFeaturesFromTaggedLine(self, line):
        words, tags = tuple(zip(*line))
        for i in range(len(line)):
            yield tags[i], self.extractFeatures(words, tags, i)

    def fitFeatures(self, inputFile, transform=True):
        with open(inputFile) as fIn:
            for line in fIn:
                splitted = line.split()
                tag, features = splitted[0], splitted[1:]
                if self.tags_dict.setdefault(tag, self.t_i) == self.t_i:
                    self.tags.append(tag)
                    self.t_i += 1

                for feature in features:
                    if self.features_dict.setdefault(feature, self.f_i) == self.f_i:
                        self.f_i += 1

                if transform:
                    sorted_features = sorted((self.features_dict[feature] for feature in features))
                    yield self.tags_dict[tag], sorted_features

        if not transform:
            return ()

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

    def setParams(self, tagsMapping, featuresMapping, modelParams=None):
        self.tags_dict = tagsMapping
        self.features_dict = featuresMapping
        self.t_i, self.f_i = len(tagsMapping), len(featuresMapping)
        self.tags = [None] * (len(self.tags_dict) + 1)
        for key, value in self.tags_dict.items():
            self.tags[value] = key
        if modelParams:
            assert self.model
            self.model.set_params(**modelParams)


class GreedyTagger(MemmTagger):

    def tagLine(self, line):
        assert self.model
        lineLength = len(line)
        tags = []
        for i in range(lineLength):
            featVec = self.transform(self.extractFeatures(line, tags, i))
            tags.append(self.tags[int(self.model.predict(featVec)[0])])
        return zip(line, tags)


class ViterbiTrigramTagger(GreedyTagger):
    TagVal = namedtuple("TagVal", "prev tag val")
    zeroTagVal = TagVal(None, "empty", -np.inf)
    startTagVal = TagVal(None, '', np.log(1.0))

    @staticmethod
    def TagValVal(tagVal):
        return tagVal.val

    def tagLine(self, line):
        if not line:
            return None
        lineLength = len(line)
        vTable = [
            defaultdict(lambda: defaultdict(lambda: self.zeroTagVal))
            for _ in range(lineLength + 1)
        ]

        maxTagVal = self.zeroTagVal
        for i in range(len(line)):
            for prevTags in product(self.tags, self.tags, [None]):
                if None in prevTags[:-1]:
                    continue
                features_vec = self.transform(self.extractFeatures(line, prevTags, i))
                all_props = self.model.predict_log_proba(features_vec)[0]

                for tag, val in zip(self.tags, all_props):
                    prevTags[-1] = tag
                    t = prevTags[-2] if i > 0 else self.startTagVal.tag
                    cell = vTable[i][t][tag] = self._calcVTableCell(vTable, i, prevTags, val)

                    if i == len(line) - 1:
                        maxTagVal = max(maxTagVal, cell, key=self.TagValVal)

        output = []
        self._appendSelectedTags(maxTagVal, line, len(line) - 1, output)
        return output

    @staticmethod
    def _calcVTableCell(vTable, i, tagsTriplet, val):
        _tagger = ViterbiTrigramTagger
        it, t, r = tagsTriplet
        if i == 0:
            VCell = _tagger.startTagVal
        elif i == 1:
            VCell = vTable[0][_tagger.startTagVal.tag][t]
        else:
            VCell = vTable[i - 1][it][t]

        return _tagger.TagVal(VCell, r, val + VCell.val)

    def _appendSelectedTags(self, tagVal, line, i, output):
        if i > 0 and tagVal.prev:
            self._appendSelectedTags(tagVal.prev, line, i - 1, output)
        output.append((line[i], tagVal.tag))
