"""
features is zero based
"""

from collections import OrderedDict, namedtuple, defaultdict
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
        self.tags = []

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
        indptr = [0, featCount]
        data = np.ones(featCount)
        return sp.csr_matrix((data, sorted_features, indptr), shape=(1, len(self.features_dict)))

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

# class ViterbiTrigramTagger(GreedyTagger):
#     TagVal = namedtuple("TagVal", "prev tag val")
#     zeroTagVal = TagVal(None, "empty", -np.inf)
#
#     @staticmethod
#     def TagValVal (tagVal):
#         return tagVal.val
#
#     def __init__ (self, featuresFuncs = None, model = None):
#         super().__init__(featuresFuncs, model)
#
#         self.startTag = "start"
#
#     def tagLine (self, line):
#         if not line:
#             return None
#         lineLength = len(line)
#         vTable = [
#             defaultdict(lambda: defaultdict(lambda: self.zeroTagVal))
#             for _ in range(lineLength + 1)
#         ]
#         for i, word in enumerate(line, 1):
#             for t, r in np.product(range(self.tagsLen), range(self.tagsLen)):
#                 for it in range(self.tagsLen):
#
#
#         vTable[0][self.startTag][self.startTag] = self.TagVal(None, "start", np.log(1.0))
#
#         maxTagVal = self.zeroTagVal
#         for i, word in enumerate(line, 1):
#             possibleIts = [self.startTag] if i <= 2 else vTable[i - 1].keys()
#             possibleTs = [self.startTag] if i == 1 else self._model.getWordTags(line[i - 2])
#             possibleRs = self._model.getWordTags(word)
#
#             for t, r in np.product(possibleTs, possibleRs):
#                 possibleValues = (self._calcVTableCell(vTable[i - 1][it][t], (it, t, r), word)
#                     for it in possibleIts)
#
#                 cell = vTable[i][t][r] = max(possibleValues, key = self.TagValVal)
#
#                 if i == lineLength:
#                     maxTagVal = max(maxTagVal, cell, key = self.TagValVal)
#
#         output = []
#         self._appendSelectedTags(maxTagVal, line, len(line) - 1, output)
#         return output
#
#     def _calcVTableCell (self, VCell, tagsTriplet, word):
#         val = self.model. + VCell.val
#         return self.TagVal(VCell, tagsTriplet[-1], val)
#
#     def _appendSelectedTags (self, tagVal, line, i, output):
#         if i > 0 and tagVal.prev:
#             self._appendSelectedTags(tagVal.prev, line, i - 1, output)
#         output.append((line[i], tagVal.tag))
