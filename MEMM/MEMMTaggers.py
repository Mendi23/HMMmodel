"""
features is zero based
"""
from functools import lru_cache
import scipy.sparse as sp
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

from utils.ETTables import EmissionTable
from utils.Viterbi import ViterbiTrigramTaggerAbstract
from utils.parsers import MappingParser


class MemmTagger:

    def __init__(self, featuresFuncs=None, model=None, parser=MappingParser()):
        self._featuresFuncs = featuresFuncs
        self.tags_dict = {}
        self.features_dict = {}
        self.t_i, self.f_i = 1, 1
        self.model: LogisticRegression = model
        self.tags = [None]
        self.word_tags = EmissionTable()
        self.MP = parser

    """
    input: list: word, list: tags, index: i
    output: list: strings of "feat=val"
    """

    def extractFeatures(self, words, tags, i):
        assert self._featuresFuncs
        return tuple(self.MP.featureValue(feat, val).lower() for feat, val in
                     filter(lambda x: x[1], ((feature[0], feature[1](words, tags, i))
                                             for feature in self._featuresFuncs)))

    """
    input: list: strings of "feat=val"
    output: list: sparse matrix libLinear
    """

    def transform(self, features):
        found_features = list(filter(lambda x: x, (self.features_dict.get(feature, None)
                                                   for feature in features)))
        vector_shape = (1, len(self.features_dict) + 1)
        featCount = len(found_features)
        data = np.ones(featCount)
        rows = np.zeros(featCount)
        return sp.coo_matrix((data, (rows, found_features)), shape=vector_shape, dtype=np.int64)

    """
    input: string: tag, list: string of "feat=val"
    output: int: tag value, sparse matrix libLinear
    """

    def transformTagged(self, tag, features):
        return self.tags_dict[tag], self.transform(features)

    """
    input: string: tagged line
    output: generator of "tag feat1=val1 feat2=val2"
    """

    def extractTagFeatString(self, line):
        return (self.MP.TagFeatToString(tag, features)
                for (tag, features) in self.extractTagFeatures(line))

    """
    input: string: tagged line
    output: generator of (tag, list: "feat=val")
    """

    def extractTagFeatures(self, line, transform=True):
        words, tags = tuple(zip(*line))
        if transform:
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

            self.word_tags.addFromIterable(((self.MP.getFeatureVal(features[0]), tag),))
            if transform:
                sorted_features = sorted((self.features_dict[feature] for feature in features))
                result.append(self.MP.TagVecToString(self.tags_dict[tag], sorted_features))
        return result

    """
    input: file where each line is "tag feat1=val1 feat2=val2"
    output: list of string: "6 5:1 7:1 9:1"
    """

    def fitFeaturesFromFile(self, inputFile, transform=True):
        with open(inputFile) as fIn:
            mapping = ((tag, self.MP.splitFeatures(features))
                       for tag, features in
                       (self.MP.splitTagFeatures(line.strip()) for line in fIn))
            return self.fitFeatures(mapping, transform)

    def fitModel(self, x_train, y_train):
        assert self.model
        self.model.fit(x_train, y_train)

    def saveParams(self, filePath, modelFile=None):
        self.MP.saveDictsToFile(filePath, [self.features_dict, self.tags_dict, self.word_tags])
        if modelFile:
            assert self.model
            self._saveModelTofile(modelFile)

    def loadParams(self, filePath, modelFile=None):
        self.features_dict, self.tags_dict, word_tags_dict = self.MP.getDictsFromFile(filePath)[:3]
        self.word_tags = EmissionTable(word_tags_dict)
        self.t_i, self.f_i = len(self.tags_dict), len(self.features_dict)
        self.tags = [None] * (len(self.tags_dict) + 1)
        for key, value in self.tags_dict.items():
            self.tags[value] = key
        if modelFile:
            self._loadModelFromFile(modelFile)

    def _saveModelTofile(self, filePath):
        pickle.dump(self.model, open(filePath, 'wb'))

    def _loadModelFromFile(self, filePath):
        self.model = pickle.load(open(filePath, 'rb'))

    def getNumOfFeatures(self):
        return len(self.features_dict)

    def getTagsMapping(self):
        return self.tags_dict

    def getFeaturesMapping(self):
        return self.features_dict

    def getModelParams(self):
        if self.model:
            return self.model.get_params()

    @lru_cache(maxsize=None)
    def _getPossibleTags(self, line, i):
        res = self.word_tags.wordTags(line[i])
        return res if res else self.tags_dict.keys()


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._viterbi = ViterbiTrigramTaggerAbstract('*start*',
            self._getPossibleTags,
            self._getCellVal_proba)

    def tagLine(self, line):
        return self._viterbi.tagLine(line)

    @lru_cache(maxsize=None)
    def _getProba(self, features):
        features_vec = self.transform(features)
        return np.array(self.model.predict_log_proba(features_vec)[0])

    def _getCellVal_proba(self, line, i, tagsTriplet):
        tag_i = self.tags_dict[tagsTriplet[-1]] - 1
        tags_window = {ii: tag for ii, tag in zip(range(i - 2, i + 1), tagsTriplet)}
        features = self.extractFeatures(line, tags_window, i)
        all_props = self._getProba(features)
        return all_props[tag_i]
