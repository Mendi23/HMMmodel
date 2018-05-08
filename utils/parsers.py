import re
from collections import deque


class TagsParser:
    """
    :param stopTags: tuple of tags used to represend end of sentence. (def: empty)
    :param wordDelim: regex of all charectars used for spliting words (def: " ")
    :param tagDelim: charectar reprents deliminator between word and its tag (def: '/')
    :param newLineDelim: newLine in file used as deliminator between sentences (def: true)
    """

    def __init__ (self, stopTags = (), wordDelim = " ", tagDelim = '/', newLineDelim = True):
        self.endLine = stopTags
        self.lineDelim = newLineDelim
        self.wordDelim = wordDelim
        self.tagDelim = tagDelim
        self.endLineToken = "$END$"

    def parseFile (self, filePath):
        tags = []
        for item in self._parseFileWords(filePath):
            if item == self.endLineToken:
                yield tags
                tags = []
            else:
                tags.append(item)
        if tags:
            yield tags

    def parseTagsFromFile (self, filePath):
        return map(lambda t: t[-1],
            filter(lambda pair: pair != self.endLineToken,
                self._parseFileWords(filePath)))

    def parseWordsFromFile (self, filePath):
        return map(lambda t: t[:-1],
            filter(lambda pair: pair != self.endLineToken,
                self._parseFileWords(filePath)))

    # def _parseFromFile(self, filePath, index):
    #     return map(lambda t: t[-1],
    #         filter(lambda pair: pair != self.endLineToken,
    #             self._parseFileWords(filePath)))

    def parseAllFromFile (self, filePath):
        return filter(lambda pair: pair != self.endLineToken,
                          self._parseFileWords(filePath))

    def _parseFileWords (self, filePath):
        with open(filePath) as f:
            for line in f:
                t = re.split(f"[{self.wordDelim}]", line.strip())
                for word in t:
                    pair = self.processWord(word)
                    yield pair
                    if pair[-1] in self.endLine:
                        yield self.endLineToken
                if self.lineDelim:
                    yield self.endLineToken

    def processWord (self, word):
        return tuple(word.rsplit(self.tagDelim, 1))


class TestParser(TagsParser):
    def __init__ (self, wordDelim = " ", splitWord = False):
        super().__init__(wordDelim = wordDelim, newLineDelim = True, stopTags = ())
        self.splitWord = splitWord

    def processWord (self, word):
        if self.splitWord:
            return "".join(word.rsplit(self.tagDelim, 1)[:-1])
        return word


class StorageParser:
    def __init__ (self, wordDelim = " ", valueDelim = "\t"):
        self.wordDelim = wordDelim
        self.valueDelim = valueDelim

    def Load (self, filePath):
        if not filePath:
            return
        with open(filePath) as f:
            for line in map(lambda x: x.split(self.valueDelim), f):
                yield tuple(line[0].split(self.wordDelim)), int(line[-1])

    def Save (self, filePath, items):
        with open(filePath, 'w') as f:
            for tags in items:
                f.write(f"{self.wordDelim.join(tags[:-1])}{self.valueDelim}{tags[-1]}\n")


class OutParser:
    def __init__ (self, filePath, wordDelim = " ", tagDelim = "/"):
        self.filePath = filePath
        self.tagDelim = tagDelim
        self.wordDelim = wordDelim

    def __enter__ (self):
        self.fd = open(self.filePath, 'w')
        return self

    def printLine (self, resultLine):
        self.fd.write(self.wordDelim.join(self.tagDelim.join(wordTag) for wordTag in resultLine))
        self.fd.write("\n")

    def __exit__ (self, exc_type, exc_val, exc_tb):
        self.fd.close()


class MappingParser:
    seperator = "------------------"
    col_delim = ' '
    feat_val_delim = '='
    label_delim = ' '
    converted_delim = ':'

    @staticmethod
    def TagFeatToString (tag, features):
        features = MappingParser.featuresToString(features)
        return f"{tag}{MappingParser.label_delim}{features}"

    @staticmethod
    def featuresToString(features):
        return MappingParser.col_delim.join(features)

    @staticmethod
    def featureValue(feature, value):
        return f"{feature}{MappingParser.feat_val_delim}{value}"

    @staticmethod
    def splitFeatures(features):
        return features.split(MappingParser.col_delim)

    @staticmethod
    def TagVecToString (tag, featVect):
        return "{tag}{delim}{features}\n".format(
            tag = tag,
            delim = MappingParser.label_delim,
            features = MappingParser.col_delim.join((
                f"{num}{MappingParser.converted_delim}1" for num in featVect
            ))
        )

    @staticmethod
    def saveDictsToFile (dicts, outFile):
        with open(outFile, "w") as fOut:
            for dict_t in dicts:
                for key, value in dict_t.items():
                    fOut.write(f"{key}{MappingParser.col_delim}{value}\n")
                fOut.write(f"{MappingParser.seperator}\n")
    @staticmethod
    def getDictsFromFile (inFile):
        with open(inFile) as fIn:
            resultDict = {}
            for line in fIn:
                line = line.strip()
                if line == MappingParser.seperator:
                    yield resultDict
                    resultDict = {}
                else:
                    key, val = line.rsplit(MappingParser.col_delim, 1)
                    resultDict[key] = int(val)

    @staticmethod
    def splitTagFeatures(line):
        return line.split(MappingParser.label_delim, 1)


    # @staticmethod
    # def splitConvTagFeatures(line):
    #     #tag, features =
    #     return line.split(MappingParser.label_delim, 1)