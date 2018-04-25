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
    def __init__ (self, wordDelim = " "):
        super().__init__(wordDelim=wordDelim, newLineDelim=True, stopTags=())

    def processWord(self, word):
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


class OutputParser:
    def __init__ (self, filePath, wordDelim = " ", tagDelim = "/", threshold = 64):
        self.threshold = threshold
        self.filePath = filePath
        self.tagDelim = tagDelim
        self.wordDelim = wordDelim
        self.first = True
        self.buff = deque()

    def __enter__ (self):
        self.fd = open(self.filePath, 'w')
        return self

    def __exit__ (self, exc_type, exc_val, exc_tb):
        self.flush()
        self.fd.close()

    def append (self, word, tag):
        if not self.first:
            self.buff.append(self.wordDelim)
        self.buff.append(f"{word}{self.tagDelim}{tag}")
        self.first = False
        self._flushIfNeeded()

    def breakLine (self):
        self.first = True
        self.buff.append("\n")
        self._flushIfNeeded()

    def flush (self):
        self.fd.write(''.join(self.buff))
        self.buff = deque()

    def _flushIfNeeded (self):
        if len(self.buff) > self.threshold:
            self.flush()
