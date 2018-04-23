import re
from collections import deque


class TagsParser:
    def __init__(self, endLineTag=(".",), wordDelim=" ", tagDelim='/'):
        self.endLine = endLineTag
        self.wordDelim = wordDelim
        self.tagDelim = tagDelim

    def parseFile(self, filePath):
        tags = []
        with open(filePath) as f:
            for line in f:
                t = re.split(f"[{self.wordDelim}]", line.strip())
                for word in t:
                    processedWord = self.processWord(word)
                    tags.append(processedWord)
                    if self.isEndLine(processedWord):
                        yield tags
                        tags = []

    def processWord(self, word):
        return tuple(word.rsplit(self.tagDelim, 1))

    def isEndLine(self, tagsPair):
        return tagsPair[-1] in self.endLine


class TestParser():
    def __init__(self, wordDelim=" "):
        self.wordDelim = wordDelim

    def parseFile(self, filePath):
        with open(filePath) as f:
            for line in f:
                yield re.split(f"[{self.wordDelim}]", line.strip())


class StorageParser:
    def __init__(self, wordDelim=" ", valueDelim="\t"):
        self.wordDelim = wordDelim
        self.valueDelim = valueDelim

    def Load(self, filePath):
        if not filePath:
            return
        with open(filePath) as f:
            for line in map(lambda x: x.split(self.valueDelim), f):
                yield tuple(line[0].split(self.wordDelim)), int(line[-1])

    def Save(self, filePath, items):
        with open(filePath, 'w') as f:
            for tags in items:
                f.write(f"{self.wordDelim.join(tags[:-1])}{self.valueDelim}{tags[-1]}\n")


class OutputParser:
    def __init__(self, filePath, wordDelim=" ", tagDelim="/", threshold=64):
        self.threshold = threshold
        self.filePath = filePath
        self.tagDelim = tagDelim
        self.wordDelim = wordDelim
        self.first = True
        self.buff = deque()

    def __enter__(self):
        self.fd = open(self.filePath, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.fd.close()

    def append(self, word, tag):
        if not self.first:
            self.buff.append(self.wordDelim)
        self.buff.append(f"{word}{self.tagDelim}{tag}")
        self.first = False
        self._flushIfNeeded()

    def breakLine(self):
        self.first = True
        self.buff.append("\n")
        self._flushIfNeeded()

    def flush(self):
        self.fd.write(''.join(self.buff))
        self.buff = deque()

    def _flushIfNeeded(self):
        if len(self.buff) > self.threshold:
            self.flush()
