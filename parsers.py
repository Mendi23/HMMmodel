import re


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
                yield from re.split(f"[{self.wordDelim}]", line.strip())


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
    def __init__(self, filePath, wordDelim=" ", tagDelim="/"):
        self.filePath = filePath
        self.tagDelim = tagDelim
        self.wordDelim = wordDelim

    def __enter__(self):
        self.fd = open(self.filePath, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fd.close()

    def append(self, word, tag, first=False):
        if not first:
            self.fd.write(self.wordDelim)
        self.fd.write(f"{word}{self.tagDelim}{tag}")

    def breakLine(self):
        self.fd.write("\n")
