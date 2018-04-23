import re


class TagsParser:
    def __init__(self, endLineTag=(".",), wordDelim=(" ",), tagDelim='/'):
        self.endLine = endLineTag
        self.wordDelim = ''.join(wordDelim)
        self.tagDelim = tagDelim

    def parseFile(self, filePath):
        tags = []
        with open(filePath) as f:
            for line in f:
                t = re.split(f"[{self.wordDelim}]", line.strip())
                for word in t:
                    tagPair = tuple(word.rsplit(self.tagDelim, 1))
                    tags.append(tagPair)
                    if tagPair[-1] in self.endLine:
                        yield tags
                        tags = []


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
