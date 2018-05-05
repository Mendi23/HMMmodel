import regex as re


class ExtractRegexFeatures:

    def __init__ (self):
        self.startTag = "start"
        self.featuresRegex = re.compile("""
        {wd}
        [^{wd}]*                #first Word
        {td}
        (?P<tpt>[^{wd}]+)   #first tag
        {wd}
        [^{wd}]*                #second word
        {td}
        (?P<pt>[^{wd}]+)    #second tag
        {wd}
        (?P<word>(?P<startWith>.)[^{wd}]*?(?P<suffix>[^{wd}]{{,3}}))        #word
        {td}
        (?P<tag>[^{wd}]+)    #tag
        """.format(td = '/', wd = "\s"), re.VERBOSE)

    def getFeatures (self, line):
        return self.featuresRegex.match(line).groupdict()

    def getFeaturesFromFile (self, filePath):
        # r = re.compile(
        #     """{wd}[^{wd}]*{td}[^{wd}]+{wd}[^{wd}]*{td}[^{wd}]+{wd}.[^{wd}]{{,3}}{td}[^{wd}]+""".format(
        #         td = '/', wd = "\s"), re.VERBOSE)
        with open(filePath) as f:
            startTags = " /{st} /{st} ".format(st = self.startTag)
            for line in f:
                #print(startTags + line)
                it = self.featuresRegex.finditer(startTags + line, overlapped = True)
                for y in it:
                    yield y.groupdict()


h = ExtractRegexFeatures()
for x in h.getFeaturesFromFile("./DataSets/ass1-small-train"):
    print(x)

    # print(x)

    #
    # params = {}
    #
    # tokens = {"word": "",}
    #
    # testVector = fe.CountVectorizer("string", analyzer = "word", lowercase = False, token_pattern =
    # "[^ ]*/[^ ]*", ngram_range = (1, 3), preprocessor = lambda x: "/START /START " + x)
    # with open("./DataSets/ass1-small-train") as f:
    #     testVector.fit_transform(f)
    #
    # for k in testVector.get_feature_names():
    #      print(k)
    #
    #
    # def my_preprocessor (x):
    #     return lambda x: "/START /START" + x

    # class FeatureVector(fe.CountVectorizer):
