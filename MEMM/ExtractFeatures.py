import MEMM.Features as Features
import parsers
import inspect

import sys

def processLine(allFeatures, words, tags):
    for i in range(len(line)):
        featuresvals = ({"name": feature[0], "val": feature[1](words, tags, i)}
                        for feature in allFeatures)

        yield "{tag} {features}".format(
            tag=tags[i],
            features=' '.join("{name}={val}".format(**v) for v in featuresvals))

if __name__ == '__main__':
    allFeatures = inspect.getmembers(Features, inspect.isfunction)

    # inputfilename, outputfilename = sys.argv[1:]
    inputfilename, outputfilename = "../DataSets/ass1-tagger-train", "features_file"

    with open(outputfilename, "w") as output:
        for line in parsers.TagsParser().parseFile(inputfilename):
            words, tags = tuple(zip(*line))
            for out in processLine(allFeatures, words, tags):
                output.write(f"{out}\n")

