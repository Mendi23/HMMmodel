# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join
root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
     sys.path.insert(0, root_path)
# -------------------------

import MEMM.Features as Features
import parsers
import inspect


def processLine(allFeatures, words, tags):
    for i in range(len(line)):
        featuresvals = ({"name": feature[0], "val": feature[1](words, tags, i)}
                        for feature in allFeatures)

        yield "{tag} {features}".format(
            tag=tags[i],
            features=' '.join("{name}={val}".format(**v) for v in featuresvals))

if __name__ == '__main__':
    allFeatures = inspect.getmembers(Features, inspect.isfunction)

    inputfilename, outputfilename = sys.argv[1:]

    with open(outputfilename, "w") as output:
        for line in parsers.TagsParser().parseFile(inputfilename):
            words, tags = tuple(zip(*line))
            for out in processLine(allFeatures, words, tags):
                output.write(f"{out}\n")

