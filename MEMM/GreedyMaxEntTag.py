import sys

from utils.measuretime import measure
from utils.parsers import MappingParser, TestParser, OutParser
from MEMM.MEMMTaggers import GreedyTagger
import MEMM.Features as Features
import inspect

@measure
def main(inputf, modelfile, mapfile, outfilename):
    tagger = GreedyTagger(inspect.getmembers(Features, inspect.isfunction))
    tagger.loadModelFromFile(modelfile)
    featuresDict, tagsDict = tuple(MappingParser.getDictsFromFile(mapfile))
    tagger.setParams(tagsDict, featuresDict)

    with OutParser(outfilename) as outF:
        for i, wordsLine in enumerate(TestParser().parseFile(inputf)):
            outF.printLine(tagger.tagLine(wordsLine))

if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, out_file_name = sys.argv[1:5]

    main(input_file_name, modelname, feature_map_file, out_file_name)
