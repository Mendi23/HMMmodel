import sys
from utils.parsers import MappingParser, TestParser, OutParser
from MEMM.MEMMTaggers import GreedyTagger
import MEMM.Features as Features
import inspect

if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, out_file_name = sys.argv[1:]
    tagger = GreedyTagger(inspect.getmembers(Features, inspect.isfunction))
    tagger.loadModelFromFile(modelname)
    featuresDict, tagsDict = tuple(MappingParser.getDictsFromFile(feature_map_file))
    tagger.setParams(tagsDict, featuresDict)

    with OutParser(out_file_name) as outF:
        for wordsLine in TestParser().parseFile(input_file_name):
            outF.printLine(tagger.tagLine(wordsLine))