import sys
from utils.parsers import TestParser, OutParser
from MEMM.MEMMTaggers import GreedyTagger
import MEMM.Features as Features
import inspect


def main(infile, modelfile, mapfile, outfile):
    tagger = GreedyTagger(inspect.getmembers(Features, inspect.isfunction))
    tagger.loadParams(mapfile, modelfile)
    with OutParser(outfile) as outF:
        for wordsLine in TestParser().parseFile(infile):
            outF.printLine(tagger.tagLine(wordsLine))


if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, out_file_name = sys.argv[1:5]
    main(input_file_name, modelname, feature_map_file, out_file_name)
