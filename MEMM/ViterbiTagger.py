import sys
from timeit import timeit

from utils.hmmModel import HmmModel
from utils.measuretime import measure
from utils.parsers import MappingParser, TestParser, OutParser
from MEMM.MEMMTaggers import ViterbiTrigramTagger
import MEMM.Features as Features
import inspect


@measure
def main(inputf, modelfile, mapfile, outfilename, emle):
    x = HmmModel(2)
    x.loadTransitions(None, emle)

    tagger = ViterbiTrigramTagger(
        featuresFuncs=inspect.getmembers(Features, inspect.isfunction),
        hmmModel=x)
    tagger.loadParams(mapfile, modelfile)

    with OutParser(outfilename) as outF:
        for i, wordsLine in enumerate(TestParser().parseFile(inputf)):
            outF.printLine(tagger.tagLine(wordsLine))


if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, e_mle_file, out_file_name = sys.argv[1:6]
    main(input_file_name, modelname, feature_map_file, out_file_name, e_mle_file)
