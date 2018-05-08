
import sys
from timeit import timeit

from utils.hmmModel import HmmModel
from utils.measuretime import measure
from utils.parsers import MappingParser, TestParser, OutParser
from MEMM.MEMMTaggers import ViterbiTrigramTagger, ViterbiTrigramTagger_other, ViterbiTrigramTagger_other2
import MEMM.Features as Features
import inspect

@measure
def main(inputf, modelfile, mapfile, outfilename, emle, TAGGER):
    x = HmmModel(2)
    x.loadTransitions(None, emle)

    tagger = TAGGER(featuresFuncs=inspect.getmembers(Features, inspect.isfunction),
                    hmmModel=x)
    tagger.loadModelFromFile(modelfile)
    featuresDict, tagsDict = tuple(MappingParser.getDictsFromFile(mapfile))
    tagger.setParams(tagsDict, featuresDict)

    with OutParser(outfilename) as outF:
        for i, wordsLine in enumerate(TestParser().parseFile(inputf)):
            outF.printLine(tagger.tagLine(wordsLine))
            if i % 10 == 0: outF.fd.flush()


if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, e_mle_file, out_file_name = sys.argv[1:6]

    main(input_file_name, modelname, feature_map_file, out_file_name, e_mle_file, ViterbiTrigramTagger)
