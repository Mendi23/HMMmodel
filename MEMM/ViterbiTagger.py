# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

import sys

from utils.hmmModel import HmmModel
from utils.measuretime import measure
from utils.parsers import TestParser, OutParser
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

    print(tagger._getProba.cache_info())

if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, e_mle_file, out_file_name = sys.argv[1:6]
    main(input_file_name, modelname, feature_map_file, out_file_name, e_mle_file)
