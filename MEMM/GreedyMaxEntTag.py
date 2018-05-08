# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

import sys

from utils.measuretime import measure
from utils.parsers import TestParser, OutParser
from MEMM.MEMMTaggers import GreedyTagger
import MEMM.Features as Features
import inspect

@measure
def main(infile, modelfile, mapfile, outfile):
    tagger = GreedyTagger(inspect.getmembers(Features, inspect.isfunction))
    tagger.loadParams(mapfile, modelfile)
    with OutParser(outfile) as outF:
        for wordsLine in TestParser().parseFile(infile):
            outF.printLine(tagger.tagLine(wordsLine))


if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, out_file_name = sys.argv[1:5]
    main(input_file_name, modelname, feature_map_file, out_file_name)
