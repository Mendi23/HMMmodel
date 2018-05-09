# for importing local modules
import sys
from inspect import currentframe, getmembers, isfunction, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

import sys
from scripts_t.measuretime import measure
from utils.parsers import TestParser, OutParser
from MEMM.MEMMTaggers import ViterbiTrigramTagger
import MEMM.Features as Features


@measure
def main(inputf, modelfile, mapfile, outfilename):
    tagger = ViterbiTrigramTagger(featuresFuncs=getmembers(Features, isfunction))
    tagger.loadParams(mapfile, modelfile)

    with OutParser(outfilename) as outF:
        for i, wordsLine in enumerate(TestParser().parseFile(inputf)):
            outF.printLine(tagger.tagLine(wordsLine))

    print(f"proba: {tagger._getProba.cache_info()}")
    print(f"posta: {tagger.getPossibleTagsForWord.cache_info()}")


if __name__ == '__main__':
    input_file_name, modelname, feature_map_file, out_file_name = sys.argv[1:5]
    main(input_file_name, modelname, feature_map_file, out_file_name)
