# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join
from MEMM.MEMMTaggers import MemmTagger
from utils.measuretime import measure

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

import MEMM.Features as Features
from utils.parsers import TagsParser
import inspect

@measure
def main(inf, outf):
    global tag, features
    t = MemmTagger(inspect.getmembers(Features, inspect.isfunction))
    with open(outf, "w") as output:
        for line in TagsParser().parseFile(inf):
            for tagFeatures in t.extractStringFromTaggedLine(line):
                output.write(tagFeatures + '\n')


if __name__ == '__main__':
    inputfilename, outputfilename = sys.argv[1:3]
    main(inputfilename, outputfilename)
